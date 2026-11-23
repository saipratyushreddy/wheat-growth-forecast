"""
Cover read-out (docs/08-generation-design.md): a ResNet-18 (ImageNet weights, microsoft/resnet-18) fine-tuned to predict
a plot's measured canopy cover from its 240 x 160 thumbnail. One per fold, trained on the fold's training years' photos
only. Gate: on real photos of the held-out year, R2 >= 0.90 and MAE <= 5 points of cover.
"""
import numpy as np
import torch
from torch import nn

import gen_data as GD

REPO = "microsoft/resnet-18"
REVISION = "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
EPOCHS, BATCH, LR = 3, 128, 3e-4
GATE_R2, GATE_MAE = 0.90, 5.0  # MAE in points of cover


class Estimator(nn.Module):
    def __init__(self, tiny=False):
        super().__init__()
        from transformers import ResNetConfig, ResNetModel
        if tiny:  # random weights, for code checks only
            self.backbone = ResNetModel(ResNetConfig(embedding_size=8, hidden_sizes=[8, 8, 16, 16], depths=[1, 1, 1, 1]))
            n = 16
        else:
            self.backbone = ResNetModel.from_pretrained(REPO, revision=REVISION)
            n = 512
        self.head = nn.Linear(n, 1)

    def forward(self, x):
        """x: uint8 (B, 160, 240, 3) -> cover (unclipped)."""
        x = (x.permute(0, 3, 1, 2).float() / 255 - MEAN.to(x.device)) / STD.to(x.device)
        return self.head(self.backbone(pixel_values=x).pooler_output.flatten(1)).squeeze(1)


class JpegSet(torch.utils.data.Dataset):
    def __init__(self, jpegs, target=None, flip=False):
        self.jpegs, self.target, self.flip = jpegs, target, flip

    def __len__(self):
        return len(self.jpegs)

    def __getitem__(self, i):
        x = GD.decode(self.jpegs[i])
        if self.flip and np.random.rand() < 0.5:
            x = x[:, ::-1].copy()
        return torch.from_numpy(x), (np.float32(self.target[i]) if self.target is not None else np.float32(0))


def train(jpegs, cover, device, workers=8, seed=0, epochs=EPOCHS, batch=BATCH, tiny=False, log=print):
    """Fit on photos with a measured cover (cover in 0-1)."""
    torch.manual_seed(seed)
    est = Estimator(tiny).to(device).train()
    opt = torch.optim.AdamW(est.parameters(), lr=LR, weight_decay=0.01)
    dl = torch.utils.data.DataLoader(JpegSet(jpegs, np.clip(cover, 0, 1), flip=True), batch_size=batch, shuffle=True,
                                     num_workers=workers, multiprocessing_context="fork" if workers else None, drop_last=True, generator=torch.Generator().manual_seed(seed))
    steps = epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps)
    use_amp = device.type == "cuda"
    k = 0
    for ep in range(epochs):
        for x, y in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16,
                                enabled=use_amp):
                loss = torch.nn.functional.l1_loss(est(x.to(device)).float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            k += 1
            if k % 200 == 0:
                log("estimator epoch", ep + 1, "step", k, "/", steps, "L1", round(float(loss), 4))
    return est.eval()


@torch.no_grad()
def predict_uint8(est, arr, device, batch=256):
    """Cover (clipped to 0-1) for an array of uint8 images (n, 160, 240, 3)."""
    out = []
    for i in range(0, len(arr), batch):
        out.append(est(torch.from_numpy(arr[i:i + batch]).to(device)).float().clamp(0, 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


@torch.no_grad()
def predict_jpegs(est, jpegs, device, workers=8, batch=256):
    dl = torch.utils.data.DataLoader(JpegSet(jpegs), batch_size=batch, shuffle=False, num_workers=workers,
                                     multiprocessing_context="fork" if workers else None)
    return np.concatenate([est(x.to(device)).float().clamp(0, 1).cpu().numpy() for x, _ in dl]) if len(jpegs) else \
        np.zeros(0, np.float32)


def gate(cover_hat, cover):
    """R2 and MAE (points of cover) of the estimator on real photos with a measured cover."""
    cover_hat, cover = np.asarray(cover_hat, float), np.clip(np.asarray(cover, float), 0, 1)
    ss = np.sum((cover - cover.mean()) ** 2)
    r2 = 1 - np.sum((cover - cover_hat) ** 2) / ss
    mae = 100 * np.mean(np.abs(cover - cover_hat))
    return dict(r2=float(r2), mae=float(mae), n=int(len(cover)), passed=bool(r2 >= GATE_R2 and mae <= GATE_MAE))
