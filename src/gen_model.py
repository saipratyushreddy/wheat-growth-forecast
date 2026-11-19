"""
Week-6 generator (docs/08-generation-design.md): InstructPix2Pix's UNet, fine-tuned to turn a plot's latest photo into
its photo at a later day, with the text conditioning replaced by a small network on the numbers (cover, height, ...).

The photos are encoded once by the frozen VAE (src/run_week6.py prep); training and sampling work on latents of size
4 x 20 x 30 (a 240 x 160 photo). Settings are fixed here, not tuned.
"""
import os

import numpy as np
import torch
from torch import nn

from gen_data import COND_DIM

REPO = "timbrooks/instruct-pix2pix"
REVISION = "31519b5cb02a7fd89b906d88731cd4d6a7bbf88d"
SCALE = 0.18215  # Stable Diffusion's latent scaling
TOKENS = 8  # conditioning tokens made from the numbers
COND_DROP = 0.1  # share of training steps where the numbers are replaced by a learned "no condition" (for guidance)
GUIDANCE = 3.0  # sampling: eps = eps(no condition) + GUIDANCE * (eps(condition) - eps(no condition)); a conventional value
SAMPLE_STEPS = 30
SCHED = dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
LR, LR_NUMBERS, WARMUP, BATCH, CLIP = 5e-5, 5e-4, 200, 32, 1.0


class NumberNet(nn.Module):
    """The numbers -> TOKENS conditioning tokens of size dim (the UNet's cross-attention input)."""

    def __init__(self, dim=768, tokens=TOKENS, n_in=COND_DIM, hidden=512):
        super().__init__()
        self.tokens, self.dim = tokens, dim
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, tokens * dim))
        self.null = nn.Parameter(torch.zeros(tokens, dim))

    def forward(self, c, drop=None):
        t = self.net(c).view(-1, self.tokens, self.dim)
        if drop is not None:
            t = torch.where(drop.view(-1, 1, 1), self.null.expand_as(t), t)
        return t


# ---------------------------------------------------------------- pretrained parts

def load_unet():
    from diffusers import UNet2DConditionModel
    return UNet2DConditionModel.from_pretrained(REPO, subfolder="unet", revision=REVISION)


def load_vae():
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(REPO, subfolder="vae", revision=REVISION).eval()


def tiny_unet(dim=32):
    """A small random UNet with the same interface (8 input channels), for code checks only."""
    from diffusers import UNet2DConditionModel
    return UNet2DConditionModel(sample_size=(20, 30), in_channels=8, out_channels=4, layers_per_block=1,
                                block_out_channels=(32, 32, 64, 64), cross_attention_dim=dim, attention_head_dim=8)


@torch.no_grad()
def encode(vae, imgs):
    """uint8 (B, 160, 240, 3) -> latents (B, 4, 20, 30): the mean of the VAE posterior, scaled."""
    x = imgs.permute(0, 3, 1, 2).float() / 127.5 - 1
    return vae.encode(x).latent_dist.mean * SCALE


@torch.no_grad()
def decode(vae, lat):
    x = vae.decode(lat / SCALE).sample
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1)


# ---------------------------------------------------------------- training

def _amp(device):
    if device.type != "cuda":
        return dict(device_type="cpu", enabled=False), None
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return dict(device_type="cuda", dtype=dt), (torch.amp.GradScaler() if dt == torch.float16 else None)


def _loss(unet, numnet, sched, la, lb, c, gen, drop_p, device, amp):
    noise = torch.randn(lb.shape, generator=gen, device="cpu").to(device)
    t = torch.randint(0, sched.config.num_train_timesteps, (lb.shape[0],), generator=gen).to(device)
    noisy = sched.add_noise(lb, noise, t)
    drop = (torch.rand(lb.shape[0], generator=gen) < drop_p).to(device)
    with torch.autocast(**amp):
        tok = numnet(c, drop)
        pred = unet(torch.cat([noisy, la], 1), t, encoder_hidden_states=tok).sample
    return torch.nn.functional.mse_loss(pred.float(), noise)


def train(data, steps, ckpt, device, unet=None, numnet=None, seed=0, batch=BATCH, ckpt_every=2000, monitor=None,
          monitor_every=1000, log=print, drop_p=COND_DROP):
    """Fine-tune on `data.batch`. Resumable: the checkpoint holds the weights, the optimiser and the step, and every
    step draws its batch and noise from (seed, step), so a resumed run is the same run."""
    from diffusers import DDPMScheduler
    unet = (unet or load_unet()).to(device).train()
    numnet = (numnet or NumberNet(dim=unet.config.cross_attention_dim)).to(device).train()
    sched = DDPMScheduler(**SCHED)
    opt = torch.optim.AdamW([{"params": unet.parameters(), "lr": LR}, {"params": numnet.parameters(), "lr": LR_NUMBERS}],
                            weight_decay=0.01)
    amp, scaler = _amp(device)
    start = 0
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location="cpu")
        unet.load_state_dict(s["unet"])
        numnet.load_state_dict(s["numnet"])
        opt.load_state_dict(s["opt"])
        start = s["step"]
        log("resumed at step", start)
    base = [LR, LR_NUMBERS]  # not read from the optimiser: after a resume its rates are already warmed up
    for step in range(start, steps):
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * min(1.0, (step + 1) / WARMUP)
        rng, gen = np.random.default_rng([seed, step]), torch.Generator().manual_seed(seed * 1_000_003 + step)
        la, lb, c = (torch.from_numpy(x).to(device) for x in data.batch(rng, batch))
        loss = _loss(unet, numnet, sched, la, lb, c, gen, drop_p, device, amp)
        if not torch.isfinite(loss):
            raise RuntimeError(f"loss is {loss.item()} at step {step}")
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(list(unet.parameters()) + list(numnet.parameters()), CLIP)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(list(unet.parameters()) + list(numnet.parameters()), CLIP)
            opt.step()
        if (step + 1) % 100 == 0:
            log("step", step + 1, "loss", round(float(loss), 4))
        if monitor is not None and (step + 1) % monitor_every == 0:
            unet.eval(); numnet.eval()
            with torch.no_grad():
                g2, rng2 = torch.Generator().manual_seed(12345), np.random.default_rng(12345)
                la, lb, c = (torch.from_numpy(x).to(device) for x in monitor.batch(rng2, 64))
                log("step", step + 1, "monitor loss (held-out plots of the training years)",
                    round(float(_loss(unet, numnet, sched, la, lb, c, g2, 0.0, device, amp)), 4))
            unet.train(); numnet.train()
        if (step + 1) % ckpt_every == 0 or step + 1 == steps:
            torch.save({"unet": unet.state_dict(), "numnet": numnet.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    return unet.eval(), numnet.eval()


def load_trained(ckpt, device, unet=None, numnet=None):
    unet = unet or load_unet()
    numnet = numnet or NumberNet(dim=unet.config.cross_attention_dim)
    s = torch.load(ckpt, map_location="cpu")
    unet.load_state_dict(s["unet"])
    numnet.load_state_dict(s["numnet"])
    return unet.to(device).eval(), numnet.to(device).eval(), s["step"]


# ---------------------------------------------------------------- sampling

@torch.no_grad()
def sample(unet, numnet, lat_a, cond, seeds, device, steps=SAMPLE_STEPS, guidance=GUIDANCE):
    """Latents of the generated photos. Each sample's noise comes from its own seed, so a result does not depend on
    which other samples share its batch."""
    from diffusers import DDIMScheduler
    sched = DDIMScheduler(**SCHED, clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    sched.set_timesteps(steps)
    x = torch.stack([torch.randn(lat_a.shape[1:], generator=torch.Generator().manual_seed(int(s))) for s in seeds]).to(device)
    lat_a, cond = lat_a.to(device), cond.to(device)
    tok_c = numnet(cond)
    tok_u = numnet(cond, torch.ones(len(cond), dtype=torch.bool, device=device))
    amp, _ = _amp(device)
    for t in sched.timesteps:
        inp = torch.cat([x, lat_a], 1)
        with torch.autocast(**amp):
            eps = unet(torch.cat([inp, inp]), t, encoder_hidden_states=torch.cat([tok_c, tok_u])).sample.float()
        e_c, e_u = eps.chunk(2)
        x = sched.step(e_u + guidance * (e_c - e_u), t, x).prev_sample
    return x
