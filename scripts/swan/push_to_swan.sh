#!/usr/bin/env bash
# From the laptop: copy the repo (code, .git, data/derived, data/fip1 tables, outputs/week1) to Swan with ONE login.
# Override with SWAN_HOST / SWAN_DIR. Never copies .venv or outputs/week2 (Swan's results are pulled, not pushed).
set -euo pipefail
HOST="${SWAN_HOST:-swan}"
REMOTE_DIR="${SWAN_DIR:-/lustre/work/cseguo/skasara2/wheat-growth-forecast}"
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$HOME/.ssh/sockets"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/sockets/swan-wheat-%C -o ControlPersist=10m"
ssh $SSH_OPTS "$HOST" "mkdir -p '$REMOTE_DIR'"   # the one password / Duo prompt
rsync -av -e "ssh $SSH_OPTS" --exclude .venv --exclude __pycache__ --exclude 'outputs/week2/' \
    "$LOCAL_DIR/" "$HOST:$REMOTE_DIR/"
ssh $SSH_OPTS -O exit "$HOST" 2>/dev/null || true
echo "Pushed to $HOST:$REMOTE_DIR"
