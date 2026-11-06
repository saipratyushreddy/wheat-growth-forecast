#!/usr/bin/env bash
# From the laptop: copy Swan's results (not in git; default outputs/week4, including SLURM logs) back with ONE login.
#   scripts/swan/pull_from_swan.sh                 # all of outputs/week4 (the current week)
#   scripts/swan/pull_from_swan.sh outputs/week2/settings.json 'outputs/week2/slurm/*'
set -euo pipefail
HOST="${SWAN_HOST:-swan}"
REMOTE_DIR="${SWAN_DIR:-/lustre/work/cseguo/skasara2/wheat-growth-forecast}"
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$HOME/.ssh/sockets"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/sockets/swan-wheat-%C -o ControlPersist=10m"
FILES=("$@")
[ "${#FILES[@]}" -gt 0 ] || FILES=(outputs/week4/)
ssh $SSH_OPTS "$HOST" true
for f in "${FILES[@]}"; do
    dest="$LOCAL_DIR/$(dirname "${f%/}")"
    mkdir -p "$dest"
    rsync -av -e "ssh $SSH_OPTS" "$HOST:$REMOTE_DIR/${f%/}" "$dest/"
done
ssh $SSH_OPTS -O exit "$HOST" 2>/dev/null || true
echo "Done."
