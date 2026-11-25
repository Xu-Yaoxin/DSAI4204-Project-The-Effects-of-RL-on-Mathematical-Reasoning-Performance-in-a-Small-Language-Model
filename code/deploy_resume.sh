#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
RLVR_DEPLOY_DIR="$(mktemp -d /tmp/rlvr-deploy.XXXXXX)"
RLVR_DEPLOY_SOCKET="$RLVR_DEPLOY_DIR/ssh"
RLVR_SSH_KEY="${RLVR_SSH_KEY:-$HOME/.ssh/dsai4207_4090}"
RLVR_REMOTE="root@xj-member.bitahub.com"
trap 'ssh -S "$RLVR_DEPLOY_SOCKET" -O exit "$RLVR_REMOTE" >/dev/null 2>&1 || true; rm -f "$RLVR_DEPLOY_SOCKET"; rmdir "$RLVR_DEPLOY_DIR" 2>/dev/null || true' EXIT
printf '%s\n' '[DEPLOY 1/4] Connect to SSH port 42258'
ssh -M -S "$RLVR_DEPLOY_SOCKET" -o ControlPersist=600 -fNT \
    -p 42258 -i "$RLVR_SSH_KEY" "$RLVR_REMOTE"
printf '%s\n' '[DEPLOY 2/4] Stop this project before changing active source files'
scp -o "ControlPath=$RLVR_DEPLOY_SOCKET" -P 42258 \
    scripts/stop_run.py "$RLVR_REMOTE:/root/code/scripts/stop_run.py"
ssh -S "$RLVR_DEPLOY_SOCKET" -p 42258 "$RLVR_REMOTE" \
    'cd /root/code && .venv/bin/python scripts/stop_run.py --root /root/code'
printf '%s\n' '[DEPLOY 3/4] Upload code and manifests; preserve server environments and outputs'
rsync -avh --progress --partial \
    --exclude='.venv/' --exclude='.bootstrap/' --exclude='__pycache__/' \
    --exclude='*.pyc' --exclude='.DS_Store' --exclude='*.log' --exclude='logs/' \
    --exclude='outputs/' --exclude='packages/linux-cp311-cu128/wheels/' \
    --exclude='models/Qwen3.5-2B-Base/' \
    -e "ssh -S $RLVR_DEPLOY_SOCKET -p 42258" \
    ./ "$RLVR_REMOTE:/root/code/"
printf '%s\n' '[DEPLOY 4/4] Preserve legacy runs and restart with checkpoint recovery'
ssh -S "$RLVR_DEPLOY_SOCKET" -p 42258 "$RLVR_REMOTE" \
    'cd /root/code && bash resume_run.sh'
printf '%s\n' '[DEPLOY COMPLETE] The server pipeline is running in the background.'
