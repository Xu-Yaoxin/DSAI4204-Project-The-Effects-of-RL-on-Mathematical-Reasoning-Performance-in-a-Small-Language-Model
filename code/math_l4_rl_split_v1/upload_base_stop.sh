#!/usr/bin/env bash
set -euo pipefail
TRANSFER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER_PORT="${1:-42172}"
TRANSFER_HOST="${2:-root@xj-member.bitahub.com}"
TRANSFER_KEY="${RLVR_SSH_KEY:-$HOME/.ssh/dsai4207_4090}"
if [[ ! "$TRANSFER_PORT" =~ ^[0-9]+$ ]] || [[ ! "$TRANSFER_HOST" =~ ^[A-Za-z0-9_.@-]+$ ]]; then
    printf '%s\n' 'Usage: bash upload_base_stop.sh [PORT [USER@HOST]]' >&2
    exit 2
fi
for TRANSFER_REQUIRED in scripts/watch_base_only.py scripts/switch_after_base.py scripts/pidfd_compat.py reference/base_manifest.json; do
    if [[ ! -f "$TRANSFER_ROOT/$TRANSFER_REQUIRED" ]]; then
        printf 'Missing control file: %s\n' "$TRANSFER_REQUIRED" >&2
        exit 2
    fi
done
printf -v TRANSFER_SSH 'ssh -p %q -i %q' "$TRANSFER_PORT" "$TRANSFER_KEY"
printf '%s\n' '[UPLOAD BASE CONTROL] Sending only external stop-after-Base tools and references.'
rsync -avh --progress --partial \
    --include='/scripts/' \
    --include='/scripts/watch_base_only.py' \
    --include='/scripts/switch_after_base.py' \
    --include='/scripts/pidfd_compat.py' \
    --include='/reference/***' \
    --exclude='*' \
    -e "$TRANSFER_SSH" \
    --rsync-path='mkdir -p /root/code/math_l4_rl_split_v1 && rsync' \
    "$TRANSFER_ROOT/" "$TRANSFER_HOST:/root/code/math_l4_rl_split_v1/"
printf '[UPLOAD COMPLETE] %s port=%s | Base code, answers and running process were not changed.\n' "$TRANSFER_HOST" "$TRANSFER_PORT"
printf '%s\n' 'Start scripts/watch_base_only.py on the existing server to arm the stop-after-Base controller.'
