#!/usr/bin/env bash
set -euo pipefail
TRANSFER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER_SHARD="${1:-}"
TRANSFER_PORT="${2:-}"
TRANSFER_HOST="${3:-root@xj-member.bitahub.com}"
TRANSFER_KEY="${RLVR_SSH_KEY:-$HOME/.ssh/dsai4207_4090}"
if [[ ! "$TRANSFER_SHARD" =~ ^[01]$ ]] || [[ ! "$TRANSFER_PORT" =~ ^[0-9]+$ ]] || [[ ! "$TRANSFER_HOST" =~ ^[A-Za-z0-9_.@-]+$ ]]; then
    printf '%s\n' 'Usage: bash upload_batch.sh SHARD_INDEX PORT [USER@HOST]; indices are 0 and 1.' >&2
    exit 2
fi
if [[ ! -f "$TRANSFER_ROOT/payload/ready.json" ]]; then
    printf '%s\n' 'Run download_current.sh first. It downloads missing selection evidence and verifies the upload payload.' >&2
    exit 2
fi
TRANSFER_MODEL="$(python3 -B "$TRANSFER_ROOT/scripts/build_payload.py" --verify-ready)"
printf -v TRANSFER_SSH 'ssh -p %q -i %q' "$TRANSFER_PORT" "$TRANSFER_KEY"
printf '[UPLOAD 1/3] shard=%s | complete Base model; remote existing files are reused.\n' "$TRANSFER_SHARD"
rsync -avh --progress --partial --exclude='.cache/' --exclude='__pycache__/' --exclude='.DS_Store' \
    -e "$TRANSFER_SSH" --rsync-path='mkdir -p /root/code/models/Qwen3.5-2B-Base && rsync' \
    "$TRANSFER_MODEL/" "$TRANSFER_HOST:/root/code/models/Qwen3.5-2B-Base/"
printf '%s\n' '[UPLOAD 2/3] Frozen experiment, data, selection evidence and dependency lock.'
rsync -avh --progress --partial -e "$TRANSFER_SSH" \
    --rsync-path='mkdir -p /root/code && rsync' \
    "$TRANSFER_ROOT/payload/code/" "$TRANSFER_HOST:/root/code/"
printf '%s\n' '[UPLOAD 3/3] Independent two-worker evaluator; current Base outputs are not uploaded.'
rsync -avh --progress --partial --exclude='/payload/' --exclude='/validation/' \
    --exclude='/outputs/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='.DS_Store' \
    -e "$TRANSFER_SSH" --rsync-path='mkdir -p /root/code/math_l4_rl_split_v1 && rsync' \
    "$TRANSFER_ROOT/" "$TRANSFER_HOST:/root/code/math_l4_rl_split_v1/"
printf '[UPLOAD COMPLETE] shard=%s | %s | port=%s\n' "$TRANSFER_SHARD" "$TRANSFER_HOST" "$TRANSFER_PORT"
printf '%s\n' 'On that server: cd /root/code/math_l4_rl_split_v1'
printf 'Then follow COMMANDS.md to run shard %s on this dedicated RL server. The original Base server uses watch_base_only.py.\n' "$TRANSFER_SHARD"
