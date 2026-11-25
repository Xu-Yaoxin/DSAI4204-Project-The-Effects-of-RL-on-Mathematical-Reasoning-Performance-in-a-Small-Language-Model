#!/usr/bin/env bash
set -euo pipefail
TRANSFER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER_PORT="${1:-42172}"
TRANSFER_HOST="${2:-root@xj-member.bitahub.com}"
TRANSFER_TARGET="${3:-$HOME/Downloads/math_l4_selected_20261104}"
TRANSFER_KEY="${RLVR_SSH_KEY:-$HOME/.ssh/dsai4207_4090}"
if [[ ! "$TRANSFER_PORT" =~ ^[0-9]+$ ]] || [[ ! "$TRANSFER_HOST" =~ ^[A-Za-z0-9_.@-]+$ ]]; then
    printf '%s\n' 'Usage: bash download_current.sh PORT USER@HOST [LOCAL_DIRECTORY]' >&2
    exit 2
fi
printf -v TRANSFER_SSH 'ssh -p %q -i %q' "$TRANSFER_PORT" "$TRANSFER_KEY"
mkdir -p "$TRANSFER_TARGET/models/Qwen3.5-2B-Base" "$TRANSFER_TARGET/math_l4_v1"
printf '%s\n' '[DOWNLOAD 1/3] Base weights and tokenizer; existing complete files are reused.'
rsync -avh --progress --partial --exclude='.cache/' --exclude='__pycache__/' --exclude='.DS_Store' \
    -e "$TRANSFER_SSH" "$TRANSFER_HOST:/root/code/models/Qwen3.5-2B-Base/" \
    "$TRANSFER_TARGET/models/Qwen3.5-2B-Base/"
printf '%s\n' '[DOWNLOAD 2/3] Data, answers, selected model and all three checkpoint-selection receipts.'
rsync -avh --progress --partial \
    --include='/outputs/math_l4_v1_fix1/' --exclude='/outputs/*/' \
    --include='/outputs/math_l4_v1_fix1/formal/checkpoint-128/' \
    --include='/outputs/math_l4_v1_fix1/formal/checkpoint-256/' \
    --include='/outputs/math_l4_v1_fix1/formal/checkpoint-384/' \
    --exclude='/outputs/math_l4_v1_fix1/formal/checkpoint-*/' \
    --exclude='/outputs/math_l4_v1_fix1/pilot/checkpoint-*/' \
    --exclude='/outputs/math_l4_v1_fix1/engineering/checkpoint-*/' \
    --exclude='.venv/' --exclude='__pycache__/' --exclude='*.pyc' \
    --exclude='.DS_Store' --exclude='*.tmp' \
    -e "$TRANSFER_SSH" "$TRANSFER_HOST:/root/code/math_l4_v1/" "$TRANSFER_TARGET/math_l4_v1/"
printf '[DOWNLOAD COMPLETE] %s\n' "$TRANSFER_TARGET"
printf '%s\n' '[DOWNLOAD 3/3] Verifying frozen evidence and preparing the two-worker upload payload.'
python3 -B "$TRANSFER_ROOT/scripts/build_payload.py" --snapshot-root "$TRANSFER_TARGET"
