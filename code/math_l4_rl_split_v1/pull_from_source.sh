#!/usr/bin/env bash
set -euo pipefail
DIRECT_BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_CODE="$(cd -- "$DIRECT_BUNDLE/.." && pwd)"
DIRECT_PORT="${1:-42172}"
DIRECT_HOST="${2:-root@xj-member.bitahub.com}"
if [[ $# -gt 2 || ! "$DIRECT_PORT" =~ ^[0-9]+$ || ! "$DIRECT_HOST" =~ ^[A-Za-z0-9_.@-]+$ ]]; then
    printf '%s\n' 'Usage on a new Linux worker: bash pull_from_source.sh [SOURCE_PORT [SOURCE_USER@HOST]]' >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    printf '%s\n' 'Direct pull runs on a new Linux x86_64 server, not the local computer.' >&2
    exit 2
fi
for DIRECT_TOOL in rsync ssh flock; do
    command -v "$DIRECT_TOOL" >/dev/null || { printf 'Required command missing: %s\n' "$DIRECT_TOOL" >&2; exit 2; }
done
if [[ -n "${RLVR_SYSTEM_PYTHON:-}" ]]; then
    DIRECT_PYTHON="$RLVR_SYSTEM_PYTHON"
elif command -v python3.11 >/dev/null; then
    DIRECT_PYTHON=python3.11
else
    DIRECT_PYTHON=python3
fi
"$DIRECT_PYTHON" - "$DIRECT_BUNDLE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
assert sys.version_info[:2] == (3, 11), 'Set RLVR_SYSTEM_PYTHON to CPython 3.11 before pulling.'
bundle = Path(sys.argv[1])
inventory = bundle / 'reference/direct_source_files.txt'
release = json.loads((bundle / 'release_manifest.json').read_text())
expected = release['files']['reference/direct_source_files.txt']['sha256']
assert hashlib.sha256(inventory.read_bytes()).hexdigest() == expected, 'Direct transfer inventory changed.'
paths = inventory.read_text().splitlines()
assert paths and paths == sorted(set(paths)), 'Inventory must be sorted and unique.'
for text in paths:
    path = Path(text)
    assert text and not path.is_absolute() and '..' not in path.parts and '\x00' not in text, 'Unsafe transfer path.'
    assert not any(part in {'.venv', 'test_base', 'test_rl', 'rl_shards', '__pycache__'} for part in path.parts), 'Active paths are forbidden.'
    assert path.name not in {'run_state.json', 'live.json', 'optimizer.pt', 'scheduler.pt'}, 'Runtime state is forbidden.'
required = {'math_l4_v1/bundle_manifest.json', 'math_l4_v1/outputs/math_l4_v1_fix1/selection.json', 'scripts/prepare_packages.py', 'requirements-gpu.txt', 'packages/linux-cp311-cu128/manifest.json', 'packages/linux-cp311-cu128/requirements-lock.txt'}
assert required <= set(paths), 'Missing direct transfer inputs.'
print(f'[DIRECT PLAN] exact_files={len(paths)} | full Base model plus frozen experiment and selection evidence', flush=True)
PY
DIRECT_LOCK="$DIRECT_CODE/math_l4_v1/outputs/.single_gpu_experiment.lock"
if [[ -L "$DIRECT_LOCK" || -L "$DIRECT_CODE/math_l4_v1" || -L "$DIRECT_CODE/math_l4_v1/outputs" ]]; then
    printf '%s\n' 'Unsafe destination path or experiment lock.' >&2
    exit 2
fi
mkdir -p "$DIRECT_CODE/math_l4_v1/outputs" "$DIRECT_CODE/models/Qwen3.5-2B-Base"
exec 9>>"$DIRECT_LOCK"
flock -n 9 || { printf '%s\n' 'An experiment is active on this destination. Do not pull over a running Base/RL job.' >&2; exit 2; }
DIRECT_SOCKET_DIR="$(mktemp -d /tmp/math-rl-pull.XXXXXXXX)"
trap 'ssh -S "$DIRECT_SOCKET_DIR/ssh" -p "$DIRECT_PORT" -O exit "$DIRECT_HOST" >/dev/null 2>&1 || true; rm -rf -- "$DIRECT_SOCKET_DIR"; flock -u 9' EXIT
DIRECT_SSH_ARGS=(ssh -p "$DIRECT_PORT" -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ControlMaster=auto -o ControlPersist=600 -o "ControlPath=$DIRECT_SOCKET_DIR/ssh")
if [[ -n "${RLVR_SOURCE_SSH_KEY:-}" ]]; then
    DIRECT_SSH_ARGS+=(-i "$RLVR_SOURCE_SSH_KEY")
fi
printf -v DIRECT_SSH '%q ' "${DIRECT_SSH_ARGS[@]}"
printf '[DIRECT DOWNLOAD 1/3] %s:%s | full Base weights; source is read only.\n' "$DIRECT_HOST" "$DIRECT_PORT"
rsync -avh --progress --partial --exclude='.cache/' --exclude='__pycache__/' --exclude='.DS_Store' \
    -e "$DIRECT_SSH" "$DIRECT_HOST:/root/code/models/Qwen3.5-2B-Base/" \
    "$DIRECT_CODE/models/Qwen3.5-2B-Base/"
printf '%s\n' '[DIRECT DOWNLOAD 2/3] Exact frozen files and checkpoints 128/256/384; active answers and run state are excluded.'
rsync -avh --progress --partial --files-from="$DIRECT_BUNDLE/reference/direct_source_files.txt" \
    -e "$DIRECT_SSH" "$DIRECT_HOST:/root/code/" "$DIRECT_CODE/"
printf '%s\n' '[DIRECT DOWNLOAD 3/3] Verifying full model hashes, source files and checkpoint selection; no GPU model is loaded.'
"$DIRECT_PYTHON" -B "$DIRECT_BUNDLE/scripts/build_payload.py" --snapshot-root "$DIRECT_CODE"
printf '%s\n' '[DIRECT READY] Source data verified. Run setup_worker.sh and run_shard.py for this worker index.'
