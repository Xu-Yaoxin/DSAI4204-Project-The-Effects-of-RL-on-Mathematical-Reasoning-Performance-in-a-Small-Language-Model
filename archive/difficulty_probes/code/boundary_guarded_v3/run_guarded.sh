#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
GUARDED_PLAN=0
for GUARDED_ARGUMENT in "$@"; do
    case "$GUARDED_ARGUMENT" in
        --plan) GUARDED_PLAN=1 ;;
        *) printf 'Usage: bash run_guarded.sh [--plan]\n' >&2; exit 2 ;;
    esac
done
if [[ -n "${RLVR_PYTHON:-}" ]]; then
    GUARDED_PYTHON="$RLVR_PYTHON"
elif [[ "$GUARDED_PLAN" == 1 ]]; then
    GUARDED_PYTHON=python3
elif [[ -x ../.venv/bin/python ]]; then
    GUARDED_PYTHON=../.venv/bin/python
elif [[ -x .venv/bin/python ]]; then
    GUARDED_PYTHON=.venv/bin/python
else
    printf 'Existing GPU environment not found: ../.venv/bin/python\n' >&2
    exit 2
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
"$GUARDED_PYTHON" -S - <<'CHECK'
import hashlib
import json
from pathlib import Path
root = Path.cwd().resolve()
manifest = json.loads((root / "bundle_manifest.json").read_text())
for relative, expected in manifest["files_sha256"].items():
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink() or not path.is_file():
        raise SystemExit(f"Invalid bundle path: {relative}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Bundle file changed or upload incomplete: {relative}")
print("[BUNDLE VERIFIED] Frozen serial protocol, data and code match their hashes", flush=True)
CHECK
GUARDED_ARGS=(--config configs/guarded_v3.json --output outputs/guarded_v3 --workers 1)
if [[ "$GUARDED_PLAN" == 1 ]]; then
    exec "$GUARDED_PYTHON" -S scripts/run_boundary_parallel.py "${GUARDED_ARGS[@]}" --plan
fi
if ! command -v flock >/dev/null 2>&1; then
    printf 'The Linux GPU launcher requires flock (util-linux).\n' >&2
    exit 2
fi
mkdir -p outputs
exec 9> outputs/.guarded-run.lock
if ! flock -n 9; then
    printf 'A guarded evaluation is already running. Check guarded.log.\n' >&2
    exit 2
fi
exec "$GUARDED_PYTHON" scripts/run_boundary_parallel.py "${GUARDED_ARGS[@]}"
