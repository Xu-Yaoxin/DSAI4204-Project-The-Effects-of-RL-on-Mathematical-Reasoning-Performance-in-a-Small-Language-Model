#!/usr/bin/env bash
set -euo pipefail
WORKER_BUNDLE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKER_CODE="$(cd -- "$WORKER_BUNDLE/.." && pwd)"
WORKER_EXPERIMENT="$WORKER_CODE/math_l4_v1"
WORKER_SHARD=0
WORKER_PLAN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment-root) WORKER_EXPERIMENT="$2"; shift 2 ;;
        --shard-index) WORKER_SHARD="$2"; shift 2 ;;
        --plan) WORKER_PLAN=1; shift ;;
        *) printf 'Unknown setup option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ "$WORKER_SHARD" != 0 && "$WORKER_SHARD" != 1 ]]; then
    printf 'Use --shard-index 0 or 1.\n' >&2
    exit 2
fi
cd -- "$WORKER_BUNDLE"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export RLVR_CPU_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
if [[ "$WORKER_PLAN" == 1 ]]; then
    "${RLVR_SYSTEM_PYTHON:-python3}" -S scripts/check_worker.py --experiment-root "$WORKER_EXPERIMENT" --shard-index "$WORKER_SHARD" --plan
    exit 0
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    printf 'Worker installation requires Linux x86_64. Use --plan for a CPU-only source check.\n' >&2
    exit 2
fi
WORKER_PYTHON="$WORKER_CODE/.venv/bin/python"
WORKER_PENDING="$WORKER_CODE/.venv/.rlvr-worker-install-incomplete"
if [[ -x "$WORKER_PYTHON" ]]; then
    printf '[SETUP 1/3] Reusing the existing project environment.\n'
elif [[ -e "$WORKER_CODE/.venv" ]]; then
    printf 'Existing .venv has no working Python; preserve it and repair the environment explicitly.\n' >&2
    exit 2
else
    WORKER_SYSTEM_PYTHON="${RLVR_SYSTEM_PYTHON:-python3.11}"
    "$WORKER_SYSTEM_PYTHON" -c 'import platform,sys; assert sys.version_info[:2]==(3,11) and platform.python_implementation()=="CPython" and platform.architecture()[0]=="64bit", "Use 64-bit CPython 3.11 via RLVR_SYSTEM_PYTHON"'
    printf '[SETUP 1/3] Creating the project CPython 3.11 environment.\n'
    "$WORKER_SYSTEM_PYTHON" -m venv "$WORKER_CODE/.venv"
    touch "$WORKER_PENDING"
fi
"$WORKER_PYTHON" -c 'import platform,sys; assert sys.version_info[:2]==(3,11) and platform.machine()=="x86_64" and platform.python_implementation()=="CPython", "Existing project Python must be CPython 3.11 x86_64"'
if [[ -f "$WORKER_PENDING" ]]; then
    printf '[SETUP 2/3] Downloading/verifying the pinned wheel bundle; resumable byte progress follows.\n'
    "$WORKER_PYTHON" "$WORKER_CODE/scripts/prepare_packages.py"
    "$WORKER_PYTHON" -m pip install --no-index --find-links "$WORKER_CODE/packages/linux-cp311-cu128/wheels" \
        --require-hashes --no-cache-dir -r "$WORKER_CODE/packages/linux-cp311-cu128/requirements-lock.txt"
    "$WORKER_PYTHON" - "$WORKER_CODE" <<'PY'
import hashlib
from pathlib import Path
import sys
root = Path(sys.argv[1])
digest = hashlib.sha256((root / 'packages/linux-cp311-cu128/requirements-lock.txt').read_bytes()).hexdigest()
(root / '.venv/.rlvr-package-lock-sha256').write_text(digest + '\n')
PY
else
    printf '[SETUP 2/3] Existing packages will be verified without reinstalling.\n'
fi
printf '[SETUP 3/3] Verifying exact dependencies and actual training/evaluation imports.\n'
"$WORKER_PYTHON" scripts/check_worker.py --experiment-root "$WORKER_EXPERIMENT" --shard-index "$WORKER_SHARD" --dependencies-only
if [[ -f "$WORKER_PENDING" ]]; then rm -- "$WORKER_PENDING"; fi
printf '[SETUP COMPLETE] Dependencies ready. GPU exclusivity is checked when the shard starts.\n'
