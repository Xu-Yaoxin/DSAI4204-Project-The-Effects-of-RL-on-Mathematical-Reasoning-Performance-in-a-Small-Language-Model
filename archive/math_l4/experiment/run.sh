#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${RLVR_PYTHON:-}" ]]; then
    MATH_PYTHON="$RLVR_PYTHON"
elif [[ -x ../.venv/bin/python ]]; then
    MATH_PYTHON=../.venv/bin/python
elif [[ "${1:-all}" == plan ]]; then
    MATH_PYTHON=python3
else
    printf 'Missing existing GPU environment ../.venv/bin/python. Run /root/code/setup_gpu.sh first.\n' >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export RLVR_CPU_THREADS="${RLVR_CPU_THREADS:-4}"
export OMP_NUM_THREADS="$RLVR_CPU_THREADS"
export MKL_NUM_THREADS="$RLVR_CPU_THREADS"
export OPENBLAS_NUM_THREADS=1
exec "$MATH_PYTHON" scripts/run_experiment.py "$@"
