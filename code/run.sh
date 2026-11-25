#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -f scripts/run_pipeline.py || ! -f configs/v1.json ]]; then
    printf '%s\n' 'The launcher must remain beside scripts/ and configs/.' >&2
    exit 2
fi
if [[ -n "${RLVR_PYTHON:-}" ]]; then
    RLVR_EXECUTABLE="$RLVR_PYTHON"
elif [[ -x .venv/bin/python ]]; then
    RLVR_EXECUTABLE=.venv/bin/python
else
    RLVR_EXECUTABLE=python3
fi
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/outputs/matplotlib-cache}"
exec "$RLVR_EXECUTABLE" scripts/run_pipeline.py "$@"
