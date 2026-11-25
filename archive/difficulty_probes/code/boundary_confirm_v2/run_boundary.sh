#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

BOUNDARY_PLAN=0
BOUNDARY_QUICK=0
BOUNDARY_PARALLEL=0
BOUNDARY_SEQUENTIAL=0
for BOUNDARY_ARGUMENT in "$@"; do
    case "$BOUNDARY_ARGUMENT" in
        --plan) BOUNDARY_PLAN=1 ;;
        --quick) BOUNDARY_QUICK=1 ;;
        --parallel) BOUNDARY_PARALLEL=1 ;;
        --sequential) BOUNDARY_SEQUENTIAL=1 ;;
        *) printf 'Usage: bash run_boundary.sh [--plan] [--quick] [--parallel [--sequential]]\n' >&2; exit 2 ;;
    esac
done
if [[ "$BOUNDARY_SEQUENTIAL" == 1 && "$BOUNDARY_PARALLEL" != 1 ]]; then
    printf '%s\n' '--sequential applies to the parallel launcher only.' >&2
    exit 2
fi

if [[ -n "${RLVR_PYTHON:-}" ]]; then
    BOUNDARY_PYTHON="$RLVR_PYTHON"
elif [[ "$BOUNDARY_PLAN" == 1 ]]; then
    BOUNDARY_PYTHON=python3
elif [[ -x .venv/bin/python ]]; then
    BOUNDARY_PYTHON=.venv/bin/python
elif [[ -x ../.venv/bin/python ]]; then
    BOUNDARY_PYTHON=../.venv/bin/python
else
    BOUNDARY_PYTHON=python3
fi

BOUNDARY_RUN=outputs/confirmation_v2
BOUNDARY_EVAL_ARGS=(--config configs/confirmation_v2.json)
if [[ "$BOUNDARY_QUICK" == 1 ]]; then
    BOUNDARY_RUN=outputs/confirmation_v2_quick
    BOUNDARY_EVAL_ARGS+=(--samples 1)
fi
BOUNDARY_PARALLEL_ARGS=(--config configs/confirmation_v2.json)
if [[ "$BOUNDARY_PARALLEL" == 1 ]]; then
    BOUNDARY_RUN=outputs/confirmation_v2_parallel
    if [[ "$BOUNDARY_QUICK" == 1 ]]; then
        BOUNDARY_RUN=outputs/confirmation_v2_parallel_quick
        BOUNDARY_PARALLEL_ARGS+=(--samples 1)
    fi
    if [[ "$BOUNDARY_SEQUENTIAL" == 1 ]]; then
        BOUNDARY_PARALLEL_ARGS+=(--workers 1)
    fi
    BOUNDARY_PARALLEL_ARGS+=(--output "$BOUNDARY_RUN")
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

if [[ -f bundle_manifest.json ]]; then
    "$BOUNDARY_PYTHON" -S - <<'PY'
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
print("[BUNDLE VERIFIED] Frozen data, scorer and evaluation code match their hashes", flush=True)
PY
fi

if [[ "$BOUNDARY_PLAN" == 1 ]]; then
    if [[ "$BOUNDARY_PARALLEL" == 1 ]]; then
        exec "$BOUNDARY_PYTHON" -S scripts/run_boundary_parallel.py "${BOUNDARY_PARALLEL_ARGS[@]}" --plan
    fi
    exec "$BOUNDARY_PYTHON" -S scripts/evaluate_boundary.py \
        --output "$BOUNDARY_RUN/base" "${BOUNDARY_EVAL_ARGS[@]}" --plan
fi
if ! command -v flock >/dev/null 2>&1; then
    printf 'The GPU launcher requires flock (util-linux) on the Linux server. Use --plan locally.\n' >&2
    exit 2
fi
mkdir -p outputs
exec 9> outputs/.boundary-run.lock
if ! flock -n 9; then
    printf 'A boundary evaluation is already running in this directory. Check boundary.log.\n' >&2
    exit 2
fi
if [[ "$BOUNDARY_PARALLEL" == 1 ]]; then
    exec "$BOUNDARY_PYTHON" scripts/run_boundary_parallel.py "${BOUNDARY_PARALLEL_ARGS[@]}"
fi
printf '%s\n' '========================================================================'
printf '[BOUNDARY 1/2] 0%% | BASE MODEL ONLY | EVALUATION WITH RESUME\n'
"$BOUNDARY_PYTHON" scripts/evaluate_boundary.py "${BOUNDARY_EVAL_ARGS[@]}" \
    --output "$BOUNDARY_RUN/base" --resume
printf '[BOUNDARY 2/2] 90%% | GROUP SUCCESS RATES AND DIAGNOSTICS\n'
"$BOUNDARY_PYTHON" -S scripts/analyze_boundary.py --input "$BOUNDARY_RUN/base" \
    --output "$BOUNDARY_RUN/analysis"
printf '✅ [BOUNDARY COMPLETE] 100%% | results=%s/analysis\n' "$BOUNDARY_RUN"
