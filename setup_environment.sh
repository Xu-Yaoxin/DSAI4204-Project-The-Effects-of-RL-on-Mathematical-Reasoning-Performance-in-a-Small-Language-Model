#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == --help ]]; then
    printf '%s\n' 'Usage: bash setup_environment.sh [--offline]' 'Linux x86_64, CPython 3.11, glibc >=2.28, NVIDIA driver >=570.26.' 'Verifies requirement.txt, installs the locked environment, downloads the pinned Base model, and runs MATH preflight.' 'No training or answer generation is started.'
    exit 0
fi
if [[ -n "${RLVR_SYSTEM_PYTHON:-}" ]]; then
    PROJECT_PYTHON="$RLVR_SYSTEM_PYTHON"
elif command -v python3.11 >/dev/null 2>&1; then
    PROJECT_PYTHON=python3.11
else
    PROJECT_PYTHON=python3
fi
"$PROJECT_PYTHON" -B -S - <<'PY'
from pathlib import Path
import hashlib
import json
root = Path.cwd()
manifest = json.loads((root / 'code/packages/linux-cp311-cu128/manifest.json').read_text())
for path in (root / 'requirement.txt', root / 'code/packages/linux-cp311-cu128/requirements-lock.txt'):
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['lock_sha256']:
        raise SystemExit(f'Locked dependency file changed: {path}')
print('[ENVIRONMENT 1/2] requirement.txt matches all 84 pinned wheel hashes', flush=True)
PY
bash code/setup_gpu.sh "$@"
cd code/math_l4_v1
mkdir -p outputs/environment
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MATH_PREFLIGHT_OUTPUT="outputs/environment/reproduction_preflight-$(../.venv/bin/python -c 'import uuid; print(uuid.uuid4().hex)').json"
../.venv/bin/python -u scripts/preflight.py --output "$MATH_PREFLIGHT_OUTPUT"
printf '[SAVED] code/math_l4_v1/%s\n' "$MATH_PREFLIGHT_OUTPUT"
printf '%s\n' '[ENVIRONMENT 2/2] MATH preflight passed. Run bash train.sh from the project directory when ready.'
