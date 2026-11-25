#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -eq 0 ]]; then
    set -- all --run-dir outputs/math_l4_reproduction
fi
exec bash code/math_l4_v1/run.sh "$@"
