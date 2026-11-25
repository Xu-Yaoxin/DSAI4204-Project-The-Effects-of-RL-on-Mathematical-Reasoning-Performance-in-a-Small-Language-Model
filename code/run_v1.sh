#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec bash run.sh all --config "${1:-configs/v1.json}" "${@:2}"
