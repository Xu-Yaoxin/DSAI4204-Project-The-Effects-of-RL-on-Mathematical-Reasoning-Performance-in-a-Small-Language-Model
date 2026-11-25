#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == --help ]]; then
    printf '%s\n' 'Usage: bash download_packages.sh [--verify] [--bundle DIRECTORY]' 'Downloads and verifies Linux x86_64 / CPython 3.11 / CUDA 12.8 wheels; does not install on the current machine.' 'Partial downloads resume; complete files are SHA-256 checked. --verify never accesses the network.' 'Set RLVR_SYSTEM_PYTHON to choose an existing Python interpreter.'
    exit 0
fi
RLVR_SYSTEM_PYTHON="${RLVR_SYSTEM_PYTHON:-python3}"
exec "$RLVR_SYSTEM_PYTHON" scripts/prepare_packages.py "$@"
