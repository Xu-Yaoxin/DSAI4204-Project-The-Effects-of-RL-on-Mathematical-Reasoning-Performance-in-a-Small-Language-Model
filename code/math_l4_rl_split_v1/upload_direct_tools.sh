#!/usr/bin/env bash
set -euo pipefail
TRANSFER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRANSFER_CODE="$(cd -- "$TRANSFER_ROOT/.." && pwd)"
TRANSFER_PORT="${1:-42172}"
TRANSFER_HOST="${2:-root@xj-member.bitahub.com}"
TRANSFER_KEY="${RLVR_SSH_KEY:-$HOME/.ssh/dsai4207_4090}"
if [[ $# -gt 2 || ! "$TRANSFER_PORT" =~ ^[0-9]+$ || ! "$TRANSFER_HOST" =~ ^[A-Za-z0-9_.@-]+$ ]]; then
    printf '%s\n' 'Usage: bash upload_direct_tools.sh [PORT [USER@HOST]]' >&2
    exit 2
fi
for TRANSFER_REQUIRED in scripts/prepare_packages.py requirements-gpu.txt packages/linux-cp311-cu128/manifest.json packages/linux-cp311-cu128/requirements-lock.txt math_l4_rl_split_v1/pull_from_source.sh math_l4_rl_split_v1/reference/direct_source_files.txt; do
    if [[ ! -f "$TRANSFER_CODE/$TRANSFER_REQUIRED" ]]; then
        printf 'Missing local deployment input: %s\n' "$TRANSFER_REQUIRED" >&2
        exit 2
    fi
done
TRANSFER_CHECKSUMS="$(mktemp /tmp/math-rl-tools-checksums.XXXXXXXX)"
trap 'rm -f -- "$TRANSFER_CHECKSUMS"' EXIT
python3 - "$TRANSFER_ROOT" "$TRANSFER_CHECKSUMS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
bundle = Path(sys.argv[1])
release = json.loads((bundle / 'release_manifest.json').read_text())
reference = json.loads((bundle / 'reference/source_identity.json').read_text())
expected = {'math_l4_rl_split_v1/' + name: row['sha256'] for name, row in release['files'].items()}
expected.update(reference['installer_files'])
expected['math_l4_rl_split_v1/release_manifest.json'] = hashlib.sha256((bundle / 'release_manifest.json').read_bytes()).hexdigest()
lines = []
for name, digest in sorted(expected.items()):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or '\n' in name or '\\' in name:
        raise SystemExit('Unsafe deployment file path: ' + name)
    source = bundle.parent / path
    if source.is_symlink() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise SystemExit('Local deployment file differs from the pinned release: ' + name)
    lines.append(digest + '  ' + name)
Path(sys.argv[2]).write_text('\n'.join(lines) + '\n')
print(f'[DIRECT TOOLS CHECK] Local release verified | files={len(lines)}', flush=True)
PY
printf -v TRANSFER_SSH 'ssh -p %q -i %q' "$TRANSFER_PORT" "$TRANSFER_KEY"
printf '%s\n' '[DIRECT TOOLS 1/2] Uploading small scripts and dependency locks; no model or active results.'
cd -- "$TRANSFER_CODE"
rsync -avh --progress --partial --relative \
    --exclude='/math_l4_rl_split_v1/payload/' --exclude='/math_l4_rl_split_v1/outputs/' \
    --exclude='/math_l4_rl_split_v1/validation/' --exclude='__pycache__/' \
    --exclude='*.pyc' --exclude='*.log' --exclude='.DS_Store' \
    -e "$TRANSFER_SSH" --rsync-path='mkdir -p /root/code && rsync' \
    math_l4_rl_split_v1/ \
    scripts/prepare_packages.py \
    requirements-gpu.txt \
    packages/linux-cp311-cu128/manifest.json \
    packages/linux-cp311-cu128/requirements-lock.txt \
    "$TRANSFER_HOST:/root/code/"
printf '%s\n' '[DIRECT TOOLS 2/2] Verifying deployed file locations and SHA256 hashes on the destination.'
ssh -p "$TRANSFER_PORT" -i "$TRANSFER_KEY" "$TRANSFER_HOST" \
    'cd /root/code && sha256sum --check --strict' < "$TRANSFER_CHECKSUMS"
printf '[DIRECT TOOLS READY] %s port=%s | existing Base and RL processes were not restarted.\n' "$TRANSFER_HOST" "$TRANSFER_PORT"
