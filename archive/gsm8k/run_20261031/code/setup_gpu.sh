#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == --help ]]; then
    printf '%s\n' 'Usage: bash setup_gpu.sh [--offline]' 'Target: Linux x86_64, CPython 3.11, glibc >=2.28, NVIDIA driver >=570.26, RTX 4090.' 'Prepares locked packages, the pinned official Qwen3.5-2B-Base snapshot, GSM8K splits, tokenizer and GPU communication checks.' 'Online mode resumes missing downloads; --offline requires complete local wheels, model files and raw data with provenance.' 'Progress shows completed setup stages, download bytes and verification bytes; stage percentages are not time estimates.' 'Set RLVR_SYSTEM_PYTHON to choose Python 3.11. Installs only into this project .venv.' 'An existing unmanaged .venv is preserved and rejected. A matching managed .venv can be resumed.' 'No training, system driver changes or optional source-built attention kernels are performed.'
    exit 0
fi
RLVR_SETUP_OFFLINE=0
if [[ "${1:-}" == --offline && $# -eq 1 ]]; then
    RLVR_SETUP_OFFLINE=1
elif [[ $# -ne 0 ]]; then
    printf '%s\n' 'Unsupported arguments. Use bash setup_gpu.sh --help.' >&2
    exit 2
fi
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=0
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
printf '\n%s\n' '========================================================================' '[SETUP 1/7] [----------------------------]   0% | HOST AND PYTHON'
if [[ "$(uname -s)" != Linux ]] || [[ "$(uname -m)" != x86_64 ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' 'GPU setup requires Linux x86_64 with an NVIDIA driver. Package downloading and CPU checks can run on this machine.' >&2
    exit 2
fi
if [[ -z "${RLVR_SYSTEM_PYTHON:-}" ]] && command -v python3.11 >/dev/null 2>&1; then
    RLVR_SYSTEM_PYTHON=python3.11
else
    RLVR_SYSTEM_PYTHON="${RLVR_SYSTEM_PYTHON:-python3}"
fi
"$RLVR_SYSTEM_PYTHON" - <<'PY'
import platform,subprocess,sys
if sys.version_info[:2] != (3,11) or sys.implementation.name != 'cpython' or sys.maxsize <= 2**32:
    raise SystemExit('The locked wheel bundle requires 64-bit CPython 3.11 exactly. Set RLVR_SYSTEM_PYTHON to its executable; no system Python will be changed.')
name,version=platform.libc_ver()
if name!='glibc' or tuple(map(int,version.split('.')[:2])) < (2,28):
    raise SystemExit('Linux glibc >=2.28 is required by the pinned CUDA PyTorch wheel.')
result=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True)
rows=[line.strip() for line in result.stdout.split('\n') if line.strip()]
if not rows:
    raise SystemExit('No NVIDIA GPU is visible to nvidia-smi.')
for row in rows:
    fields=[field.strip() for field in row.split(',')]
    if tuple(map(int,fields[1].split('.'))) < (570,26):
        raise SystemExit('This CUDA 12.8 environment requires NVIDIA driver >=570.26; update the host driver first.')
    print('[GPU]',row)
print(f'[HOST VERIFIED] Python={platform.python_version()} glibc={version} physical_gpus={len(rows)}; training GPU selection is separate.')
PY
RLVR_LOCK_SHA256="$("$RLVR_SYSTEM_PYTHON" -c 'import hashlib; from pathlib import Path; print(hashlib.sha256(Path("packages/linux-cp311-cu128/requirements-lock.txt").read_bytes()).hexdigest())')"
if [[ -L .venv ]]; then
    printf '%s\n' 'The project .venv is a symlink. It was preserved; use a local project environment.' >&2
    exit 2
elif [[ -e .venv ]]; then
    if [[ ! -f .venv/.rlvr-package-lock-sha256 ]] || [[ "$(cat .venv/.rlvr-package-lock-sha256)" != "$RLVR_LOCK_SHA256" ]]; then
        printf '%s\n' 'An unmanaged .venv or a different dependency lock already exists. It was preserved; use a clean project copy or explicitly move that environment.' >&2
        exit 2
    fi
else
    if ! "$RLVR_SYSTEM_PYTHON" -c 'import venv, ensurepip'; then
        printf '%s\n' 'Python 3.11 venv/ensurepip support is missing. Install the Python 3.11 venv support package on the server and rerun; no wheels were downloaded.' >&2
        exit 2
    fi
    "$RLVR_SYSTEM_PYTHON" -m venv --without-pip .venv
    printf '%s\n' "$RLVR_LOCK_SHA256" > .venv/.rlvr-package-lock-sha256
fi
if [[ ! -x .venv/bin/python ]]; then
    printf '%s\n' 'The managed .venv Python cannot run. Virtual environments cannot be copied between machines; preserve or move it explicitly, then rerun setup.' >&2
    exit 2
fi
.venv/bin/python - <<'PY'
import sys
from pathlib import Path
if sys.version_info[:2] != (3,11) or sys.implementation.name != 'cpython' or sys.maxsize <= 2**32:
    raise SystemExit('The existing environment requires 64-bit CPython 3.11.')
if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != Path('.venv').resolve():
    raise SystemExit('The managed interpreter does not belong to this project .venv; no packages were downloaded or installed.')
PY
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    if ! .venv/bin/python -m ensurepip --upgrade; then
        printf '%s\n' 'Offline pip bootstrap failed. Install Python 3.11 venv/ensurepip support and rerun; this managed .venv can be resumed and no wheels were downloaded.' >&2
        exit 2
    fi
fi
.venv/bin/python -m pip --version
printf '%s\n' '[ENVIRONMENT VERIFIED] Project Python and pip are ready before package downloads.'
mkdir -p outputs/environment
RLVR_SETUP_REPORT_DIR="$(mktemp -d outputs/environment/setup-XXXXXXXX)"
export RLVR_SETUP_REPORT_DIR
printf '\n%s\n' '[SETUP 2/7] [####------------------------]  14% | DEPENDENCY DOWNLOAD AND VERIFICATION'
if [[ "$RLVR_SETUP_OFFLINE" -eq 1 ]]; then
    "$RLVR_SYSTEM_PYTHON" scripts/prepare_packages.py --verify
else
    "$RLVR_SYSTEM_PYTHON" scripts/prepare_packages.py
fi
printf '\n%s\n' '[SETUP 3/7] [########--------------------]  29% | PACKAGE INSTALLATION'
.venv/bin/python -m pip install --no-cache-dir --no-index --find-links packages/linux-cp311-cu128/wheels --require-hashes -r packages/linux-cp311-cu128/requirements-lock.txt
.venv/bin/python -m pip check
printf '\n%s\n' '[SETUP 4/7] [############----------------]  43% | CUDA AND DEPENDENCY CHECKS'
.venv/bin/python - <<'PY'
import hashlib,importlib.metadata,json,os,platform
from pathlib import Path
import torch
from peft import LoraConfig
from transformers import Qwen3_5ForCausalLM
from trl import GRPOConfig,GRPOTrainer
manifest=json.loads(Path('packages/linux-cp311-cu128/manifest.json').read_text())
for entry in manifest['packages']:
    installed=importlib.metadata.version(entry['name'])
    if installed!=entry['version']:
        raise SystemExit(f"Installed version mismatch: {entry['name']} expected {entry['version']}, got {installed}")
if torch.__version__!='2.10.0+cu128' or torch.version.cuda!='12.8' or not torch.cuda.is_available():
    raise SystemExit('Pinned CUDA PyTorch did not initialize. Check the host driver and GPU visibility.')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('Visible GPU does not support the required BF16 mode.')
x=torch.ones((32,32),device='cuda',dtype=torch.bfloat16)
y=x@x
torch.cuda.synchronize()
assert y[0,0].item()==32
report={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'package_manifest_sha256':hashlib.sha256(Path('packages/linux-cp311-cu128/manifest.json').read_bytes()).hexdigest(),
        'dependency_versions_verified':True,'cuda_bf16_matmul_passed':True,
        'visible_gpus':[{'name':torch.cuda.get_device_name(i),'capability':list(torch.cuda.get_device_capability(i))} for i in range(torch.cuda.device_count())],
        'qwen_model_loaded':False,'training_executed':False,'optional_fast_kernels_installed':False}
output=Path(os.environ['RLVR_SETUP_REPORT_DIR'])/'gpu_setup.json'
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(report,indent=2)+'\n')
print('[SETUP VERIFIED] Pinned packages, Qwen3.5/GRPO imports, CUDA and BF16 matrix multiplication passed.')
print('[SAVED]',output)
PY
.venv/bin/python -m pip freeze > "$RLVR_SETUP_REPORT_DIR/requirements-installed.txt"
printf '\n%s\n' '[SETUP 5/7] [################------------]  57% | PINNED MODEL DOWNLOAD AND VERIFICATION'
if [[ "$RLVR_SETUP_OFFLINE" -eq 1 ]]; then
    .venv/bin/python scripts/download_model.py --offline
else
    .venv/bin/python scripts/download_model.py
fi
printf '\n%s\n' '[SETUP 6/7] [####################--------]  71% | GSM8K DOWNLOAD AND SPLIT PREPARATION'
if [[ "$RLVR_SETUP_OFFLINE" -eq 1 ]]; then
    .venv/bin/python scripts/download_data.py --offline
else
    .venv/bin/python scripts/download_data.py
fi
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/build_variants.py
printf '\n%s\n' '[SETUP 7/7] [########################----]  86% | TOKENIZER, DATA AND GPU COMMUNICATION'
RLVR_PYTHON="$PWD/.venv/bin/python" bash run.sh gpucheck --run-dir "$RLVR_SETUP_REPORT_DIR"
.venv/bin/python - <<'PY'
import hashlib,json,os,runpy,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from transformers import AutoTokenizer

root=Path.cwd()
output=Path(os.environ['RLVR_SETUP_REPORT_DIR'])
config=json.loads(Path('configs/v1.json').read_text())
model_dir=Path(config['model']['path'])
locked_model=json.loads(Path('models/qwen3.5-2b-base-manifest.json').read_text())
if config['model']['repo_id']!=locked_model['repo_id'] or model_dir!=Path('models/Qwen3.5-2B-Base'):
    raise SystemExit('The setup model lock and experiment config disagree; no readiness report was saved.')
model_receipt=json.loads((model_dir/'source_manifest.json').read_text())
if (model_receipt.get('status')!='verified' or model_receipt.get('all_files_verified') is not True
        or model_receipt.get('repo_id')!=locked_model['repo_id'] or model_receipt.get('revision')!=locked_model['revision']
        or model_receipt.get('manifest_sha256')!=hashlib.sha256(Path('models/qwen3.5-2b-base-manifest.json').read_bytes()).hexdigest()):
    raise SystemExit('The pinned model download did not produce a matching verification receipt.')
tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
if tokenizer.eos_token_id is None or not tokenizer.encode('Compute 1 + 1.\n#### 2',add_special_tokens=False):
    raise SystemExit('Local tokenizer sanity check failed.')
runtime=runpy.run_path(str(root/'scripts/resolve_gpus.py'),init_globals={'REQUESTED_GPUS':'auto'})['runtime']
environment={**os.environ,**runtime['environment']}
subprocess.run([sys.executable,'scripts/doctor.py','--gpu','--world-size',str(runtime['world_size']),
                '--output',str(output/'readiness.json')],env=environment,check=True)
readiness=json.loads((output/'readiness.json').read_text())
hardware=json.loads((output/'hardware_check.json').read_text())
if readiness.get('ready') is not True or hardware.get('passed') is not True:
    raise SystemExit('Readiness or GPU communication checks failed.')
report={'checked_utc':datetime.now(timezone.utc).isoformat(),'ready':True,
        'repo_id':locked_model['repo_id'],'revision':locked_model['revision'],
        'model_manifest_sha256':hashlib.sha256(Path('models/qwen3.5-2b-base-manifest.json').read_bytes()).hexdigest(),
        'model_files_verified':True,'tokenizer_checked':True,'data_ready':True,
        'selected_gpu_count':runtime['world_size'],'hardware_checked':True,
        'qwen_model_loaded':False,'training_executed':False,
        'note':'Dependencies, pinned model files, tokenizer, data and small GPU operations verified. Run engineering to measure model memory and training.'}
(output/'ready.json').write_text(json.dumps(report,indent=2)+'\n')
print('[SAVED]',output/'ready.json')
PY
printf '\n%s\n' '[SETUP 7/7] [############################] 100% | ALL PREPARATION CHECKS PASSED' '✅ [SETUP COMPLETE] Dependencies, model and data are ready. Run: bash run.sh all' "[SAVED] $RLVR_SETUP_REPORT_DIR/ready.json"
