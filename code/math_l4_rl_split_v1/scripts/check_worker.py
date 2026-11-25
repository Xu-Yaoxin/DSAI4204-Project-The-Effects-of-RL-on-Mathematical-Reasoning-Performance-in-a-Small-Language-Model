"""Verify a serial RTX 4090 worker against the frozen Base evaluation environment."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import runpy
import socket
import subprocess
import sys

bundle = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment-root', default='../math_l4_v1')
parser.add_argument('--run-dir', default='outputs/math_l4_v1_fix1')
parser.add_argument('--shard-index', required=True, type=int, choices=(0, 1))
parser.add_argument('--output')
mode = parser.add_mutually_exclusive_group()
mode.add_argument('--plan', action='store_true')
mode.add_argument('--dependencies-only', action='store_true')
args = parser.parse_args()
raw_experiment = Path(args.experiment_root).expanduser()
experiment = (raw_experiment if raw_experiment.is_absolute() else bundle / raw_experiment).resolve()
run = (experiment / args.run_dir).resolve()
if not run.is_relative_to(experiment / 'outputs'):
    raise SystemExit('Run directory must belong to the frozen experiment outputs.')
reference_path = bundle / 'reference/base_manifest.json'
reference_bytes = reference_path.read_bytes()
reference = json.loads(reference_bytes)
cfg_bytes = (experiment / 'configs/math_l4.json').read_bytes()
cfg = json.loads(cfg_bytes)
if (reference.get('model_label') != 'base' or reference.get('split') != 'test'
        or reference.get('synthetic') is not False or reference.get('config') != cfg
        or reference.get('model_identity') != cfg['model']['expected_identity']
        or reference.get('gpu') != 'NVIDIA GeForce RTX 4090'
        or reference.get('execution_cpu_threads') != {'intraop': 4, 'interop': 1}):
    raise SystemExit('Frozen Base reference must declare the original RTX 4090, four-thread test protocol.')
settings = reference['settings']
if settings.get('dtype') != 'bfloat16' or settings.get('backend') != 'transformers-sdpa' or settings.get('batch_size') != 1 or settings.get('world_size') != 1:
    raise SystemExit('Unsupported Base generation environment.')
expected_versions = settings['package_versions']
for optional in ('flash-linear-attention', 'fla-core', 'causal-conv1d'):
    if expected_versions.get(optional) is not None:
        raise SystemExit('This worker preserves the frozen fallback backend; optional fast kernels must be absent.')
frozen = json.loads((experiment / 'bundle_manifest.json').read_text())
for relative, expected in frozen['files_sha256'].items():
    path = experiment / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('Frozen experiment file differs: ' + relative)
report = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'shard_index': args.shard_index,
          'experiment_root': str(experiment), 'reference_base_manifest_sha256': hashlib.sha256(reference_bytes).hexdigest(),
          'config_sha256': hashlib.sha256(cfg_bytes).hexdigest(), 'source_gpu': reference['gpu'],
          'source_execution_cpu_threads': reference['execution_cpu_threads'], 'expected_package_versions': expected_versions,
          'gpu_checked': False, 'model_loaded': False, 'dependencies_checked': False,
          'scope': 'Source-only plan' if args.plan else 'Environment and independent worker readiness'}
if args.plan:
    print(json.dumps(report, indent=2), flush=True)
    print('[WORKER PLAN COMPLETE] Frozen source verified; no model packages or GPU imported.', flush=True)
    raise SystemExit(0)
if platform.system() != 'Linux' or platform.machine() != 'x86_64' or platform.python_implementation() != 'CPython' or sys.version_info[:2] != (3, 11):
    raise SystemExit('Worker requires Linux x86_64 CPython 3.11.')
if any(int(os.environ.get(name, str(expected))) != expected for name, expected in [('WORLD_SIZE', 1), ('LOCAL_WORLD_SIZE', 1), ('RANK', 0), ('LOCAL_RANK', 0)]):
    raise SystemExit('Use one serial GPU process on each worker; do not use torchrun/DDP.')
code = bundle.parent
lock_path = code / 'packages/linux-cp311-cu128/requirements-lock.txt'
lock_manifest = json.loads((code / 'packages/linux-cp311-cu128/manifest.json').read_text())
lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
if lock_hash != lock_manifest['lock_sha256']:
    raise SystemExit('Package lock hash differs from the pinned wheel manifest.')
versions = {}
for package in lock_manifest['packages']:
    try:
        observed = importlib.metadata.version(package['name'])
    except importlib.metadata.PackageNotFoundError:
        observed = None
    versions[package['name']] = observed
    if observed != package['version']:
        raise SystemExit(f"Pinned dependency differs: {package['name']} expected={package['version']} observed={observed}. Existing environments are not silently reinstalled.")
for name, expected in expected_versions.items():
    try:
        observed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        observed = None
    if observed != expected:
        raise SystemExit(f'Base/worker package mismatch: {name} expected={expected} observed={observed}')
pip_check = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True, timeout=60)
if pip_check.returncode:
    raise SystemExit('Dependency consistency check failed: ' + pip_check.stdout + pip_check.stderr)
import torch
import pandas
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
os.environ.update({'RLVR_CPU_THREADS': '4', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1'})
report.update(dependencies_checked=True, versions=versions, pip_check=pip_check.stdout.strip(),
              python=sys.version, executable=sys.executable, cpu_threads=torch.get_num_threads(),
              interop_threads=torch.get_num_interop_threads())
if args.dependencies_only:
    print('[DEPENDENCIES COMPLETE] Exact wheel versions, Base package versions, pip check and real imports passed; no GPU allocation.', flush=True)
    raise SystemExit(0)
# Take the occupancy snapshot before CUDA can create this process's context.
# NVML may report host-namespace PIDs while Python sees container-namespace PIDs.
gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20, check=True)
process_result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20, check=True)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Expose exactly one CUDA GPU on this worker.')
gpu_rows = [[part.strip() for part in line.split(',')] for line in gpu_result.stdout.splitlines() if line.strip()]
properties = torch.cuda.get_device_properties(0)
gpu_uuid = str(getattr(properties, 'uuid', ''))
matches = [row for row in gpu_rows if row[0] == gpu_uuid]
if not matches and len(gpu_rows) == 1:
    matches = gpu_rows
if len(matches) != 1 or matches[0][1] != reference['gpu'] or torch.cuda.get_device_name(0) != reference['gpu']:
    raise SystemExit('This matched evaluation requires one NVIDIA GeForce RTX 4090, with a uniquely identified GPU UUID.')
gpu_uuid, gpu_name, driver, memory_mib = matches[0]
compute_processes = []
for line in process_result.stdout.splitlines():
    fields = [part.strip() for part in line.split(',')]
    if len(fields) >= 4 and fields[0] == gpu_uuid:
        try:
            pid = int(fields[1])
        except ValueError:
            raise SystemExit('Cannot reliably identify a GPU compute process; refusing concurrent work.')
        compute_processes.append({'pid': pid, 'process_name': fields[2], 'used_gpu_memory_mib': fields[3]})
if compute_processes:
    raise SystemExit('GPU has other compute processes. Wait for the original job to stop: ' + json.dumps(compute_processes))
driver_parts = tuple(int(x) for x in re.findall(r'\d+', driver)[:2])
if driver_parts < (570, 26):
    raise SystemExit('Pinned CUDA 12.8 wheels require NVIDIA driver >=570.26.')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('BF16 support is required.')
x = torch.ones((128, 128), dtype=torch.bfloat16, device='cuda')
if not bool(((x @ x) == 128).all()):
    raise SystemExit('BF16 execution check failed.')
del x
torch.cuda.synchronize()
torch.cuda.empty_cache()
free_gpu, total_gpu = torch.cuda.mem_get_info()
mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line)
ram = int(mem['MemAvailable'].split()[0]) * 1024
for limit_path, used_path in [(Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory.current')),
                             (Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'), Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'))]:
    if limit_path.is_file() and used_path.is_file() and limit_path.read_text().strip() != 'max':
        ram = min(ram, max(0, int(limit_path.read_text()) - int(used_path.read_text())))
if free_gpu < cfg['execution']['required_free_gpu_gib'] * 2**30 or ram < cfg['execution']['required_available_ram_gib'] * 2**30:
    raise SystemExit('Insufficient free GPU or system memory for the frozen evaluation.')
model_path = (experiment / cfg['model']['path']).resolve()
fingerprint = runpy.run_path(str(experiment / 'scripts/fingerprint_model.py'), init_globals={'MODEL_PATH': model_path, 'HASH_MODEL': True})
if fingerprint['model_identity'] != reference['model_identity']:
    raise SystemExit('Local Base weights differ from the original evaluation model.')
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
if tokenizer.eos_token_id != cfg['model']['expected_eos_token_id']:
    raise SystemExit('Tokenizer EOS identity differs from the reference.')
host = {'hostname': socket.gethostname(), 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}
for path in [Path('/proc/cpuinfo'), Path('/sys/fs/cgroup/cpu.max'), Path('/sys/fs/cgroup/cpuset.cpus.effective')]:
    if path.is_file():
        host[str(path)] = path.read_text()
report.update(ready=True, gpu_checked=True, gpu=gpu_name, gpu_uuid=gpu_uuid, driver_version=driver,
              free_gpu_gib=free_gpu / 2**30, total_gpu_gib=total_gpu / 2**30, available_ram_gib=ram / 2**30,
              compute_processes=compute_processes, host=host, model_identity=fingerprint['model_identity'],
              package_lock_sha256=lock_hash, torch_build_cuda=torch.version.cuda,
              cpu_configuration={'RLVR_CPU_THREADS': 4, 'OMP_NUM_THREADS': 4, 'MKL_NUM_THREADS': 4, 'OPENBLAS_NUM_THREADS': 1, 'interop_threads': 1})
destination = Path(args.output).expanduser() if args.output else run / 'rl_shards' / f'worker_{args.shard_index}_hardware.json'
if not destination.is_absolute():
    destination = bundle / destination
destination = destination.absolute()
if any(path.is_symlink() for path in (destination, *destination.parents)) or not destination.resolve().is_relative_to(run):
    raise SystemExit('Hardware audit must be a regular file within this experiment run.')
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(destination.name + f'.{os.getpid()}.tmp')
with temporary.open('x') as stream:
    stream.write(json.dumps(report, indent=2) + '\n')
temporary.replace(destination)
print('[WORKER READY] RTX 4090 | BF16 passed | one active worker | threads=4/1 | [SAVED] ' + str(destination), flush=True)
