"""Verify the frozen bundle and existing single-GPU environment without training."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', default='configs/math_l4.json')
parser.add_argument('--cpu-only', action='store_true')
parser.add_argument('--output')
args = parser.parse_args()
config_path=(root/args.config).absolute()
if any(p.is_symlink() for p in (config_path,*config_path.parents)) or not config_path.resolve().is_relative_to(root):
    raise SystemExit('Configuration must stay within the package without symlinks')
config = json.loads(config_path.read_text())
if (root / 'bundle_manifest.json').is_file():
    bundle = json.loads((root / 'bundle_manifest.json').read_text())
    for relative, expected in bundle['files_sha256'].items():
        path = root / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or any(p.is_symlink() for p in (path,*path.parents)) or not path.is_file():
            raise SystemExit(f'Unsafe/missing bundle file: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Incomplete upload or modified bundle file: {relative}')
    print(f'[BUNDLE VERIFIED] {len(bundle["files_sha256"])} files', flush=True)
else:
    raise SystemExit('Missing bundle_manifest.json; upload the complete released package')
data = runpy.run_path(str(root / 'scripts/doctor.py'), init_globals={'DOCTOR_CONFIG': config})['report']
if not data['data_ready']:
    raise SystemExit('Prepared data integrity failed: ' + json.dumps(data))
report = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': hashlib.sha256((root / args.config).read_bytes()).hexdigest(),
          'data_ready': True, 'gpu_checked': False, 'model_loaded': False}
if not args.cpu_only:
    if platform.system() != 'Linux' or sys.version_info[:2] != (3,11):
        raise SystemExit('Use the existing Linux CPython3.11 GPU environment: ../.venv/bin/python')
    if any(int(os.environ.get(key, default)) != value for key,default,value in [('WORLD_SIZE','1',1),('LOCAL_WORLD_SIZE','1',1),('RANK','0',0),('LOCAL_RANK','0',0)]):
        raise SystemExit('Use a single GPU process, not torchrun/DDP')
    versions = {}
    for name, expected in {'torch':'2.10.0','transformers':'5.3.0','trl':'1.16.0','peft':'0.18.1','accelerate':'1.10.1'}.items():
        versions[name] = importlib.metadata.version(name)
        if versions[name].split('+')[0] != expected:
            raise SystemExit(f'Expected {name}=={expected}, found {versions[name]}')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise SystemExit('Expose exactly one real BF16 CUDA GPU with CUDA_VISIBLE_DEVICES=0')
    x = torch.ones((128,128), dtype=torch.bfloat16, device='cuda')
    assert bool(((x @ x) == 128).all()), 'BF16 matrix check failed'
    del x
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line)
    ram = int(mem['MemAvailable'].strip().split()[0])*1024
    for limit_path,used_path in ((Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/memory.current')),
        (Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'),Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'))):
        if limit_path.is_file() and used_path.is_file() and limit_path.read_text().strip() != 'max':
            ram = min(ram,max(0,int(limit_path.read_text())-int(used_path.read_text())))
    if free_gpu < config['execution']['required_free_gpu_gib']*1024**3 or ram < config['execution']['required_available_ram_gib']*1024**3:
        raise SystemExit('Insufficient free GPU/RAM; stop other model jobs before starting this serial experiment')
    model_path = (root / config['model']['path']).resolve()
    fingerprint = runpy.run_path(str(root/'scripts/fingerprint_model.py'), init_globals={'MODEL_PATH':model_path,'HASH_MODEL':True})
    if fingerprint['model_identity'] != config['model']['expected_identity']:
        raise SystemExit('Local base model differs from the frozen model identity')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    if tokenizer.eos_token_id != config['model']['expected_eos_token_id']:
        raise SystemExit('Unexpected tokenizer EOS identity')
    prompt_counts = {}
    for split in ('train','dev','test'):
        records = [json.loads(line) for line in (root/config['data']['output_dir']/f'{split}.jsonl').read_text().splitlines() if line.strip()]
        counts = [len(tokenizer(config['prompt']['template'].format(question=r['question']),add_special_tokens=True)['input_ids']) for r in records]
        limit = min(config['training']['max_prompt_length'],config['evaluation']['max_prompt_length'])
        if max(counts)>limit:
            raise SystemExit(f'{split}: prompt length exceeds frozen limit; no silent truncation')
        prompt_counts[split]={'count':len(counts),'max_prompt_tokens':max(counts)}
    report.update(gpu_checked=True, gpu=torch.cuda.get_device_name(), free_gpu_gib=free_gpu/1024**3,
        total_gpu_gib=total_gpu/1024**3, available_ram_gib=ram/1024**3, versions=versions,
        model_identity=fingerprint['model_identity'], prompt_preflight=prompt_counts)
report['ready'] = True
if args.output:
    target = (root / args.output).absolute()
    if any(p.is_symlink() for p in (target,*target.parents)):
        raise SystemExit('Preflight output path must not contain symlinks')
    target=target.resolve()
    if not target.is_relative_to(root/'outputs'):
        raise SystemExit('Preflight output must be under outputs/')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
print('✅ [PREPARATION COMPLETE] Frozen data/code verified; existing environment ready. GPU training itself remains gated by engineering/pilot.',flush=True)
