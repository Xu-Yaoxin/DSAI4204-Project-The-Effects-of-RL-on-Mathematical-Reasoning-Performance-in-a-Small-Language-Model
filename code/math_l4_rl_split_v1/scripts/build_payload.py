"""Verify downloaded model/selection evidence and stage immutable worker inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys

bundle = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--snapshot-root', default=str(Path.home()/'Downloads/math_l4_selected_20261104'))
parser.add_argument('--plan', action='store_true')
parser.add_argument('--verify-ready', action='store_true')
args = parser.parse_args()
reference = json.loads((bundle/'reference/source_identity.json').read_text())
ready_path = bundle/'payload/ready.json'
if args.verify_ready:
    ready = json.loads(ready_path.read_text())
    if ready.get('complete') is not True or ready.get('selection_id') != reference['selection_id']:
        raise SystemExit('Upload payload has no matching completed verification.')
    payload_root = bundle/'payload/code'
    actual_paths = list(payload_root.rglob('*'))
    if any(path.is_symlink() for path in (payload_root, *payload_root.parents, *actual_paths)):
        raise SystemExit('Payload contains an unsafe symlink.')
    actual_files = {str(path.relative_to(payload_root)) for path in actual_paths if path.is_file()}
    if actual_files != set(ready['files']):
        raise SystemExit('Payload inventory changed; active outputs or unmanaged files must not be uploaded.')
    for relative, expected in ready['files'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise SystemExit('Unsafe payload inventory path.')
        path = bundle/'payload/code'/relative
        if path.is_symlink() or not path.is_file():
            raise SystemExit('Payload file missing: '+relative)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(8*1024*1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise SystemExit('Payload file changed: '+relative)
    model = Path(ready['model_source'])
    if (ready.get('model_identity') != reference['model_identity'] or
            hashlib.sha256(json.dumps(ready['model_files'],sort_keys=True).encode()).hexdigest() != reference['model_identity']):
        raise SystemExit('Base model inventory differs from the frozen identity.')
    for name, expected in ready['model_files'].items():
        if Path(name).name != name:
            raise SystemExit('Unsafe Base model inventory path.')
        path = model/name
        if path.is_symlink() or not path.is_file():
            raise SystemExit('Base model file missing: '+name)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(8*1024*1024):
                digest.update(chunk)
        if path.stat().st_size != expected['bytes'] or digest.hexdigest() != expected['sha256']:
            raise SystemExit('Base model file changed: '+name)
    print(model)
    raise SystemExit(0)

snapshot = Path(args.snapshot_root).expanduser().resolve()
experiment = snapshot/'math_l4_v1'
run = experiment/'outputs/math_l4_v1_fix1'
frozen_path = experiment/'bundle_manifest.json'
if hashlib.sha256(frozen_path.read_bytes()).hexdigest() != reference['bundle_manifest_sha256']:
    raise SystemExit('Downloaded experiment differs from the pinned math-l4-1.0.1 bundle.')
frozen = json.loads(frozen_path.read_text())
for relative, expected in frozen['files_sha256'].items():
    path = experiment/relative
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('Frozen experiment file differs: '+relative)
selection_path = run/'selection.json'
if hashlib.sha256(selection_path.read_bytes()).hexdigest() != reference['selection_sha256']:
    raise SystemExit('This deployment is bound to the supplied checkpoint-384 selection.')
selection = json.loads(selection_path.read_text())
needed = []
for candidate in selection['candidates']:
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'resume_state.json'):
        path = experiment/candidate['adapter_path']/name
        if path.is_symlink() or not path.is_file():
            needed.append(str(path))
if args.plan:
    print(json.dumps({'plan_only':True, 'frozen_files_verified':len(frozen['files_sha256']),
        'selected_step':selection['selected_step'], 'batches':2, 'questions_each':100,
        'responses_each':1600, 'missing_selection_files':needed,
        'payload_ready':False, 'gpu_used':False},indent=2))
    raise SystemExit(0)
if needed:
    raise SystemExit('Selection evidence is incomplete. Run download_current.sh to fetch all three candidates:\n'+'\n'.join(needed))
print('[PACKAGE 1/4] Verifying all development answers and the frozen selection.',flush=True)
validated = runpy.run_path(str(experiment/'scripts/validate_selection.py'),init_globals={
    'SELECTION_PATH':selection_path})['SELECTION_RESULT']
if validated['selection_id'] != reference['selection_id'] or validated['selected_step'] != 384:
    raise SystemExit('Validated selection changed.')
print('[PACKAGE 2/4] Hashing the complete Base weights and tokenizer; no model/GPU is loaded.',flush=True)
model = snapshot/'models/Qwen3.5-2B-Base'
identity = runpy.run_path(str(experiment/'scripts/fingerprint_model.py'),init_globals={'MODEL_PATH':model})
if identity['model_identity'] != reference['model_identity']:
    raise SystemExit('Base model bytes differ from the experiment.')

files = {'math_l4_v1/bundle_manifest.json':frozen_path}
for relative in frozen['files_sha256']:
    files['math_l4_v1/'+relative] = experiment/relative
files['math_l4_v1/outputs/math_l4_v1_fix1/selection.json'] = selection_path
files['math_l4_v1/outputs/math_l4_v1_fix1/formal/report.json'] = run/'formal/report.json'
training = json.loads((run/'formal/report.json').read_text())
for relative in training['completion_files']:
    files['math_l4_v1/outputs/math_l4_v1_fix1/formal/'+relative] = run/'formal'/relative
for candidate in validated['candidates']:
    for name in ('adapter_config.json','adapter_model.safetensors','resume_state.json'):
        relative = candidate['adapter_path']+'/'+name
        files['math_l4_v1/'+relative] = experiment/relative
    for name in ('manifest.json','samples.jsonl'):
        relative = candidate['dev_output']+'/'+name
        files['math_l4_v1/'+relative] = experiment/relative
project = bundle.parent
for relative in ('scripts/prepare_packages.py','requirements-gpu.txt',
        'packages/linux-cp311-cu128/manifest.json','packages/linux-cp311-cu128/requirements-lock.txt'):
    files[relative] = project/relative
    if hashlib.sha256(files[relative].read_bytes()).hexdigest() != reference['installer_files'][relative]:
        raise SystemExit('Pinned installation input changed: '+relative)
print('[PACKAGE 3/4] Staging only frozen inputs; active run state and test answers are excluded from uploads.',flush=True)
destination = bundle/'payload/code'
inventory = {}
for relative, source in sorted(files.items()):
    if Path(relative).is_absolute() or '..' in Path(relative).parts or source.is_symlink():
        raise SystemExit('Unsafe package path: '+relative)
    target = destination/relative
    if any(path.is_symlink() for path in (target,*target.parents)):
        raise SystemExit('Unsafe destination path: '+str(target))
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    inventory[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
actual = {str(path.relative_to(destination)) for path in destination.rglob('*') if path.is_file()}
if actual != set(inventory):
    raise SystemExit('Payload contains unmanaged extra files; preserve it and inspect payload/code before uploading.')
ready = {'schema_version':1,'complete':True,'created_utc':datetime.now(timezone.utc).isoformat(),
    'selection_id':validated['selection_id'],'selected_step':validated['selected_step'],
    'model_source':str(model),'model_identity':identity['model_identity'],'model_files':identity['model_files'],
    'files':inventory,'base_reference_manifest_sha256':hashlib.sha256((bundle/'reference/base_manifest.json').read_bytes()).hexdigest(),
    'uploads_active_test_outputs':False,'gpu_executed':False}
ready_path.write_text(json.dumps(ready,indent=2)+'\n')
print(f'[PACKAGE 4/4] READY | files={len(inventory)} | batches=2 | questions/worker=100 | responses/worker=1600',flush=True)
print('[SAVED] '+str(ready_path),flush=True)
