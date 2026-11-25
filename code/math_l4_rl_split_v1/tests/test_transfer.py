"""Synthetic CPU transfer fixtures; real rsync, packager and tiny-weight fingerprinting."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile

bundle = Path(__file__).resolve().parents[1]
frozen = bundle.parent / 'math_l4_v1'
checks = 0
for filename in ('scripts/build_payload.py', 'tests/test_transfer.py'):
    tree = ast.parse((bundle / filename).read_text())
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
    checks += 1

download_script = (bundle / 'download_current.sh').read_text().replace('\\\n', ' ')
rsync_lines = [line for line in download_script.splitlines() if line.strip().startswith('rsync ')]
assert len(rsync_lines) == 2
transfer_filters = [argument for argument in shlex.split(rsync_lines[1])
    if argument.startswith(('--include=', '--exclude='))]
assert shutil.which('rsync'), 'A local rsync executable is required; no network is used.'
checks += 2

with tempfile.TemporaryDirectory(prefix='math-rl-transfer-fixture-') as temporary:
    root = Path(temporary).resolve()
    source = root / 'rsync-source'
    destination = root / 'rsync-destination'
    keep = [
        'configs/math_l4.json', 'data/processed/train.jsonl', 'data/raw/math.parquet',
        'outputs/math_l4_v1_fix1/selection.json',
        'outputs/math_l4_v1_fix1/formal/report.json',
        'outputs/math_l4_v1_fix1/formal/reward_audit.rank-0.jsonl',
        'outputs/math_l4_v1_fix1/test_base/samples.jsonl',
        'outputs/math_l4_v1_fix1/test_rl/samples.jsonl',
    ]
    for step in (128, 256, 384):
        keep += [f'outputs/math_l4_v1_fix1/formal/checkpoint-{step}/{name}'
            for name in ('adapter_config.json', 'adapter_model.safetensors', 'resume_state.json')]
        keep += [f'outputs/math_l4_v1_fix1/dev/checkpoint-{step}/{name}'
            for name in ('manifest.json', 'samples.jsonl')]
    omit = [
        'outputs/math_l4_v1_fix1/formal/checkpoint-16/optimizer.pt',
        'outputs/math_l4_v1_fix1/formal/checkpoint-272/adapter_model.safetensors',
        'outputs/math_l4_v1_fix1/pilot/checkpoint-96/optimizer.pt',
        'outputs/math_l4_v1_fix1/engineering/checkpoint-2/optimizer.pt',
        'outputs/old_run/test_base/samples.jsonl', '.venv/bin/python',
        '__pycache__/module.pyc', '.DS_Store',
        'outputs/math_l4_v1_fix1/test_base/manifest.json.tmp',
    ]
    for relative in keep + omit:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Synthetic local transfer fixture.\n')
    destination.mkdir()
    result = subprocess.run(['rsync', '-a', *transfer_filters, str(source) + '/', str(destination) + '/'],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.stdout, result.stderr)
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob('*') if path.is_file()}
    assert actual == set(keep), {'unexpected': sorted(actual - set(keep)), 'missing': sorted(set(keep) - actual)}
    checks += 2

    control_script = (bundle / 'upload_base_stop.sh').read_text().replace('\\\n', ' ')
    control_lines = [line for line in control_script.splitlines() if line.strip().startswith('rsync ')]
    assert len(control_lines) == 1
    control_arguments = shlex.split(control_lines[0])
    assert not any(argument.startswith('--delete') for argument in control_arguments)
    control_filters = [argument for argument in control_arguments
        if argument.startswith(('--include=', '--exclude='))]
    checks += 2
    control_source = root / 'base-control-source'
    remote_code = root / 'simulated-remote/code'
    control_destination = remote_code / 'math_l4_rl_split_v1'
    control_keep = [
        'scripts/watch_base_only.py', 'scripts/switch_after_base.py', 'scripts/pidfd_compat.py',
        'reference/base_manifest.json', 'reference/source_identity.json',
        'reference/nested/fixture.json',
    ]
    control_omit = [
        'scripts/run_shard.py', 'scripts/evaluate_shard.py', 'README.md',
        'math_l4_v1/scripts/evaluate.py', 'math_l4_v1/configs/math_l4.json',
        'models/Qwen3.5-2B-Base/model.safetensors',
        'payload/code/math_l4_v1/outputs/math_l4_v1_fix1/run_state.json',
        'outputs/existing-shard-result.json', 'outputs/must-not-upload.json',
        '.venv/bin/python',
    ]
    for relative in control_keep + control_omit:
        path = control_source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('New controller-source fixture: ' + relative + '\n')
    protected_files = [
        remote_code / 'math_l4_v1/scripts/evaluate.py',
        remote_code / 'math_l4_v1/outputs/math_l4_v1_fix1/run_state.json',
        remote_code / 'math_l4_v1/outputs/math_l4_v1_fix1/test_base/samples.jsonl',
        remote_code / 'math_l4_v1/outputs/math_l4_v1_fix1/test_rl/samples.jsonl',
        remote_code / 'math_l4_v1/outputs/math_l4_v1_fix1/rl_shards/shard_0/samples.jsonl',
        remote_code / 'models/Qwen3.5-2B-Base/model.safetensors',
        control_destination / 'outputs/existing-shard-result.json',
        control_destination / 'outputs/local-only-result.json',
    ]
    protected_bytes = {}
    for path in protected_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Existing remote data must remain byte-identical: ' + path.name + '\n')
        protected_bytes[path] = path.read_bytes()
    for repeat in range(2):
        result = subprocess.run(['rsync', '-a', *control_filters,
            str(control_source) + '/', str(control_destination) + '/'],
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (result.stdout, result.stderr)
        actual = {path.relative_to(control_destination).as_posix()
            for path in control_destination.rglob('*') if path.is_file()}
        expected = set(control_keep) | {'outputs/existing-shard-result.json', 'outputs/local-only-result.json'}
        assert actual == expected, {'unexpected': sorted(actual - expected), 'missing': sorted(expected - actual)}
        assert all(path.read_bytes() == expected_bytes for path, expected_bytes in protected_bytes.items())
        assert all((control_destination / relative).read_bytes() == (control_source / relative).read_bytes()
            for relative in control_keep)
        checks += 4

    project = root / 'code'
    extension = project / 'math_l4_rl_split_v1'
    snapshot = root / 'downloaded-snapshot'
    experiment = snapshot / 'math_l4_v1'
    run = experiment / 'outputs/math_l4_v1_fix1'
    model = snapshot / 'models/Qwen3.5-2B-Base'
    for path in (extension / 'scripts', extension / 'reference', experiment / 'scripts',
            experiment / 'configs', experiment / 'data/processed', run / 'formal', model):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle / 'scripts/build_payload.py', extension / 'scripts/build_payload.py')

    # Only the original selection reconstruction is a trusted, hash-bound fixture.
    # Production payload construction, hashing and original fingerprinting execute.
    selector_fixture = '''import hashlib,json
from pathlib import Path
experiment = Path(__file__).resolve().parents[1]
SELECTION_RESULT = json.loads(Path(SELECTION_PATH).read_text())
for candidate in SELECTION_RESULT['candidates']:
    for name, expected in candidate['adapter_files'].items():
        assert hashlib.sha256((experiment/candidate['adapter_path']/name).read_bytes()).hexdigest() == expected
    dev = experiment/candidate['dev_output']
    manifest = json.loads((dev/'manifest.json').read_text())
    assert manifest['complete'] is True
    assert hashlib.sha256((dev/'samples.jsonl').read_bytes()).hexdigest() == manifest['samples_sha256']
'''
    (experiment / 'scripts/validate_selection.py').write_text(selector_fixture)
    shutil.copyfile(frozen / 'scripts/fingerprint_model.py', experiment / 'scripts/fingerprint_model.py')
    (experiment / 'configs/math_l4.json').write_text('{"fixture": true}\n')
    (experiment / 'data/processed/test.jsonl').write_text('{"id":"synthetic-1","answer":"1"}\n')
    model_content = {
        'config.json': b'{"model_type":"synthetic_cpu_fixture"}\n',
        'tokenizer_config.json': b'{"eos_token_id":1}\n',
        'tokenizer.json': b'{"synthetic":true}\n',
    }
    tensor_header = json.dumps({'fixture.weight': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
    model_content['model.safetensors'] = len(tensor_header).to_bytes(8, 'little') + tensor_header + b'\x00\x00'
    for name, content in model_content.items():
        (model / name).write_bytes(content)
    model_files = {name: {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        for name, content in model_content.items()}
    model_identity = hashlib.sha256(json.dumps(model_files, sort_keys=True).encode()).hexdigest()
    fingerprint = runpy.run_path(str(experiment / 'scripts/fingerprint_model.py'), init_globals={'MODEL_PATH': model})
    assert fingerprint['model_files'] == model_files and fingerprint['model_identity'] == model_identity
    checks += 1

    candidates = []
    for step in (128, 256, 384):
        adapter_relative = f'outputs/math_l4_v1_fix1/formal/checkpoint-{step}'
        dev_relative = f'outputs/math_l4_v1_fix1/dev/checkpoint-{step}'
        adapter = experiment / adapter_relative
        dev = experiment / dev_relative
        adapter.mkdir(parents=True)
        dev.mkdir(parents=True)
        (adapter / 'adapter_config.json').write_text('{"fixture":true}\n')
        (adapter / 'adapter_model.safetensors').write_bytes(f'Synthetic adapter {step}; never loaded'.encode())
        (adapter / 'resume_state.json').write_text(json.dumps({'step': step, 'synthetic': True}) + '\n')
        samples = (json.dumps({'id': 'synthetic-1', 'sample_index': 0, 'correct': True}) + '\n').encode()
        (dev / 'samples.jsonl').write_bytes(samples)
        (dev / 'manifest.json').write_text(json.dumps({'complete': True,
            'samples_sha256': hashlib.sha256(samples).hexdigest()}) + '\n')
        candidates.append({'step': step, 'adapter_path': adapter_relative, 'dev_output': dev_relative,
            'adapter_files': {name: hashlib.sha256((adapter / name).read_bytes()).hexdigest()
                for name in ('adapter_config.json', 'adapter_model.safetensors')}})
    selection = {'selected_step': 384, 'model_identity': model_identity, 'candidates': candidates,
        'adapter_path': candidates[-1]['adapter_path']}
    selection['selection_id'] = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    selection_path = run / 'selection.json'
    selection_path.write_text(json.dumps(selection, indent=2) + '\n')
    completion_files = {}
    for relative, content in {'manifest.json': '{"synthetic":true}\n', 'trainer_state.json': '{"global_step":384}\n'}.items():
        path = run / 'formal' / relative
        path.write_text(content)
        completion_files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / 'formal/report.json').write_text(json.dumps({'completion_files': completion_files}) + '\n')

    frozen_files = ['scripts/validate_selection.py', 'scripts/fingerprint_model.py',
        'configs/math_l4.json', 'data/processed/test.jsonl']
    frozen_manifest = {'files_sha256': {name: hashlib.sha256((experiment / name).read_bytes()).hexdigest()
        for name in frozen_files}}
    frozen_manifest_path = experiment / 'bundle_manifest.json'
    frozen_manifest_path.write_text(json.dumps(frozen_manifest, indent=2) + '\n')
    installer_files = {}
    for relative in ('scripts/prepare_packages.py', 'requirements-gpu.txt',
            'packages/linux-cp311-cu128/manifest.json', 'packages/linux-cp311-cu128/requirements-lock.txt'):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Synthetic installer input: ' + relative + '\n')
        installer_files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    reference = {'bundle_manifest_sha256': hashlib.sha256(frozen_manifest_path.read_bytes()).hexdigest(),
        'selection_sha256': hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        'selection_id': selection['selection_id'], 'model_identity': model_identity,
        'installer_files': installer_files}
    (extension / 'reference/source_identity.json').write_text(json.dumps(reference, indent=2) + '\n')
    (extension / 'reference/base_manifest.json').write_text('{"synthetic":true}\n')

    active_paths = ['run_state.json', 'test_base/samples.jsonl', 'test_rl/samples.jsonl',
        'test_base/live.json', 'test_rl/live.json']
    for relative in active_paths:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"active_fixture":true}\n')
    (experiment / 'experiment-fix1.log').write_text('Synthetic active experiment log.\n')
    command = [sys.executable, '-S', str(extension / 'scripts/build_payload.py'), '--snapshot-root', str(snapshot)]
    result = subprocess.run([*command, '--plan'], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.stdout, result.stderr)
    plan = json.loads(result.stdout)
    assert plan['selected_step'] == 384 and plan['missing_selection_files'] == []
    assert plan['payload_ready'] is False and not (extension / 'payload/ready.json').exists()
    checks += 3

    missing_path = experiment / candidates[0]['adapter_path'] / 'resume_state.json'
    original_missing = missing_path.read_bytes()
    missing_path.unlink()
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and 'Selection evidence is incomplete' in result.stderr
    missing_path.write_bytes(original_missing)
    checks += 1

    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.stdout, result.stderr)
    ready_path = extension / 'payload/ready.json'
    ready = json.loads(ready_path.read_text())
    payload = extension / 'payload/code'
    assert ready['complete'] is True and ready['selected_step'] == 384
    assert ready['model_files'] == model_files and ready['model_identity'] == model_identity
    assert ready['uploads_active_test_outputs'] is False and ready['gpu_executed'] is False
    assert all(set(metadata) == {'bytes', 'sha256'} for metadata in ready['model_files'].values())
    actual = {path.relative_to(payload).as_posix() for path in payload.rglob('*') if path.is_file()}
    assert actual == set(ready['files'])
    assert not any('/test_base/' in name or '/test_rl/' in name or 'run_state' in name
        or 'live' in Path(name).name or 'experiment' in Path(name).name for name in actual)
    for step in (128, 256, 384):
        assert f'math_l4_v1/outputs/math_l4_v1_fix1/formal/checkpoint-{step}/adapter_model.safetensors' in actual
        assert f'math_l4_v1/outputs/math_l4_v1_fix1/dev/checkpoint-{step}/samples.jsonl' in actual
    checks += 9
    verify_command = [sys.executable, '-S', str(extension / 'scripts/build_payload.py'), '--verify-ready']
    result = subprocess.run(verify_command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout.strip() == str(model), (result.stdout, result.stderr)
    checks += 1

    # Every mutation is restored before the next independent rejection check.
    mutation_paths = [selection_path, experiment / 'configs/math_l4.json', frozen_manifest_path,
        project / 'requirements-gpu.txt', model / 'model.safetensors']
    for path in mutation_paths:
        original = path.read_bytes()
        path.write_bytes(original[:-1] + b'\x01' if path == model / 'model.safetensors' else original + b' ')
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0, ('Unexpectedly accepted changed source', str(path), result.stdout)
        path.write_bytes(original)
        checks += 1
    for path in (model / 'model.safetensors', payload / 'math_l4_v1/outputs/math_l4_v1_fix1/selection.json',
            payload / 'math_l4_v1/configs/math_l4.json'):
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        result = subprocess.run(verify_command, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0, ('Unexpectedly verified changed file', str(path), result.stdout)
        path.write_bytes(original)
        checks += 1
    original_ready = ready_path.read_bytes()
    for field in ('bytes', 'sha256'):
        altered = json.loads(original_ready)
        altered['model_files']['model.safetensors'][field] = 999 if field == 'bytes' else '0' * 64
        ready_path.write_text(json.dumps(altered))
        result = subprocess.run(verify_command, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0, ('Unexpectedly accepted wrong model metadata', field)
        ready_path.write_bytes(original_ready)
        checks += 1

    weight = model / 'model.safetensors'
    original_weight = weight.read_bytes()
    weight.write_bytes(original_weight[:-1] + b'\x01')
    altered = json.loads(original_ready)
    altered['model_files']['model.safetensors']['sha256'] = hashlib.sha256(weight.read_bytes()).hexdigest()
    altered['model_identity'] = hashlib.sha256(json.dumps(altered['model_files'], sort_keys=True).encode()).hexdigest()
    ready_path.write_text(json.dumps(altered))
    result = subprocess.run(verify_command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, 'Self-consistent replacement Base weights bypassed the pinned reference identity.'
    weight.write_bytes(original_weight)
    ready_path.write_bytes(original_ready)
    checks += 1

    extra = payload / 'math_l4_v1/outputs/math_l4_v1_fix1/test_base/samples.jsonl'
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text('{"must_not_upload":true}\n')
    result = subprocess.run(verify_command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, 'Upload verification accepted an unmanaged active result file.'
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, 'Rebuild accepted an unmanaged active result file.'
    extra.unlink()
    checks += 2
    result = subprocess.run(verify_command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.stdout, result.stderr)
    checks += 1

print(f'[PASS] {checks} transfer checks | real local rsync and payload builder; synthetic selection fixture and tiny valid weights; no GPU/model inference/network')
