"""Check direct-server transfer selection using real local rsync and isolated files."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

bundle = Path(__file__).resolve().parents[1]
checks = 0
assert shutil.which('rsync'), 'A local rsync is required; no network is used.'
file_list = bundle / 'reference/direct_source_files.txt'
selected = file_list.read_text().splitlines()
assert selected and len(selected) == len(set(selected))
assert selected == sorted(selected)
assert all(not Path(name).is_absolute() and '..' not in Path(name).parts and '\\' not in name
           and name.strip() == name and not any(ord(c) < 32 for c in name) for name in selected)
checks += 3
run = 'math_l4_v1/outputs/math_l4_v1_fix1'
required = {
    'math_l4_v1/bundle_manifest.json', 'math_l4_v1/configs/math_l4.json',
    'math_l4_v1/scripts/evaluate.py', 'math_l4_v1/scripts/validate_selection.py',
    'math_l4_v1/data/processed/test.jsonl', run + '/selection.json', run + '/formal/report.json',
    'scripts/prepare_packages.py', 'requirements-gpu.txt',
    'packages/linux-cp311-cu128/manifest.json', 'packages/linux-cp311-cu128/requirements-lock.txt',
}
for step in (128, 256, 384):
    required.update(f'{run}/formal/checkpoint-{step}/{name}' for name in
                    ('adapter_config.json', 'adapter_model.safetensors', 'resume_state.json'))
    required.update(f'{run}/dev/checkpoint-{step}/{name}' for name in ('manifest.json', 'samples.jsonl'))
assert required <= set(selected), sorted(required - set(selected))
checks += 1
assert not any('/test_base/' in name or '/test_rl/' in name or '/rl_shards/' in name
               or Path(name).name in {'run_state.json', 'optimizer.pt', 'scheduler.pt'}
               or 'rng_state' in name or name.endswith('.log') or '/.venv/' in name
               for name in selected)
assert not any('/checkpoint-' in name and not any(f'/checkpoint-{step}/' in name for step in (128, 256, 384))
               for name in selected)
assert not any(name.startswith('models/') for name in selected)
checks += 3
upload_text = (bundle / 'upload_direct_tools.sh').read_text().replace('\\\n', ' ')
pull_text = (bundle / 'pull_from_source.sh').read_text().replace('\\\n', ' ')
upload_lines = [line for line in upload_text.splitlines() if line.strip().startswith('rsync ')]
pull_lines = [line for line in pull_text.splitlines() if line.strip().startswith('rsync ')]
assert len(upload_lines) == 1 and len(pull_lines) == 2
upload_arguments = shlex.split(upload_lines[0])
assert '--relative' in upload_arguments or '-R' in upload_arguments
assert not any(arg.startswith('--delete') for line in upload_lines + pull_lines for arg in shlex.split(line))
assert '--files-from=' in pull_lines[1] and 'direct_source_files.txt' in pull_text
assert '--snapshot-root' in pull_text and 'build_payload.py' in pull_text
checks += 5

# Use the production source arguments, including its working-directory contract.
assert 'cd -- "$TRANSFER_CODE"' in upload_text
assert upload_text.index('cd -- "$TRANSFER_CODE"') < upload_text.index(upload_lines[0])
checks += 1
upload_sources = []
skip_value = False
for argument in upload_arguments[1:-1]:
    if skip_value:
        skip_value = False
        continue
    if argument in {'-e', '--rsh', '--rsync-path'}:
        skip_value = True
    elif argument.startswith('-'):
        continue
    else:
        upload_sources.append(argument)
assert upload_sources and not skip_value
assert all(not Path(argument).is_absolute() and '/./' not in argument
           and '$' not in argument and '..' not in Path(argument).parts for argument in upload_sources)
checks += 1
assert upload_arguments[-1] == '$TRANSFER_HOST:/root/code/'
assert 'math_l4_rl_split_v1/' in upload_sources and 'scripts/prepare_packages.py' in upload_sources
checks += 1
assert 'sha256sum --check --strict' in upload_text
assert upload_text.index('sha256sum --check --strict') < upload_text.index('[DIRECT TOOLS READY]')
checks += 1

with tempfile.TemporaryDirectory(prefix='math-direct-transfer-') as temporary:
    root = Path(temporary)
    inventory_bundle = root / 'inventory-check-bundle'
    (inventory_bundle / 'reference').mkdir(parents=True)
    inventory_path = inventory_bundle / 'reference/direct_source_files.txt'
    checker = re.search(r"<<'PY'\n(.*?)\nPY", pull_text, re.S).group(1)
    for scenario in ['valid', 'hash_changed', 'absolute', 'traversal', 'active_answers',
                     'runtime_state', 'duplicate', 'missing_required']:
        candidate = list(selected)
        if scenario == 'hash_changed': candidate.append('unexpected.txt')
        if scenario == 'absolute': candidate.append('/root/.ssh/id_ed25519')
        if scenario == 'traversal': candidate.append('../outside.txt')
        if scenario == 'active_answers': candidate.append(run + '/test_base/samples.jsonl')
        if scenario == 'runtime_state': candidate.append(run + '/run_state.json')
        if scenario == 'duplicate': candidate.append(candidate[0])
        if scenario == 'missing_required': candidate.remove('scripts/prepare_packages.py')
        content = ('\n'.join(sorted(candidate)) + '\n').encode()
        inventory_path.write_bytes(content)
        expected = file_list.read_bytes() if scenario == 'hash_changed' else content
        (inventory_bundle / 'release_manifest.json').write_text(json.dumps({'files': {
            'reference/direct_source_files.txt': {'sha256': hashlib.sha256(expected).hexdigest()}}}))
        result = subprocess.run([sys.executable, '-', str(inventory_bundle)], input=checker,
                                capture_output=True, text=True, timeout=15)
        if scenario == 'valid':
            assert result.returncode == 0 and '[DIRECT PLAN]' in result.stdout, result.stderr
        else:
            assert result.returncode != 0, (scenario, result.stdout, result.stderr)
            assert any(message in result.stderr for message in ['inventory changed', 'Unsafe transfer path',
                       'Active paths are forbidden', 'Runtime state is forbidden',
                       'Inventory must be sorted and unique', 'Missing direct transfer inputs']), result.stderr
        checks += 1
    source = root / 'old-server/code'
    destination = root / 'new-server/code'
    omit = [run + '/test_base/samples.jsonl', run + '/test_base/manifest.json',
            run + '/test_rl/samples.jsonl', run + '/rl_shards/shard_0/samples.jsonl',
            run + '/rl_shards/shard_1/samples.jsonl', run + '/run_state.json',
            run + '/formal/checkpoint-384/optimizer.pt', run + '/formal/checkpoint-384/rng_state.pth',
            run + '/formal/checkpoint-16/adapter_model.safetensors',
            'models/Qwen3.5-2B-Base/model.safetensors', '.venv/bin/python',
            'math_l4_v1/experiment-fix1.log']
    for name in selected + omit:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Old-server synthetic file: ' + name + '\n')
    protected = {name: ('Existing destination output: ' + name + '\n').encode() for name in omit}
    for name, content in protected.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    source_hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in selected + omit}
    for repeat in range(2):
        result = subprocess.run(['rsync', '-a', '--files-from=' + str(file_list), str(source) + '/', str(destination) + '/'],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (result.stdout, result.stderr)
        actual = {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
        assert actual == set(selected) | set(protected)
        assert all((destination / name).read_bytes() == (source / name).read_bytes() for name in selected)
        assert all((destination / name).read_bytes() == content for name, content in protected.items())
        assert all(hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
                   for name, digest in source_hashes.items())
        checks += 5

    missing_candidate = source / run / 'formal/checkpoint-256/resume_state.json'
    saved_candidate = missing_candidate.read_bytes()
    missing_candidate.unlink()
    result = subprocess.run(['rsync', '-a', '--files-from=' + str(file_list), str(source) + '/', str(destination) + '/'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0, 'Missing candidate evidence must fail the transfer.'
    assert all((destination / name).read_bytes() == content for name, content in protected.items())
    missing_candidate.write_bytes(saved_candidate)
    checks += 2

    light_source = root / 'local source with spaces/code'
    light_destination = root / 'new-light-code'
    keep = ['math_l4_rl_split_v1/scripts/evaluate_shard.py',
            'math_l4_rl_split_v1/scripts/build_payload.py',
            'math_l4_rl_split_v1/reference/direct_source_files.txt',
            'math_l4_rl_split_v1/reference/source_identity.json',
            'math_l4_rl_split_v1/setup_worker.sh', 'math_l4_rl_split_v1/pull_from_source.sh',
            'math_l4_rl_split_v1/README.md', 'math_l4_rl_split_v1/release_manifest.json',
            'scripts/prepare_packages.py', 'requirements-gpu.txt',
            'packages/linux-cp311-cu128/manifest.json', 'packages/linux-cp311-cu128/requirements-lock.txt']
    light_omit = ['math_l4_rl_split_v1/payload/code/models/model.safetensors',
                  'math_l4_rl_split_v1/outputs/worker_0.json',
                  'math_l4_rl_split_v1/validation/report.json',
                  'math_l4_rl_split_v1/__pycache__/example.pyc',
                  'math_l4_rl_split_v1/rl-batch-0.log', 'math_l4_rl_split_v1/.DS_Store',
                  'models/Qwen3.5-2B-Base/model.safetensors',
                  'math_l4_v1/outputs/math_l4_v1_fix1/test_base/samples.jsonl',
                  '.venv/bin/python', 'packages/linux-cp311-cu128/wheels/torch.whl']
    for name in keep + light_omit:
        path = light_source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('New lightweight code: ' + name + '\n')
    light_destination.mkdir()
    light_protected = {name: ('Preserve new-server file: ' + name + '\n').encode() for name in light_omit}
    for name, content in light_protected.items():
        path = light_destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    filters = [arg for arg in upload_arguments if arg.startswith(('--include=', '--exclude='))]
    for repeat in range(2):
        result = subprocess.run(['rsync', '-a', '--relative', *filters, *upload_sources, str(light_destination) + '/'],
                                cwd=light_source, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (result.stdout, result.stderr)
        actual = {p.relative_to(light_destination).as_posix() for p in light_destination.rglob('*') if p.is_file()}
        assert actual == set(keep) | set(light_protected)
        assert all((light_destination / name).read_bytes() == (light_source / name).read_bytes() for name in keep)
        assert all((light_destination / name).read_bytes() == content for name, content in light_protected.items())
        checks += 4

for filename in ('upload_direct_tools.sh', 'pull_from_source.sh'):
    result = subprocess.run(['bash', '-n', str(bundle / filename)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(['bash', str(bundle / filename), 'not-a-port', 'root@example.invalid'], capture_output=True, text=True)
    assert result.returncode != 0, (result.stdout, result.stderr)
    checks += 2
tree = ast.parse(Path(__file__).read_text())
assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
checks += 1
print(f'[PASS] {checks} direct-transfer checks | exact source selection, real local rsync, repeated destination preservation, production upload source arguments, spaced paths and remote-check ordering | no SSH/GPU/network')
