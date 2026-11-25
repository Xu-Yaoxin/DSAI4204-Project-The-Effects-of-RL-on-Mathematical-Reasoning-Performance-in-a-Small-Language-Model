"""CPU fixtures exercise real planning, full-grid merge, tamper rejection and subprocess recovery."""
import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import itertools
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, patch

bundle = Path(__file__).resolve().parents[1]
frozen = bundle.parent / 'math_l4_v1'
real_run = runpy.run_path
checks = 0
for name in ('evaluate_shard.py', 'merge_shards.py', 'run_shard.py'):
    tree = ast.parse((bundle / 'scripts' / name).read_text())
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
    checks += 1

with tempfile.TemporaryDirectory(prefix='math-rl-split-contract-') as temporary:
    base = Path(temporary).resolve()
    root = base / 'math_l4_v1'
    extension = base / 'math_l4_rl_split_v1'
    source_manifest = json.loads((frozen / 'bundle_manifest.json').read_text())
    for relative in ['bundle_manifest.json', *source_manifest['files_sha256']]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(frozen / relative, target)
    (extension / 'scripts').mkdir(parents=True)
    (extension / 'reference').mkdir()
    for name in ('evaluate_shard.py', 'merge_shards.py', 'run_shard.py'):
        shutil.copyfile(bundle / 'scripts' / name, extension / 'scripts' / name)
    baseline = json.loads((bundle / 'reference/base_manifest.json').read_text())
    selection = json.loads((bundle / 'reference/selection.json').read_text())
    run = root / 'outputs/math_l4_v1_fix1'
    run.mkdir(parents=True)
    (run / 'selection.json').write_text(json.dumps(selection))
    (extension / 'reference/selection.json').write_text(json.dumps(selection))
    (extension / 'reference/base_manifest.json').write_text(json.dumps(baseline))
    descriptions = []
    for index in (0, 1):
        command = [sys.executable, '-S', str(extension / 'scripts/evaluate_shard.py'),
            '--experiment-root', str(root), '--shard-index', str(index), '--plan']
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert 'assigned_questions=100/200' in result.stdout and 'responses=1600' in result.stdout
        descriptions.append(real_run(str(extension / 'scripts/evaluate_shard.py'), init_globals={
            'DESCRIBE_ONLY': True, 'SHARD_ARGUMENTS': ['--experiment-root', str(root), '--shard-index', str(index)]})['SHARD_EXTENSION'])
        checks += 2
    assert not (run / 'rl_shards').exists()
    assert descriptions[0]['transformed_evaluator_sha256'] == descriptions[1]['transformed_evaluator_sha256']
    assert descriptions[0]['assigned_question_ids_sha256'] != descriptions[1]['assigned_question_ids_sha256']
    checks += 3
    for mutation in ('dataset', 'selection', 'source'):
        path = {'dataset': root / 'data/processed/test.jsonl', 'selection': run / 'selection.json',
            'source': root / 'scripts/evaluate.py'}[mutation]
        original = path.read_bytes()
        path.write_bytes(original + b' ')
        if mutation == 'selection':
            changed = json.loads(original)
            changed['selected_step'] = 128
            path.write_text(json.dumps(changed))
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0, mutation
        path.write_bytes(original)
        checks += 1

    # Only development selection reconstruction is stubbed here. All production
    # merge, original sample validators, scoring, source and adapter hashes run.
    adapter = run / 'formal/checkpoint-384'
    adapter.mkdir(parents=True)
    (adapter / 'adapter_config.json').write_text('{}')
    (adapter / 'adapter_model.safetensors').write_bytes(b'CPU fixture adapter, never loaded')
    digest = hashlib.sha256()
    adapter_files = {}
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        content = (adapter / name).read_bytes()
        digest.update(name.encode())
        digest.update(content)
        adapter_files[name] = hashlib.sha256(content).hexdigest()
    selection['adapter_files'] = adapter_files
    selection['adapter_identity'] = digest.hexdigest()
    selection_payload = {key: value for key, value in selection.items() if key != 'selection_id'}
    selection['selection_id'] = hashlib.sha256(json.dumps(selection_payload, sort_keys=True).encode()).hexdigest()
    (run / 'selection.json').write_text(json.dumps(selection))
    (extension / 'reference/selection.json').write_text(json.dumps(selection))
    test_rows = [json.loads(line) for line in (root / 'data/processed/test.jsonl').read_text().splitlines()]
    verifier = compile((root / 'scripts/verify_answers.py').read_text(), str(root / 'scripts/verify_answers.py'), 'exec')
    config = json.loads((root / 'configs/math_l4.json').read_text())
    for index in (0, 1):
        descriptor = real_run(str(extension / 'scripts/evaluate_shard.py'), init_globals={
            'DESCRIBE_ONLY': True, 'SHARD_ARGUMENTS': ['--experiment-root', str(root), '--shard-index', str(index)]})['SHARD_EXTENSION']
        output = run / 'rl_shards' / f'shard_{index}'
        output.mkdir(parents=True)
        records = []
        for position, row in enumerate(test_rows):
            if position % 2 != index:
                continue
            text = '#### ' + row['answer']
            namespace = {'SCORE_INPUT': [{'text': text, 'answer': row['answer'], 'truncated': False}]}
            exec(verifier, namespace)
            score = namespace['SCORE_OUTPUT'][0]
            assert score['correct'] is True
            for sample in range(16):
                seed = int.from_bytes(hashlib.sha256(f"{config['evaluation']['seed']}:{row['id']}:{sample}".encode()).digest()[:4], 'big')
                records.append({**row, **score, 'sample_index': sample, 'sample_seed': seed,
                    'text': text, 'raw_text': text, 'output_token_ids': [17, 248044], 'output_tokens': 2,
                    'prompt_tokens': 5, 'generation_limit': 262139, 'seconds': 1.0,
                    'termination_reason': 'eos', 'scoring_evaluable': True, 'guard_evidence': None, 'timeout_evidence': None})
        samples = ''.join(json.dumps(row) + '\n' for row in records).encode()
        manifest = copy.deepcopy(baseline)
        manifest.update(model_label='rl', adapter_path=selection['adapter_path'], adapter_files=adapter_files,
            adapter_identity=selection['adapter_identity'], selection_identity=selection['selection_id'],
            selection_path='outputs/math_l4_v1_fix1/selection.json', complete=True, total_samples=1600,
            samples_sha256=hashlib.sha256(samples).hexdigest(), execution_extension=descriptor,
            sharding={'count': 2, 'index': index, 'assignment': 'question_index_modulo'})
        (output / 'samples.jsonl').write_bytes(samples)
        (output / 'manifest.json').write_text(json.dumps(manifest))
    merge_argv = [str(extension / 'scripts/merge_shards.py'), '--experiment-root', str(root)]
    with patch.object(sys, 'argv', merge_argv), patch('runpy.run_path', wraps=real_run,
            side_effect=itertools.chain([DEFAULT, DEFAULT, {'SELECTION_RESULT': selection}], itertools.repeat(DEFAULT))):
        real_run(merge_argv[0], run_name='__main__')
    merged = run / 'test_rl_split_merged'
    result = real_run(str(root / 'scripts/validate_evaluation.py'), init_globals={
        'EVALUATION_PATH': merged, 'VALIDATE_CONFIG': config})['EVALUATION_RESULT']
    assert len(result['rows']) == 3200 and result['manifest'].get('sharding') is None
    assert result['manifest']['protocol_id'] == baseline['protocol_id']
    assert result['manifest']['execution_extension']['sample_content_modified'] is False
    assert len(result['manifest']['execution_extension']['sources']) == 2
    checks += 5
    shard = run / 'rl_shards/shard_0'
    sample_original = (shard / 'samples.jsonl').read_bytes()
    manifest_original = (shard / 'manifest.json').read_bytes()
    for mutation in ('wrong_shard', 'incomplete', 'duplicate', 'seed', 'score', 'adapter'):
        manifest = json.loads(manifest_original)
        records = [json.loads(line) for line in sample_original.splitlines()]
        if mutation == 'wrong_shard':
            manifest['sharding']['index'] = 1
        elif mutation == 'incomplete':
            manifest['complete'] = False
        elif mutation == 'duplicate':
            records[1] = dict(records[0])
        elif mutation == 'seed':
            records[0]['sample_seed'] += 1
        elif mutation == 'score':
            records[0]['correct'] = False
        elif mutation == 'adapter':
            (adapter / 'adapter_model.safetensors').write_bytes(b'tampered')
        raw = ''.join(json.dumps(row) + '\n' for row in records).encode()
        manifest['samples_sha256'] = hashlib.sha256(raw).hexdigest()
        (shard / 'manifest.json').write_text(json.dumps(manifest))
        (shard / 'samples.jsonl').write_bytes(raw)
        failed = False
        with patch.object(sys, 'argv', merge_argv), patch('runpy.run_path', wraps=real_run,
                side_effect=itertools.chain([DEFAULT, DEFAULT, {'SELECTION_RESULT': selection}], itertools.repeat(DEFAULT))):
            try:
                real_run(merge_argv[0], run_name='__main__')
            except ValueError:
                failed = True
        assert failed, mutation
        (shard / 'manifest.json').write_bytes(manifest_original)
        (shard / 'samples.jsonl').write_bytes(sample_original)
        (adapter / 'adapter_model.safetensors').write_bytes(b'CPU fixture adapter, never loaded')
        checks += 1

    # Execute the actual transformed evaluator over 1,598 saved answers. Only
    # selection reconstruction, model identity/loading and CUDA packages are
    # mocked; the frozen cache, seed, streamer, scorer and append logic execute.
    resume_prefix = b'\n'.join(sample_original.splitlines()[:-2]) + b'\n'
    resume_manifest = json.loads(manifest_original)
    resume_manifest['complete'] = False
    (shard / 'samples.jsonl').write_bytes(resume_prefix)
    (shard / 'manifest.json').write_text(json.dumps(resume_manifest))
    (shard / '.evaluation.lock').touch()
    identity_fixture = {'model_identity': baseline['model_identity'], 'model_files': baseline['model_files']}
    package_names = ('torch', 'transformers', 'peft', 'accelerate', 'datasets', 'trl', 'safetensors',
        'flash-linear-attention', 'fla-core', 'causal-conv1d')
    package_values = [baseline['settings']['package_versions'][name] if baseline['settings']['package_versions'][name] is not None
        else importlib.metadata.PackageNotFoundError(name) for name in package_names]
    evaluation_argv = [str(extension / 'scripts/evaluate_shard.py'), '--experiment-root', str(root), '--shard-index', '0']
    resume_hook_results = [DEFAULT] * 6 + [{'SELECTION_RESULT': selection}, identity_fixture]
    captured = io.StringIO()
    with patch.object(sys, 'argv', evaluation_argv + ['--cache-only']), contextlib.redirect_stdout(captured), \
            patch('importlib.metadata.version', side_effect=list(package_values)), \
            patch('runpy.run_path', wraps=real_run,
                side_effect=itertools.chain(resume_hook_results, itertools.repeat(DEFAULT))) as routed:
        try:
            real_run(evaluation_argv[0], run_name='__main__')
        except SystemExit as error:
            assert error.code == 3
        else:
            raise AssertionError('Incomplete real evaluator cache must exit 3.')
    assert 'verified=1598/1600' in captured.getvalue()
    assert not any(str(call.args[0]).endswith('/load_model.py') for call in routed.call_args_list)
    assert (shard / 'samples.jsonl').read_bytes() == resume_prefix
    checks += 3

    input_tensor = MagicMock()
    input_tensor.shape = (1, 5)
    input_tensor.to.return_value = input_tensor
    input_tensor.tolist.return_value = [[100] * 5]
    token_tensor = MagicMock()
    token_tensor.tolist.return_value = [17, 248044]
    returned_tensor = MagicMock()
    returned_tensor.__getitem__.return_value.tolist.return_value = [17, 248044]
    tokenizer = MagicMock(return_value={'input_ids': input_tensor, 'attention_mask': input_tensor})
    tokenizer.eos_token_id = 248044
    tokenizer.pad_token_id = 248044
    tokenizer.decode.return_value = '#### 0'
    generate = MagicMock()
    producer_threads = []
    generate.side_effect = (
        (producer_threads.append(threading.current_thread().name),
         generate.call_args.kwargs['streamer'].put(input_tensor),
         generate.call_args.kwargs['streamer'].put(token_tensor),
         generate.call_args.kwargs['streamer'].end(), returned_tensor)[-1]
        for _ in range(2)
    )
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=262144), generate=generate)
    loaded_fixture = {**identity_fixture, 'adapter_identity': selection['adapter_identity'],
        'package_versions': baseline['settings']['package_versions'], 'model': model, 'tokenizer': tokenizer}
    cuda = SimpleNamespace(get_device_name=MagicMock(return_value='NVIDIA GeForce RTX 4090'),
        reset_peak_memory_stats=MagicMock(), max_memory_allocated=MagicMock(return_value=0))
    torch_fixture = SimpleNamespace(cuda=cuda, set_num_threads=MagicMock(), set_num_interop_threads=MagicMock())
    seed_setter = MagicMock()
    transformers_fixture = SimpleNamespace(set_seed=seed_setter,
        MaxTimeCriteria=MagicMock(return_value=SimpleNamespace(max_time=300)))
    captured = io.StringIO()
    with patch.object(sys, 'argv', evaluation_argv), contextlib.redirect_stdout(captured), \
            patch.dict(os.environ, {'RLVR_CPU_THREADS': '4'}), \
            patch.dict(sys.modules, {'torch': torch_fixture, 'transformers': transformers_fixture}), \
            patch('importlib.metadata.version', side_effect=list(package_values)), \
            patch('runpy.run_path', wraps=real_run,
                side_effect=itertools.chain(resume_hook_results + [loaded_fixture], itertools.repeat(DEFAULT))):
        real_run(evaluation_argv[0], run_name='__main__')
    completed_bytes = (shard / 'samples.jsonl').read_bytes()
    resumed_rows = [json.loads(line) for line in completed_bytes.splitlines()]
    assert generate.call_count == 2 and len(producer_threads) == 2
    assert all(name != threading.current_thread().name for name in producer_threads)
    assert completed_bytes.startswith(resume_prefix) and len(resumed_rows) == 1600
    assert len({(row['id'], row['sample_index']) for row in resumed_rows}) == 1600
    prior_rows = [json.loads(line) for line in resume_prefix.splitlines()]
    assert resumed_rows[:1598] == prior_rows
    missing_keys = [(row['id'], row['sample_index']) for row in [json.loads(line) for line in sample_original.splitlines()][-2:]]
    assert [(row['id'], row['sample_index']) for row in resumed_rows[-2:]] == missing_keys
    assert [call.args[0] for call in seed_setter.call_args_list] == [row['sample_seed'] for row in resumed_rows[-2:]]
    assert json.loads((shard / 'manifest.json').read_text())['complete'] is True
    assert 'remaining=2' in captured.getvalue() and '1600 model answers' in captured.getvalue()
    checks += 8

    captured = io.StringIO()
    generate.reset_mock()
    with patch.object(sys, 'argv', evaluation_argv), contextlib.redirect_stdout(captured), \
            patch('importlib.metadata.version', side_effect=list(package_values)), \
            patch('runpy.run_path', wraps=real_run,
                side_effect=itertools.chain(resume_hook_results, itertools.repeat(DEFAULT))) as routed:
        try:
            real_run(evaluation_argv[0], run_name='__main__')
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError('Complete real evaluator cache must return before loading a model.')
    assert '[EVAL CACHED]' in captured.getvalue() and '1600 answers verified' in captured.getvalue()
    assert not any(str(call.args[0]).endswith('/load_model.py') for call in routed.call_args_list)
    assert generate.call_count == 0 and (shard / 'samples.jsonl').read_bytes() == completed_bytes
    checks += 3

    worker_fixture = r'''import json,os,sys,time
from pathlib import Path
args=sys.argv[1:]
root=Path(args[args.index('--experiment-root')+1])
case=os.environ.get('SHARD_FIXTURE_CASE','normal')
for key,value in {'RLVR_CPU_THREADS':'4','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1','PYTORCH_ALLOC_CONF':'expandable_segments:True'}.items():
 assert os.environ.get(key)==value,(key,os.environ.get(key))
with (root/'events.jsonl').open('a') as f:f.write(json.dumps({'name':Path(__file__).stem,'args':args,'pid':os.getpid()})+'\n')
if Path(__file__).stem=='check_worker':raise SystemExit(9 if case=='preflight_fail' else 0)
if '--plan' in args:raise SystemExit(0)
out=root/'outputs/math_l4_v1_fix1/rl_shards/shard_0'
out.mkdir(parents=True,exist_ok=True)
sample=out/'samples.jsonl'
rows=sample.read_text().splitlines() if sample.exists() else []
if '--cache-only' in args:raise SystemExit(0 if (out/'complete').exists() else 3)
if case=='hang':
 (root/'child.pid').write_text(str(os.getpid()))
 time.sleep(60)
elif case=='eval_fail':raise SystemExit(7)
elif case=='unsafe75':raise SystemExit(75)
elif case in ('restart','limit','complete75') and (not rows or case=='limit'):
 with sample.open('a') as f:f.write(json.dumps({'id':'fixture','sample_index':len(rows)})+'\n');f.flush();os.fsync(f.fileno())
 if case=='complete75':(out/'complete').write_text('done')
 raise SystemExit(75)
else:
 with sample.open('a') as f:f.write(json.dumps({'id':'fixture','sample_index':len(rows)})+'\n')
 (out/'complete').write_text('done')
'''
    bootstrap = "import runpy,sys;sys.platform='linux';sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
    for case in ('normal', 'restart', 'limit', 'complete75', 'unsafe75', 'preflight_fail', 'eval_fail', 'hang'):
        case_root = base / ('supervisor-' + case) / 'math_l4_v1'
        case_bundle = case_root.parent / 'math_l4_rl_split_v1'
        (case_root / 'configs').mkdir(parents=True)
        (case_root / 'configs/math_l4.json').write_text(json.dumps(config))
        (case_bundle / 'scripts').mkdir(parents=True)
        shutil.copyfile(bundle / 'scripts/run_shard.py', case_bundle / 'scripts/run_shard.py')
        for name in ('check_worker.py', 'evaluate_shard.py'):
            (case_bundle / 'scripts' / name).write_text(worker_fixture)
        command = [sys.executable, '-S', '-c', bootstrap, str(case_bundle / 'scripts/run_shard.py'),
            '--experiment-root', str(case_root), '--shard-index', '0']
        env = dict(os.environ, SHARD_FIXTURE_CASE=case, PYTHONUNBUFFERED='1')
        if case == 'hang':
            with (case_root / 'capture.log').open('w') as stream:
                child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 15
                while not (case_root / 'child.pid').exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(.05)
                assert (case_root / 'child.pid').is_file(), (case_root / 'capture.log').read_text()
                worker_pid = int((case_root / 'child.pid').read_text())
                second = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
                assert second.returncode != 0
                child.send_signal(signal.SIGTERM)
                child.wait(timeout=20)
                try:
                    os.kill(worker_pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise AssertionError('GPU worker survived supervisor SIGTERM.')
            state = json.loads((case_root / 'outputs/math_l4_v1_fix1/rl_shards/worker_0_state.json').read_text())
            assert state['status'] == 'interrupted' and state['child_pid'] is None
            checks += 3
            continue
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
        success = case in ('normal', 'restart', 'complete75')
        assert (result.returncode == 0) == success, (case, result.stdout, result.stderr)
        state = json.loads((case_root / 'outputs/math_l4_v1_fix1/rl_shards/worker_0_state.json').read_text())
        assert state['status'] == ('completed' if success else 'failed')
        if case == 'restart':
            assert state['restarts'] == 1
        if case == 'limit':
            assert state['restarts'] == 2
        if case == 'preflight_fail':
            events = [json.loads(line) for line in (case_root / 'events.jsonl').read_text().splitlines()]
            assert [item['name'] for item in events] == ['check_worker']
        checks += 2

print(f'✅ [RL SHARDS] {checks} CPU fixture checks | actual frozen-data plans, 3200-answer validated merge, real evaluator 1598-to-1600 resume and complete-cache reuse, tamper rejection, durable restart limits, GPU lock and SIGTERM cleanup | no model/GPU/network')
