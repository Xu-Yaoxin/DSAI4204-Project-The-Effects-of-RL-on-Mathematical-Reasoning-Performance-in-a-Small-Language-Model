"""CPU-only worker setup and process-switch safety checks with isolated process fixtures."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

bundle = Path(__file__).resolve().parents[1]
real_run_path = runpy.run_path
path_new = Path.__new__
checks = 0
with tempfile.TemporaryDirectory(prefix='rl worker tools ') as temporary:
    fixture = Path(temporary)
    code = fixture / 'code'
    worker = code / 'math_l4_rl_split_v1'
    experiment = code / 'math_l4_v1'
    run = experiment / 'outputs/math_l4_v1_fix1'
    proc_root = fixture / 'proc'
    for directory in [worker / 'scripts', worker / 'reference', experiment / 'scripts', experiment / 'configs', run / 'test_base', proc_root, code / 'packages/linux-cp311-cu128']:
        directory.mkdir(parents=True, exist_ok=True)
    for name in ['check_worker.py', 'switch_after_base.py', 'watch_base_only.py', 'pidfd_compat.py']:
        shutil.copy2(bundle / 'scripts' / name, worker / 'scripts' / name)
    reference = json.loads((bundle / 'reference/base_manifest.json').read_text())
    cfg = reference['config']
    (worker / 'reference/base_manifest.json').write_text(json.dumps(reference))
    (experiment / 'configs/math_l4.json').write_text(json.dumps(cfg))
    for name in ['run_experiment.py', 'evaluate.py', 'analyze.py', 'validate_evaluation.py', 'validate_selection.py']:
        (experiment / 'scripts' / name).write_text('# Isolated process identity fixture.\n')
    frozen_files = {path.relative_to(experiment).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in experiment.rglob('*') if path.is_file()}
    (experiment / 'bundle_manifest.json').write_text(json.dumps({'files_sha256': frozen_files}))
    identity = {'config_sha256': hashlib.sha256((experiment / 'configs/math_l4.json').read_bytes()).hexdigest(),
                'bundle_sha256': hashlib.sha256((experiment / 'bundle_manifest.json').read_bytes()).hexdigest()}
    for name in ['manifest.json', 'requirements-lock.txt']:
        shutil.copy2(bundle.parent / 'packages/linux-cp311-cu128' / name, code / 'packages/linux-cp311-cu128' / name)
    lock = json.loads((code / 'packages/linux-cp311-cu128/manifest.json').read_text())
    installed = {row['name']: row['version'] for row in lock['packages']}
    installed.update(reference['settings']['package_versions'])
    (run / '.experiment.lock').touch()
    (experiment / 'outputs/.single_gpu_experiment.lock').touch()

    result = subprocess.run([sys.executable, '-S', str(worker / 'scripts/check_worker.py'), '--experiment-root', str(experiment), '--shard-index', '0', '--plan'], capture_output=True, text=True)
    assert result.returncode == 0 and 'WORKER PLAN COMPLETE' in result.stdout, result.stderr
    assert not (run / 'rl_shards').exists()
    checks += 2
    for key, bad in [('execution_cpu_threads', 4), ('gpu', 'NVIDIA A100 80GB PCIe')]:
        altered = dict(reference)
        altered[key] = bad
        (worker / 'reference/base_manifest.json').write_text(json.dumps(altered))
        result = subprocess.run([sys.executable, '-S', str(worker / 'scripts/check_worker.py'), '--experiment-root', str(experiment), '--shard-index', '0', '--plan'], capture_output=True, text=True)
        assert result.returncode != 0 and 'Frozen Base reference' in result.stderr
        checks += 1
    (worker / 'reference/base_manifest.json').write_text(json.dumps(reference))
    result = subprocess.run([sys.executable, '-S', str(worker / 'scripts/check_worker.py'), '--experiment-root', str(experiment), '--shard-index', '1', '--run-dir', '../../escape', '--plan'], capture_output=True, text=True)
    assert result.returncode != 0 and 'Run directory' in result.stderr
    checks += 1
    before = (experiment / 'scripts/evaluate.py').read_bytes()
    (experiment / 'scripts/evaluate.py').write_text('# Changed fixture.\n')
    result = subprocess.run([sys.executable, '-S', str(worker / 'scripts/check_worker.py'), '--experiment-root', str(experiment), '--shard-index', '0', '--plan'], capture_output=True, text=True)
    assert result.returncode != 0 and 'Frozen experiment file differs' in result.stderr
    (experiment / 'scripts/evaluate.py').write_bytes(before)
    checks += 1

    cuda_order = []
    fake_torch = SimpleNamespace(set_num_threads=MagicMock(), set_num_interop_threads=MagicMock(),
        get_num_threads=lambda: 4, get_num_interop_threads=lambda: 1, ones=MagicMock(),
        cuda=SimpleNamespace(is_available=lambda: (cuda_order.append('cuda'), True)[1], device_count=lambda: 1,
             get_device_properties=lambda index: SimpleNamespace(uuid='GPU-fixture'),
             get_device_name=lambda index: 'NVIDIA GeForce RTX 4090'))
    modules = {'torch': fake_torch, 'pandas': SimpleNamespace(), 'datasets': SimpleNamespace(Dataset=object),
               'peft': SimpleNamespace(PeftModel=object), 'transformers': SimpleNamespace(AutoTokenizer=object),
               'trl': SimpleNamespace(GRPOConfig=object, GRPOTrainer=object)}
    responses = [SimpleNamespace(returncode=0, stdout='No broken requirements found.', stderr=''),
                 SimpleNamespace(returncode=0, stdout='GPU-fixture, NVIDIA GeForce RTX 4090, 580.20, 24564\n', stderr=''),
                 SimpleNamespace(returncode=0, stdout='GPU-fixture, 987654, python, 5000\n', stderr='')]
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, 'argv', [str(worker / 'scripts/check_worker.py'), '--experiment-root', str(experiment), '--shard-index', '0']))
        stack.enter_context(patch.object(sys, 'version_info', (3, 11, 0)))
        stack.enter_context(patch('platform.system', return_value='Linux'))
        stack.enter_context(patch('platform.machine', return_value='x86_64'))
        stack.enter_context(patch('platform.python_implementation', return_value='CPython'))
        stack.enter_context(patch('importlib.metadata.version', side_effect=lambda name: installed[name]))
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(patch('subprocess.run', side_effect=lambda command, **kw: (cuda_order.append('snapshot') if '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory' in command else None, responses.pop(0))[1]))
        stack.enter_context(redirect_stdout(io.StringIO()))
        try:
            real_run_path(str(worker / 'scripts/check_worker.py'), run_name='__main__')
        except SystemExit as error:
            assert 'other compute processes' in str(error), str(error)
        else:
            raise AssertionError('A concurrent GPU process was accepted.')
    fake_torch.ones.assert_not_called()
    assert cuda_order == ['snapshot', 'cuda'], cuda_order
    checks += 3

    base_complete = dict(reference)
    base_complete.update(complete=True, samples_sha256='a' * 64, total_samples=3200)
    selection = {'selected_step': 384, 'adapter_path': 'outputs/math_l4_v1_fix1/formal/checkpoint-384', 'selection_id': 'b' * 64}
    scenarios = ['plan_incomplete', 'failed_incomplete', 'bad_base_validation', 'wrong_state_identity',
                 'unrelated_pid', 'wrong_runner_directory', 'pid_reused_at_open', 'unexpected_child',
                 'switch_rl_child', 'switch_completed_base_child', 'switch_interrupted', 'completed_no_runner',
                 'stop_only_complete', 'stop_only_repeat', 'stop_only_incomplete']
    for scenario in scenarios:
        for entry in proc_root.iterdir():
            if entry.is_dir(): shutil.rmtree(entry)
            else: entry.unlink()
        for path in (worker / 'outputs').glob('*.json') if (worker / 'outputs').exists() else []:
            path.unlink()
        state = {'identity': identity, 'status': 'running', 'pid': 40001, 'child_pid': 40002, 'phase': 'TEST BASE'}
        manifest = dict(base_complete)
        if scenario in ['plan_incomplete', 'failed_incomplete', 'stop_only_incomplete']:
            manifest['complete'] = False
        if scenario == 'failed_incomplete': state['status'] = 'failed'
        if scenario == 'wrong_state_identity': state['identity'] = {'invalid': True}
        if scenario in ['completed_no_runner', 'stop_only_repeat']: state['status'] = 'completed'
        (run / 'run_state.json').write_text(json.dumps(state))
        (run / 'test_base/manifest.json').write_text(json.dumps(manifest))
        parent_command = [sys.executable, 'scripts/run_experiment.py', 'all', '--run-dir', 'outputs/math_l4_v1_fix1']
        if scenario == 'unrelated_pid': parent_command[1] = 'unrelated.py'
        if scenario == 'wrong_runner_directory': parent_command[-1] = 'outputs/unrelated'
        child_script = 'unrelated.py' if scenario == 'unexpected_child' else 'scripts/evaluate.py'
        child_output = 'outputs/math_l4_v1_fix1/test_base' if scenario == 'switch_completed_base_child' else 'outputs/math_l4_v1_fix1/test_rl'
        for pid, ppid, command in [(40001, 1, parent_command), (40002, 40001, [sys.executable, child_script, '--output', child_output])]:
            if scenario in ['completed_no_runner', 'stop_only_repeat']: continue
            directory = proc_root / str(pid)
            directory.mkdir()
            fields = ['S', str(ppid), str(pid)] + ['0'] * 16 + ['123456'] + ['0'] * 10
            (directory / 'stat').write_text(str(pid) + ' (python) ' + ' '.join(fields))
            (directory / 'cmdline').write_bytes(b'\0'.join(x.encode() for x in command) + b'\0')
            (directory / 'cwd').symlink_to(experiment, target_is_directory=True)
        events = []
        argv = [str(worker / 'scripts/switch_after_base.py'), '--experiment-root', str(experiment), '--shard-index', '0', '--poll-seconds', '0.2']
        if scenario == 'plan_incomplete': argv.append('--plan')
        if scenario.startswith('stop_only_'): argv.append('--stop-only')
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch('platform.system', return_value='Linux'))
            stack.enter_context(patch('signal.signal'))
            stack.enter_context(patch.object(Path, '__new__', staticmethod(lambda cls, *parts, **kw: path_new(cls, *(str(proc_root) + str(parts[0])[5:] if parts and str(parts[0]).startswith('/proc') else parts[0], *parts[1:]), **kw) if parts else path_new(cls, **kw))))
            validator = MagicMock(side_effect=(ValueError('Invalid complete Base sample hash') if scenario == 'bad_base_validation' else lambda path, init_globals: {'EVALUATION_RESULT': {'manifest': base_complete}} if str(path).endswith('validate_evaluation.py') else {'SELECTION_RESULT': selection}))
            stack.enter_context(patch('runpy.run_path', side_effect=lambda path, init_globals: ({'PIDFD_RESULT': {'backend': 'fixture'}} if init_globals.get('PIDFD_OPERATION') == 'probe' else real_run_path(path, init_globals=init_globals)) if str(path).endswith('pidfd_compat.py') else validator(path, init_globals=init_globals)))
            opened = stack.enter_context(patch('os.pidfd_open', create=True, side_effect=lambda pid, flags: ((proc_root / str(pid) / 'stat').write_text((proc_root / str(pid) / 'stat').read_text().replace('123456', '999999')) if scenario == 'pid_reused_at_open' else None, pid)[1]))
            sent = stack.enter_context(patch('signal.pidfd_send_signal', create=True, side_effect=lambda fd, sig: (
                events.append((fd, sig)),
                (proc_root / str(fd) / 'stat').write_text((proc_root / str(fd) / 'stat').read_text().replace(') S ', ') T ')) if sig == signal.SIGSTOP else None,
                (proc_root / '40002/stat').write_text((proc_root / '40002/stat').read_text().replace(') S ', ') Z ')) if sig == signal.SIGSTOP and scenario == 'switch_completed_base_child' else None,
                (proc_root / str(fd) / 'stat').write_text((proc_root / str(fd) / 'stat').read_text().replace(') S ', ') Z ').replace(') T ', ') Z ')) if sig in {signal.SIGTERM, signal.SIGKILL} else None,
                (_ for _ in ()).throw(KeyboardInterrupt()) if scenario == 'switch_interrupted' and fd == 40002 and sig == signal.SIGTERM else None
            )))
            stack.enter_context(patch('os.close'))
            launched = stack.enter_context(patch('os.execv', side_effect=SystemExit(0)))
            stack.enter_context(patch('time.sleep', side_effect=SystemExit(0) if scenario == 'stop_only_incomplete' else None))
            stack.enter_context(redirect_stdout(io.StringIO()))
            raised = None
            try:
                real_run_path(str(worker / 'scripts/switch_after_base.py'), run_name='__main__')
                if scenario == 'stop_only_repeat':
                    real_run_path(str(worker / 'scripts/switch_after_base.py'), run_name='__main__')
            except (SystemExit, RuntimeError, ValueError, KeyboardInterrupt) as error:
                raised = error
            if scenario == 'stop_only_incomplete':
                assert isinstance(raised, SystemExit) and raised.code == 0 and not events and not launched.called
                assert not validator.called and not opened.called
            elif scenario.startswith('stop_only_'):
                assert raised is None and not launched.called, (scenario, repr(raised))
                receipt = json.loads((worker / 'outputs/base_stop_receipt.json').read_text())
                assert receipt['status'] == 'stopped_after_base' and receipt['complete'] is True and receipt['rl_started'] is False
                assert receipt['base_total_samples'] == 3200 and receipt['base_samples_sha256'] == base_complete['samples_sha256']
                if scenario == 'stop_only_repeat': assert not events
                else: assert (40002, signal.SIGTERM) in events and (40001, signal.SIGTERM) in events
            elif scenario in ['switch_rl_child', 'switch_completed_base_child', 'completed_no_runner']:
                assert isinstance(raised, SystemExit) and raised.code == 0 and launched.call_count == 1, (scenario, repr(raised))
                command = launched.call_args.args[1]
                assert command[-2:] == ['--shard-index', '0'] and '--run-dir' in command
                if scenario == 'switch_completed_base_child':
                    assert not any(pid == 40002 for pid, sig in events), events
                if scenario == 'switch_rl_child':
                    assert (40002, signal.SIGTERM) in events and events.index((40002, signal.SIGTERM)) < events.index((40001, signal.SIGTERM))
            elif scenario == 'switch_interrupted':
                assert isinstance(raised, KeyboardInterrupt) and not launched.called
                assert events == [(40001, signal.SIGSTOP), (40002, signal.SIGTERM), (40001, signal.SIGCONT)], events
            elif scenario == 'unexpected_child':
                assert isinstance(raised, RuntimeError) and not launched.called
                assert events == [(40001, signal.SIGSTOP), (40001, signal.SIGCONT)], events
            else:
                assert raised is not None and not events and not launched.called, (scenario, repr(raised), events)
            if scenario in ['plan_incomplete', 'failed_incomplete', 'wrong_state_identity']:
                assert not validator.called and not opened.called
        checks += 1

    for script_name in ['switch_after_base.py', 'watch_base_only.py']:
        state = {'identity': identity, 'status': 'running', 'pid': 40001, 'phase': 'TEST BASE'}
        (run / 'run_state.json').write_text(json.dumps(state))
        (run / 'test_base/manifest.json').write_text(json.dumps(dict(base_complete, complete=False)))
        before_base = (run / 'test_base/manifest.json').read_bytes()
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', [str(worker / 'scripts' / script_name), '--experiment-root', str(experiment)]))
            stack.enter_context(patch('runpy.run_path', side_effect=OSError(38, 'pidfd self-check failed: syscall unavailable')))
            sent = stack.enter_context(patch('signal.pidfd_send_signal', create=True))
            launched = stack.enter_context(patch('os.execv'))
            try:
                real_run_path(str(worker / 'scripts' / script_name), run_name='__main__')
            except OSError as error:
                assert error.errno == 38
            else:
                raise AssertionError('A failed capability probe was ignored.')
            assert not sent.called and not launched.called
            assert (run / 'test_base/manifest.json').read_bytes() == before_base
        checks += 1

    watcher_scenarios = ['incomplete_no_watcher', 'old_watcher_incomplete', 'old_stop_only', 'stranger',
                         'wrong_run', 'pid_reused', 'exec_to_rl', 'existing_rl', 'plan']
    for scenario in watcher_scenarios:
        for entry in proc_root.iterdir():
            if entry.is_dir(): shutil.rmtree(entry)
            else: entry.unlink()
        state = {'identity': identity, 'status': 'running', 'pid': 40001, 'child_pid': 40002, 'phase': 'TEST BASE'}
        (run / 'run_state.json').write_text(json.dumps(state))
        incomplete = dict(base_complete, complete=False)
        (run / 'test_base/manifest.json').write_text(json.dumps(incomplete))
        old_script = str(worker / 'scripts/switch_after_base.py')
        if scenario == 'stranger': old_script = str(fixture / 'other/scripts/switch_after_base.py')
        if scenario == 'existing_rl': old_script = str(worker / 'scripts/run_shard.py')
        old_command = [sys.executable, old_script, '--experiment-root', str(experiment), '--run-dir', 'outputs/math_l4_v1_fix1']
        if scenario == 'wrong_run': old_command[-1] = 'outputs/unrelated'
        if scenario == 'old_stop_only': old_command.append('--stop-only')
        proc_commands = [(40001, [sys.executable, str(experiment / 'scripts/run_experiment.py'), 'all']),
                         (40002, [sys.executable, str(experiment / 'scripts/evaluate.py'), '--output', str(run / 'test_base')])]
        if scenario != 'incomplete_no_watcher': proc_commands.append((40003, old_command))
        for pid, command in proc_commands:
            directory = proc_root / str(pid)
            directory.mkdir()
            fields = ['S', '1', str(pid)] + ['0'] * 16 + ['123456'] + ['0'] * 10
            (directory / 'stat').write_text(str(pid) + ' (python) ' + ' '.join(fields))
            (directory / 'cmdline').write_bytes(b'\0'.join(x.encode() for x in command) + b'\0')
            (directory / 'cwd').symlink_to(worker, target_is_directory=True)
        events = []
        argv = [str(worker / 'scripts/watch_base_only.py'), '--experiment-root', str(experiment), '--run-dir', 'outputs/math_l4_v1_fix1']
        if scenario == 'plan': argv.append('--plan')
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', argv))
            stack.enter_context(patch('platform.system', return_value='Linux'))
            stack.enter_context(patch('signal.signal'))
            stack.enter_context(patch.object(Path, '__new__', staticmethod(lambda cls, *parts, **kw: path_new(cls, *(str(proc_root) + str(parts[0])[5:] if parts and str(parts[0]).startswith('/proc') else parts[0], *parts[1:]), **kw) if parts else path_new(cls, **kw))))
            stack.enter_context(patch('runpy.run_path', side_effect=lambda path, init_globals: {'PIDFD_RESULT': {'backend': 'fixture'}} if init_globals.get('PIDFD_OPERATION') == 'probe' else real_run_path(path, init_globals=init_globals)))
            stack.enter_context(patch('os.pidfd_open', create=True, side_effect=lambda pid, flags: ((proc_root / str(pid) / 'stat').write_text((proc_root / str(pid) / 'stat').read_text().replace('123456', '999999')) if scenario == 'pid_reused' else None, pid)[1]))
            stack.enter_context(patch('signal.pidfd_send_signal', create=True, side_effect=lambda fd, sig: (
                events.append((fd, sig)),
                (proc_root / str(fd) / 'stat').write_text((proc_root / str(fd) / 'stat').read_text().replace(') S ', ') T ')) if sig == signal.SIGSTOP else None,
                (proc_root / str(fd) / 'cmdline').write_bytes(b'\0'.join(x.encode() for x in [sys.executable, str(worker / 'scripts/run_shard.py'), '--shard-index', '0']) + b'\0') if scenario == 'exec_to_rl' and sig == signal.SIGSTOP else None,
                (proc_root / str(fd) / 'stat').write_text((proc_root / str(fd) / 'stat').read_text().replace(') S ', ') Z ').replace(') T ', ') Z ')) if sig == signal.SIGTERM else None
            )))
            stack.enter_context(patch('os.close'))
            launched = stack.enter_context(patch('os.execv', side_effect=SystemExit(0)))
            stack.enter_context(patch('time.sleep'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            raised = None
            try:
                real_run_path(str(worker / 'scripts/watch_base_only.py'), run_name='__main__')
            except (SystemExit, RuntimeError) as error:
                raised = error
            assert not any(pid in {40001, 40002} for pid, sig in events), events
            if scenario in ['pid_reused', 'exec_to_rl', 'existing_rl']:
                assert isinstance(raised, RuntimeError) and not launched.called, (scenario, repr(raised))
                expected = [(40003, signal.SIGSTOP), (40003, signal.SIGCONT)] if scenario == 'exec_to_rl' else []
                assert events == expected, (scenario, events)
            elif scenario == 'plan':
                assert raised is None and not events and not launched.called
            else:
                assert isinstance(raised, SystemExit) and raised.code == 0 and launched.called, (scenario, repr(raised))
                assert launched.call_args.args[1][-1] == '--stop-only'
                if scenario in ['old_watcher_incomplete', 'old_stop_only']:
                    assert events == [(40003, signal.SIGSTOP), (40003, signal.SIGTERM), (40003, signal.SIGCONT)], events
                else: assert not events, (scenario, events)
        checks += 1
    result = subprocess.run(['bash', '-n', str(bundle / 'setup_worker.sh')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    checks += 1
print(f'[WORKER TOOLS PASS] {checks} CPU scenarios; real source-plan subprocesses, isolated dependency/GPU occupancy checks and pidfd process fixtures; no GPU, downloads or real signals.')
