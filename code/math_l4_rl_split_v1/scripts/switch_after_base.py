"""Wait for durable Base completion, then stop its verified runner or continue with one RL shard."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time

bundle = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment-root', default='../math_l4_v1')
parser.add_argument('--run-dir', default='outputs/math_l4_v1_fix1')
parser.add_argument('--shard-index', type=int, choices=(0, 1), default=0)
parser.add_argument('--poll-seconds', type=float, default=5)
parser.add_argument('--plan', action='store_true')
parser.add_argument('--stop-only', action='store_true', help='Leave this host idle after the current Base evaluation finishes.')
args = parser.parse_args()
if not 0.2 <= args.poll_seconds <= 60:
    raise SystemExit('--poll-seconds must be between 0.2 and 60.')
raw_experiment = Path(args.experiment_root).expanduser()
experiment = (raw_experiment if raw_experiment.is_absolute() else bundle / raw_experiment).resolve()
run = (experiment / args.run_dir).resolve()
if not run.is_relative_to(experiment / 'outputs'):
    raise SystemExit('Run must belong to the frozen experiment outputs.')
cfg_path = experiment / 'configs/math_l4.json'
cfg = json.loads(cfg_path.read_text())
expected_state_identity = {'config_sha256': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                           'bundle_sha256': hashlib.sha256((experiment / 'bundle_manifest.json').read_bytes()).hexdigest()}
reference = json.loads((bundle / 'reference/base_manifest.json').read_text())
if reference.get('config') != cfg or reference.get('model_label') != 'base' or reference.get('split') != 'test':
    raise SystemExit('Frozen Base reference and original experiment differ.')
frozen = json.loads((experiment / 'bundle_manifest.json').read_text())
for relative, expected in frozen['files_sha256'].items():
    path = experiment / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts or path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit('Frozen experiment source differs: ' + relative)
state_path = run / 'run_state.json'
base_path = run / 'test_base'
base_manifest_path = base_path / 'manifest.json'
pidfd_helper = bundle / 'scripts/pidfd_compat.py'
if not args.plan:
    pidfd_probe = runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'probe'})['PIDFD_RESULT']
    print('[PIDFD READY] ' + json.dumps(pidfd_probe) + ' | self-check only; Base was not signalled', flush=True)
while True:
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    if state and state.get('identity') != expected_state_identity:
        raise SystemExit('Saved runner identity differs; no process will be interrupted.')
    base_manifest = json.loads(base_manifest_path.read_text()) if base_manifest_path.is_file() else {}
    base_ready = base_manifest.get('complete') is True
    if args.plan:
        print('[SWITCH PLAN] ' + json.dumps({'phase': state.get('phase'), 'status': state.get('status'),
              'base_complete_marker': base_ready, 'shard_index': args.shard_index,
              'action': 'Stop original runner after Base' if args.stop_only else 'Switch original runner to RL after Base',
              'signals_sent': 0}), flush=True)
        if not base_ready:
            raise SystemExit(0)
    if base_ready:
        validated = runpy.run_path(str(experiment / 'scripts/validate_evaluation.py'), init_globals={
            'EVALUATION_PATH': base_path, 'VALIDATE_CONFIG': cfg})['EVALUATION_RESULT']
        observed = validated['manifest']
        if observed.get('model_label') != 'base' or observed.get('split') != 'test' or observed.get('protocol_id') != reference.get('protocol_id'):
            raise SystemExit('Completed Base evaluation differs from the frozen reference; no process will be interrupted.')
        selection = runpy.run_path(str(experiment / 'scripts/validate_selection.py'), init_globals={
            'SELECTION_PATH': run / 'selection.json', 'SELECTION_CONFIG': cfg})['SELECTION_RESULT']
        if selection.get('selected_step') != 384 or (experiment / selection['adapter_path']).resolve() != run / 'formal/checkpoint-384':
            raise SystemExit('This RL continuation requires the already selected checkpoint-384.')
        print('[BASE COMPLETE] Complete answer grid, sample hashes and checkpoint selection passed.', flush=True)
        break
    if state.get('status') in {'failed', 'interrupted', 'completed'}:
        raise SystemExit('Original runner is not running and Base is incomplete; resolve its failure before switching.')
    if not state or state.get('status') != 'running':
        raise SystemExit('Cannot establish a running original Base pipeline; no process will be interrupted.')
    print('[WAIT BASE] phase=' + str(state.get('phase')) + ' | incomplete | no signals sent', flush=True)
    time.sleep(args.poll_seconds)
if args.plan:
    print('[SWITCH PLAN COMPLETE] Base and selection verified; no signals sent and no worker launched.', flush=True)
    raise SystemExit(0)

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
audit_dir = bundle / 'outputs'
audit_dir.mkdir(exist_ok=True)
audit_path = audit_dir / f'switch_{stamp}_{os.getpid()}.json'
audit = {'started_utc': datetime.now(timezone.utc).isoformat(), 'experiment_root': str(experiment),
         'run_dir': str(run), 'shard_index': args.shard_index, 'base_samples_sha256': observed['samples_sha256'],
         'base_total_samples': observed['total_samples'], 'selection_identity': selection['selection_id'],
         'mode': 'stop_after_base' if args.stop_only else 'switch_to_rl',
         'signals': [], 'processes': [], 'status': 'checking_original_runner'}
parent_fd = None
parent_stopped = False
fds = []
held_locks = []
signal.signal(signal.SIGTERM, signal.default_int_handler)
try:
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    if state and state.get('identity') != expected_state_identity:
        raise RuntimeError('Runner identity changed during Base verification.')
    pid = state.get('pid')
    parent = None
    proc = Path('/proc') / str(pid) if type(pid) is int and pid > 0 else None
    if proc is not None and (proc / 'stat').is_file():
        raw_stat = (proc / 'stat').read_text()
        fields = raw_stat[raw_stat.rfind(')') + 2:].split()
        if fields[0] != 'Z':
            command = (proc / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
            cwd = (proc / 'cwd').resolve()
            script_indexes = [i for i, token in enumerate(command) if token.endswith('run_experiment.py') and (cwd / token).resolve() == experiment / 'scripts/run_experiment.py']
            run_options = []
            for index, token in enumerate(command):
                if token == '--run-dir' and index + 1 < len(command):
                    run_options.append(command[index + 1])
                elif token.startswith('--run-dir='):
                    run_options.append(token.split('=', 1)[1])
            declared_run = (cwd / (run_options[0] if run_options else 'outputs/math_l4_v1')).resolve()
            python_matches = bool(command) and (cwd / command[0]).resolve() == Path(sys.executable).resolve()
            options_match = len(script_indexes) == 1 and all(token in {'-S', '-u', '-B', '-s', '-E', '-I'} for token in command[1:script_indexes[0]])
            if not python_matches or not options_match or cwd != experiment or len(script_indexes) != 1 or len(run_options) > 1 or declared_run != run:
                raise RuntimeError('Saved PID is not the exact original experiment runner; refusing to signal it.')
            parent_fd = runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'open', 'PIDFD_PID': pid})['PIDFD_RESULT']['fd']
            fds.append(parent_fd)
            after = (proc / 'stat').read_text()
            after_fields = after[after.rfind(')') + 2:].split()
            if fields[19] != after_fields[19] or after_fields[0] == 'Z':
                raise RuntimeError('Original runner identity changed while opening its pidfd.')
            parent = {'pid': pid, 'start_ticks': fields[19], 'cwd': str(cwd), 'command': command, 'pidfd': parent_fd}
            audit['processes'].append(parent)
            runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'send', 'PIDFD_FD': parent_fd, 'PIDFD_SIGNAL': signal.SIGSTOP})
            parent_stopped = True
            audit['signals'].append({'pid': pid, 'signal': 'SIGSTOP', 'via': 'pidfd'})
            stop_deadline = time.monotonic() + 5
            while True:
                current = (proc / 'stat').read_text()
                current_fields = current[current.rfind(')') + 2:].split()
                if current_fields[19] != parent['start_ticks']:
                    raise RuntimeError('Runner PID changed after stop request.')
                if current_fields[0] in {'T', 't'}:
                    break
                if time.monotonic() >= stop_deadline:
                    raise RuntimeError('Original runner did not stop safely; no child will be interrupted.')
                time.sleep(0.05)
            children = []
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    raw = (entry / 'stat').read_text()
                    child_fields = raw[raw.rfind(')') + 2:].split()
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if int(child_fields[1]) != pid or child_fields[0] == 'Z':
                    continue
                child_pid = int(entry.name)
                command = (entry / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
                cwd = (entry / 'cwd').resolve()
                child_script_indexes = [i for i, token in enumerate(command) if (cwd / token).resolve() in {experiment / 'scripts/evaluate.py', experiment / 'scripts/analyze.py'}]
                outputs = []
                for index, token in enumerate(command):
                    if token == '--output' and index + 1 < len(command):
                        outputs.append((cwd / command[index + 1]).resolve())
                    elif token.startswith('--output='):
                        outputs.append((cwd / token.split('=', 1)[1]).resolve())
                allowed = {run / 'test_base', run / 'test_rl', run / 'comparison'}
                python_matches = bool(command) and (cwd / command[0]).resolve() == Path(sys.executable).resolve()
                options_match = len(child_script_indexes) == 1 and all(token in {'-S', '-u', '-B', '-s', '-E', '-I'} for token in command[1:child_script_indexes[0]])
                if not python_matches or not options_match or cwd != experiment or len(child_script_indexes) != 1 or len(outputs) != 1 or outputs[0] not in allowed or int(child_fields[2]) != child_pid:
                    raise RuntimeError('Original runner has an unexpected child; it will not be signalled.')
                fd = runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'open', 'PIDFD_PID': child_pid})['PIDFD_RESULT']['fd']
                fds.append(fd)
                after = (entry / 'stat').read_text()
                after_fields = after[after.rfind(')') + 2:].split()
                if child_fields[19] != after_fields[19] or int(after_fields[1]) != pid:
                    raise RuntimeError('Child identity changed while opening its pidfd.')
                child = {'pid': child_pid, 'pidfd': fd, 'start_ticks': child_fields[19], 'cwd': str(cwd),
                         'command': command, 'output': str(outputs[0]), 'is_base': outputs[0] == run / 'test_base'}
                children.append(child)
                audit['processes'].append(child)
            for child in children:
                if not child['is_base']:
                    runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'send', 'PIDFD_FD': child['pidfd'], 'PIDFD_SIGNAL': signal.SIGTERM})
                    audit['signals'].append({'pid': child['pid'], 'signal': 'SIGTERM', 'via': 'pidfd'})
                deadline = time.monotonic() + 30
                while True:
                    stat_path = Path('/proc') / str(child['pid']) / 'stat'
                    if not stat_path.is_file():
                        break
                    raw = stat_path.read_text()
                    fields = raw[raw.rfind(')') + 2:].split()
                    if fields[19] != child['start_ticks'] or fields[0] == 'Z':
                        break
                    if time.monotonic() >= deadline:
                        if child['is_base']:
                            raise RuntimeError('Completed Base process has not exited; it was not interrupted. Retry after it exits.')
                        runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'send', 'PIDFD_FD': child['pidfd'], 'PIDFD_SIGNAL': signal.SIGKILL})
                        audit['signals'].append({'pid': child['pid'], 'signal': 'SIGKILL', 'via': 'pidfd'})
                        break
                    time.sleep(0.2)
            runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'send', 'PIDFD_FD': parent_fd, 'PIDFD_SIGNAL': signal.SIGTERM})
            audit['signals'].append({'pid': pid, 'signal': 'SIGTERM', 'via': 'pidfd'})
            try:
                runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'send', 'PIDFD_FD': parent_fd, 'PIDFD_SIGNAL': signal.SIGCONT})
            except ProcessLookupError:
                pass
            parent_stopped = False
            audit['signals'].append({'pid': pid, 'signal': 'SIGCONT', 'via': 'pidfd'})
            deadline = time.monotonic() + 30
            while (proc / 'stat').is_file():
                raw = (proc / 'stat').read_text()
                fields = raw[raw.rfind(')') + 2:].split()
                if fields[19] != parent['start_ticks'] or fields[0] == 'Z':
                    break
                if time.monotonic() >= deadline:
                    runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'send', 'PIDFD_FD': parent_fd, 'PIDFD_SIGNAL': signal.SIGKILL})
                    audit['signals'].append({'pid': pid, 'signal': 'SIGKILL', 'via': 'pidfd'})
                    break
                time.sleep(0.2)
    for lock_path in (experiment / 'outputs/.single_gpu_experiment.lock', run / '.experiment.lock'):
        if lock_path.is_symlink():
            raise RuntimeError('Unsafe original experiment lock.')
        if lock_path.is_file():
            stream = lock_path.open('r')
            held_locks.append(stream)
            deadline = time.monotonic() + 10
            while True:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Original experiment lock is still held; new shard was not launched.')
                    time.sleep(0.2)
    audit['status'] = 'original_runner_released'
    audit['finished_utc'] = datetime.now(timezone.utc).isoformat()
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    for stream in held_locks:
        stream.close()
    held_locks.clear()
    for fd in fds:
        os.close(fd)
    fds.clear()
    parent_fd = None
    os.environ.update({'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
                       'RLVR_CPU_THREADS': '4', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1',
                       'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
    if args.stop_only:
        audit.update(status='stopped_after_base', complete=True, rl_started=False)
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
        receipt_path = audit_dir / 'base_stop_receipt.json'
        if receipt_path.is_symlink():
            raise RuntimeError('Unsafe Base stop receipt path.')
        temporary = receipt_path.with_name(receipt_path.name + f'.{os.getpid()}.tmp')
        with temporary.open('x') as stream:
            stream.write(json.dumps(audit, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(receipt_path)
        print('[STOPPED AFTER BASE] Base is complete; original runner released; no RL launched on this host. | [SAVED] ' + str(receipt_path), flush=True)
    else:
        command = [sys.executable, str(bundle / 'scripts/run_shard.py'), '--experiment-root', str(experiment),
                   '--run-dir', args.run_dir, '--shard-index', str(args.shard_index)]
        print('[SWITCH COMPLETE] Original job released; starting RL shard ' + str(args.shard_index) + ' | [SAVED] ' + str(audit_path), flush=True)
        os.execv(sys.executable, command)
except BaseException as error:
    audit['status'] = 'failed'
    audit['error'] = repr(error)
    audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    raise
finally:
    if parent_stopped and parent_fd is not None:
        try:
            runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'send', 'PIDFD_FD': parent_fd, 'PIDFD_SIGNAL': signal.SIGCONT})
        except ProcessLookupError:
            pass
    for stream in held_locks:
        stream.close()
    for fd in fds:
        os.close(fd)
