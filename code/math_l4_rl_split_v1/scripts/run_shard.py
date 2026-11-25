"""Supervise one serial GPU shard with durable recovery and process-group cleanup."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

bundle_root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experiment-root', default=str(bundle_root.parent / 'math_l4_v1'))
parser.add_argument('--run-dir', default='outputs/math_l4_v1_fix1')
parser.add_argument('--shard-index', type=int, choices=(0, 1), required=True)
parser.add_argument('--plan', action='store_true')
args = parser.parse_args()
experiment = Path(args.experiment_root).expanduser().absolute()
run = (experiment / args.run_dir).absolute()
if any(path.is_symlink() for path in (run, *run.parents)):
    raise ValueError('Experiment/run paths must not contain symlinks.')
experiment = experiment.resolve()
run = run.resolve()
if not run.is_relative_to(experiment / 'outputs') or run == experiment / 'outputs':
    raise ValueError('Run must remain under experiment outputs/.')
base_command = [sys.executable, str(bundle_root / 'scripts/evaluate_shard.py'),
    '--experiment-root', str(experiment), '--run-dir', str(run.relative_to(experiment)),
    '--shard-index', str(args.shard_index)]
if args.plan:
    raise SystemExit(subprocess.run(base_command + ['--plan'], cwd=bundle_root).returncode)
if sys.platform != 'linux':
    raise SystemExit('Run inference on the Linux GPU server; use --plan for local inspection.')
os.environ.update({'RLVR_CPU_THREADS': '4', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
    'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'})
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
config = json.loads((experiment / 'configs/math_l4.json').read_text())
max_restarts = config['execution']['max_eval_worker_restarts']
if type(max_restarts) is not int or max_restarts != 2:
    raise ValueError('This release preserves the frozen maximum of two automatic worker restarts.')
(experiment / 'outputs').mkdir(exist_ok=True)
global_lock_path = experiment / 'outputs/.single_gpu_experiment.lock'
if global_lock_path.is_symlink():
    raise ValueError('Unsafe global GPU lock.')
lock_stream = global_lock_path.open('a')
fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
parent = run / 'rl_shards'
if any(path.is_symlink() for path in (parent, *parent.parents)):
    raise ValueError('Unsafe shard parent path.')
parent.mkdir(parents=True, exist_ok=True)
supervisor_lock_path = parent / f'.worker_{args.shard_index}.lock'
if supervisor_lock_path.is_symlink():
    raise ValueError('Unsafe supervisor lock.')
supervisor_lock = supervisor_lock_path.open('a')
fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state_path = parent / f'worker_{args.shard_index}_state.json'
temporary_state = state_path.with_suffix('.json.tmp')
if any(path.is_symlink() for path in (state_path, temporary_state)):
    raise ValueError('Unsafe worker state path.')
output = parent / f'shard_{args.shard_index}'
sample_path = output / 'samples.jsonl'
state = {'status': 'preflight', 'shard_index': args.shard_index, 'pid': os.getpid(),
    'child_pid': None, 'restarts': 0, 'completed_answers': 0, 'target_answers': 1600,
    'started_at_unix': time.time()}
signal.signal(signal.SIGTERM, signal.default_int_handler)
child = None
try:
    commands = [('PREFLIGHT', [sys.executable, str(bundle_root / 'scripts/check_worker.py'),
        '--experiment-root', str(experiment), '--run-dir', str(run.relative_to(experiment)),
        '--shard-index', str(args.shard_index)])]
    phase = 'PREFLIGHT'
    while commands:
        phase, command = commands.pop(0)
        before_bytes = sample_path.read_bytes() if sample_path.is_file() else b''
        if before_bytes and not before_bytes.endswith(b'\n'):
            raise ValueError('Incomplete JSONL tail: preserve the file and inspect it before resuming.')
        before = len([line for line in before_bytes.splitlines() if line.strip()])
        print(f'[SHARD START] shard={args.shard_index} | phase={phase} | completed={before}/1600 | restarts={state["restarts"]}', flush=True)
        child = subprocess.Popen(command, cwd=bundle_root, stdin=subprocess.DEVNULL, start_new_session=True)
        state.update(status='running', phase=phase, child_pid=child.pid, completed_answers=before)
        phase_start = time.monotonic()
        while child.poll() is None:
            state.update(updated_at_unix=time.time(), phase_elapsed_seconds=time.monotonic()-phase_start)
            temporary_state.write_text(json.dumps(state, indent=2) + '\n')
            temporary_state.replace(state_path)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                print(f'[SHARD HEARTBEAT] shard={args.shard_index} | phase={phase} | elapsed={time.monotonic()-phase_start:.0f}s | pid={child.pid}', flush=True)
        exit_code = child.returncode
        child = None
        state['child_pid'] = None
        after_bytes = sample_path.read_bytes() if sample_path.is_file() else b''
        if after_bytes and not after_bytes.endswith(b'\n'):
            raise ValueError('Worker left an incomplete JSONL tail; no automatic overwrite.')
        after = len([line for line in after_bytes.splitlines() if line.strip()])
        state['completed_answers'] = after
        if phase == 'PREFLIGHT':
            if exit_code != 0:
                raise RuntimeError(f'GPU/environment preflight exited {exit_code}; generation was not started.')
            commands.append(('EVALUATE', base_command))
        elif phase == 'CACHE CHECK':
            if exit_code == 0:
                state['status'] = 'completed'
            elif exit_code == 3:
                if state['restarts'] >= max_restarts:
                    raise RuntimeError('Automatic restart limit reached; saved answers are preserved.')
                state['restarts'] += 1
                commands.append(('EVALUATE', base_command))
            else:
                raise RuntimeError(f'Guard recovery cache validation exited {exit_code}.')
        elif exit_code == 0:
            state['status'] = 'completed'
        elif exit_code == 75:
            if after <= before or not after_bytes.startswith(before_bytes):
                raise RuntimeError('Exit 75 without an append-only durable answer; refusing automatic restart.')
            commands.append(('CACHE CHECK', base_command + ['--cache-only']))
        else:
            raise RuntimeError(f'Evaluator exited {exit_code}; saved answers are preserved.')
    print(f'✅ [SHARD COMPLETE] index={args.shard_index}/2 | responses={state["completed_answers"]}/1600 | saved={output}', flush=True)
except KeyboardInterrupt:
    state['status'] = 'interrupted'
    print('[SHARD INTERRUPTED] Stopping child process group; completed answers are preserved.', flush=True)
    raise SystemExit(130)
except BaseException as error:
    state.update(status='failed', error=f'{type(error).__name__}: {error}')
    raise
finally:
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
    state.update(child_pid=None, updated_at_unix=time.time())
    temporary_state.write_text(json.dumps(state, indent=2) + '\n')
    temporary_state.replace(state_path)
    supervisor_lock.close()
    lock_stream.close()
