"""Replace an older Base-to-RL watcher, then stop this host after its current Base work."""
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
parser.add_argument('--poll-seconds', type=float, default=5)
parser.add_argument('--plan', action='store_true')
args = parser.parse_args()
if not 0.2 <= args.poll_seconds <= 60:
    raise SystemExit('--poll-seconds must be between 0.2 and 60.')
raw = Path(args.experiment_root).expanduser()
experiment = (raw if raw.is_absolute() else bundle / raw).resolve()
run = (experiment / args.run_dir).resolve()
if not run.is_relative_to(experiment / 'outputs'):
    raise SystemExit('Run must belong to the frozen experiment outputs.')
state_path = run / 'run_state.json'
state = json.loads(state_path.read_text()) if state_path.is_file() else {}
identity = {'config_sha256': hashlib.sha256((experiment / 'configs/math_l4.json').read_bytes()).hexdigest(),
            'bundle_sha256': hashlib.sha256((experiment / 'bundle_manifest.json').read_bytes()).hexdigest()}
if state and state.get('identity') != identity:
    raise SystemExit('Original run identity differs; no watcher will be interrupted.')
pidfd_helper = bundle / 'scripts/pidfd_compat.py'
if not args.plan:
    pidfd_probe = runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'probe'})['PIDFD_RESULT']
    print('[PIDFD READY] ' + json.dumps(pidfd_probe) + ' | self-check only; Base was not signalled', flush=True)
audit = {'started_utc': datetime.now(timezone.utc).isoformat(), 'mode': 'base_only',
         'experiment_root': str(experiment), 'run_dir': str(run), 'signals': [], 'watchers': [],
         'base_restarted': False, 'rl_launched': False, 'status': 'checking_previous_watchers'}
audit_path = None
held_lock = None
fds = []
stopped = set()
signal.signal(signal.SIGTERM, signal.default_int_handler)
try:
    if not args.plan:
        audit_dir = bundle / 'outputs'
        audit_dir.mkdir(exist_ok=True)
        lock_path = audit_dir / '.watch_base_only.lock'
        if lock_path.is_symlink():
            raise RuntimeError('Unsafe watcher replacement lock.')
        held_lock = lock_path.open('a+')
        try:
            fcntl.flock(held_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another Base-only watcher replacement is in progress; no process was signalled.')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        audit_path = audit_dir / f'watch_base_only_{stamp}_{os.getpid()}.json'
    targets = []
    proc_root = Path('/proc')
    for entry in proc_root.iterdir() if proc_root.is_dir() else []:
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw_stat = (entry / 'stat').read_text()
            fields = raw_stat[raw_stat.rfind(')') + 2:].split()
            if fields[0] == 'Z':
                continue
            command_bytes = (entry / 'cmdline').read_bytes()
            command = command_bytes.rstrip(b'\0').decode().split('\0')
            cwd = (entry / 'cwd').resolve()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeError):
            continue
        candidates = [(index, (cwd / token).resolve()) for index, token in enumerate(command)
                      if token.endswith(('switch_after_base.py', 'watch_base_only.py', 'run_shard.py'))
                      and (cwd / token).resolve() in {bundle / 'scripts/switch_after_base.py',
                           bundle / 'scripts/watch_base_only.py', bundle / 'scripts/run_shard.py'}]
        if len(candidates) != 1 or not command or (cwd / command[0]).resolve() != Path(sys.executable).resolve():
            continue
        script_index, script = candidates[0]
        if not all(token in {'-S', '-u', '-B', '-s', '-E', '-I'} for token in command[1:script_index]):
            continue
        parsed = {}
        ambiguous = False
        for index, token in enumerate(command[script_index + 1:], start=script_index + 1):
            for option in ('--experiment-root', '--run-dir'):
                if token == option or token.startswith(option + '='):
                    if option in parsed or (token == option and index + 1 >= len(command)):
                        ambiguous = True
                    else:
                        parsed[option] = token.split('=', 1)[1] if '=' in token else command[index + 1]
        if ambiguous:
            continue
        old_raw = Path(parsed.get('--experiment-root', '../math_l4_v1')).expanduser()
        old_experiment = (old_raw if old_raw.is_absolute() else bundle / old_raw).resolve()
        old_run = (old_experiment / parsed.get('--run-dir', 'outputs/math_l4_v1_fix1')).resolve()
        if old_experiment != experiment or old_run != run:
            continue
        if script.name == 'run_shard.py':
            raise RuntimeError(f'RL shard is already running on this host (PID {entry.name}). It was left intact; Base-only transition was not applied. Do not start another shard here.')
        targets.append({'pid': int(entry.name), 'start_ticks': fields[19], 'command': command,
                        'command_bytes': command_bytes, 'cwd': str(cwd), 'script': str(script)})
    audit['watchers'] = [{key: value for key, value in row.items() if key != 'command_bytes'} for row in targets]
    if args.plan:
        print('[BASE ONLY PLAN] ' + json.dumps({'phase': state.get('phase'), 'status': state.get('status'),
              'old_watcher_pids': [row['pid'] for row in targets], 'signals_sent': 0,
              'action': 'Replace matching watchers; keep current Base; stop original pipeline after validated completion'}), flush=True)
    else:
        for target in targets:
            proc = proc_root / str(target['pid'])
            try:
                fd = runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'open', 'PIDFD_PID': target['pid']})['PIDFD_RESULT']['fd']
            except ProcessLookupError:
                continue
            fds.append(fd)
            try:
                raw_stat = (proc / 'stat').read_text()
                fields = raw_stat[raw_stat.rfind(')') + 2:].split()
                current_command = (proc / 'cmdline').read_bytes()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if fields[19] != target['start_ticks'] or current_command != target['command_bytes']:
                raise RuntimeError('Previous watcher identity changed before replacement; no TERM was sent. Inspect whether an RL shard has already started.')
            if fields[0] == 'Z':
                continue
            runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'send', 'PIDFD_FD': fd, 'PIDFD_SIGNAL': signal.SIGSTOP})
            stopped.add(fd)
            audit['signals'].append({'pid': target['pid'], 'signal': 'SIGSTOP', 'via': 'pidfd'})
            deadline = time.monotonic() + 5
            while True:
                raw_stat = (proc / 'stat').read_text()
                fields = raw_stat[raw_stat.rfind(')') + 2:].split()
                if fields[19] != target['start_ticks'] or (proc / 'cmdline').read_bytes() != target['command_bytes']:
                    raise RuntimeError('Watcher exec/identity changed during replacement; it will be resumed without TERM. An already started RL shard must be handled separately.')
                if fields[0] in {'T', 't'}:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Previous watcher did not pause safely; it will be resumed without TERM.')
                time.sleep(0.05)
            runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'send', 'PIDFD_FD': fd, 'PIDFD_SIGNAL': signal.SIGTERM})
            audit['signals'].append({'pid': target['pid'], 'signal': 'SIGTERM', 'via': 'pidfd'})
            try:
                runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'send', 'PIDFD_FD': fd, 'PIDFD_SIGNAL': signal.SIGCONT})
            except ProcessLookupError:
                pass
            stopped.discard(fd)
            audit['signals'].append({'pid': target['pid'], 'signal': 'SIGCONT', 'via': 'pidfd'})
            deadline = time.monotonic() + 35
            while (proc / 'stat').is_file():
                raw_stat = (proc / 'stat').read_text()
                fields = raw_stat[raw_stat.rfind(')') + 2:].split()
                if fields[19] != target['start_ticks'] or fields[0] == 'Z':
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Previous watcher has not exited; no new monitor started. It was not force-killed because cleanup may need to resume the original runner.')
                time.sleep(0.2)
            print(f'[OLD WATCHER STOPPED] PID={target["pid"]} | current Base was not signalled', flush=True)
        audit.update(status='monitoring_current_base', finished_utc=datetime.now(timezone.utc).isoformat())
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
        for fd in fds:
            os.close(fd)
        fds.clear()
        held_lock.close()
        held_lock = None
        command = [sys.executable, str(bundle / 'scripts/switch_after_base.py'), '--experiment-root', str(experiment),
                   '--run-dir', args.run_dir, '--poll-seconds', str(args.poll_seconds), '--stop-only']
        print('[BASE ONLY WATCH] Continuing the current Base job; stop after completion; no RL will be launched here.', flush=True)
        os.execv(sys.executable, command)
except BaseException as error:
    if audit_path is not None:
        audit.update(status='failed', error=repr(error))
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    raise
finally:
    for fd in stopped:
        try:
            runpy.run_path(str(pidfd_helper), init_globals={'PIDFD_OPERATION': 'send', 'PIDFD_FD': fd, 'PIDFD_SIGNAL': signal.SIGCONT})
        except ProcessLookupError:
            pass
    for fd in fds:
        os.close(fd)
    if held_lock is not None:
        held_lock.close()
