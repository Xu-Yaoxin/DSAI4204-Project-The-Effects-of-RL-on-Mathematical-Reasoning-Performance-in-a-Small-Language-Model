"""Stop only experiment processes belonging to one explicit project directory."""

import argparse
import os
from pathlib import Path
import re
import signal

import psutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", required=True)
parser.add_argument("--check-only", action="store_true")
args = parser.parse_args()
root = Path(args.root).expanduser().resolve(strict=True)
if not root.is_dir() or not (root / "scripts/run_pipeline.py").is_file():
    parser.error("--root must name the experiment code directory.")
current = psutil.Process(os.getpid())
protected = {current.pid, *(process.pid for process in current.parents())}
known = {"scripts/run_pipeline.py", "scripts/train_grpo.py", "scripts/evaluate.py",
         "scripts/select_checkpoint.py", "scripts/check_gpu.py", "scripts/doctor.py", "tests/run_all.py"}
targets = {}
pipelines = {}
for process in psutil.process_iter(["pid", "cwd", "cmdline", "status"]):
    try:
        command = process.info.get("cmdline") or []
        cwd = process.info.get("cwd")
        if (process.pid in protected or process.info.get("status") == psutil.STATUS_ZOMBIE
                or not cwd or Path(cwd).resolve() != root or not command):
            continue
        executable = Path(command[0]).name
        if not (re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable) or executable == "torchrun"):
            continue
        script_tokens = []
        if "-c" in command[1:3]:
            continue
        if "-m" in command[1:4]:
            module_index = command.index("-m") + 1
            if module_index >= len(command) or command[module_index] != "torch.distributed.run":
                continue
            script_tokens = [token for token in command[module_index + 1:] if token.endswith(".py")][:1]
        elif executable == "torchrun":
            script_tokens = [token for token in command[1:] if token.endswith(".py")][:1]
        else:
            script_tokens = [token for token in command[1:] if not token.startswith("-")][:1]
        scripts = []
        for token in script_tokens:
            if not token.endswith(".py"):
                continue
            path = Path(token)
            path = path if path.is_absolute() else root / path
            resolved = path.resolve()
            if resolved.is_relative_to(root) and resolved.relative_to(root).as_posix() in known:
                scripts.append(resolved.relative_to(root).as_posix())
        if not scripts:
            continue
        targets[process.pid] = process
        if "scripts/run_pipeline.py" in scripts:
            pipelines[process.pid] = process
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
        continue
for process in list(targets.values()):
    try:
        for child in process.children(recursive=True):
            if child.pid not in protected and child.status() != psutil.STATUS_ZOMBIE:
                targets[child.pid] = child
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        continue
if args.check_only:
    if targets:
        print(f"[ACTIVE] project={root} | pids={','.join(map(str, sorted(targets)))}", flush=True)
        raise SystemExit(2)
    print(f"[IDLE] No active experiment processes in {root}", flush=True)
    raise SystemExit(0)
if not targets:
    print(f"[STOP] No active experiment processes in {root}", flush=True)
    raise SystemExit(0)

suspended = []
signal_errors = []
remaining = []
try:
    # Freeze the stage launcher before it can start another child process.
    for process in pipelines.values():
        try:
            process.send_signal(signal.SIGSTOP)
            suspended.append(process)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            signal_errors.append(str(error))
    for process in list(targets.values()):
        try:
            for child in process.children(recursive=True):
                if child.pid not in protected and child.status() != psutil.STATUS_ZOMBIE:
                    targets[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    children = [process for pid, process in targets.items() if pid not in pipelines]
    print(f"[STOP] project={root} | selected_pids={','.join(map(str, sorted(targets)))}", flush=True)
    for process in reversed(children):
        try:
            process.send_signal(signal.SIGINT)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            signal_errors.append(str(error))
    _, alive = psutil.wait_procs(children, timeout=8)
    for process in [*alive, *pipelines.values()]:
        try:
            process.send_signal(signal.SIGTERM)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            signal_errors.append(str(error))
    for process in suspended:
        try:
            process.send_signal(signal.SIGCONT)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            signal_errors.append(str(error))
    _, alive = psutil.wait_procs(list(targets.values()), timeout=5)
    for process in alive:
        try:
            process.send_signal(signal.SIGKILL)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as error:
            signal_errors.append(str(error))
    _, candidates = psutil.wait_procs(alive, timeout=3)
    for process in candidates:
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                remaining.append(process)
        except psutil.NoSuchProcess:
            pass
finally:
    for process in suspended:
        try:
            process.send_signal(signal.SIGCONT)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
if remaining or signal_errors:
    raise SystemExit(f"Could not stop every selected process: remaining={[process.pid for process in remaining]}, errors={signal_errors}")
print("[STOP COMPLETE] Selected experiment processes exited; unrelated processes were not signalled.", flush=True)
