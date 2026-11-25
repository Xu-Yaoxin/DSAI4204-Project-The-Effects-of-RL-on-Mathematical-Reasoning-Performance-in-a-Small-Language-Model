"""Verify safe stop selection and lossless upgrade preparation using isolated fixtures."""

import ast
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

root = Path(__file__).resolve().parents[1]
checks = 0
for name in ("stop_run.py", "prepare_resume_upgrade.py"):
    tree = ast.parse((root / "scripts" / name).read_text())
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
    checks += 1

with tempfile.TemporaryDirectory(prefix="rlvr-upgrade-preservation-") as temporary:
    fixture = Path(temporary) / "code"
    (fixture / "scripts").mkdir(parents=True)
    shutil.copy2(root / "scripts/prepare_resume_upgrade.py", fixture / "scripts/prepare_resume_upgrade.py")
    run = fixture / "outputs/v1_2"
    run.mkdir(parents=True)
    for stage, mode in (("engineering_train", "engineering"), ("formal", "formal")):
        (run / stage).mkdir()
        (run / stage / "report.json").write_text(json.dumps({"mode": mode, "status": "failed"}))
        (run / stage / "original-state.bin").write_bytes(b"retain legacy tensors and logs")
    (run / "pilot").mkdir()
    pilot_bytes = json.dumps({"mode": "pilot", "resume_schema_version": 1, "status": "running"}).encode()
    (run / "pilot/report.json").write_bytes(pilot_bytes)
    for stage in ("selection", "test_base", "test_rl", "main_analysis"):
        (run / stage).mkdir()
        (run / stage / "evidence.bin").write_bytes(stage.encode())
    baseline = run / "engineering_base"
    baseline.mkdir()
    manifest_bytes = json.dumps({"complete": False, "code_hashes": {"evaluate.py": "unchanged"}}).encode()
    (baseline / "manifest.json").write_bytes(manifest_bytes)
    complete_line = b'{"id":"q1","sample_index":0}\n'
    original = complete_line + b'{"id":"q2"'
    (baseline / "samples.jsonl").write_bytes(original)
    result = subprocess.run([sys.executable, "scripts/prepare_resume_upgrade.py"], cwd=fixture, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    archives = list(run.glob("legacy-before-resume-*"))
    assert len(archives) == 1
    archive = archives[0]
    for stage in ("engineering_train", "formal", "selection", "test_base", "test_rl", "main_analysis"):
        assert (archive / stage).is_dir() and not (run / stage).exists()
    assert (archive / "formal/original-state.bin").read_bytes() == b"retain legacy tensors and logs"
    assert (run / "pilot/report.json").read_bytes() == pilot_bytes
    assert (baseline / "manifest.json").read_bytes() == manifest_bytes
    assert (baseline / "samples.jsonl").read_bytes() == complete_line
    backups = list((baseline / "recovery_backups").glob("*.jsonl"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    checks += 3
    result = subprocess.run([sys.executable, "scripts/prepare_resume_upgrade.py"], cwd=fixture, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert list(run.glob("legacy-before-resume-*")) == archives
    assert list((baseline / "recovery_backups").glob("*.jsonl")) == backups
    assert (baseline / "samples.jsonl").read_bytes() == complete_line
    checks += 1

for label, samples, complete, accepted, expected in (
        ("valid-tail", b'{"id":"q1"}', False, True, b'{"id":"q1"}\n'),
        ("complete-tail", b'{"id":"q1"}', True, True, b'{"id":"q1"}'),
        ("middle-invalid", b'broken\n{"id":"q1"}\n', False, False, None),
        ("terminated-invalid", b'{"id":"q1"}\nbroken\n', False, False, None),
        ("complete-corrupt", b'{"id":"q1"}\nbroken', True, False, None),
        ("partial-unicode", b'{"id":"q1"}\n{"text":"\xe4\xb8', False, True, b'{"id":"q1"}\n'),
        ("empty", b'', False, True, b'')):
    with tempfile.TemporaryDirectory(prefix="rlvr-upgrade-tail-") as temporary:
        fixture = Path(temporary) / "code"
        (fixture / "scripts").mkdir(parents=True)
        shutil.copy2(root / "scripts/prepare_resume_upgrade.py", fixture / "scripts/prepare_resume_upgrade.py")
        run = fixture / "outputs/v1_2"
        baseline = run / "engineering_base"
        baseline.mkdir(parents=True)
        manifest_bytes = json.dumps({"complete": complete}).encode()
        (baseline / "manifest.json").write_bytes(manifest_bytes)
        (baseline / "samples.jsonl").write_bytes(samples)
        (run / "formal").mkdir()
        (run / "formal/report.json").write_text(json.dumps({"mode": "formal"}))
        result = subprocess.run([sys.executable, "scripts/prepare_resume_upgrade.py"], cwd=fixture, text=True, capture_output=True)
        assert (result.returncode == 0) == accepted, (label, result.stdout, result.stderr)
        assert (baseline / "manifest.json").read_bytes() == manifest_bytes
        if accepted:
            assert (baseline / "samples.jsonl").read_bytes() == expected, label
        else:
            assert (baseline / "samples.jsonl").read_bytes() == samples
            assert (run / "formal/report.json").is_file()
            assert not list(run.glob("legacy-before-resume-*"))
            assert not (baseline / "recovery_backups").exists()
        checks += 1

with tempfile.TemporaryDirectory(prefix="rlvr-upgrade-rejection-") as temporary:
    fixture = Path(temporary) / "code"
    (fixture / "scripts").mkdir(parents=True)
    shutil.copy2(root / "scripts/prepare_resume_upgrade.py", fixture / "scripts/prepare_resume_upgrade.py")
    formal = fixture / "outputs/v1_2/formal"
    formal.mkdir(parents=True)
    for bad in ("{}", "not-json", "[]", '{"mode":"pilot"}'):
        (formal / "report.json").write_text(bad)
        result = subprocess.run([sys.executable, "scripts/prepare_resume_upgrade.py"], cwd=fixture, text=True, capture_output=True)
        assert result.returncode != 0 and (formal / "report.json").read_text() == bad
        assert not list(formal.parent.glob("legacy-before-resume-*"))
        checks += 1
    outside = Path(temporary) / "outside-report.json"
    outside.write_text('{"mode":"formal"}')
    (formal / "report.json").unlink()
    (formal / "report.json").symlink_to(outside)
    result = subprocess.run([sys.executable, "scripts/prepare_resume_upgrade.py"], cwd=fixture, text=True, capture_output=True)
    assert result.returncode != 0 and outside.read_text() == '{"mode":"formal"}'
    checks += 1

# A fake psutil module records every signal; no real process is stopped by this test.
with tempfile.TemporaryDirectory(prefix="rlvr-stop-filter-") as temporary:
    fixture = Path(temporary) / "code"
    (fixture / "scripts").mkdir(parents=True)
    (fixture / "scripts/run_pipeline.py").write_text("fixture")
    selected_root = str(fixture.resolve())
    process_specs = {
        42: (selected_root, ["python", "scripts/run_pipeline.py"]),
        70: (selected_root, ["python3.11", "scripts/run_pipeline.py", "all"]),
        71: (selected_root, ["python3.11", "scripts/evaluate.py", "--model", "base"]),
        72: ("/elsewhere", ["worker-helper"]),
        73: (selected_root, ["python", "-m", "torch.distributed.run", "--nproc_per_node=2", "scripts/train_grpo.py"]),
        99: ("/different/project", ["python", "scripts/run_pipeline.py"]),
        100: (selected_root, ["python", "unrelated.py", "scripts/run_pipeline.py"]),
        101: (selected_root, ["python", "-c", "scripts/run_pipeline.py"]),
        102: (selected_root, ["bash", "scripts/evaluate.py"]),
        103: (selected_root, ["python", "-m", "unrelated.module", "scripts/train_grpo.py"]),
        104: (selected_root, ["python", "scripts/train_grpo.py"]),
        105: ("/elsewhere", ["dead-worker"]),
    }
    alive = set(process_specs)
    events = []
    waits = []
    processes = {}
    children = {70: [71], 71: [72, 105], 73: []}
    for pid, (cwd, command) in process_specs.items():
        processes[pid] = SimpleNamespace(pid=pid, info={"pid": pid, "cwd": cwd, "cmdline": command})
        processes[pid].status = lambda pid=pid: processes[pid].info.get("status", "running")
        processes[pid].is_running = lambda pid=pid: pid in alive
        processes[pid].children = lambda recursive, pid=pid: [processes[child] for child in children.get(pid, []) if child in alive]
        processes[pid].send_signal = lambda sig, pid=pid: (events.append((pid, sig)),
            alive.discard(pid) if sig == signal.SIGKILL or (pid != 72 and sig in {signal.SIGINT, signal.SIGTERM}) else None)
    processes[104].info["status"] = "zombie"
    processes[105].info["status"] = "zombie"
    fake = SimpleNamespace(NoSuchProcess=ProcessLookupError, AccessDenied=PermissionError, ZombieProcess=ChildProcessError, STATUS_ZOMBIE="zombie",
        Process=lambda pid: SimpleNamespace(pid=pid, parents=lambda: [processes[42]]),
        process_iter=lambda attrs: iter([process for pid, process in processes.items() if pid in alive]),
        wait_procs=lambda items, timeout: (waits.append(timeout),
            ([process for process in items if process.pid not in alive], [process for process in items if process.pid in alive]))[1])
    saved_module = sys.modules.get("psutil")
    saved_argv = sys.argv
    try:
        sys.modules["psutil"] = fake
        sys.argv = ["stop_run.py", "--root", selected_root, "--check-only"]
        try:
            runpy.run_path(str(root / "scripts/stop_run.py"), run_name="__main__")
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError("Active experiment processes were not reported")
        assert events == []
        checks += 1
        sys.argv = ["stop_run.py", "--root", selected_root]
        runpy.run_path(str(root / "scripts/stop_run.py"), run_name="__main__")
        assert {pid for pid, _ in events} == {70, 71, 72, 73}, events
        assert events[0] == (70, signal.SIGSTOP)
        assert (72, signal.SIGKILL) in events
        assert waits == [8, 5, 3]
        assert {42, 99, 100, 101, 102, 103, 104, 105} <= alive
        checks += 4
        signals_before = list(events)
        sys.argv = ["stop_run.py", "--root", selected_root, "--check-only"]
        try:
            runpy.run_path(str(root / "scripts/stop_run.py"), run_name="__main__")
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError("Idle check did not finish")
        assert events == signals_before
        checks += 1
    finally:
        sys.argv = saved_argv
        if saved_module is None:
            sys.modules.pop("psutil", None)
        else:
            sys.modules["psutil"] = saved_module

print(f"[PASS] {checks} resume deployment scenarios; legacy preservation, baseline tail recovery and isolated process selection; no real jobs signalled")
