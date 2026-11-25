"""Check deployment ordering and detached resume startup with isolated shell fixtures."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

root = Path(__file__).resolve().parents[1]
checks = 0
for name in ("deploy_resume.sh", "resume_run.sh"):
    result = subprocess.run(["bash", "-n", str(root / name)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    checks += 1
with tempfile.TemporaryDirectory(prefix="rlvr-deploy-shell-") as temporary:
    fixture = Path(temporary)
    (fixture / "scripts").mkdir()
    (fixture / "scripts/stop_run.py").write_text("fixture only\n")
    shutil.copy2(root / "deploy_resume.sh", fixture / "deploy_resume.sh")
    bins = fixture / "bin"
    bins.mkdir()
    log = fixture / "commands.jsonl"
    command_source = f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\nwith Path(os.environ['FIXTURE_COMMAND_LOG']).open('a') as f: f.write(json.dumps([Path(sys.argv[0]).name,*sys.argv[1:]])+'\\n')\nif Path(sys.argv[0]).name==os.environ.get('FIXTURE_FAIL_COMMAND'): raise SystemExit(19)\n"
    for command in ("ssh", "scp", "rsync"):
        path = bins / command
        path.write_text(command_source)
        path.chmod(0o755)
    env = {**os.environ, "PATH": str(bins) + os.pathsep + os.environ["PATH"],
           "FIXTURE_COMMAND_LOG": str(log), "RLVR_SSH_KEY": str(fixture / "key with spaces")}
    result = subprocess.run(["bash", "deploy_resume.sh"], cwd=fixture, env=env, text=True, capture_output=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert [command[0] for command in commands] == ["ssh", "scp", "ssh", "rsync", "ssh", "ssh"]
    assert commands[0][commands[0].index("-i") + 1] == str(fixture / "key with spaces")
    assert "stop_run.py --root /root/code" in commands[2][-1]
    assert "--delete" not in commands[3]
    for excluded in (".venv/", ".bootstrap/", "outputs/", "models/Qwen3.5-2B-Base/", "packages/linux-cp311-cu128/wheels/"):
        assert "--exclude=" + excluded in commands[3]
    assert "bash resume_run.sh" in commands[4][-1]
    assert commands[5][commands[5].index("-O") + 1] == "exit"
    checks += 9
    log.unlink()
    result = subprocess.run(["bash", "deploy_resume.sh"], cwd=fixture,
        env={**env, "FIXTURE_FAIL_COMMAND": "scp"}, text=True, capture_output=True)
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert result.returncode == 19
    assert not any(command[0] == "rsync" or "bash resume_run.sh" in command[-1] for command in commands)
    checks += 2

with tempfile.TemporaryDirectory(prefix="rlvr-resume-shell-") as temporary:
    fixture = Path(temporary)
    shutil.copy2(root / "resume_run.sh", fixture / "resume_run.sh")
    (fixture / ".venv/bin").mkdir(parents=True)
    (fixture / "bin").mkdir()
    log = fixture / "preflight.jsonl"
    python_stub = fixture / ".venv/bin/python"
    python_stub.write_text(f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\nwith Path(os.environ['FIXTURE_COMMAND_LOG']).open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\nif 'stop_run.py' in sys.argv[1] and os.environ.get('FIXTURE_BUSY'): raise SystemExit(3)\n")
    python_stub.chmod(0o755)
    setsid = fixture / "bin/setsid"
    setsid.write_text(f"#!{sys.executable}\nimport os,sys\nos.setsid()\nos.execvp(sys.argv[1],sys.argv[1:])\n")
    setsid.chmod(0o755)
    (fixture / "run.sh").write_text('exec "$FIXTURE_PYTHON" -c "import json,sys,time; from pathlib import Path; Path(\'invocation.json\').write_text(json.dumps(sys.argv[1:])); time.sleep(30)" "$@"\n')
    env = {**os.environ, "PATH": str(fixture / "bin") + os.pathsep + os.environ["PATH"],
           "FIXTURE_COMMAND_LOG": str(log), "FIXTURE_PYTHON": sys.executable}
    result = subprocess.run(["bash", "resume_run.sh"], cwd=fixture, env=env, text=True, capture_output=True)
    pid = int((fixture / "logs/latest.pid").read_text())
    try:
        assert result.returncode == 0, (result.stdout, result.stderr)
        deadline = time.monotonic() + 5
        while not (fixture / "invocation.json").is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert os.getpgid(pid) == pid, (pid, os.getpgid(pid), result.stdout, (fixture / "train-current.log").read_text())
        assert json.loads((fixture / "invocation.json").read_text()) == ["all", "--resume"]
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        assert calls[0][0:2] == ["scripts/stop_run.py", "--root"] and calls[0][3:] == ["--check-only"]
        assert Path(calls[0][2]).resolve() == fixture.resolve()
        assert calls[1] == ["scripts/prepare_resume_upgrade.py"]
        assert (fixture / "train-current.log").is_symlink()
        assert (fixture / "train-current.log").resolve().parent == (fixture / "logs").resolve()
        checks += 7
        previous_log = os.readlink(fixture / "train-current.log")
        result = subprocess.run(["bash", "resume_run.sh"], cwd=fixture, env={**env, "FIXTURE_BUSY": "1"}, text=True, capture_output=True)
        assert result.returncode == 3
        assert os.readlink(fixture / "train-current.log") == previous_log
        checks += 2
    finally:
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
print(f"[PASS] {checks} deployment/startup checks; mocked SSH only, isolated sleeper terminated, no remote actions")
