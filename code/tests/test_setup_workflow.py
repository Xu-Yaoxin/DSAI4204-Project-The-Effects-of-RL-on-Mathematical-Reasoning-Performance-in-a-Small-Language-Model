"""Exercise setup orchestration and readiness failure gates without GPUs or downloads."""

import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import MagicMock, patch


root = Path(__file__).resolve().parents[1]
setup_text = (root / "setup_gpu.sh").read_text()
inline_blocks = re.findall(r"<<'PY'\n(.*?)\nPY", setup_text, flags=re.DOTALL)
assert len(inline_blocks) == 4, "Update setup fixture stages when inline Python stages change."
for block in inline_blocks:
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   for node in ast.walk(ast.parse(block)))
lock_payload = (root / "packages/linux-cp311-cu128/requirements-lock.txt").read_bytes()
lock_sha256 = hashlib.sha256(lock_payload).hexdigest()
checks = 0

with tempfile.TemporaryDirectory(prefix="dsai4207-setup-workflow-") as directory:
    fixture_root = Path(directory)
    fake_bin = fixture_root / "bin"
    fake_bin.mkdir()
    (fake_bin / "uname").write_text('#!/bin/sh\nif [ "$1" = "-s" ]; then echo Linux; else echo x86_64; fi\n')
    (fake_bin / "nvidia-smi").write_text('#!/bin/sh\nexit 0\n')
    fake_driver = fixture_root / "fixture_driver.py"
    fake_python = fake_bin / "fixture-python"
    fake_python.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' '
                           + shlex.quote(str(fake_driver)) + ' "$@"\n')
    fake_driver.write_text(r'''
import json
import os
from pathlib import Path
import sys

root = Path(os.environ['SETUP_FIXTURE_ROOT'])
failure = os.environ['SETUP_FIXTURE_FAILURE']
args = sys.argv[1:]
calls_path = root / 'calls.jsonl'
previous = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
body = sys.stdin.read() if args == ['-'] else ''
stage = None
if args == ['-']:
    stage = ('host', 'venv', 'cuda', 'readiness')[sum(call['args'] == ['-'] for call in previous) % 4]
elif args[:1] == ['-c']:
    stage = 'lock_hash'
elif args[:3] == ['-m', 'pip', '--version']:
    stage = 'pip_version'
elif args[:3] == ['-m', 'pip', 'install']:
    stage = 'install'
elif args[:3] == ['-m', 'pip', 'check']:
    stage = 'pip_check'
elif args[:3] == ['-m', 'pip', 'freeze']:
    stage = 'freeze'
elif args[:1] == ['scripts/prepare_packages.py']:
    stage = 'packages'
elif args[:1] == ['scripts/download_model.py']:
    stage = 'model'
elif args[:1] == ['scripts/download_data.py']:
    stage = 'data'
elif args[:1] == ['scripts/prepare_data.py']:
    stage = 'splits'
elif args[:1] == ['scripts/build_variants.py']:
    stage = 'variants'
elif args[:2] == ['scripts/run_pipeline.py', 'gpucheck']:
    stage = 'gpucheck'
else:
    raise SystemExit('Unexpected invocation; no real command was executed: ' + repr(args))
with calls_path.open('a') as output:
    output.write(json.dumps({'args': args, 'stage': stage, 'stdin': body,
                            'cwd': str(Path.cwd()), 'progress': os.environ.get('HF_HUB_DISABLE_PROGRESS_BARS'),
                            'device_order': os.environ.get('CUDA_DEVICE_ORDER'),
                            'report_dir': os.environ.get('RLVR_SETUP_REPORT_DIR')}) + '\n')
if stage == failure:
    print('[FIXTURE FAILURE] ' + stage, file=sys.stderr)
    raise SystemExit(73)
if stage == 'lock_hash':
    print(os.environ['SETUP_FIXTURE_LOCK_SHA256'])
elif stage == 'freeze':
    print('fixture-package==1.0')
else:
    print('[FIXTURE PASSED] ' + stage)
''')
    for executable in fake_bin.iterdir():
        executable.chmod(0o755)
    scenarios = [("online", ""), ("offline", ""), ("online_order_override", "")]
    scenarios += [("online", stage) for stage in
                  ("packages", "install", "pip_check", "cuda", "model", "data", "splits", "variants", "gpucheck", "readiness")]
    for mode, failure in scenarios:
        project = fixture_root / (mode + " " + (failure or "success"))
        project.mkdir()
        shutil.copy2(root / "setup_gpu.sh", project / "setup_gpu.sh")
        shutil.copy2(root / "run.sh", project / "run.sh")
        (project / "scripts").mkdir()
        (project / "scripts/run_pipeline.py").write_text("# The fake interpreter intercepts this entry.\n")
        (project / "configs").mkdir()
        shutil.copy2(root / "configs/v1.json", project / "configs/v1.json")
        lock_dir = project / "packages/linux-cp311-cu128"
        lock_dir.mkdir(parents=True)
        (lock_dir / "requirements-lock.txt").write_bytes(lock_payload)
        (project / ".venv/bin").mkdir(parents=True)
        shutil.copy2(fake_python, project / ".venv/bin/python")
        (project / ".venv/.rlvr-package-lock-sha256").write_text(lock_sha256 + "\n")
        environment = {**os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
                       "RLVR_SYSTEM_PYTHON": str(fake_python), "RLVR_PYTHON": str(fake_python),
                       "SETUP_FIXTURE_ROOT": str(project), "SETUP_FIXTURE_FAILURE": failure,
                       "SETUP_FIXTURE_LOCK_SHA256": lock_sha256,
                       "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
        environment.pop("CUDA_DEVICE_ORDER", None)
        if mode == "online_order_override":
            environment["CUDA_DEVICE_ORDER"] = "FASTEST_FIRST"
        command = ["bash", str(project / "setup_gpu.sh")] + (["--offline"] if mode == "offline" else [])
        completed = subprocess.run(command, env=environment, text=True, capture_output=True, cwd=fixture_root)
        calls = [json.loads(line) for line in (project / "calls.jsonl").read_text().splitlines()]
        stages = [call["stage"] for call in calls]
        assert all(Path(call["cwd"]).resolve() == project.resolve() for call in calls)
        assert all(call["progress"] == "0" for call in calls)
        expected_device_order = "FASTEST_FIRST" if mode == "online_order_override" else "PCI_BUS_ID"
        assert all(call["device_order"] == expected_device_order for call in calls)
        expected_order = ["host", "lock_hash", "venv", "pip_version", "pip_version", "packages", "install", "pip_check",
                          "cuda", "freeze", "model", "data", "splits", "variants", "gpucheck", "readiness"]
        if failure:
            expected_order = expected_order[:expected_order.index(failure) + 1]
            assert completed.returncode != 0, (mode, failure, completed.stdout, completed.stderr)
            assert "✅ [SETUP COMPLETE]" not in completed.stdout
            assert "100% | ALL PREPARATION CHECKS PASSED" not in completed.stdout
            assert not list(project.glob("outputs/environment/setup-*/ready.json"))
        else:
            assert completed.returncode == 0, (mode, completed.stdout, completed.stderr)
            assert "✅ [SETUP COMPLETE]" in completed.stdout
            assert "100% | ALL PREPARATION CHECKS PASSED" in completed.stdout
            for phase in range(1, 8):
                assert f"[SETUP {phase}/7]" in completed.stdout
            gpu_call = next(call for call in calls if call["stage"] == "gpucheck")
            report_dir = gpu_call["args"][gpu_call["args"].index("--run-dir") + 1]
            assert report_dir == gpu_call["report_dir"]
            assert not Path(report_dir).is_absolute()
            assert (project / report_dir / "requirements-installed.txt").is_file()
        assert stages == expected_order, (mode, failure, stages, completed.stdout, completed.stderr)
        for stage, flag in (("packages", "--verify"), ("model", "--offline"), ("data", "--offline")):
            for call in calls:
                if call["stage"] == stage:
                    assert (flag in call["args"]) == (mode == "offline"), (mode, stage, call["args"])
        checks += 1
        if mode == "online" and not failure:
            prior_report_dirs = set(project.glob("outputs/environment/setup-*"))
            repeated = subprocess.run(command, env=environment, text=True, capture_output=True, cwd=fixture_root)
            assert repeated.returncode == 0, (repeated.stdout, repeated.stderr)
            assert len(prior_report_dirs) == 1
            assert len(set(project.glob("outputs/environment/setup-*")) - prior_report_dirs) == 1
            checks += 1

    # Execute the actual final Python block against isolated tokenizer and report fixtures.
    for mode in ("ready", "doctor_failed", "hardware_failed", "doctor_exception", "missing_eos", "empty_tokens", "wrong_model",
                 "unverified_receipt", "wrong_revision", "wrong_receipt_hash"):
        project = fixture_root / ("readiness " + mode)
        project.mkdir()
        (project / "configs").mkdir()
        config = {"model": {"path": "models/Qwen3.5-2B-Base", "repo_id": "Qwen/Qwen3.5-2B-Base"}}
        if mode == "wrong_model":
            config["model"]["repo_id"] = "fixture/wrong-model"
        (project / "configs/v1.json").write_text(json.dumps(config))
        (project / "models/Qwen3.5-2B-Base").mkdir(parents=True)
        locked_model = {"repo_id": "Qwen/Qwen3.5-2B-Base", "revision": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"}
        (project / "models/qwen3.5-2b-base-manifest.json").write_text(json.dumps(locked_model))
        receipt = {**locked_model, "status": "verified", "all_files_verified": mode != "unverified_receipt",
                   "manifest_sha256": hashlib.sha256((project / "models/qwen3.5-2b-base-manifest.json").read_bytes()).hexdigest()}
        if mode == "wrong_revision":
            receipt["revision"] = "0" * 40
        if mode == "wrong_receipt_hash":
            receipt["manifest_sha256"] = "0" * 64
        (project / "models/Qwen3.5-2B-Base/source_manifest.json").write_text(json.dumps(receipt))
        output_dir = project / "outputs/environment/test"
        output_dir.mkdir(parents=True)
        (output_dir / "readiness.json").write_text(json.dumps({"ready": mode != "doctor_failed"}))
        (output_dir / "hardware_check.json").write_text(json.dumps({"passed": mode != "hardware_failed"}))
        transformers = MagicMock()
        tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
        tokenizer.eos_token_id = None if mode == "missing_eos" else 1
        tokenizer.encode.return_value = [] if mode == "empty_tokens" else [1, 2, 3]
        runtime = {"world_size": 2, "environment": {"CUDA_VISIBLE_DEVICES": "1,3"}}
        original_cwd = Path.cwd()
        failure_seen = None
        try:
            os.chdir(project)
            with patch.dict(sys.modules, {"transformers": transformers}), \
                    patch.dict(os.environ, {"RLVR_SETUP_REPORT_DIR": "outputs/environment/test"}), \
                    patch("runpy.run_path", return_value={"runtime": runtime}), \
                    patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["doctor.py"]) if mode == "doctor_exception" else None) as doctor, \
                    contextlib.redirect_stdout(io.StringIO()):
                try:
                    exec(compile(inline_blocks[-1], "setup_gpu.sh:readiness", "exec"), {})
                except (SystemExit, subprocess.CalledProcessError) as error:
                    failure_seen = error
        finally:
            os.chdir(original_cwd)
        if mode == "ready":
            assert failure_seen is None, str(failure_seen)
            report = json.loads((output_dir / "ready.json").read_text())
            assert report["ready"] is True and report["selected_gpu_count"] == 2
            assert report["qwen_model_loaded"] is False and report["training_executed"] is False
            assert report["revision"] == locked_model["revision"]
            assert doctor.call_args.kwargs["check"] is True
            assert doctor.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1,3"
            assert "--world-size" in doctor.call_args.args[0] and "2" in doctor.call_args.args[0]
            assert transformers.AutoTokenizer.from_pretrained.call_args.kwargs == {"local_files_only": True, "trust_remote_code": False}
        else:
            assert failure_seen is not None, mode
            assert not (output_dir / "ready.json").exists(), mode
        checks += 1

print(f"[PASS] {checks} setup workflow/readiness checks; all GPU, tokenizer and download calls were isolated fixtures")
