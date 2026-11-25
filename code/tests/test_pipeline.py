"""Check portable staged commands without loading models or downloading data."""
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
for stage in ["check", "prepare", "gpucheck", "engineering", "pilot", "formal", "select", "evaluate", "analyze", "structural", "all"]:
    result = subprocess.run(["bash", "run.sh", stage, "--plan"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, (stage, result.stdout, result.stderr)
    assert str(root) not in result.stdout, "Launch plan must contain relative project paths"
    assert "executed_commands=0" in result.stdout
    if stage == "formal":
        assert "outputs/v1_2/pilot/report.json" in result.stdout
    if stage in {"evaluate", "structural"}:
        assert "--selection outputs/v1_2/selection/selection.json" in result.stdout
for stage in ("all", "engineering", "pilot", "formal"):
    result = subprocess.run(["bash", "run.sh", stage, "--resume", "--gpus", "2", "--plan"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0 and "executed_commands=0" in result.stdout, (stage, result.stdout, result.stderr)
    assert "[GPU PLAN]" in result.stdout and "preview_processes=2" in result.stdout
    resumable_lines = [line for line in result.stdout.splitlines() if line.startswith("$ ")
                       and any(script in line for script in ("scripts/train_grpo.py", "scripts/evaluate.py", "scripts/select_checkpoint.py"))]
    assert resumable_lines and all(line.count("--resume") == 1 for line in resumable_lines), (stage, resumable_lines)
    training_lines = [line for line in resumable_lines if "scripts/train_grpo.py" in line]
    assert len(training_lines) == (3 if stage == "all" else 1)
    assert all("--nproc_per_node=2" in line for line in training_lines)
    if stage in {"all", "engineering"}:
        assert any("--split engineering" in line for line in resumable_lines)
for stage in ("check", "prepare", "gpucheck"):
    result = subprocess.run(["bash", "run.sh", stage, "--resume", "--plan"], cwd=root, capture_output=True, text=True)
    assert result.returncode != 0 and "--resume" in result.stderr
result = subprocess.run(["bash", "run.sh", "evaluate", "--resume", "--plan", "--run-dir", "outputs/custom run"], cwd=root, capture_output=True, text=True)
assert result.returncode == 0 and "--resume" in result.stdout and "outputs/custom run" in result.stdout
for forbidden in ["/tmp/external-run", "../external-run", ".", "models/replace"]:
    result = subprocess.run(["bash", "run.sh", "check", "--plan", "--run-dir", forbidden], cwd=root, capture_output=True, text=True)
    assert result.returncode != 0, forbidden
with tempfile.TemporaryDirectory(prefix="rlvr-cwd-test-") as temporary:
    result = subprocess.run(["bash", str(root / "run.sh"), "check", "--plan"], cwd=temporary, capture_output=True, text=True)
    assert result.returncode == 0 and "executed_commands=0" in result.stdout
    assert str(root) not in result.stdout
with tempfile.TemporaryDirectory(prefix="rlvr-launch-layout-") as temporary:
    parent = Path(temporary) / "workspace with spaces"
    for layout in (Path("project/code"), Path("code"), Path("renamed experiment")):
        relocated = parent / layout
        (relocated / "scripts").mkdir(parents=True)
        (relocated / "configs").mkdir()
        for relative in ("run.sh", "scripts/run_pipeline.py", "configs/v1.json"):
            shutil.copyfile(root / relative, relocated / relative)
        result = subprocess.run(["bash", str(layout / "run.sh"), "prepare", "--plan"],
                                cwd=parent, capture_output=True, text=True,
                                env={**os.environ, "RLVR_PYTHON": sys.executable})
        assert result.returncode == 0, (str(layout), result.stdout, result.stderr)
        assert "scripts/download_data.py" in result.stdout and "executed_commands=0" in result.stdout, result.stdout
        assert str(parent) not in result.stdout, "Relocated plans must contain relative project paths"
        assert not (relocated / "outputs").exists() and not (relocated / "data").exists()
result = subprocess.run(["bash", "run_v1.sh", "configs/v1.json", "--plan"], cwd=root, capture_output=True, text=True)
assert result.returncode == 0 and "FORMAL" in result.stdout
result = subprocess.run(["bash", "setup_gpu.sh", "--help"], cwd=root, capture_output=True, text=True)
assert result.returncode == 0 and "Linux" in result.stdout
configuration = json.loads((root / "configs/v1.json").read_text())
assert configuration["training"]["beta"] == 0
assert configuration["training"]["num_generations"] == 8
assert configuration["training"]["gradient_accumulation_steps"] == 32
assert configuration["training"]["steps_per_generation"] == 8
for gpu_count in (1, 2, 4):
    result = subprocess.run(["bash", "run.sh", "all", "--gpus", str(gpu_count), "--plan"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "scripts/check_gpu.py" in result.stdout
    training_lines = [line for line in result.stdout.splitlines() if line.startswith("$ ") and "scripts/train_grpo.py" in line]
    assert len(training_lines) == 3
    for line in training_lines:
        assert ("torch.distributed.run" in line) == (gpu_count > 1)
        if gpu_count > 1:
            assert f"--nproc_per_node={gpu_count}" in line
    evaluation_lines = [line for line in result.stdout.splitlines() if line.startswith("$ ") and ("scripts/evaluate.py" in line or "scripts/select_checkpoint.py" in line)]
    assert evaluation_lines and all("torch.distributed.run" not in line for line in evaluation_lines)
result = subprocess.run(["bash", "run.sh", "pilot", "--gpus", "3", "--plan"], cwd=root, capture_output=True, text=True)
assert result.returncode != 0
for rank_key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE", "RANK", "LOCAL_RANK"):
    result = subprocess.run(["bash", "run.sh", "check", "--plan"], cwd=root, capture_output=True, text=True, env={**os.environ, rank_key: "2"})
    assert result.returncode != 0 and "launcher once" in result.stderr

saved_environment = {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE")}
try:
    for key in saved_environment:
        os.environ.pop(key, None)
    for available, expected in ((1, 1), (2, 2), (3, 2), (4, 4), (8, 4)):
        inventory = [{"name": "NVIDIA GeForce RTX 4090", "bf16": True, "ordinal": index} for index in range(available)]
        runtime = runpy.run_path(str(root / "scripts/resolve_gpus.py"), init_globals={"REQUESTED_GPUS": "auto", "GPU_INVENTORY": inventory})["runtime"]
        assert runtime["world_size"] == expected
        assert runtime["gradient_accumulation_steps_per_rank"] * expected == 32
        if expected > 1:
            assert runtime["environment"]["NCCL_P2P_DISABLE"] == "1"
            assert runtime["environment"]["NCCL_IB_DISABLE"] == "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-first,GPU-second,GPU-third"
    runtime = runpy.run_path(str(root / "scripts/resolve_gpus.py"), init_globals={"REQUESTED_GPUS": "auto", "GPU_INVENTORY": inventory[:3]})["runtime"]
    assert runtime["selected_device_ids"] == ["GPU-first", "GPU-second"]
    assert runtime["evaluation_device_id"] == "GPU-first"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    runtime = runpy.run_path(str(root / "scripts/resolve_gpus.py"), init_globals={"REQUESTED_GPUS": "2", "GPU_INVENTORY": inventory[:3]})["runtime"]
    assert runtime["environment"]["NCCL_P2P_DISABLE"] == "0"
    for request, devices in (("4", inventory[:2]), ("auto", []), ("1", [{"name": "old GPU", "bf16": False}])):
        rejected = False
        try:
            runpy.run_path(str(root / "scripts/resolve_gpus.py"), init_globals={"REQUESTED_GPUS": request, "GPU_INVENTORY": devices})
        except (ValueError, RuntimeError):
            rejected = True
        assert rejected
finally:
    for key, value in saved_environment.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
for stage in ("select", "evaluate"):
    result = subprocess.run(["bash", "run.sh", stage, "--resume", "--run-dir", "outputs/absent-resume-fixture"], cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr, result.stderr
    assert "Real runs require Linux" not in result.stderr and "CUDA is unavailable" not in result.stderr
with tempfile.TemporaryDirectory(prefix="rlvr-worker-exit-") as temporary:
    fixture = Path(temporary)
    (fixture / "scripts").mkdir()
    (fixture / "configs").mkdir()
    (fixture / "configs/v1.json").write_text("{}")
    shutil.copyfile(root / "scripts/run_pipeline.py", fixture / "scripts/run_pipeline.py")
    (fixture / "scripts/resolve_gpus.py").write_text('import json\nprint(json.dumps({"world_size":2,"visible_gpu_count":2,"selected_device_ids":["GPU-a","GPU-b"],"environment":{"CUDA_VISIBLE_DEVICES":"GPU-a,GPU-b"},"evaluation_device_id":"GPU-a"}))\n')
    (fixture / "scripts/check_gpu.py").write_text('import sys\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index("--output")+1]); p.parent.mkdir(parents=True,exist_ok=True)\nwith p.open("x") as f: f.write("fixture communication check, not GPU execution")\n')
    (fixture / "scripts/train_grpo.py").write_text('import sys,json\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index("--output")+1]); p.mkdir(parents=True,exist_ok=False)\n(p/"report.json").write_text(json.dumps({"status":"running","gates":{"all_passed":False}}))\n(p/"failure.rank-1.json").write_text("fixture worker error")\nraise SystemExit(7)\n')
    # This tiny module models a worker exit, not DDP computation or communication.
    (fixture / "torch/distributed").mkdir(parents=True)
    (fixture / "torch/__init__.py").write_text("")
    (fixture / "torch/distributed/__init__.py").write_text("")
    (fixture / "torch/distributed/run.py").write_text('import subprocess,sys\ni=next(i for i,x in enumerate(sys.argv) if x.startswith("scripts/"))\nraise SystemExit(subprocess.run([sys.executable,*sys.argv[i:]]).returncode)\n')
    result = subprocess.run([sys.executable, "scripts/run_pipeline.py", "pilot", "--gpus", "2"], cwd=fixture, capture_output=True, text=True)
    assert result.returncode == 7, (result.stdout, result.stderr)
    failed_report = json.loads((fixture / "outputs/v1_2/pilot/report.json").read_text())
    assert failed_report["status"] == "failed" and failed_report["launcher_exit_code"] == 7
    assert (fixture / "outputs/v1_2/pilot/failure.rank-1.json").read_text() == "fixture worker error"
    previous_hardware = (fixture / "outputs/v1_2/pilot_hardware.json").read_bytes()
    result = subprocess.run([sys.executable, "scripts/run_pipeline.py", "pilot", "--gpus", "2"], cwd=fixture, capture_output=True, text=True)
    assert result.returncode != 0 and "FileExistsError" in result.stderr
    assert (fixture / "outputs/v1_2/pilot_hardware.json").read_bytes() == previous_hardware
    assert len(list((fixture / "outputs/v1_2").glob("pilot_hardware.*.json"))) == 1
    assert json.loads((fixture / "outputs/v1_2/pilot/report.json").read_text()) == failed_report
    (fixture / "outputs/v1_2/selection").mkdir()
    (fixture / "outputs/v1_2/selection/selection.json").write_text(json.dumps({"adapter_path": "outputs/v1_2/formal/adapter"}))
    (fixture / "scripts/evaluate.py").write_text('import json,os\nfrom pathlib import Path\nwith Path("evaluation-env.jsonl").open("a") as f: f.write(json.dumps({k:os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES","RANK","LOCAL_RANK","WORLD_SIZE"]})+"\\n")\n')
    result = subprocess.run([sys.executable, "scripts/run_pipeline.py", "evaluate"], cwd=fixture, capture_output=True, text=True,
                            env={**os.environ, "RANK": "0", "LOCAL_RANK": "-1", "WORLD_SIZE": "1"})
    assert result.returncode == 0, result.stderr
    evaluation_environments = [json.loads(line) for line in (fixture / "evaluation-env.jsonl").read_text().split("\n") if line]
    assert len(evaluation_environments) == 2
    assert all(env == {"CUDA_VISIBLE_DEVICES": "GPU-a", "RANK": None, "LOCAL_RANK": None, "WORLD_SIZE": None} for env in evaluation_environments)
with tempfile.TemporaryDirectory(prefix="rlvr-resume-pipeline-") as temporary:
    fixture = Path(temporary)
    for directory in ("scripts", "configs", "tests", "torch/distributed", "outputs/v1_2/engineering_base", "outputs/v1_2/pilot"):
        (fixture / directory).mkdir(parents=True)
    (fixture / "configs/v1.json").write_text(json.dumps({"evaluation": {"bootstrap_replicates": 2}}))
    shutil.copyfile(root / "scripts/run_pipeline.py", fixture / "scripts/run_pipeline.py")
    (fixture / "scripts/resolve_gpus.py").write_text('import json,sys\nfrom pathlib import Path\nwith Path("probes.jsonl").open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\nprint(json.dumps({"world_size":2,"visible_gpu_count":2,"selected_device_ids":["GPU-a","GPU-b"],"environment":{"CUDA_VISIBLE_DEVICES":"GPU-a,GPU-b"},"evaluation_device_id":"GPU-a"}))\n')
    fixture_stage = '''import json,os,sys
from pathlib import Path
name=Path(__file__).name
with Path("commands.jsonl").open("a") as stream:
    stream.write(json.dumps({"script":name,"args":sys.argv[1:],"environment":{key:os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES","RANK","LOCAL_RANK","WORLD_SIZE")}})+"\\n")
if name in {"train_grpo.py","evaluate.py","select_checkpoint.py"}:
    assert sys.argv.count("--resume")==1
if "--output" in sys.argv:
    output=Path(sys.argv[sys.argv.index("--output")+1])
    if name in {"doctor.py","check_gpu.py"}:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("x") as stream: stream.write("fixture hardware check")
    elif name=="select_checkpoint.py":
        output.mkdir(parents=True,exist_ok=True)
        selection=output/"selection.json"
        if not selection.exists():
            selection.write_text(json.dumps({"adapter_path":"outputs/v1_2/formal/adapter"}))
'''
    for name in ("doctor.py", "download_data.py", "prepare_data.py", "build_variants.py", "check_gpu.py", "evaluate.py", "train_grpo.py", "select_checkpoint.py", "analyze.py", "plot_results.py"):
        (fixture / "scripts" / name).write_text(fixture_stage)
    (fixture / "tests/run_all.py").write_text(fixture_stage)
    (fixture / "torch/__init__.py").write_text("")
    (fixture / "torch/distributed/__init__.py").write_text("")
    (fixture / "torch/distributed/run.py").write_text('import json,subprocess,sys\nfrom pathlib import Path\nwith Path("workers.jsonl").open("a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\ni=next(i for i,x in enumerate(sys.argv) if x.startswith("scripts/"))\nraise SystemExit(subprocess.run([sys.executable,*sys.argv[i:]]).returncode)\n')
    baseline_path = fixture / "outputs/v1_2/engineering_base/samples.jsonl"
    pilot_path = fixture / "outputs/v1_2/pilot/report.json"
    baseline_path.write_text("preserved baseline fixture\n")
    pilot_path.write_text("preserved completed pilot fixture\n")
    for attempt in range(2):
        result = subprocess.run([sys.executable, "scripts/run_pipeline.py", "all", "--resume"], cwd=fixture, capture_output=True, text=True,
                                env={**os.environ, "RANK": "0", "LOCAL_RANK": "-1", "WORLD_SIZE": "1"})
        assert result.returncode == 0 and "executed_commands=18" in result.stdout, (result.stdout, result.stderr)
        assert baseline_path.read_text() == "preserved baseline fixture\n"
        assert pilot_path.read_text() == "preserved completed pilot fixture\n"
    probes = [json.loads(line) for line in (fixture / "probes.jsonl").read_text().splitlines()]
    assert probes == [["--gpus", "auto"], ["--gpus", "auto"]], probes
    calls = [json.loads(line) for line in (fixture / "commands.jsonl").read_text().splitlines()]
    assert len(calls) == 36
    for call in calls:
        assert all(call["environment"][key] is None for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")), call
        if call["script"] == "train_grpo.py":
            assert call["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
        elif call["script"] in {"evaluate.py", "select_checkpoint.py"}:
            assert call["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-a"
    for attempt in range(2):
        attempt_calls = calls[attempt * 18:(attempt + 1) * 18]
        train_calls = [call for call in attempt_calls if call["script"] == "train_grpo.py"]
        assert [call["args"][call["args"].index("--mode") + 1] for call in train_calls] == ["engineering", "pilot", "formal"]
        baseline_calls = [call for call in attempt_calls if call["script"] == "evaluate.py" and "engineering" in call["args"]]
        assert len(baseline_calls) == 1 and "--resume" in baseline_calls[0]["args"]
    workers = [json.loads(line) for line in (fixture / "workers.jsonl").read_text().splitlines()]
    assert len(workers) == 12 and all("--nproc_per_node=2" in command for command in workers), workers
    for name in ("engineering_hardware", "pilot_hardware", "formal_hardware", "environment"):
        assert (fixture / "outputs/v1_2" / (name + ".json")).is_file()
        assert len(list((fixture / "outputs/v1_2").glob(name + ".*.json"))) == 1
print("[PASS] 54 launcher/GPU-selection scenarios; resumed stage dispatch, relocated layouts, masks, worker failures, retry preservation and serial evaluation verified")
