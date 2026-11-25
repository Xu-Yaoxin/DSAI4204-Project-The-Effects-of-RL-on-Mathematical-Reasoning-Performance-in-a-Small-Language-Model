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
result = subprocess.run(["bash", "run.sh", "all", "--resume", "--plan"], cwd=root, capture_output=True, text=True)
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
print("[PASS] 46 launcher/GPU-selection scenarios; relocated layouts, plans, masks, worker failures, retry preservation and serial evaluation verified")
