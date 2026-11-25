"""Run explicit experiment stages using paths relative to the code directory."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", nargs="?", default="check", choices=[
    "check", "prepare", "gpucheck", "engineering", "pilot", "formal", "select", "evaluate", "analyze", "structural", "all"])
parser.add_argument("--config", default="configs/v1.json")
parser.add_argument("--run-dir", default="outputs/v1_2")
parser.add_argument("--gpus", choices=("auto", "1", "2", "4"), default="auto", help="Auto selects up to four visible GPUs; evaluation uses one")
parser.add_argument("--resume", action="store_true", help="Resume evaluation/selection; never resume training")
parser.add_argument("--plan", action="store_true", help="Print relative commands without executing or writing")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1
        or int(os.environ.get("RANK", "0")) not in (-1, 0) or int(os.environ.get("LOCAL_RANK", "0")) not in (-1, 0)):
    parser.error("Run this launcher once; it starts torchrun workers for training itself.")
if Path.cwd().resolve() != root:
    parser.error("Run from project/code: bash run.sh <stage>.")
config_path = Path(args.config)
run_dir = Path(args.run_dir)
if config_path.is_absolute() or run_dir.is_absolute() or ".." in run_dir.parts or ".." in config_path.parts:
    parser.error("Use relative paths within project/code for --config and --run-dir.")
if run_dir == Path(".") or run_dir.parts[0] in {"scripts", "configs", "tests", "models", "data"}:
    parser.error("Use a dedicated output directory, e.g. outputs/v1_2.")
if not config_path.is_file():
    parser.error(f"Config file missing: {config_path}")
config = json.loads(config_path.read_text())
stages = ["check", "prepare", "engineering", "pilot", "formal", "select", "evaluate", "analyze"] if args.stage == "all" else [args.stage]
if args.resume and args.stage not in {"select", "evaluate", "structural", "analyze"}:
    parser.error("--resume is for select/evaluate/structural/analyze. Completed training is reused through later stages.")
python_command = sys.executable
try:
    relative_python = Path(sys.executable).relative_to(root)
    display_python = str(relative_python)
except ValueError:
    display_python = "python3"
selection_path = run_dir / "selection/selection.json"
adapter_argument = "<selected-adapter-from-selection.json>"
commands_executed = 0
gpu_stages = {"gpucheck", "engineering", "pilot", "formal", "select", "evaluate", "structural"}
parallel_stages = {"gpucheck", "engineering", "pilot", "formal"}
needs_gpu = bool(set(stages) & gpu_stages) and not args.resume
needs_parallel = bool(set(stages) & parallel_stages)
world_size = int(args.gpus) if args.gpus != "auto" and needs_parallel else 1
runtime = None
if needs_gpu and not args.plan:
    probe = subprocess.run([python_command, "scripts/resolve_gpus.py", "--gpus", args.gpus if needs_parallel else "1"],
                           cwd=root, text=True, capture_output=True)
    if probe.returncode:
        print(probe.stderr, end="", file=sys.stderr)
        raise SystemExit(probe.returncode)
    runtime = json.loads(probe.stdout)
    world_size = runtime["world_size"]
    print(f"[GPU SELECT] visible={runtime['visible_gpu_count']} | training_processes={world_size} | "
          f"devices={','.join(runtime['selected_device_ids'])} | evaluation_processes=1", flush=True)
    if runtime["visible_gpu_count"] > world_size and needs_parallel:
        print("[GPU SELECT] Extra visible GPUs are unused to preserve four questions per optimizer update.", flush=True)
elif needs_gpu and args.plan:
    print(f"[GPU PLAN] requested={args.gpus} | preview_processes={world_size} | no hardware probed; "
          "auto resolves on the server, explicit --gpus 2/4 previews distributed commands.", flush=True)
for stage_index, stage in enumerate(stages, 1):
    print(f"\n{'='*72}\n[{stage_index:02d}/{len(stages):02d}] {stage.upper()} | config={args.config} | run={run_dir}", flush=True)
    common = ["--config", str(config_path)]
    commands = []
    if stage == "check":
        commands = [["tests/run_all.py"], ["scripts/doctor.py", *common]]
    elif stage == "prepare":
        commands = [["scripts/download_data.py", *common], ["scripts/prepare_data.py", *common], ["scripts/build_variants.py", *common]]
    elif stage == "gpucheck":
        commands = [["scripts/check_gpu.py", "--output", str(run_dir / "hardware_check.json")]]
    elif stage == "engineering":
        commands = [["scripts/check_gpu.py", "--output", str(run_dir / "engineering_hardware.json")],
                    ["scripts/doctor.py", *common, "--gpu", "--world-size", str(world_size), "--output", str(run_dir / "environment.json")],
                    ["scripts/evaluate.py", *common, "--model", "base", "--split", "engineering", "--samples", "8", "--output", str(run_dir / "engineering_base")],
                    ["scripts/train_grpo.py", *common, "--mode", "engineering", "--output", str(run_dir / "engineering_train")]]
    elif stage in {"pilot", "formal"}:
        command = ["scripts/train_grpo.py", *common, "--mode", stage, "--output", str(run_dir / stage)]
        if stage == "formal":
            command += ["--pilot-report", str(run_dir / "pilot/report.json")]
        commands = [["scripts/check_gpu.py", "--output", str(run_dir / f"{stage}_hardware.json")], command]
    elif stage == "select":
        commands = [["scripts/select_checkpoint.py", *common, "--run", str(run_dir / "formal"), "--output", str(run_dir / "selection")]]
        if args.resume:
            commands[0].append("--resume")
    elif stage in {"evaluate", "structural"}:
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text())
            selected_adapter = Path(selection["adapter_path"])
            if selected_adapter.is_absolute():
                try:
                    selected_adapter = selected_adapter.relative_to(root)
                except ValueError as error:
                    raise ValueError("Selection references an external absolute adapter; use a portable run created by v1.1.") from error
            adapter_argument = str(selected_adapter)
        elif not args.plan:
            raise FileNotFoundError(f"Run the select stage first: missing {selection_path}")
        split = "test" if stage == "evaluate" else "structural"
        for model in ["base", "rl"]:
            command = ["scripts/evaluate.py", *common, "--model", model, "--split", split,
                       "--selection", str(selection_path), "--output", str(run_dir / f"{split}_{model}")]
            if model == "rl":
                command += ["--adapter", adapter_argument]
            if args.resume:
                command.append("--resume")
            commands.append(command)
        if stage == "structural":
            commands += [["scripts/analyze.py", "--base", str(run_dir / "structural_base"), "--rl", str(run_dir / "structural_rl"),
                          "--output", str(run_dir / "structural_analysis"), "--bootstrap", str(config["evaluation"]["bootstrap_replicates"])],
                         ["scripts/plot_results.py", "--summary", str(run_dir / "structural_analysis/summary.json"),
                          "--output", str(run_dir / "structural_analysis/overview.png")]]
    elif stage == "analyze":
        commands = [["scripts/analyze.py", "--base", str(run_dir / "test_base"), "--rl", str(run_dir / "test_rl"),
                     "--output", str(run_dir / "main_analysis"), "--bootstrap", str(config["evaluation"]["bootstrap_replicates"])],
                    ["scripts/plot_results.py", "--summary", str(run_dir / "main_analysis/summary.json"),
                     "--output", str(run_dir / "main_analysis/overview.png")]]
    for command in commands:
        if not args.plan and command[0] in {"scripts/check_gpu.py", "scripts/doctor.py"} and "--output" in command:
            output_index = command.index("--output") + 1
            previous_output = Path(command[output_index])
            if previous_output.exists():
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                command[output_index] = str(previous_output.with_name(previous_output.stem + "." + timestamp + previous_output.suffix))
        distributed = world_size > 1 and command[0] in {"scripts/train_grpo.py", "scripts/check_gpu.py"}
        prefix = ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc_per_node={world_size}", "--max_restarts=0"] if distributed else []
        child_env = os.environ.copy()
        for rank_key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
            child_env.pop(rank_key, None)
        if runtime and stage in gpu_stages:
            child_env.update(runtime["environment"])
        if command[0] in {"scripts/evaluate.py", "scripts/select_checkpoint.py"}:
            child_env["CUDA_VISIBLE_DEVICES"] = (runtime["evaluation_device_id"] if runtime
                else child_env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip())
        print("$ " + shlex.join([display_python, *prefix, *command]), flush=True)
        if args.plan:
            continue
        result = subprocess.run([python_command, *prefix, *command], cwd=root, env=child_env)
        if result.returncode != 0 and command[0] == "scripts/train_grpo.py":
            failed_output = Path(command[command.index("--output") + 1])
            failed_report_path = failed_output / "report.json"
            if failed_report_path.is_file():
                failed_report = json.loads(failed_report_path.read_text())
                if failed_report.get("status") == "running":
                    failed_report.update(status="failed", gates={"all_passed": False},
                        launcher_exit_code=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat(),
                        error="Training workers exited unsuccessfully; inspect failure.rank-*.json and the terminal log.")
                    temporary_report = failed_report_path.with_suffix(".launcher.tmp")
                    temporary_report.write_text(json.dumps(failed_report, indent=2) + "\n")
                    temporary_report.replace(failed_report_path)
        if stage == "check" and command[0] == "scripts/doctor.py" and result.returncode == 2:
            print("[CHECK] GPU/model environment is not ready; model-free checks above remain valid.", flush=True)
        elif result.returncode != 0:
            raise SystemExit(result.returncode)
        commands_executed += 1
print(f"\n✅ [{'PLAN' if args.plan else 'STAGE COMPLETE'}] {args.stage} | executed_commands={commands_executed}", flush=True)
