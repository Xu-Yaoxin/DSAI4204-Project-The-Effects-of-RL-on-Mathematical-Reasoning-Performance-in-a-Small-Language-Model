"""Supervise two resumable baseline shards on one GPU; merge only complete runs."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/confirmation_v2.json")
parser.add_argument("--output", default="outputs/confirmation_v2_parallel")
parser.add_argument("--samples", type=int)
parser.add_argument("--workers", type=int, choices=(1, 2), default=2,
                    help="Use 1 to resume the same two shards sequentially")
parser.add_argument("--plan", action="store_true")
args = parser.parse_args()
output = (root / args.output).absolute()
if any(path.is_symlink() for path in (output, *output.parents)):
    raise SystemExit("Output must not contain symlinks")
output = output.resolve()
if not output.is_relative_to(root / "outputs") or output == root / "outputs":
    raise SystemExit("Use a dedicated directory under outputs/")
config_path = (root / args.config).resolve()
if not config_path.is_relative_to(root):
    raise SystemExit("Configuration must be within the code directory")
cfg = json.loads(config_path.read_text())
samples = cfg["evaluation"]["n_samples"] if args.samples is None else args.samples
commands = []
for index in range(2):
    command = [sys.executable, str(root / "scripts/evaluate_boundary.py"), "--config", args.config,
               "--output", str(output / f"worker_{index}"), "--num-shards", "2", "--shard-index", str(index), "--resume"]
    if args.samples is not None:
        command += ["--samples", str(samples)]
    commands.append(command)
    subprocess.run([sys.executable, "-S", *command[1:], "--plan"], cwd=root, check=True)
print(f"[PARALLEL PLAN] shards=2 | concurrent_workers={args.workers} | same visible GPU | batch/worker=1", flush=True)
if args.plan:
    print("✅ [PARALLEL PLAN COMPLETE] No weights, GPU or output files touched", flush=True)
    raise SystemExit(0)
cache_results = []
for index, command in enumerate(commands):
    probe = subprocess.run([*command, "--cache-only"], cwd=root, text=True, capture_output=True)
    print(probe.stdout, end="", flush=True)
    if probe.stderr:
        print(probe.stderr, end="", file=sys.stderr, flush=True)
    if probe.returncode not in (0, 3):
        raise SystemExit(f"Worker {index} cached data validation failed; no GPU workers started")
    cache_results.append(probe.returncode)
if cache_results == [0, 0]:
    subprocess.run([sys.executable, "-S", str(root / "scripts/merge_boundary.py"), "--inputs",
                    str(output / "worker_0"), str(output / "worker_1"), "--output", str(output / "base")], cwd=root, check=True)
    subprocess.run([sys.executable, "-S", str(root / "scripts/analyze_boundary.py"), "--input", str(output / "base"),
                    "--output", str(output / "analysis")], cwd=root, check=True)
    print(f"✅ [PARALLEL COMPLETE] 100% | CACHED | no GPU initialized | [SAVED] {output / 'analysis/report.md'}", flush=True)
    raise SystemExit(0)
if sys.platform != "linux":
    raise SystemExit("Execute GPU evaluation on the Linux server; use --plan locally")
if any(int(os.environ.get(name, default)) != expected for name, default, expected in (
        ("WORLD_SIZE", "1", 1), ("LOCAL_WORLD_SIZE", "1", 1), ("RANK", "0", 0), ("LOCAL_RANK", "0", 0))):
    raise SystemExit("Launch independent single-GPU processes, not torchrun/DDP")
cpu_capacity = float(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
quota_path = Path("/sys/fs/cgroup/cpu.max")
if quota_path.is_file():
    quota, period = quota_path.read_text().split()
    if quota != "max":
        cpu_capacity = min(cpu_capacity, int(quota) / int(period))
else:
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = quota_path.with_name("cpu.cfs_period_us")
    if quota_path.is_file() and period_path.is_file() and int(quota_path.read_text()) > 0:
        cpu_capacity = min(cpu_capacity, int(quota_path.read_text()) / int(period_path.read_text()))
if cpu_capacity < args.workers:
    raise SystemExit("Insufficient CPU quota for the requested workers; use --sequential")
threads = int(os.environ.get("RLVR_CPU_THREADS", min(4, max(1, math.floor(cpu_capacity / 2)))))
if not 1 <= threads <= max(1, math.floor(cpu_capacity / args.workers)):
    raise SystemExit("RLVR_CPU_THREADS exceeds the per-worker CPU quota")
worker_env = dict(os.environ)
worker_env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", RLVR_CPU_THREADS=str(threads),
                  OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = worker_env[name]
os.environ["CUDA_VISIBLE_DEVICES"] = worker_env["CUDA_VISIBLE_DEVICES"]
import torch
torch.set_num_threads(1)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
    raise SystemExit("Expose exactly one BF16-capable GPU with CUDA_VISIBLE_DEVICES")
gpu_free, gpu_total = torch.cuda.mem_get_info(0)
memory_info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
ram_available = int(memory_info["MemAvailable"].strip().split()[0]) * 1024
for limit_path, used_path in ((Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
                              (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))):
    if limit_path.is_file() and used_path.is_file():
        limit = limit_path.read_text().strip()
        if limit != "max":
            ram_available = min(ram_available, max(0, int(limit) - int(used_path.read_text())))
required_gpu = 20 if args.workers == 2 else 11
required_ram = 32 if args.workers == 2 else 16
resources = {"checked_utc": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(0),
             "gpu_free_gib": gpu_free / 1024**3, "gpu_total_gib": gpu_total / 1024**3,
             "ram_available_gib": ram_available / 1024**3, "cpu_capacity": cpu_capacity,
             "cpu_threads_per_worker": threads, "concurrent_workers": args.workers,
             "required_free_gpu_gib": required_gpu, "required_available_ram_gib": required_ram}
print("[PARALLEL RESOURCES] " + json.dumps(resources), flush=True)
if gpu_free < required_gpu * 1024**3 or ram_available < required_ram * 1024**3:
    raise SystemExit("Insufficient headroom under the conservative startup policy. Finish other GPU jobs, or use --sequential. No workers started.")
output.mkdir(parents=True, exist_ok=True)
lock_path = output / ".supervisor.lock"
if lock_path.is_symlink():
    raise SystemExit("Unsafe supervisor lock")
lock_stream = lock_path.open("a")
fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
invocation = output / ("invocation-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + f"-{os.getpid()}.json")
resources.update(status="running", supervisor_pid=os.getpid())
invocation.write_text(json.dumps(resources, indent=2) + "\n")
questions = [json.loads(line) for line in (root / cfg["data"]["path"]).read_text().split("\n") if line.strip()]
total = len(questions) * samples
initial_counts = []
initial_tokens = 0
for index in range(2):
    sample_path = output / f"worker_{index}/samples.jsonl"
    saved_rows = [json.loads(line) for line in sample_path.read_text().split("\n") if line.strip()] if sample_path.is_file() else []
    initial_counts.append(len(saved_rows))
    initial_tokens += sum(row.get("output_tokens", 0) for row in saved_rows)
children = []
log_streams = []
started = time.perf_counter()
last_progress = -15.0
next_worker = 0
status = "failed"
signal.signal(signal.SIGTERM, signal.default_int_handler)
try:
    while next_worker < 2 or any(child.poll() is None for child in children):
        for index, child in enumerate(children):
            if child.poll() not in (None, 0):
                raise RuntimeError(f"Worker {index} failed (exit {child.returncode}); see {output / f'worker_{index}.log'}. Partial answers are preserved.")
        while next_worker < 2 and sum(child.poll() is None for child in children) < args.workers:
            log_path = output / f"worker_{next_worker}.log"
            if log_path.is_symlink():
                raise ValueError("Unsafe worker log path")
            log_stream = log_path.open("a", buffering=1)
            log_streams.append(log_stream)
            log_stream.write(f"\n[INVOCATION] {invocation.name}\n")
            child = subprocess.Popen(commands[next_worker], cwd=root, env=worker_env, stdout=log_stream,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            children.append(child)
            print(f"[PARALLEL START] worker={next_worker} | pid={child.pid} | log={log_path}", flush=True)
            next_worker += 1
        elapsed = time.perf_counter() - started
        if elapsed - last_progress >= 15:
            counts = []
            for index in range(2):
                path = output / f"worker_{index}/samples.jsonl"
                counts.append(path.read_bytes().count(b"\n") if path.is_file() else 0)
            completed = sum(counts)
            fresh = completed - sum(initial_counts)
            eta = f"{(total-completed)*elapsed/fresh/60:.1f}min" if fresh > 0 else "warming-up"
            print(f"[PARALLEL PROGRESS] {100*completed/total:.1f}% | {completed}/{total} | worker_counts={counts} | elapsed={elapsed:.0f}s | ETA={eta}", flush=True)
            last_progress = elapsed
        time.sleep(2)
    if any(child.returncode != 0 for child in children):
        raise RuntimeError("A worker failed; merge and analysis were not run. Inspect worker logs.")
    evaluation_seconds = time.perf_counter() - started
    print("[PARALLEL MERGE] 95% | checking complete, disjoint, identical-protocol shards", flush=True)
    subprocess.run([sys.executable, "-S", str(root / "scripts/merge_boundary.py"), "--inputs",
                    str(output / "worker_0"), str(output / "worker_1"), "--output", str(output / "base")], cwd=root, check=True)
    subprocess.run([sys.executable, "-S", str(root / "scripts/analyze_boundary.py"), "--input", str(output / "base"),
                    "--output", str(output / "analysis")], cwd=root, check=True)
    merged_rows = [json.loads(line) for line in (output / "base/samples.jsonl").read_text().split("\n") if line.strip()]
    fresh_tokens = sum(row["output_tokens"] for row in merged_rows) - initial_tokens
    resources.update(evaluation_wall_seconds=evaluation_seconds, new_answers=len(merged_rows)-sum(initial_counts),
                     new_output_tokens=fresh_tokens, aggregate_new_tokens_per_second=fresh_tokens/max(evaluation_seconds, 1e-9))
    status = "completed"
    print(f"✅ [PARALLEL COMPLETE] 100% | answers={len(merged_rows)} | aggregate_new_tokens/s={resources['aggregate_new_tokens_per_second']:.2f} | [SAVED] {output / 'analysis/report.md'}", flush=True)
finally:
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while any(child.poll() is None for child in children) and time.monotonic() < deadline:
        time.sleep(0.1)
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait()
    for stream in log_streams:
        stream.close()
    resources.update(status=status, wall_seconds=time.perf_counter()-started, worker_exit_codes=[child.returncode for child in children])
    invocation.with_suffix(".json.tmp").write_text(json.dumps(resources, indent=2) + "\n")
    invocation.with_suffix(".json.tmp").replace(invocation)
    lock_stream.close()
