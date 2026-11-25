"""Supervise two strictly sequential guarded shards with durable progress and bounded recovery."""
import argparse
from datetime import datetime, timezone
from collections import Counter
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
parser.add_argument("--config", default="configs/guarded_v3.json")
parser.add_argument("--output", default="outputs/guarded_v3")
parser.add_argument("--samples", type=int)
parser.add_argument("--workers", type=int, choices=(1,), default=1,
                    help="Only 1 is allowed: fixed sequential evaluation")
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
guard = cfg["evaluation"].get("generation_guard", {})
max_restarts = guard.get("max_worker_restarts", 2)
if type(max_restarts) is not int or not 0 <= max_restarts <= 10:
    raise SystemExit("max_worker_restarts must be an integer from 0 to 10")
commands = []
for index in range(2):
    command = [sys.executable, str(root / "scripts/evaluate_boundary.py"), "--config", args.config,
               "--output", str(output / f"worker_{index}"), "--num-shards", "2", "--shard-index", str(index), "--resume"]
    if args.samples is not None:
        command += ["--samples", str(samples)]
    commands.append(command)
    subprocess.run([sys.executable, "-S", *command[1:], "--plan"], cwd=root, check=True)
print(f"[SERIAL PLAN] shards=2 | concurrent_workers=1 | one model process at a time | batch/worker=1", flush=True)
if args.plan:
    print("✅ [SERIAL PLAN COMPLETE] No weights, GPU or output files touched", flush=True)
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
    cached_counts = Counter()
    cached_total = 0
    with (output / "base/samples.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                cached_total += 1
                cached_counts[row.get("termination_reason", "unspecified")] += 1
    cached_live = output / "live"
    if cached_live.is_symlink():
        raise ValueError("Unsafe live directory")
    cached_live.mkdir(exist_ok=True)
    for name in ("current.json", "current.json.tmp", "current.txt", "current.txt.tmp"):
        if (cached_live / name).is_symlink():
            raise ValueError("Unsafe live summary path")
    cached_state = {"schema_version": 1, "kind": "guarded_supervisor_live", "status": "completed",
                    "updated_at_unix": time.time(), "handled_attempts": cached_total,
                    "total_attempts": cached_total, "progress_percent": 100.0, "sequential": True,
                    "restarts_used": 0, "cached": True, "termination_counts": dict(cached_counts),
                    "workers": [{"worker_index": index, "state": "completed", "pid": None} for index in range(2)]}
    (cached_live / "current.json.tmp").write_text(json.dumps(cached_state, indent=2) + "\n")
    (cached_live / "current.json.tmp").replace(cached_live / "current.json")
    (cached_live / "current.txt.tmp").write_text(f"[GUARDED COMPLETED] cached | handled={cached_total}/{cached_total}\n")
    (cached_live / "current.txt.tmp").replace(cached_live / "current.txt")
    print(f"✅ [SERIAL COMPLETE] 100% | CACHED | no GPU initialized | [SAVED] {output / 'analysis/report.md'}", flush=True)
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
    raise SystemExit("Insufficient CPU quota: serial evaluation requires at least one available CPU core")
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
required_gpu = 11
required_ram = 16
resources = {"checked_utc": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(0),
             "gpu_free_gib": gpu_free / 1024**3, "gpu_total_gib": gpu_total / 1024**3,
             "ram_available_gib": ram_available / 1024**3, "cpu_capacity": cpu_capacity,
             "cpu_threads_per_worker": threads, "concurrent_workers": args.workers,
             "required_free_gpu_gib": required_gpu, "required_available_ram_gib": required_ram}
print("[SERIAL RESOURCES] " + json.dumps(resources), flush=True)
if gpu_free < required_gpu * 1024**3 or ram_available < required_ram * 1024**3:
    raise SystemExit("Insufficient headroom under the conservative startup policy. Finish other GPU jobs or free host memory, then rerun the serial command. No worker started.")
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
counts = []
file_offsets = []
last_records = [None, None]
termination_counts = [Counter(), Counter()]
for index in range(2):
    sample_path = output / f"worker_{index}/samples.jsonl"
    count = 0
    offset = 0
    if sample_path.is_file():
        with sample_path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    raise ValueError("Incomplete existing sample line; preserve and inspect the worker output")
                if line.strip():
                    row = json.loads(line)
                    count += 1
                    initial_tokens += row.get("output_tokens", 0)
                    termination_counts[index][row.get("termination_reason", "unspecified")] += 1
                    last_records[index] = row
                offset += len(line)
    initial_counts.append(count)
    counts.append(count)
    file_offsets.append(offset)
live_dir = output / "live"
if live_dir.is_symlink():
    raise ValueError("Unsafe live directory")
live_dir.mkdir(exist_ok=True)
for name in ("current.json", "current.json.tmp", "current.txt", "current.txt.tmp"):
    if (live_dir / name).is_symlink():
        raise ValueError("Unsafe live summary path")
children = []
log_streams = []
attempts = []
restart_counts = [0, 0]
restarts_used = 0
finished = [False, False]
active = None
started = time.perf_counter()
last_progress = -15.0
status = "failed"
summary = None
signal.signal(signal.SIGTERM, signal.default_int_handler)
try:
    while not all(finished):
        observed_exit = active["child"].poll() if active is not None else None
        for index in range(2):
            sample_path = output / f"worker_{index}/samples.jsonl"
            if not sample_path.exists():
                continue
            if sample_path.is_symlink() or sample_path.stat().st_size < file_offsets[index]:
                raise ValueError("Worker sample file changed identity or shrank during supervision")
            with sample_path.open("rb") as stream:
                stream.seek(file_offsets[index])
                while True:
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    if line.strip():
                        row = json.loads(line)
                        counts[index] += 1
                        termination_counts[index][row.get("termination_reason", "unspecified")] += 1
                        last_records[index] = row
                    file_offsets[index] += len(line)
        if active is not None and observed_exit is not None:
            child = active["child"]
            index = active["index"]
            active["record"].update(exit_code=child.returncode, finished_at_unix=time.time(), handled_after=counts[index])
            if child.returncode == 0:
                finished[index] = True
            elif child.returncode == 75:
                last = last_records[index]
                durable_guarded = (counts[index] > active["handled_before"] and isinstance(last, dict)
                    and last.get("termination_reason") in {"no_token_timeout", "answer_timeout", "repetition_guard"}
                    and ((last.get("termination_reason") == "answer_timeout" and last.get("correct") is False and last.get("reward") == 0)
                         or (last.get("termination_reason") in {"no_token_timeout", "repetition_guard"} and last.get("correct") is None and last.get("reward") is None)))
                if not durable_guarded:
                    raise RuntimeError(f"Worker {index} exited75 without a newly saved guarded result; refusing an unsafe restart")
                probe = subprocess.run([*commands[index], "--cache-only"], cwd=root, env=worker_env, text=True, capture_output=True)
                print(probe.stdout, end="", flush=True)
                if probe.stderr:
                    print(probe.stderr, end="", file=sys.stderr, flush=True)
                if probe.returncode not in (0, 3):
                    raise RuntimeError(f"Worker {index} guarded result failed cache validation; no restart performed")
                active["record"]["post_exit_cache_code"] = probe.returncode
                if probe.returncode == 0:
                    finished[index] = True
                    print(f"[GUARDED SHARD COMPLETE] worker={index} | final guarded attempt saved and cache verified | no restart needed", flush=True)
                else:
                    if restarts_used >= max_restarts:
                        raise RuntimeError(f"Worker restart limit reached ({max_restarts} per invocation). Saved answers are preserved; inspect worker_{index}.log")
                    restarts_used += 1
                    restart_counts[index] += 1
                    print(f"[GUARDED RESTART] worker={index} | reason={last['termination_reason']} | handled={counts[index]} | restart={restarts_used}/{max_restarts} | resume=yes", flush=True)
            else:
                raise RuntimeError(f"Worker {index} failed (exit {child.returncode}); no automatic retry. See {output / f'worker_{index}.log'}. Saved answers are preserved.")
            active = None
        if active is None and not all(finished):
            index = next(index for index in range(2) if not finished[index])
            log_path = output / f"worker_{index}.log"
            if log_path.is_symlink():
                raise ValueError("Unsafe worker log path")
            log_stream = log_path.open("a", buffering=1)
            log_streams.append(log_stream)
            log_stream.write(f"\n[INVOCATION] {invocation.name} | worker={index} | restart={restart_counts[index]}\n")
            launch_time = time.time()
            child = subprocess.Popen(commands[index], cwd=root, env=worker_env, stdout=log_stream,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            children.append(child)
            record = {"worker_index": index, "pid": child.pid, "started_at_unix": launch_time,
                      "restart_number": restart_counts[index], "handled_before": counts[index], "exit_code": None}
            attempts.append(record)
            active = {"child": child, "index": index, "started_at_unix": launch_time,
                      "handled_before": counts[index], "record": record}
            print(f"[GUARDED START] worker={index} | pid={child.pid} | sequential=yes | log={log_path}", flush=True)
        elapsed = time.perf_counter() - started
        now = time.time()
        worker_states = []
        for index in range(2):
            is_active = active is not None and active["index"] == index
            live_path = output / f"worker_{index}/live/current.json"
            live = {}
            warning = None
            if live_path.is_file() and not live_path.is_symlink():
                try:
                    candidate = json.loads(live_path.read_text())
                    if isinstance(candidate, dict):
                        live = candidate
                except (OSError, ValueError):
                    warning = "live_snapshot_temporarily_unavailable"
            if is_active and (live.get("updated_at_unix", 0) < active["started_at_unix"]
                    or live.get("worker_pid", active["child"].pid) != active["child"].pid):
                live = {}
                warning = "waiting_for_current_process_heartbeat"
            token_time = live.get("last_token_unix")
            token_age = max(0, now - token_time) if type(token_time) in (int, float) and math.isfinite(token_time) else None
            state = {"worker_index": index, "pid": active["child"].pid if is_active else None,
                     "state": "running" if is_active else "completed" if finished[index] else "queued",
                     "handled_attempts": counts[index], "restart_count": restart_counts[index],
                     "active_question_id": live.get("id") if is_active else None,
                     "sample_index": live.get("sample_index") if is_active else None,
                     "generated_tokens": live.get("generated_tokens", 0) if is_active else 0,
                     "last_token_age_seconds": token_age if is_active else None,
                     "live_status": live.get("status") if is_active else None,
                     "answer_elapsed_seconds": live.get("elapsed_seconds") if is_active else None,
                     "tokens_per_second": live.get("tokens_per_second") if is_active else None,
                     "answer_timeout_seconds": guard.get("answer_timeout_seconds", 300),
                     "answer_remaining_seconds": max(0, guard.get("answer_timeout_seconds", 300) - (now - live["started_at_unix"])) if is_active and type(live.get("started_at_unix")) in (int, float) else None,
                     "partial_text": live.get("partial_text", "") if is_active else "",
                     "termination_counts": dict(termination_counts[index]), "warning": warning}
            worker_states.append(state)
        combined = termination_counts[0] + termination_counts[1]
        handled = sum(counts)
        fresh = handled - sum(initial_counts)
        eta_seconds = (total - handled) * elapsed / fresh if fresh > 0 else None
        summary = {"schema_version": 1, "kind": "guarded_supervisor_live", "updated_at_unix": now,
                   "status": "running" if not all(finished) else "merging", "total_attempts": total,
                   "handled_attempts": handled, "progress_percent": 100 * handled / total,
                   "elapsed_seconds": elapsed, "eta_seconds": eta_seconds,
                   "eta_reliability": "low: answer lengths, repetition and timeout rates vary",
                   "sequential": True, "restart_limit_per_invocation": max_restarts,
                   "restarts_used": restarts_used, "termination_counts": dict(combined), "workers": worker_states}
        (live_dir / "current.json.tmp").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        (live_dir / "current.json.tmp").replace(live_dir / "current.json")
        eta_text = f"{eta_seconds/60:.1f}min (rough)" if eta_seconds is not None else "warming-up"
        lines = [f"[GUARDED LIVE] {summary['status']} | handled={handled}/{total} ({summary['progress_percent']:.1f}%) | elapsed={elapsed:.0f}s | ETA={eta_text}",
                 "Termination counts: " + json.dumps(dict(combined), sort_keys=True)]
        for state in worker_states:
            age_text = f"{state['last_token_age_seconds']:.1f}s" if state['last_token_age_seconds'] is not None else "unknown"
            lines.append(f"worker={state['worker_index']} | {state['state']} | question={state['active_question_id']} | sample={state['sample_index']} | tokens={state['generated_tokens']} | last_token_age={age_text} | restarts={state['restart_count']}")
            if state["partial_text"]:
                lines.append("Partial: " + str(state["partial_text"]).replace("\x1b", "")[-512:])
        (live_dir / "current.txt.tmp").write_text("\n".join(lines) + "\n")
        (live_dir / "current.txt.tmp").replace(live_dir / "current.txt")
        if elapsed - last_progress >= 15:
            print(lines[0], flush=True)
            for state in worker_states:
                if state["state"] == "running":
                    print(f"[LIVE WORKER] worker={state['worker_index']} | question={state['active_question_id']} | sample={state['sample_index']} | tokens={state['generated_tokens']} | no_token_age={state['last_token_age_seconds']} | terminal={dict(combined)}", flush=True)
            last_progress = elapsed
        if not all(finished):
            time.sleep(0.5)
    evaluation_seconds = time.perf_counter() - started
    print("[SERIAL MERGE] 95% | checking complete, disjoint, identical-protocol shards", flush=True)
    subprocess.run([sys.executable, "-S", str(root / "scripts/merge_boundary.py"), "--inputs",
                    str(output / "worker_0"), str(output / "worker_1"), "--output", str(output / "base")], cwd=root, check=True)
    subprocess.run([sys.executable, "-S", str(root / "scripts/analyze_boundary.py"), "--input", str(output / "base"),
                    "--output", str(output / "analysis")], cwd=root, check=True)
    merged_tokens = 0
    merged_count = 0
    with (output / "base/samples.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                merged_count += 1
                merged_tokens += row["output_tokens"]
    fresh_tokens = merged_tokens - initial_tokens
    resources.update(evaluation_wall_seconds=evaluation_seconds, new_answers=merged_count-sum(initial_counts),
                     new_output_tokens=fresh_tokens, aggregate_new_tokens_per_second=fresh_tokens/max(evaluation_seconds, 1e-9))
    status = "completed"
    print(f"✅ [SERIAL COMPLETE] 100% | handled={merged_count} | aggregate_new_tokens/s={resources['aggregate_new_tokens_per_second']:.2f} | [SAVED] {output / 'analysis/report.md'}", flush=True)
except KeyboardInterrupt:
    status = "interrupted"
    raise
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
    for record, child in zip(attempts, children):
        record.update(exit_code=child.returncode)
    for stream in log_streams:
        stream.close()
    resources.update(status=status, wall_seconds=time.perf_counter()-started,
                     worker_exit_codes=[child.returncode for child in children], worker_attempts=attempts,
                     restarts_used=restarts_used, restart_counts=restart_counts,
                     restart_limit_per_invocation=max_restarts, handled_attempts=sum(counts))
    invocation.with_suffix(".json.tmp").write_text(json.dumps(resources, indent=2) + "\n")
    invocation.with_suffix(".json.tmp").replace(invocation)
    if summary is not None:
        summary.update(status=status, updated_at_unix=time.time(), handled_attempts=sum(counts),
                       progress_percent=100*sum(counts)/total, restarts_used=restarts_used,
                       termination_counts=dict(termination_counts[0] + termination_counts[1]))
        for state in summary["workers"]:
            index = state["worker_index"]
            state.update(handled_attempts=counts[index], restart_count=restart_counts[index],
                         termination_counts=dict(termination_counts[index]))
            if state["state"] == "running":
                state["state"] = status
                state["pid"] = None
        (live_dir / "current.json.tmp").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        (live_dir / "current.json.tmp").replace(live_dir / "current.json")
        (live_dir / "current.txt.tmp").write_text(f"[GUARDED {status.upper()}] handled={sum(counts)}/{total} | restarts={restarts_used}/{max_restarts}\n" + "\n".join(lines[1:]) + "\n")
        (live_dir / "current.txt.tmp").replace(live_dir / "current.txt")
    lock_stream.close()
