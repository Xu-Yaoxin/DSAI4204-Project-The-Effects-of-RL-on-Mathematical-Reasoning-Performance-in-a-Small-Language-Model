"""Execute a small BF16 CUDA and NCCL all-reduce check before any model allocation."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
output = (root / args.output).resolve()
world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if platform.system() != "Linux" or world_size not in (1, 2, 4):
    raise RuntimeError("GPU preflight requires Linux and 1, 2 or 4 local processes.")
if output.exists():
    raise FileExistsError(f"Preserve the existing hardware report; use a new output directory: {output}")
import torch
import torch.distributed as dist

if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
    raise RuntimeError("The number of visible CUDA GPUs must equal the selected process count.")
torch.cuda.set_device(local_rank)
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The selected GPU does not support BF16.")
if world_size > 1:
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120), device_id=torch.device("cuda", local_rank))
matrix = torch.ones((128, 128), dtype=torch.bfloat16, device=local_rank)
product = matrix @ matrix
if not bool(torch.isfinite(product).all()) or not bool((product == 128).all()):
    raise RuntimeError("BF16 CUDA matrix multiplication returned an invalid result.")
total = torch.tensor([rank + 1], dtype=torch.float32, device=local_rank)
if world_size > 1:
    dist.all_reduce(total)
expected = world_size * (world_size + 1) / 2
if total.item() != expected:
    raise RuntimeError("NCCL all-reduce returned an invalid result.")
torch.cuda.synchronize()
free_bytes, total_bytes = torch.cuda.mem_get_info(local_rank)
device = {"rank": rank, "local_rank": local_rank, "name": torch.cuda.get_device_name(local_rank),
          "capability": list(torch.cuda.get_device_capability(local_rank)),
          "free_bytes_after_check": free_bytes, "total_bytes": total_bytes,
          "bf16_matmul_passed": True, "all_reduce_sum": total.item()}
gathered = [None] * world_size
if world_size > 1:
    dist.all_gather_object(gathered, device)
else:
    gathered = [device]
if rank == 0:
    driver = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
                            text=True, capture_output=True, timeout=20)
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
              "world_size": world_size, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "nccl": list(torch.cuda.nccl.version()) if isinstance(torch.cuda.nccl.version(), tuple) else torch.cuda.nccl.version(),
              "devices": gathered, "nvidia_smi": driver.stdout.strip(),
              "environment": {name: os.environ.get(name) for name in ("CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "PYTORCH_ALLOC_CONF")},
              "scope": "Small BF16 operation and process communication only; model memory and training not tested."}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(f"✅ [GPU CHECK 100%] ranks={world_size} | BF16=passed | collective=passed | saved={args.output}", flush=True)
if world_size > 1 and dist.is_initialized():
    dist.destroy_process_group()
