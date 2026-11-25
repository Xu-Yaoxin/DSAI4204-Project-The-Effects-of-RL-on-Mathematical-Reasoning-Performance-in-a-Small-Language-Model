"""Select a supported single-node GPU topology while preserving the user's device mask."""
import argparse
import json
import os
import platform

internal = "REQUESTED_GPUS" in globals()
if internal:
    requested = str(REQUESTED_GPUS)
else:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", choices=("auto", "1", "2", "4"), default="auto")
    requested = parser.parse_args().gpus
if requested not in ("auto", "1", "2", "4"):
    raise ValueError("GPU count must be auto, 1, 2 or 4.")
if "GPU_INVENTORY" in globals():
    devices = GPU_INVENTORY
else:
    if platform.system() != "Linux":
        raise RuntimeError("Real runs require Linux NVIDIA GPUs. Use check/prepare or --plan on this machine.")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the prepared environment and check the NVIDIA driver.")
    devices = []
    for index in range(torch.cuda.device_count()):
        torch.cuda.set_device(index)
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        devices.append({"ordinal": index, "name": properties.name,
                        "capability": [properties.major, properties.minor],
                        "total_bytes": total_bytes, "free_bytes": free_bytes,
                        "bf16": properties.major >= 8 and bool(torch.cuda.is_bf16_supported())})
if not devices:
    raise RuntimeError("No visible CUDA GPUs. Check CUDA_VISIBLE_DEVICES and your server allocation.")
visible_count = len(devices)
world_size = max(count for count in (1, 2, 4) if count <= visible_count) if requested == "auto" else int(requested)
if world_size > visible_count:
    raise ValueError(f"Requested {world_size} GPUs, but only {visible_count} are visible.")
selected = devices[:world_size]
if any(device.get("bf16") is not True for device in selected):
    raise RuntimeError("Every selected GPU must support native BF16; no precision substitution is allowed.")
device_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
if device_mask is None:
    visible_ids = [str(index) for index in range(visible_count)]
else:
    visible_ids = [item.strip() for item in device_mask.split(",")]
    if len(visible_ids) < visible_count or any(not item for item in visible_ids) or len(set(visible_ids)) != len(visible_ids):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain unique, nonempty GPU indices or UUIDs.")
selected_ids = visible_ids[:world_size]
environment = {"CUDA_VISIBLE_DEVICES": ",".join(selected_ids)}
if world_size > 1 and any("RTX 40" in device["name"] for device in selected):
    environment["NCCL_P2P_DISABLE"] = os.environ.get("NCCL_P2P_DISABLE", "1")
    environment["NCCL_IB_DISABLE"] = os.environ.get("NCCL_IB_DISABLE", "1")
runtime = {"requested_gpus": requested, "visible_gpu_count": visible_count, "world_size": world_size,
           "selected_device_ids": selected_ids, "selected_devices": selected, "environment": environment,
           "evaluation_device_id": selected_ids[0], "global_responses_per_update": 32,
           "global_prompts_per_update": 4, "gradient_accumulation_steps_per_rank": 32 // world_size,
           "note": "DDP replicates the model; GPU memory is not pooled. Auto uses at most four GPUs to keep the update batch fixed."}
if not internal:
    print(json.dumps(runtime, indent=2))
