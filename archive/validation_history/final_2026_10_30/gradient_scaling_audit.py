"""Reproduce the accumulation divisor audit with real CPU autograd and the locked Accelerate wheel."""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
args = parser.parse_args()
code_root = Path(__file__).resolve().parents[3]
package_root = code_root / "packages/linux-cp311-cu128"
package_manifest = json.loads((package_root / "manifest.json").read_text())
locked_packages = {item["name"]: item for item in package_manifest["packages"]}
locked_accelerate = locked_packages["accelerate"]
assert locked_accelerate["version"] == "1.10.1"
wheel = package_root / "wheels" / locked_accelerate["filename"]
wheel_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
assert wheel_hash == locked_accelerate["sha256"]
sys.path.insert(0, str(wheel))

import torch
import accelerate
from accelerate import Accelerator

assert accelerate.__version__ == "1.10.1"
assert str(wheel) in accelerate.__file__, accelerate.__file__
results = []
for world_size in (1, 2, 4):
    accumulation = 32 // world_size
    for corrected in (False, True):
        rank_gradients = []
        for rank in range(world_size):
            accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulation)
            assert accelerator.gradient_accumulation_steps == accumulation
            if corrected:
                accelerator.gradient_accumulation_steps = 1
            model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
            with torch.no_grad():
                model.weight.zero_()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            model, optimizer = accelerator.prepare(model, optimizer)
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=2)
            sync_events = []
            generation_events = []
            for microstep in range(accumulation):
                do_sync_step = (microstep + 1) % accumulation == 0
                accelerator.gradient_state._set_sync_gradients(do_sync_step)
                sync_events.append(accelerator.sync_gradients)
                if microstep % 8 == 0:
                    generation_events.append(microstep)
                sync_context = contextlib.nullcontext() if do_sync_step else accelerator.no_sync(model)
                with sync_context:
                    target = rank * accumulation + microstep + 1
                    per_token_loss = (model(torch.ones(1, 1, dtype=torch.float64)) - target).square()
                    mask = torch.ones_like(per_token_loss)
                    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
                    loss = loss / accumulation
                    accelerator.backward(loss)
            gradient = model.weight.grad.item()
            rank_gradients.append(gradient)
            optimizer.step()
            scheduler.step()
            assert math.isclose(model.weight.item(), -0.01 * gradient, rel_tol=1e-12)
            assert scheduler.last_epoch == 1
            assert sync_events == [False] * (accumulation - 1) + [True]
            assert generation_events == list(range(0, accumulation, 8))
            accelerator.free_memory()
        global_gradient = sum(rank_gradients) / world_size
        expected = -33.0 if corrected else -33.0 / accumulation
        assert math.isclose(global_gradient, expected, rel_tol=1e-12), (global_gradient, expected)
        results.append({"simulated_world_size": world_size, "trainer_accumulation_steps": accumulation,
            "accelerate_backward_divisor": 1 if corrected else accumulation, "correction_enabled": corrected,
            "cpu_rank_gradients": rank_gradients, "mean_of_cpu_rank_gradients": global_gradient,
            "expected_global_mean_gradient": -33.0, "optimizer_steps_per_rank": 1,
            "scheduler_steps_per_rank": 1, "simulated_generation_groups_per_rank": accumulation // 8})

audit = {"schema_version": 1, "checked_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
    "case_count": len(results), "python": sys.version, "torch_cpu_version": torch.__version__,
    "accelerate_executed_version": accelerate.__version__,
    "accelerate_wheel": str(wheel.relative_to(code_root)), "accelerate_wheel_sha256": wheel_hash,
    "reproducer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "reviewed_locked_wheels": {name: {key: locked_packages[name][key] for key in ["version", "filename", "sha256"]}
        for name in ["accelerate", "transformers", "trl"]},
    "source_review": {
        "transformers/trainer.py:750-807": "Copies Trainer accumulation into Accelerate's plugin.",
        "transformers/trainer.py:1690-1772": "Trainer manually owns prefetch, synchronization, optimizer and scheduler boundaries.",
        "transformers/trainer.py:1925-1934": "compute_loss_func set by TRL disables Trainer's additional loss normalization.",
        "trl/trainer/base_trainer.py:73-79": "Pre-5.19 Transformers compatibility disables Trainer normalization for GRPO.",
        "trl/trainer/grpo_trainer.py:1538-1546": "Generation buffers depend on the GRPO microstep counter and steps_per_generation.",
        "trl/trainer/grpo_trainer.py:3153-3157": "GRPO already divides the loss by current_gradient_accumulation_steps.",
        "accelerate/accelerator.py:1242-1247,2702-2735": "Changing plugin num_steps to one removes only the extra backward division in this Trainer path."},
    "scope": "Actual CPU torch autograd and locked Accelerate.backward; GRPO normalization is reproduced as its scalar expression. Rank gradients are averaged arithmetically.",
    "limitations": "Transformers/TRL training, actual DDP, CUDA, BF16, model generation, model memory and throughput are not executed by this audit.",
    "actual_gpu_execution": False, "actual_ddp_execution": False, "actual_model_training": False,
    "results": results}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(audit, indent=2) + "\n")
print(f"[AUDIT PASSED] {len(results)} CPU scalar-gradient cases; GPU=False; DDP=False; saved={args.output}")
