"""Execute the pinned Trainer optimizer/scheduler loader on trusted CPU fixtures."""
import argparse
import ast
import copy
import glob
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import warnings
import zipfile

from packaging.version import Version
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--transformers-wheel", default="/private/tmp/rlvr-final-audit/transformers-5.3.0-py3-none-any.whl")
parser.add_argument("--output", default=str(Path(__file__).with_suffix(".json")))
args = parser.parse_args()
assert Version(torch.__version__.split("+")[0]) >= Version("2.6")
root = Path(__file__).resolve().parents[3]
manifest = json.loads((root / "packages/linux-cp311-cu128/manifest.json").read_text())
package = next(entry for entry in manifest["packages"] if entry["name"] == "transformers")
wheel = Path(args.transformers_wheel)
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == package["sha256"]
namespace = {"torch": torch, "os": os, "glob": glob, "warnings": warnings,
    "OPTIMIZER_NAME": "optimizer.pt", "OPTIMIZER_NAME_BIN": "optimizer.bin", "SCHEDULER_NAME": "scheduler.pt",
    "is_sagemaker_mp_enabled": lambda: False, "is_torch_xla_available": lambda: False,
    "is_torch_greater_or_equal": lambda requested: Version(torch.__version__.split("+")[0]) >= Version(requested)}
sources = []
with zipfile.ZipFile(wheel) as archive:
    for member, target in (
        ("transformers/utils/import_utils.py", "check_torch_load_is_safe"),
        ("transformers/trainer_pt_utils.py", "reissue_pt_warnings"),
        ("transformers/trainer.py", "_load_optimizer_and_scheduler"),
    ):
        raw = archive.read(member)
        tree = ast.parse(raw.decode())
        if target == "_load_optimizer_and_scheduler":
            owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
            node = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == target)
        else:
            node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == target)
        exec(compile(ast.Module(body=[node], type_ignores=[]), member, "exec"), namespace)
        sources.append({"member": member, "member_sha256": hashlib.sha256(raw).hexdigest(), "executed_function": target})

cases = []
with tempfile.TemporaryDirectory(prefix="rlvr-cpu-optimizer-resume-") as temporary:
    directory = Path(temporary)
    torch.manual_seed(20261030)
    weight = torch.nn.Parameter(torch.tensor([0.4, -0.8], dtype=torch.float64))
    optimizer = torch.optim.AdamW([weight], lr=0.01, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    for step in range(1, 13):
        target = torch.randn(2, dtype=torch.float64)
        loss = ((weight - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if step in (1, 4, 8, 12):
            checkpoint = directory / f"checkpoint-{step}"
            checkpoint.mkdir()
            torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
            torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
            torch.save({"weight": weight.detach().clone(), "rng": torch.get_rng_state()}, checkpoint / "fixture.pt")
    expected_weight = weight.detach().clone()
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    expected_rng = torch.get_rng_state().clone()
    expected_next_draw = torch.rand(8)
    for checkpoint_step in (1, 4, 8, 12):
        checkpoint = directory / f"checkpoint-{checkpoint_step}"
        for world_size in (1, 2, 4):
            saved = torch.load(checkpoint / "fixture.pt", weights_only=True)
            resumed_weight = torch.nn.Parameter(saved["weight"].clone())
            resumed_optimizer = torch.optim.AdamW([resumed_weight], lr=0.9, weight_decay=0)
            resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lr_lambda=lambda step: 1.0)
            trainer = SimpleNamespace(is_deepspeed_enabled=False, is_fsdp_xla_v1_enabled=False, is_fsdp_enabled=False,
                args=SimpleNamespace(world_size=world_size, device=torch.device("cpu")),
                optimizer=resumed_optimizer, lr_scheduler=resumed_scheduler)
            namespace["_load_optimizer_and_scheduler"](trainer, str(checkpoint))
            assert resumed_optimizer.param_groups[0]["lr"] == 0.01
            assert resumed_scheduler.last_epoch == checkpoint_step
            torch.set_rng_state(saved["rng"])
            for step in range(checkpoint_step + 1, 13):
                target = torch.randn(2, dtype=torch.float64)
                loss = ((resumed_weight - target) ** 2).sum()
                loss.backward()
                resumed_optimizer.step()
                resumed_scheduler.step()
                resumed_optimizer.zero_grad()
            actual_optimizer = resumed_optimizer.state_dict()
            assert torch.equal(resumed_weight.detach(), expected_weight)
            assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
            assert actual_optimizer["state"].keys() == expected_optimizer["state"].keys()
            for index, state in expected_optimizer["state"].items():
                for key, expected in state.items():
                    actual = actual_optimizer["state"][index][key]
                    assert torch.equal(actual, expected) if isinstance(expected, torch.Tensor) else actual == expected
            assert resumed_scheduler.state_dict() == expected_scheduler
            assert torch.equal(torch.get_rng_state(), expected_rng)
            assert torch.equal(torch.rand(8), expected_next_draw)
            cases.append({"checkpoint_step": checkpoint_step, "world_size_argument": world_size,
                "actual_distributed_execution": False, "cpu_map_location_branch": True,
                "exact_parameter_parity": True, "exact_adamw_state_parity": True,
                "exact_scheduler_state_parity": True, "exact_rng_and_next_draw_parity": True})

report = {"status": "passed", "host_torch": torch.__version__, "transformers_executed_version": package["version"],
    "transformers_wheel_sha256": package["sha256"], "sources": sources, "case_count": len(cases), "cases": cases,
    "scope": "Actual hash-locked Trainer optimizer/scheduler loading and CPU AdamW continuation on trusted synthetic tensors",
    "limitations": "No full Trainer, PEFT adapter loading, language model, BF16, GPU, or DDP execution; model tensor and RNG restored directly for the fixture.",
    "gpu_executed": False, "full_trainer_executed": False, "peft_load_executed": False,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print(f"[OPTIMIZER RESUME AUDIT PASSED] cases={len(cases)} | actual Trainer loader + CPU AdamW")
print(f"[SAVED] {args.output}")
