"""Reproduce batching checks from verified upstream wheel classes on CPU only."""
import argparse
import ast
from collections.abc import Sized
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import zipfile

import torch
from torch.utils.data import BatchSampler, Sampler

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trl-wheel", required=True, help="Local hash-locked TRL wheel; never downloaded by this script")
parser.add_argument("--output", default=str(Path(__file__).with_suffix(".json")))
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
manifest_path = root / "packages/linux-cp311-cu128/manifest.json"
manifest = json.loads(manifest_path.read_text())
packages = {entry["name"]: entry for entry in manifest["packages"]}
namespace = {"torch": torch, "Sampler": Sampler, "Sized": Sized, "BatchSampler": BatchSampler}
sources = []
for package, member, class_name, path in (
    ("trl", "trl/trainer/utils.py", "RepeatSampler", Path(args.trl_wheel)),
    ("accelerate", "accelerate/data_loader.py", "BatchSamplerShard",
     root / "packages/linux-cp311-cu128/wheels" / packages["accelerate"]["filename"]),
):
    entry = packages[package]
    wheel_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    assert wheel_sha256 == entry["sha256"], f"Unverified upstream wheel: {path}"
    with zipfile.ZipFile(path) as archive:
        source_bytes = archive.read(member)
    source = source_bytes.decode()
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), member, "exec"), namespace)
    sources.append({"package": package, "version": entry["version"], "wheel_url": entry["url"],
        "wheel_sha256": wheel_sha256, "member": member,
        "member_sha256": hashlib.sha256(source_bytes).hexdigest(), "executed_class": class_name})

cases = []
for count in (32, 128, 512):
    for world_size in (1, 2, 4):
        accumulation = 32 // world_size
        per_rank_groups = []
        for rank in range(world_size):
            repeat = namespace["RepeatSampler"](list(range(count)), mini_repeat_count=8,
                batch_size=world_size, repeat_count=8, shuffle=True, seed=42)
            shard = namespace["BatchSamplerShard"](BatchSampler(repeat, batch_size=8, drop_last=False),
                num_processes=world_size, process_index=rank, split_batches=False, even_batches=True)
            batches = list(shard)
            assert len(shard) == len(batches) == count * 8 // world_size
            assert all(len(batch) == 8 and len(set(batch)) == 1 for batch in batches)
            rank_groups = []
            for start in range(0, len(batches), 8):
                assert batches[start:start + 8] == [batches[start]] * 8
                rank_groups.append(batches[start][0])
            per_rank_groups.append(rank_groups)
        questions = []
        for update in range(count // 4):
            window = []
            for rank in range(world_size):
                groups_per_update = accumulation // 8
                window += per_rank_groups[rank][update * groups_per_update:(update + 1) * groups_per_update]
            assert len(window) == len(set(window)) == 4
            questions += window
        assert len(questions) == len(set(questions)) == count
        cases.append({"questions": count, "world_size": world_size,
            "accumulation_per_rank": accumulation, "complete_local_groups": True,
            "unique_questions_per_update": 4, "responses_per_update": 32,
            "complete_epoch_coverage": True, "optimizer_steps_for_three_full_epochs": count // 4 * 3})

report = {"status": "passed", "scope": "CPU execution of the actual upstream sampler classes",
    "gpu_executed": False, "ddp_collectives_executed": False, "model_executed": False,
    "python": platform.python_version(), "host_torch": importlib.metadata.version("torch"),
    "sources": sources, "cases": cases, "case_count": len(cases),
    "engineering_note": "The 32-question case checks a complete sampler epoch; actual engineering training stops after two updates.",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
output = Path(args.output)
with output.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(report, indent=2) + "\n")
print(f"[SAMPLER AUDIT PASSED] cases={len(cases)} | GPU=not executed | saved={output}")
