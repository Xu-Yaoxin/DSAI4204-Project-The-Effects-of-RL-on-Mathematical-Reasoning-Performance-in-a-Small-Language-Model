"""Run verified upstream sampler and GRPO-buffer resume checks on CPU only."""
import argparse
import ast
from collections.abc import Sequence, Sized
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import zipfile

import torch
from torch.utils.data import BatchSampler, Sampler

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trl-wheel", default="/private/tmp/rlvr-final-audit/trl-1.16.0-py3-none-any.whl")
parser.add_argument("--transformers-wheel", default="/private/tmp/rlvr-final-audit/transformers-5.3.0-py3-none-any.whl")
parser.add_argument("--output", default=str(Path(__file__).with_suffix(".json")))
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
manifest = json.loads((root / "packages/linux-cp311-cu128/manifest.json").read_text())
packages = {entry["name"]: entry for entry in manifest["packages"]}
namespace = {"torch": torch, "Sampler": Sampler, "Sized": Sized, "BatchSampler": BatchSampler,
             "Sequence": Sequence, "Any": Any}
sources = []
source_texts = {}
for package, member, path in (
    ("trl", "trl/trainer/utils.py", Path(args.trl_wheel)),
    ("trl", "trl/trainer/grpo_trainer.py", Path(args.trl_wheel)),
    ("transformers", "transformers/trainer.py", Path(args.transformers_wheel)),
    ("accelerate", "accelerate/data_loader.py", root / "packages/linux-cp311-cu128/wheels" / packages["accelerate"]["filename"]),
):
    entry = packages[package]
    wheel_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert wheel_hash == entry["sha256"], str(path)
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(member)
    source = raw.decode()
    source_texts[member] = source
    sources.append({"package": package, "version": entry["version"], "wheel_sha256": wheel_hash,
                    "member": member, "member_sha256": hashlib.sha256(raw).hexdigest()})
    tree = ast.parse(source)
    if member.endswith("utils.py"):
        names = {"RepeatSampler", "shuffle_sequence_dict", "split_tensor_dict",
                 "split_pixel_values_by_grid", "unsplit_pixel_values_by_grid"}
    elif package == "accelerate":
        names = {"BatchSamplerShard"}
    else:
        names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), member, "exec"), namespace)
    if member == "trl/trainer/grpo_trainer.py":
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "_prepare_inputs")
        method.decorator_list = []
        exec(compile(ast.Module(body=[method], type_ignores=[]), member, "exec"), namespace)
        # Pinned GRPO only initializes its private counter once, then increments it.
        counter_assignments = [ast.unparse(node) for node in ast.walk(klass)
                               if isinstance(node, (ast.Assign, ast.AugAssign))
                               and any(isinstance(child, ast.Attribute) and child.attr == "_step"
                                       for child in (node.targets if isinstance(node, ast.Assign) else [node.target]))]
        assert counter_assignments == ["self._step = 0", "self._step += 1"], counter_assignments

sampler_cases = []
broken_epoch_resumes = 0
for questions in (32, 128, 512):
    updates_per_epoch = questions // 4
    total_updates = updates_per_epoch * 3
    checkpoints = sorted({0, 1, updates_per_epoch - 1, updates_per_epoch,
                          updates_per_epoch + 1, total_updates - 1, total_updates})
    for world_size in (1, 2, 4):
        accumulation = 32 // world_size
        for rank in range(world_size):
            sampler = namespace["RepeatSampler"](range(questions), mini_repeat_count=8,
                batch_size=world_size, repeat_count=8, shuffle=True, seed=42)
            shard = namespace["BatchSamplerShard"](BatchSampler(sampler, batch_size=8, drop_last=False),
                num_processes=world_size, process_index=rank, split_batches=False, even_batches=True)
            full = [batch for epoch in range(3) for batch in list(shard)]
            for checkpoint in checkpoints:
                epochs_trained, updates_in_epoch = divmod(checkpoint, updates_per_epoch)
                resumed = namespace["RepeatSampler"](range(questions), mini_repeat_count=8,
                    batch_size=world_size, repeat_count=8, shuffle=True, seed=42)
                for epoch in range(epochs_trained):
                    torch.randperm(questions, generator=resumed.generator)
                resumed_shard = namespace["BatchSamplerShard"](BatchSampler(resumed, batch_size=8, drop_last=False),
                    num_processes=world_size, process_index=rank, split_batches=False, even_batches=True)
                remaining = []
                for epoch in range(epochs_trained, 3):
                    batches = list(resumed_shard)
                    remaining.extend(batches[updates_in_epoch * accumulation:] if epoch == epochs_trained else batches)
                assert remaining == full[checkpoint * accumulation:], (questions, world_size, rank, checkpoint)
                assert checkpoint * accumulation % 8 == 0
                sampler_cases.append({"questions": questions, "world_size": world_size, "rank": rank,
                    "checkpoint_step": checkpoint, "restored_epoch_permutation": True,
                    "remaining_microbatches": len(remaining), "buffer_boundary": True})
            if world_size == 1:
                reset_sampler = namespace["RepeatSampler"](range(questions), mini_repeat_count=8,
                    batch_size=world_size, repeat_count=8, shuffle=True, seed=42)
                wrong = list(BatchSampler(reset_sampler, batch_size=8, drop_last=False))
                one_epoch = questions * 8
                assert wrong != full[one_epoch:2 * one_epoch]
                broken_epoch_resumes += 1

buffer_cases = []
for world_size in (1, 2, 4):
    accumulation = 32 // world_size
    total_updates = 24
    checkpoints = (0, 1, 7, 8, 9, 15, 16, 23, 24)
    torch.manual_seed(12345)
    reference = SimpleNamespace(model=SimpleNamespace(training=True), args=SimpleNamespace(steps_per_generation=8),
                                num_iterations=1, _step=0, _buffered_inputs=None,
                                _generate_and_score_completions=lambda inputs: {"tokens": torch.randint(1000, (8, 4))})
    values = []
    states = {0: torch.get_rng_state().clone()}
    for microstep in range(total_updates * accumulation):
        values.append(namespace["_prepare_inputs"](reference, {})["tokens"].clone())
        reference._step += 1
        if reference._step % accumulation == 0:
            states[reference._step // accumulation] = torch.get_rng_state().clone()
    for checkpoint in checkpoints:
        torch.set_rng_state(states[checkpoint])
        resumed = SimpleNamespace(model=SimpleNamespace(training=True), args=SimpleNamespace(steps_per_generation=8),
                                  num_iterations=1, _step=checkpoint * accumulation, _buffered_inputs=None,
                                  _generate_and_score_completions=lambda inputs: {"tokens": torch.randint(1000, (8, 4))})
        restored = []
        for microstep in range(checkpoint * accumulation, total_updates * accumulation):
            restored.append(namespace["_prepare_inputs"](resumed, {})["tokens"].clone())
            resumed._step += 1
        expected = values[checkpoint * accumulation:]
        assert len(restored) == len(expected) and all(torch.equal(a, b) for a, b in zip(restored, expected))
        assert torch.equal(torch.get_rng_state(), states[total_updates])
        actual_next = torch.rand(8)
        torch.set_rng_state(states[total_updates])
        assert torch.equal(actual_next, torch.rand(8))
        buffer_cases.append({"world_size": world_size, "checkpoint_step": checkpoint,
            "actual_upstream_prepare_inputs": True, "same_remaining_slices": True,
            "same_final_rng_state": True, "same_next_random_draw": True})

trainer_source = source_texts["transformers/trainer.py"]
assert 'model.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)' in trainer_source
assert 'steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps' in trainer_source
assert 'self._save_checkpoint(model, trial)\n            self.control = self.callback_handler.on_save' in trainer_source
report = {"status": "passed", "scope": "Actual hash-locked upstream sampler, sharding, GRPO buffer helpers and CPU RNG",
    "gpu_executed": False, "model_executed": False, "optimizer_restore_executed": False,
    "ddp_collectives_executed": False, "generation": "random CPU tensor fixture, not a language model",
    "host_torch": torch.__version__, "sources": sources, "sampler_case_count": len(sampler_cases),
    "sampler_cases": sampler_cases, "unfixed_epoch_resume_failures_reproduced": broken_epoch_resumes,
    "buffer_rng_case_count": len(buffer_cases), "buffer_rng_cases": buffer_cases,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print(f"[UPSTREAM RESUME AUDIT PASSED] sampler={len(sampler_cases)} buffer_rng={len(buffer_cases)} broken_epoch_order={broken_epoch_resumes}")
print(f"[SAVED] {args.output}")
