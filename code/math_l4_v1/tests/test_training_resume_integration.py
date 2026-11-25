"""Execute production checkpoint callbacks and resume wiring with small CPU-only fixtures."""

import ast
import contextlib
import copy
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
source = root / "scripts/train_grpo.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
callback_nodes = [node for node in ast.walk(tree) if (
    isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "checkpoint_callback"
        or isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
        and target.value.id == "checkpoint_callback" and target.attr == "on_save"
        for target in node.targets)
    or isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name)
    and node.value.func.value.id == "trainer" and node.value.func.attr == "add_callback"
)]
callback_nodes.sort(key=lambda node: node.lineno)
assert len(callback_nodes) == 3
callback_code = compile(ast.Module(body=callback_nodes, type_ignores=[]), str(source), "exec")
train_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "train_result" for target in node.targets)]
assert len(train_nodes) == 1
train_code = compile(ast.Module(body=train_nodes, type_ignores=[]), str(source), "exec")
resume_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name) and node.test.id == "resume_checkpoint"]
assert len(resume_nodes) == 1
resume_code = compile(ast.Module(body=resume_nodes, type_ignores=[]), str(source), "exec")
checks = 0

for world_size in (1, 2, 4):
    for scenario in ("valid", "wrong_counter", "rollout_boundary", "missing_optimizer", "missing_audit"):
        with tempfile.TemporaryDirectory(prefix="rlvr-save-event-") as temporary:
            output = Path(temporary) / "formal"
            checkpoint = output / "checkpoint-1"
            checkpoint.mkdir(parents=True)
            required_files = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt"]
            required_files += ([f"rng_state_{rank}.pth" for rank in range(world_size)]
                               if world_size > 1 else ["rng_state.pth"])
            for filename in required_files:
                (checkpoint / filename).write_bytes(b"CPU fixture; no tensor or pickle loading\n")
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1, "max_steps": 4}))
            for rank in range(world_size):
                rows = [{"rank": rank, "optimizer_step_before_update": 0, "question_id": f"question-{rank}-{index}"}
                        for index in range(4 // world_size)]
                (output / f"reward_audit.rank-{rank}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            if scenario == "missing_optimizer":
                (checkpoint / "optimizer.pt").unlink()
            if scenario == "missing_audit":
                (output / "reward_audit.rank-0.jsonl").unlink()
            binding = {"mode": "formal", "expected_optimizer_steps": 4,
                       "topology": {"world_size": world_size, "per_rank_unique_prompts_per_update": 4 // world_size}}
            marker = checkpoint / "resume_state.json"
            events = []
            packets = []
            callbacks = []
            trainer = SimpleNamespace(_step=32 // world_size + int(scenario == "wrong_counter"), add_callback=callbacks.append)
            state = SimpleNamespace(global_step=1, max_steps=4)
            control = SimpleNamespace(should_training_stop=False)
            fake_torch = SimpleNamespace(distributed=SimpleNamespace(
                barrier=lambda: events.append(("barrier", marker.exists())),
                broadcast_object_list=lambda packet, src: (
                    events.append(("broadcast", marker.exists(), src)), packets.append(copy.deepcopy(packet[0])))))
            namespace = {"TrainerCallback": SimpleNamespace, "runpy": runpy, "CODE_ROOT": root,
                         "trainer": trainer, "torch": fake_torch, "rank": 0, "world_size": world_size,
                         "output_dir": output, "resume_binding": binding, "accumulation_per_rank": 32 // world_size,
                         "steps_per_generation": 3 if scenario == "rollout_boundary" else 8}
            exec(callback_code, namespace)
            assert len(callbacks) == 1 and callbacks[0] is namespace["checkpoint_callback"]
            if scenario == "valid":
                with contextlib.redirect_stdout(io.StringIO()) as captured:
                    returned = callbacks[0].on_save(None, state, control, model="unused fixture")
                assert returned is control and "[CHECKPOINT SAVED] step=1/4" in captured.getvalue()
                receipt = json.loads(marker.read_text())
                assert receipt["extra_state"] == {"grpo_step": 32 // world_size, "optimizer_step": 1}
                assert receipt["binding"] == binding and set(receipt["audits"]) == {str(rank) for rank in range(world_size)}
                assert set(required_files) <= set(receipt["files"])
                if world_size > 1:
                    assert events == [("barrier", False), ("broadcast", True, 0)], events
                    assert packets == [{"checkpoint": str(checkpoint.resolve())}]
                else:
                    assert events == [] and packets == []
            else:
                try:
                    callbacks[0].on_save(None, state, control)
                except RuntimeError as error:
                    assert "Checkpoint sealing failed" in str(error), str(error)
                    expected = ("rollout/update boundaries" if scenario in {"wrong_counter", "rollout_boundary"}
                                else "Checkpoint is incomplete" if scenario == "missing_optimizer" else "rank audit")
                    assert expected in str(error), str(error)
                else:
                    raise AssertionError(f"Checkpoint callback accepted {scenario}")
                assert not marker.exists()
                if world_size > 1:
                    assert events == [("barrier", False), ("broadcast", False, 0)], events
                    assert len(packets) == 1 and "error" in packets[0]
                else:
                    assert events == [] and packets == []
            checks += 1
            before_files = {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()}
            for rank in range(1, world_size):
                events.clear()
                namespace["rank"] = rank
                namespace["torch"] = SimpleNamespace(distributed=SimpleNamespace(
                    barrier=lambda: events.append("barrier"),
                    broadcast_object_list=lambda packet, src: (
                        events.append(("broadcast", src)), packet.__setitem__(0, copy.deepcopy(packets[0])))))
                if scenario == "valid":
                    with contextlib.redirect_stdout(io.StringIO()) as captured:
                        returned = callbacks[0].on_save(None, state, control)
                    assert returned is control and captured.getvalue() == ""
                else:
                    try:
                        callbacks[0].on_save(None, state, control)
                    except RuntimeError as error:
                        assert "Checkpoint sealing failed" in str(error)
                    else:
                        raise AssertionError("Nonzero rank ignored a checkpoint failure broadcast")
                assert events == ["barrier", ("broadcast", 0)], events
                assert before_files == {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()}
                checks += 1

for checkpoint in (None, "/fixture/formal/checkpoint-16"):
    calls = []
    expected_result = object()
    namespace = {"trainer": SimpleNamespace(train=lambda **kwargs: (calls.append(kwargs), expected_result)[1]),
                 "resume_checkpoint": checkpoint}
    exec(train_code, namespace)
    assert calls == [{"resume_from_checkpoint": checkpoint}] and namespace["train_result"] is expected_result
    checks += 1

# Execute the production counter and sampler-restoration block. Sampler numerical
# parity is tested separately against the pinned upstream implementation.
with tempfile.TemporaryDirectory(prefix="rlvr-resume-wiring-") as temporary:
    code_root = Path(temporary)
    for world_size in (1, 2, 4):
        for step in (16, 128, 144, 384):
            rng_advances = []
            generator = object()
            dataset = object()
            sampler = SimpleNamespace(num_samples=512, shuffle=True, generator=generator)
            trainer = SimpleNamespace(_step=-1, _buffered_inputs=object(), _get_train_sampler=lambda train_dataset=None: sampler)
            checkpoint = code_root / f"outputs/formal/checkpoint-{step}"
            expected_grpo_step = step * (32 // world_size)
            namespace = {"trainer": trainer, "resume_checkpoint": str(checkpoint), "resume_step": step,
                         "accumulation_per_rank": 32 // world_size, "steps_per_generation": 8,
                         "expected_optimizer_steps": 384, "optimizer_steps_per_epoch": 128,
                         "restore_packet": [{"receipt": {"extra_state": {"grpo_step": expected_grpo_step, "optimizer_step": step}}}],
                         "train_dataset": dataset, "records": list(range(512)),
                         "type": lambda obj: SimpleNamespace(__name__="RepeatSampler") if obj is sampler else type(obj),
                         "torch": SimpleNamespace(randperm=lambda num_samples, generator: rng_advances.append((num_samples, generator))),
                         "report": {"resume_history": []}, "CODE_ROOT": code_root, "Path": Path,
                         "datetime": datetime, "timezone": timezone, "args": SimpleNamespace(mode="formal")}
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                exec(resume_code, namespace)
            assert trainer._step == expected_grpo_step and trainer._buffered_inputs is None
            assert rng_advances == [(512, generator)] * (step // 128)
            assert trainer._get_train_sampler(dataset) is sampler and trainer._get_train_sampler() is sampler
            history = namespace["report"]["resume_history"]
            assert len(history) == 1 and history[0]["checkpoint"] == f"outputs/formal/checkpoint-{step}"
            assert history[0]["optimizer_step"] == step and history[0]["grpo_step"] == expected_grpo_step
            assert f"step={step}/384" in captured.getvalue()
            checks += 1

    for scenario in ("wrong_grpo_step", "wrong_optimizer_step", "past_budget", "rollout_boundary", "sampler_name", "sampler_length", "sampler_shuffle"):
        sampler = SimpleNamespace(num_samples=511 if scenario == "sampler_length" else 512,
                                  shuffle=scenario != "sampler_shuffle", generator=object())
        trainer = SimpleNamespace(_step=0, _buffered_inputs=object(), _get_train_sampler=lambda train_dataset=None: sampler)
        step = 385 if scenario == "past_budget" else 16
        extra_state = {"grpo_step": step * 32 + int(scenario == "wrong_grpo_step"),
                       "optimizer_step": step + int(scenario == "wrong_optimizer_step")}
        namespace = {"trainer": trainer, "resume_checkpoint": str(code_root / f"outputs/formal/checkpoint-{step}"),
                     "resume_step": step, "accumulation_per_rank": 32,
                     "steps_per_generation": 3 if scenario == "rollout_boundary" else 8,
                     "expected_optimizer_steps": 384, "optimizer_steps_per_epoch": 128,
                     "restore_packet": [{"receipt": {"extra_state": extra_state}}],
                     "train_dataset": object(), "records": list(range(512)),
                     "type": lambda obj: SimpleNamespace(__name__="UnexpectedSampler" if scenario == "sampler_name" else "RepeatSampler") if obj is sampler else type(obj),
                     "torch": SimpleNamespace(randperm=lambda num_samples, generator: None),
                     "report": {"resume_history": []}, "CODE_ROOT": code_root, "Path": Path,
                     "datetime": datetime, "timezone": timezone, "args": SimpleNamespace(mode="formal")}
        try:
            exec(resume_code, namespace)
        except (ValueError, RuntimeError) as error:
            assert "GRPO progress" in str(error) or "GRPO sampler" in str(error), str(error)
        else:
            raise AssertionError(f"Production resume wiring accepted {scenario}")
        assert namespace["report"]["resume_history"] == []
        checks += 1

trainer = SimpleNamespace(_step=0, _buffered_inputs=object())
initial_buffer = trainer._buffered_inputs
namespace = {"resume_checkpoint": None, "trainer": trainer}
exec(resume_code, namespace)
assert trainer._step == 0 and trainer._buffered_inputs is initial_buffer
checks += 1
print(f"[PASS] {checks} training-resume integration checks; production callbacks, commit ordering, counter restoration and Trainer arguments; no GPU execution")
