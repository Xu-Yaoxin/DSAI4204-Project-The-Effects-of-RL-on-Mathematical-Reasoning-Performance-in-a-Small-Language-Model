"""Exercise checkpoint commits and recovery with real small files, without GPU packages."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import tempfile

script = Path(__file__).resolve().parents[1] / "scripts/training_resume.py"
tree = ast.parse(script.read_text(encoding="utf-8"))
assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
checks = 1

for world_size in (1, 2, 4):
    with tempfile.TemporaryDirectory(prefix="rlvr-checkpoint-recovery-") as temporary:
        parent = Path(temporary)
        output = parent / "formal"
        binding = {"mode": "formal", "config_training_hash": "a" * 64, "expected_optimizer_steps": 4,
                   "model_identity": {"sha256": "b" * 64}, "package_versions": {"torch": "fixture"},
                   "topology": {"world_size": world_size, "per_rank_unique_prompts_per_update": 4 // world_size},
                   "initial_state": {"base_sha256": "c" * 64, "adapter_sha256": "d" * 64}}
        request = {"operation": "inspect", "output_dir": output, "binding": binding}
        result = runpy.run_path(str(script), init_globals={"RESUME_REQUEST": request})["RESUME_RESULT"]
        assert result["checkpoint"] is None and not output.exists()
        checks += 1
        output.mkdir()
        required = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt"]
        required += ([f"rng_state_{rank}.pth" for rank in range(world_size)] if world_size > 1 else ["rng_state.pth"])
        first = output / "checkpoint-1"
        first.mkdir()
        for name in required:
            (first / name).write_bytes(b"fixture-only: no pickle or tensors loaded\n")
        (first / "trainer_state.json").write_text(json.dumps({"global_step": 1, "max_steps": 4}))
        snapshots = {}
        for rank in range(world_size):
            lines = [json.dumps({"rank": rank, "optimizer_step_before_update": 0,
                                "question_id": f"q-{rank}-{index}", "rewards": [0, 1] * 4,
                                "samples": [{"text": "fixture"}] * 8}) + "\n"
                     for index in range(4 // world_size)]
            snapshots[rank] = "".join(lines).encode()
            (output / f"reward_audit.rank-{rank}.jsonl").write_bytes(snapshots[rank])
        seal = {**request, "operation": "seal", "step": 1,
                "extra_state": {"grpo_step": 32 // world_size, "optimizer_step": 1}}
        result = runpy.run_path(str(script), init_globals={"RESUME_REQUEST": seal})["RESUME_RESULT"]
        assert result["step"] == 1 and result["receipt"]["binding"] == binding
        assert result["receipt"]["extra_state"] == seal["extra_state"]
        assert all((first / f"resume_audit.rank-{rank}.jsonl").read_bytes() == snapshots[rank] for rank in range(world_size))
        committed = (first / "resume_state.json").read_bytes()
        repeated = runpy.run_path(str(script), init_globals={"RESUME_REQUEST": seal})["RESUME_RESULT"]
        assert repeated["receipt"] == result["receipt"] and (first / "resume_state.json").read_bytes() == committed
        checks += 2

        unsealed = output / "checkpoint-2"
        unsealed.mkdir()
        (unsealed / "optimizer.pt").write_bytes(b"interrupted write")
        inspected = runpy.run_path(str(script), init_globals={"RESUME_REQUEST": request})["RESUME_RESULT"]
        assert inspected["step"] == 1 and unsealed.is_dir()
        checks += 1
        current_bytes = {}
        for rank in range(world_size):
            current_bytes[rank] = snapshots[rank] + b'{"interrupted":'
            (output / f"reward_audit.rank-{rank}.jsonl").write_bytes(current_bytes[rank])
        (output / "adapter").mkdir()
        (output / "adapter" / "adapter_model.safetensors").write_bytes(b"unfinished final model")
        for name in ("reward_audit.jsonl", "trainer_logs.json", "trainer_state.json", "module_audit.json", "failure.rank-0.json"):
            (output / name).write_text("partial derived artifact")
        restored = runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {**request, "operation": "restore"}})["RESUME_RESULT"]
        archived = Path(restored["recovery_dir"])
        assert archived.is_dir() and (archived / "checkpoint-2/optimizer.pt").read_bytes() == b"interrupted write"
        assert (archived / "adapter/adapter_model.safetensors").read_bytes() == b"unfinished final model"
        assert not unsealed.exists() and not (output / "reward_audit.jsonl").exists()
        for rank in range(world_size):
            assert (output / f"reward_audit.rank-{rank}.jsonl").read_bytes() == snapshots[rank]
            assert (archived / f"reward_audit.rank-{rank}.jsonl").read_bytes() == current_bytes[rank]
        again = runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {**request, "operation": "restore"}})["RESUME_RESULT"]
        assert again["recovery_dir"] is None
        assert len(list((output / "recoveries").iterdir())) == 1
        checks += 2

        for rank in range(world_size):
            target = output / f"reward_audit.rank-{rank}.jsonl"
            target.write_bytes(snapshots[rank][:17])
        recovered_short = runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {**request, "operation": "restore"}})["RESUME_RESULT"]
        assert recovered_short["recovery_dir"] is not None
        assert all((output / f"reward_audit.rank-{rank}.jsonl").read_bytes() == snapshots[rank] for rank in range(world_size))
        (output / "reward_audit.rank-0.jsonl").unlink()
        recovered_missing = runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {**request, "operation": "restore"}})["RESUME_RESULT"]
        assert (output / "reward_audit.rank-0.jsonl").read_bytes() == snapshots[0]
        checks += 2

        second = output / "checkpoint-2"
        second.mkdir()
        for name in required:
            (second / name).write_bytes(b"second complete checkpoint\n")
        (second / "trainer_state.json").write_text(json.dumps({"global_step": 2, "max_steps": 4}))
        second_snapshots = {}
        for rank in range(world_size):
            lines = [json.dumps({"rank": rank, "optimizer_step_before_update": 1,
                                "question_id": f"next-{rank}-{index}"}) + "\n" for index in range(4 // world_size)]
            second_snapshots[rank] = snapshots[rank] + "".join(lines).encode()
            (output / f"reward_audit.rank-{rank}.jsonl").write_bytes(second_snapshots[rank])
        runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {**seal, "step": 2,
            "extra_state": {"grpo_step": 64 // world_size, "optimizer_step": 2}}})
        latest = runpy.run_path(str(script), init_globals={"RESUME_REQUEST": request})["RESUME_RESULT"]
        assert latest["step"] == 2 and Path(latest["checkpoint"]) == second.resolve()
        checks += 1

        before_files = {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in output.rglob("*") if path.is_file()}
        changed_binding = copy.deepcopy(binding)
        changed_binding["config_training_hash"] = "f" * 64
        try:
            runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {**request, "operation": "restore", "binding": changed_binding}})
        except ValueError as error:
            assert "binding changed" in str(error)
        else:
            raise AssertionError("Changed training protocol was accepted")
        assert before_files == {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in output.rglob("*") if path.is_file()}
        checks += 1

        original_optimizer = (second / "optimizer.pt").read_bytes()
        (second / "optimizer.pt").write_bytes(b"X" + original_optimizer[1:])
        for operation in ("inspect", "restore"):
            try:
                runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {**request, "operation": operation}})
            except ValueError as error:
                assert "hash changed" in str(error)
            else:
                raise AssertionError("Tampered committed state fell back to an older checkpoint")
            checks += 1
        (second / "optimizer.pt").write_bytes(original_optimizer)
        rank_zero = output / "reward_audit.rank-0.jsonl"
        rank_zero.write_bytes(b"X" + second_snapshots[0][1:])
        archives_before = set((output / "recoveries").iterdir())
        try:
            runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {**request, "operation": "restore"}})
        except ValueError as error:
            assert "diverges" in str(error)
        else:
            raise AssertionError("Changed audit prefix was overwritten")
        assert set((output / "recoveries").iterdir()) == archives_before
        assert rank_zero.read_bytes().startswith(b"X")
        rank_zero.write_bytes(second_snapshots[0])
        checks += 1

        rng_name = "rng_state_0.pth" if world_size > 1 else "rng_state.pth"
        rng_bytes = (second / rng_name).read_bytes()
        (second / rng_name).unlink()
        try:
            runpy.run_path(str(script), init_globals={"RESUME_REQUEST": request})
        except ValueError as error:
            assert "incomplete" in str(error)
        else:
            raise AssertionError("Missing rank RNG state was accepted")
        (second / rng_name).write_bytes(rng_bytes)
        checks += 1

        external = parent / "outside"
        external.mkdir()
        (output / "adapter").symlink_to(external, target_is_directory=True)
        rank_zero.write_bytes(second_snapshots[0] + b"partial tail")
        try:
            runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {**request, "operation": "restore"}})
        except ValueError as error:
            assert "symlink" in str(error)
        else:
            raise AssertionError("Symlinked recovery artifact was accepted")
        assert rank_zero.read_bytes().endswith(b"partial tail") and not list(external.iterdir())
        (output / "adapter").unlink()
        (output / "checkpoint-3").symlink_to(external, target_is_directory=True)
        try:
            runpy.run_path(str(script), init_globals={"RESUME_REQUEST": request})
        except ValueError as error:
            assert "Unsafe checkpoint" in str(error)
        else:
            raise AssertionError("Symlinked checkpoint was accepted")
        (output / "checkpoint-3").unlink()
        checks += 2

with tempfile.TemporaryDirectory(prefix="rlvr-uncommitted-recovery-") as temporary:
    output = Path(temporary)
    binding = {"expected_optimizer_steps": 4, "topology": {"world_size": 1, "per_rank_unique_prompts_per_update": 2}}
    checkpoint = output / "checkpoint-1"
    checkpoint.mkdir()
    for name in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (checkpoint / name).write_bytes(b"fixture state")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1, "max_steps": 4}))
    good = [{"rank": 0, "optimizer_step_before_update": 0, "question_id": "q1"},
            {"rank": 0, "optimizer_step_before_update": 0, "question_id": "q2"}]
    bad_rows = [good[:1], [good[0], good[0]], [good[0], {**good[1], "rank": 1}],
                [good[0], {**good[1], "optimizer_step_before_update": 1}]]
    for rows in bad_rows:
        (output / "reward_audit.rank-0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        try:
            runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {
                "operation": "seal", "output_dir": output, "binding": binding, "step": 1}})
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete or invalid reward groups were committed")
        assert not (checkpoint / "resume_state.json").exists()
        inspected = runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {
            "operation": "inspect", "output_dir": output, "binding": binding}})["RESUME_RESULT"]
        assert inspected["checkpoint"] is None
        checks += 1
    (output / "reward_audit.rank-0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in good).rstrip("\n"))
    try:
        runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {
            "operation": "seal", "output_dir": output, "binding": binding, "step": 1}})
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("A partial JSONL record was committed")
    checks += 1
    (output / "reward_audit.rank-0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in good))
    sealed = runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {
        "operation": "seal", "output_dir": output, "binding": binding, "step": 1}})["RESUME_RESULT"]
    assert sealed["step"] == 1
    checks += 1
    original_marker = (checkpoint / "resume_state.json").read_bytes()
    invalid = json.loads(original_marker)
    invalid["files"]["../outside"] = {"size": 1, "sha256": "0" * 64}
    (checkpoint / "resume_state.json").write_text(json.dumps(invalid))
    try:
        runpy.run_path(str(script), init_globals={"RESUME_REQUEST": {
            "operation": "inspect", "output_dir": output, "binding": binding}})
    except ValueError as error:
        assert "inventory changed" in str(error) or "unsafe" in str(error)
    else:
        raise AssertionError("Unsafe receipt path was accepted")
    checks += 1

print(f"[PASS] {checks} checkpoint recovery scenarios; sealed states, rank audits, interruption, integrity and archive safety; no model or GPU execution")
