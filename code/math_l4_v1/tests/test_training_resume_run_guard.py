"""Execute the production training run guard with real sealed CPU-only file fixtures."""
import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import runpy
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
source = root / "scripts/train_grpo.py"
tree = ast.parse(source.read_text())
main = next(node for node in tree.body if isinstance(node, ast.If) and "REWARD_REQUEST" in ast.unparse(node.test)).orelse
first = next(index for index, node in enumerate(main) if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "resume_packet" for target in node.targets))
last = next(index for index, node in enumerate(main[first:], first) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "started" for target in node.targets))
guard = compile(ast.Module(body=main[first:last], type_ignores=[]), str(source), "exec")
original_run_path = runpy.run_path
scenarios = (
    "fresh", "fresh-resume", "existing-without-resume", "changed-protocol", "changed-data",
    "changed-topology", "old-schema", "before-first-checkpoint", "incomplete-with-checkpoint",
    "completed", "changed-completed-gate", "changed-completed-step", "changed-completed-artifact",
    "changed-package", "changed-model", "changed-checkpoint", "active-run-lock",
)
checks = 0
for scenario in scenarios:
    with tempfile.TemporaryDirectory(prefix="rlvr-training-run-guard-") as temporary:
        parent = Path(temporary).resolve()
        output = parent / "formal"
        topology = {"world_size": 1, "per_rank_unique_prompts_per_update": 4}
        split_hashes = {"train": "a" * 64, "dev": "b" * 64, "test": "c" * 64}
        identity = {"sha256": "d" * 64}
        protocol_hash = "e" * 64
        binding = {"mode": "formal", "config_training_hash": protocol_hash,
                   "expected_optimizer_steps": 4, "topology": topology, "model_identity": identity}
        report = {"mode": "formal", "config_training_hash": protocol_hash, "split_sha256": split_hashes,
                  "prepared_manifest_sha256": "f" * 64, "topology": topology, "resume_schema_version": 1,
                  "status": "completed", "gates": {"all_passed": True, "training_completed": True},
                  "optimizer_steps": 4, "model_identity": identity, "package_versions": {"torch": "fixture-locked"}}
        manifest = {"config_training_hash": protocol_hash, "resume_binding": binding}
        is_fresh = scenario in {"fresh", "fresh-resume"}
        if not is_fresh:
            output.mkdir()
            (output / "preserved.txt").write_text("Existing run evidence must not be discarded.\n")
            if scenario == "before-first-checkpoint":
                report.update(status="running", gates={"all_passed": False})
            else:
                checkpoint_step = 2 if scenario == "incomplete-with-checkpoint" else 4
                checkpoint = output / f"checkpoint-{checkpoint_step}"
                checkpoint.mkdir()
                for name in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                    (checkpoint / name).write_bytes(b"trusted tiny fixture; no model or pickle load\n")
                (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": checkpoint_step, "max_steps": 4}))
                audit = "".join(json.dumps({"rank": 0, "optimizer_step_before_update": step,
                    "question_id": f"q-{step}-{index}"}) + "\n"
                    for step in range(checkpoint_step) for index in range(4))
                (output / "reward_audit.rank-0.jsonl").write_text(audit)
                original_run_path(str(root / "scripts/training_resume.py"), init_globals={"RESUME_REQUEST": {
                    "operation": "seal", "output_dir": output, "binding": binding, "step": checkpoint_step,
                    "extra_state": {"grpo_step": checkpoint_step * 32, "optimizer_step": checkpoint_step}}})
                (output / "manifest.json").write_text(json.dumps(manifest))
                report["completion_files"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                    for name in ("manifest.json", "preserved.txt", "reward_audit.rank-0.jsonl")}
                if scenario == "incomplete-with-checkpoint":
                    report.update(status="failed", gates={"all_passed": False}, optimizer_steps=2)
            if scenario == "changed-protocol":
                report["config_training_hash"] = "0" * 64
            elif scenario == "changed-data":
                report["split_sha256"] = {**split_hashes, "train": "0" * 64}
            elif scenario == "changed-topology":
                report["topology"] = {**topology, "world_size": 2}
            elif scenario == "old-schema":
                report.pop("resume_schema_version")
            elif scenario == "changed-completed-gate":
                report["gates"]["training_completed"] = False
            elif scenario == "changed-completed-step":
                report["optimizer_steps"] = 3
            elif scenario == "changed-completed-artifact":
                (output / "preserved.txt").write_text("Changed completed artifact.\n")
            elif scenario == "changed-checkpoint":
                (checkpoint / "optimizer.pt").write_bytes(b"damaged checkpoint state")
            (output / "report.json").write_text(json.dumps(report))
        before = ({path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in output.rglob("*") if path.is_file()} if output.exists() else {})
        held_lock = None
        if scenario == "active-run-lock":
            held_lock = (parent / ".formal.lock").open("a")
            fcntl.flock(held_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        namespace = {"Path": Path, "json": json, "hashlib": hashlib, "fcntl": fcntl, "shutil": shutil,
            "datetime": datetime, "timezone": timezone, "importlib": importlib, "runpy": runpy,
            "CODE_ROOT": root, "resume_script": root / "scripts/training_resume.py", "output_dir": output,
            "rank": 0, "distributed": False, "torch": SimpleNamespace(),
            "args": SimpleNamespace(mode="formal", resume=scenario not in {"fresh", "existing-without-resume"}),
            "config": {"model": {"path": "models/Qwen3.5-2B-Base"}}, "topology": topology,
            "config_training_hash": protocol_hash, "split_hashes": split_hashes, "expected_optimizer_steps": 4,
            "data_readiness": {"data": {"prepared_manifest_sha256": "f" * 64}}}
        observed_identity = {"sha256": "0" * 64} if scenario == "changed-model" else identity
        observed_version = "changed-version" if scenario == "changed-package" else "fixture-locked"
        captured = io.StringIO()
        failure = None
        try:
            with patch.object(runpy, "run_path", side_effect=lambda path, init_globals=None:
                    {"model_identity": observed_identity} if Path(path).name == "fingerprint_model.py"
                    else original_run_path(path, init_globals=init_globals)), \
                    patch.object(importlib.metadata, "version", return_value=observed_version), redirect_stdout(captured):
                exec(guard, namespace)
        except (RuntimeError, SystemExit) as error:
            failure = error
        finally:
            if "run_lock" in namespace:
                namespace["run_lock"].close()
            if held_lock is not None:
                held_lock.close()
        after = ({path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in output.rglob("*") if path.is_file()} if output.exists() else {})
        if is_fresh:
            assert failure is None and output.is_dir() and after == {}, (scenario, failure)
            assert namespace["recovery"]["checkpoint"] is None and namespace["recovery"]["step"] == 0
        elif scenario == "before-first-checkpoint":
            assert failure is None and output.is_dir() and after == {}, failure
            archived = Path(namespace["recovery"]["restart_archive"])
            archived_files = {path.relative_to(archived).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in archived.rglob("*") if path.is_file()}
            assert archived_files == before and "restarting from step 0" in captured.getvalue()
        elif scenario == "incomplete-with-checkpoint":
            assert failure is None, failure
            assert namespace["recovery"]["step"] == 2 and namespace["recovery"]["checkpoint"] == str(checkpoint.resolve())
            assert all(after.get(name) == digest for name, digest in before.items())
            archives = list((output / "recoveries").glob("attempt-*"))
            assert len(archives) == 1
            assert (archives[0] / "report.json").read_bytes() == (output / "report.json").read_bytes()
            assert (archives[0] / "manifest.json").read_bytes() == (output / "manifest.json").read_bytes()
        elif scenario == "completed":
            assert isinstance(failure, SystemExit) and failure.code == 0, failure
            assert "[TRAIN CACHED]" in captured.getvalue() and after == before
        else:
            assert isinstance(failure, RuntimeError), (scenario, failure)
            assert after == before, f"Rejected {scenario} mutated training artifacts"
            assert not list(parent.glob("formal.before-first-checkpoint-*")), scenario
        checks += 1
print(f"[PASS] {checks} production run-guard scenarios | fresh, lock, completed reuse, archive and rejected changes | no GPU/model execution")
