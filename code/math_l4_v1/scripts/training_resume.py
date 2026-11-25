"""Seal and restore training checkpoints without importing model packages."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

if "RESUME_REQUEST" not in globals():
    raise SystemExit("Internal script: provide RESUME_REQUEST through runpy.")
request = RESUME_REQUEST
operation = request["operation"]
if operation not in {"inspect", "seal", "restore"}:
    raise ValueError("Unknown checkpoint recovery operation.")
if operation == "seal" and not isinstance(request.get("extra_state", {}), dict):
    raise ValueError("Checkpoint extra state must be an object.")
raw_output = Path(request["output_dir"]).expanduser().absolute()
if raw_output.is_symlink():
    raise ValueError("Training output must not be a symlink.")
output = raw_output.resolve()
binding = request["binding"]
if not isinstance(binding, dict):
    raise ValueError("Checkpoint binding must be an object.")
json.dumps(binding, sort_keys=True, allow_nan=False)
topology = binding["topology"]
world_size = topology["world_size"]
groups_per_step = topology["per_rank_unique_prompts_per_update"]
expected_steps = binding["expected_optimizer_steps"]
if (type(world_size) is not int or world_size not in {1, 2, 4}
        or type(groups_per_step) is not int or groups_per_step < 1
        or type(expected_steps) is not int or expected_steps < 1):
    raise ValueError("Invalid recovery topology or optimizer-step budget.")
required_files = ["adapter_model.safetensors", "adapter_config.json", "trainer_state.json", "optimizer.pt", "scheduler.pt"]
required_files += ([f"rng_state_{rank}.pth" for rank in range(world_size)]
                   if world_size > 1 else ["rng_state.pth"])
checkpoints = []
if output.exists():
    if not output.is_dir():
        raise ValueError("Training output must be a directory.")
    for candidate in output.glob("checkpoint-*"):
        suffix = candidate.name.removeprefix("checkpoint-")
        if (not suffix.isdigit() or str(int(suffix)) != suffix or int(suffix) < 1
                or candidate.is_symlink() or not candidate.is_dir()):
            raise ValueError(f"Unsafe checkpoint entry: {candidate}")
        if (candidate / "resume_state.json").is_symlink():
            raise ValueError("Checkpoint commit marker must not be a symlink.")
        checkpoints.append((int(suffix), candidate))
elif operation != "inspect":
    raise FileNotFoundError(f"Training output does not exist: {output}")

RESUME_RESULT = {"checkpoint": None, "step": 0, "receipt": None, "recovery_dir": None}
checkpoint = None
if operation == "seal":
    step = request["step"]
    if type(step) is not int or not 1 <= step <= expected_steps:
        raise ValueError("Checkpoint step is outside the training budget.")
    checkpoint = output / f"checkpoint-{step}"
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is missing or unsafe: {checkpoint}")
else:
    sealed = [(step, path) for step, path in checkpoints if (path / "resume_state.json").is_file()]
    if sealed:
        step, checkpoint = max(sealed, key=lambda item: item[0])
    elif operation == "restore":
        raise FileNotFoundError("No committed checkpoint is available for recovery.")

if checkpoint is not None:
    if not 1 <= step <= expected_steps:
        raise ValueError("Committed checkpoint step is outside the training budget.")
    marker = checkpoint / "resume_state.json"
    for path in checkpoint.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Unsafe checkpoint content: {path}")
    missing = [name for name in required_files if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0]
    if missing:
        raise ValueError(f"Checkpoint is incomplete: {missing}")
    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if (type(trainer_state.get("global_step")) is not int or trainer_state["global_step"] != step
            or type(trainer_state.get("max_steps")) is not int or trainer_state["max_steps"] != expected_steps):
        raise ValueError("Trainer state disagrees with the checkpoint step or training budget.")

    if operation == "seal" and not marker.exists():
        audits = {}
        for rank in range(world_size):
            source = output / f"reward_audit.rank-{rank}.jsonl"
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"Missing or unsafe rank audit: {source}")
            snapshot = checkpoint / f"resume_audit.rank-{rank}.jsonl"
            pending = snapshot.with_suffix(".jsonl.pending")
            if pending.is_symlink():
                raise ValueError("Audit snapshot temporary path must not be a symlink.")
            digest = hashlib.sha256()
            snapshot_bytes = 0
            previous_questions = set()
            with source.open("rb") as original, pending.open("wb") as saved:
                for ordinal in range(step * groups_per_step):
                    line = original.readline()
                    if not line.endswith(b"\n"):
                        raise ValueError("Rank audit is incomplete at the checkpoint boundary.")
                    group = json.loads(line)
                    group_step = ordinal // groups_per_step
                    if ordinal % groups_per_step == 0:
                        previous_questions = set()
                    if (type(group.get("rank")) is not int or group["rank"] != rank
                            or type(group.get("optimizer_step_before_update")) is not int
                            or group["optimizer_step_before_update"] != group_step
                            or not isinstance(group.get("question_id"), str) or not group["question_id"]
                            or group["question_id"] in previous_questions):
                        raise ValueError("Rank audit does not contain complete ordered optimizer-step groups.")
                    previous_questions.add(group["question_id"])
                    saved.write(line)
                    digest.update(line)
                    snapshot_bytes += len(line)
                saved.flush()
                os.fsync(saved.fileno())
            pending.replace(snapshot)
            audits[str(rank)] = {"snapshot": snapshot.name, "target": source.name,
                                 "groups": step * groups_per_step, "bytes": snapshot_bytes,
                                 "sha256": digest.hexdigest()}
        file_records = {}
        for path in sorted(checkpoint.rglob("*")):
            if path.is_file() and path.name not in {"resume_state.json", "resume_state.json.pending"}:
                if path.name.endswith(".pending"):
                    raise ValueError("An unfinished checkpoint snapshot remains; refusing to commit.")
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                file_records[path.relative_to(checkpoint).as_posix()] = {"size": path.stat().st_size, "sha256": digest.hexdigest()}
        receipt = {"schema_version": 1, "step": step, "checkpoint_name": checkpoint.name,
                   "binding": binding, "extra_state": request.get("extra_state", {}),
                   "files": file_records, "audits": audits,
                   "sealed_utc": datetime.now(timezone.utc).isoformat()}
        pending_marker = checkpoint / "resume_state.json.pending"
        if pending_marker.is_symlink():
            raise ValueError("Checkpoint commit temporary path must not be a symlink.")
        with pending_marker.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        pending_marker.replace(marker)
    else:
        receipt = json.loads(marker.read_text(encoding="utf-8"))

    if (type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1 or type(receipt.get("step")) is not int
            or receipt["step"] != step or receipt.get("checkpoint_name") != checkpoint.name
            or receipt.get("binding") != binding or not isinstance(receipt.get("extra_state"), dict)):
        raise ValueError("Checkpoint identity, protocol or resume binding changed.")
    file_records = receipt.get("files")
    if not isinstance(file_records, dict) or not set(required_files) <= set(file_records):
        raise ValueError("Checkpoint receipt lacks required state-file hashes.")
    observed_files = {path.relative_to(checkpoint).as_posix() for path in checkpoint.rglob("*")
                      if path.is_file() and path != marker}
    if observed_files != set(file_records):
        raise ValueError("Checkpoint file inventory changed after commit.")
    for relative, record in file_records.items():
        relative_path = Path(relative)
        if (relative_path.is_absolute() or ".." in relative_path.parts
                or relative_path.as_posix() != relative or relative in {"", "."}):
            raise ValueError("Checkpoint receipt contains an unsafe relative path.")
        path = checkpoint / relative_path
        if (not isinstance(record, dict) or type(record.get("size")) is not int
                or record["size"] < 0 or path.is_symlink() or not path.is_file()
                or path.stat().st_size != record["size"]):
            raise ValueError(f"Checkpoint file size or path changed: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != record.get("sha256"):
            raise ValueError(f"Checkpoint file hash changed: {relative}")
    audits = receipt.get("audits")
    if not isinstance(audits, dict) or set(audits) != {str(rank) for rank in range(world_size)}:
        raise ValueError("Checkpoint must contain a snapshot for every rank.")
    for rank in range(world_size):
        audit = audits[str(rank)]
        snapshot_name = f"resume_audit.rank-{rank}.jsonl"
        target_name = f"reward_audit.rank-{rank}.jsonl"
        if (not isinstance(audit, dict) or audit.get("snapshot") != snapshot_name or audit.get("target") != target_name
                or audit.get("groups") != step * groups_per_step or type(audit.get("groups")) is not int
                or snapshot_name not in file_records or audit.get("bytes") != file_records[snapshot_name]["size"]
                or audit.get("sha256") != file_records[snapshot_name]["sha256"]):
            raise ValueError("Checkpoint audit snapshot metadata changed.")
    RESUME_RESULT.update(checkpoint=str(checkpoint), step=step, receipt=receipt)

    if operation == "restore":
        archive_targets = []
        replacements = []
        for rank in range(world_size):
            audit = audits[str(rank)]
            snapshot = checkpoint / audit["snapshot"]
            target = output / audit["target"]
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError("Current rank audit must be a regular file.")
            if target.exists():
                shared_bytes = min(target.stat().st_size, audit["bytes"])
                remaining = shared_bytes
                with target.open("rb") as current, snapshot.open("rb") as saved:
                    while remaining:
                        size = min(1024 * 1024, remaining)
                        if current.read(size) != saved.read(size):
                            raise ValueError("Current rank audit diverges before the committed checkpoint boundary.")
                        remaining -= size
                if target.stat().st_size == audit["bytes"]:
                    continue
                archive_targets.append(target)
            replacements.append((snapshot, target))
        for candidate_step, candidate in checkpoints:
            if not (candidate / "resume_state.json").is_file():
                archive_targets.append(candidate)
        for name in ("adapter", "reward_audit.jsonl", "trainer_logs.json", "trainer_state.json", "module_audit.json"):
            path = output / name
            if path.exists() or path.is_symlink():
                archive_targets.append(path)
        archive_targets += sorted(output.glob("failure.rank-*.json"))
        for path in archive_targets:
            if path.is_symlink() or (path.is_dir() and any(child.is_symlink() for child in path.rglob("*"))):
                raise ValueError("Recovery refuses symlinked artifacts; no files were changed.")
        recovery_root = output / "recoveries"
        if recovery_root.is_symlink() or (recovery_root.exists() and not recovery_root.is_dir()):
            raise ValueError("Recovery archive must be a local directory.")
        for _, target in replacements:
            if target.with_suffix(".jsonl.restore-pending").is_symlink():
                raise ValueError("Recovery temporary path must not be a symlink.")
        if archive_targets:
            recovery_root.mkdir(exist_ok=True)
            recovery_dir = Path(tempfile.mkdtemp(prefix=f"before-step-{step}-", dir=recovery_root))
            for path in archive_targets:
                path.rename(recovery_dir / path.name)
            (recovery_dir / "recovery.json").write_text(json.dumps({
                "restored_step": step, "checkpoint": checkpoint.name,
                "archived": [path.name for path in archive_targets],
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=2) + "\n", encoding="utf-8")
            RESUME_RESULT["recovery_dir"] = str(recovery_dir)
        for snapshot, target in replacements:
            pending = target.with_suffix(".jsonl.restore-pending")
            with snapshot.open("rb") as source, pending.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            pending.replace(target)
