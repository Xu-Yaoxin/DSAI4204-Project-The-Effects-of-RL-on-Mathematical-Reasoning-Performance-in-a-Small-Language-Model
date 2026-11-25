"""Rebuild or validate deterministic checkpoint selection using development data only."""
from pathlib import Path
import hashlib
import json
import runpy

root = Path(__file__).resolve().parents[1]
cfg = globals().get("SELECTION_CONFIG")
if cfg is None:
    cfg = json.loads((root / "configs/math_l4.json").read_text())
stored = None
if globals().get("SELECTION_PATH") is not None:
    selection_path = (root / Path(SELECTION_PATH)).absolute()
    if any(p.is_symlink() for p in (selection_path, *selection_path.parents)) or not selection_path.is_file():
        raise ValueError("Frozen selection is missing or unsafe.")
    stored = json.loads(selection_path.read_text())
    report_path = (root / stored["training_report_path"]).resolve()
    dev_root = (root / stored["dev_root"]).resolve()
else:
    report_path = (root / Path(TRAINING_REPORT_PATH)).resolve()
    dev_root = (root / Path(DEVELOPMENT_ROOT)).resolve()
if not report_path.is_relative_to(root / "outputs") or not dev_root.is_relative_to(root / "outputs"):
    raise ValueError("Training and development artifacts must belong to this experiment.")
if report_path.is_symlink() or not report_path.is_file():
    raise ValueError("Completed formal training report is missing.")
report_bytes = report_path.read_bytes()
report = json.loads(report_bytes)
if (report.get("mode") != "formal" or report.get("status") != "completed"
        or report.get("gates", {}).get("all_passed") is not True
        or not report.get("gates") or not all(v is True for v in report["gates"].values())
        or report.get("model_identity") != cfg["model"]["expected_identity"]):
    raise ValueError("Checkpoint selection requires an accepted formal training run.")
training_dir = report_path.parent
training_manifest_path = training_dir / "manifest.json"
training_bytes = training_manifest_path.read_bytes()
training = json.loads(training_bytes)
if (training.get("config") != cfg or training.get("mode") != "formal"
        or training.get("fresh_adapter_from_base") is not True
        or training.get("model_identity") != report["model_identity"]
        or training.get("config_training_hash") != report.get("config_training_hash")):
    raise ValueError("Formal training manifest differs from its report/configuration.")
training_protocol = training["training_protocol"]
if hashlib.sha256(json.dumps(training_protocol, sort_keys=True, ensure_ascii=False).encode()).hexdigest() != report["config_training_hash"]:
    raise ValueError("Training protocol hash is invalid.")
for field, filename in (("verifier_sha256", "verify_answers.py"), ("training_script_sha256", "train_grpo.py"),
        ("loader_sha256", "load_model.py"), ("fingerprint_sha256", "fingerprint_model.py"),
        ("doctor_sha256", "doctor.py"), ("resume_helper_sha256", "training_resume.py"),
        ("checkpoint_event_sha256", "training_checkpoint_event.py")):
    if training_protocol.get(field) != hashlib.sha256((root / "scripts" / filename).read_bytes()).hexdigest():
        raise ValueError(f"Training protocol source changed: {filename}")
for relative, expected in report.get("completion_files", {}).items():
    path = training_dir / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink() or not path.is_file():
        raise ValueError("Unsafe or missing training completion artifact.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Training completion artifact changed: {relative}")
if report.get("completion_files", {}).get("manifest.json") != hashlib.sha256(training_bytes).hexdigest():
    raise ValueError("Training completion report does not bind its manifest.")
data_dir = root / cfg["data"]["output_dir"]
prepared_hash = hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest()
if report.get("prepared_manifest_sha256") != prepared_hash:
    raise ValueError("Training provenance changed.")
for split in ("engineering", "pilot", "train", "dev", "test"):
    if report.get("split_sha256", {}).get(split) != hashlib.sha256((data_dir / (split + ".jsonl")).read_bytes()).hexdigest():
        raise ValueError("Training and evaluation data differ.")
responses_per_update = cfg["training"]["per_device_train_batch_size"] * cfg["training"]["gradient_accumulation_steps"]
questions_per_update = responses_per_update // cfg["training"]["num_generations"]
steps_per_epoch = cfg["data"]["train_size"] // questions_per_update
expected_steps = [steps_per_epoch * epoch for epoch in range(1, cfg["training"]["epochs"] + 1)]
checkpoints = report.get("checkpoints", [])
if (sorted(c.get("step") for c in checkpoints) != expected_steps
        or report.get("optimizer_steps") != expected_steps[-1]):
    raise ValueError("All predeclared epoch candidates are required; no test-dependent candidate omission.")
candidates = []
common_protocol = None
for item in sorted(checkpoints, key=lambda c: c["step"]):
    step = item["step"]
    adapter = (root / item["path"]).resolve()
    if adapter != training_dir / f"checkpoint-{step}":
        raise ValueError("Epoch candidate must belong to the accepted training run.")
    receipt_path = adapter / "resume_state.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("Epoch checkpoint is not sealed.")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("step") != step or receipt.get("binding") != training.get("resume_binding")
            or receipt.get("checkpoint_name") != adapter.name):
        raise ValueError("Epoch checkpoint seal differs from its training binding.")
    evaluated = runpy.run_path(str(root / "scripts/validate_evaluation.py"), init_globals={
        "EVALUATION_PATH": dev_root / adapter.name, "VALIDATE_CONFIG": cfg})["EVALUATION_RESULT"]
    manifest, rows = evaluated["manifest"], evaluated["rows"]
    if (manifest["model_label"] != "rl" or manifest["split"] != "dev"
            or manifest["n_samples"] != cfg["evaluation"]["dev_samples"]
            or (root / manifest["adapter_path"]).resolve() != adapter
            or manifest["settings"]["package_versions"] != report.get("package_versions")):
        raise ValueError("Selection accepts only the fixed development protocol for each epoch adapter.")
    if common_protocol is None:
        common_protocol = manifest["protocol_id"]
    elif common_protocol != manifest["protocol_id"]:
        raise ValueError("Candidate development protocols differ.")
    for filename, expected in manifest["adapter_files"].items():
        if receipt.get("files", {}).get(filename, {}).get("sha256") != expected:
            raise ValueError("Evaluated adapter differs from its sealed epoch checkpoint.")
    candidates.append({"step": step, "adapter_path": str(adapter.relative_to(root)),
        "adapter_identity": manifest["adapter_identity"], "adapter_files": manifest["adapter_files"],
        "checkpoint_seal_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "dev_output": str((dev_root / adapter.name).relative_to(root)),
        "dev_manifest_sha256": evaluated["manifest_sha256"], "dev_samples_sha256": evaluated["samples_sha256"],
        "correct_count": sum(row["correct"] is True for row in rows), "requested_count": len(rows),
        "timeout_count": sum(row["termination_reason"] == "answer_timeout" for row in rows),
        "unknown_count": sum(row["correct"] is None for row in rows),
        "protocol_id": manifest["protocol_id"]})
selected = min(candidates, key=lambda c: (-c["correct_count"] / c["requested_count"], c["step"]))
selection = {"schema_version": 1, "complete": True, "model_identity": cfg["model"]["expected_identity"],
    "training_report_path": str(report_path.relative_to(root)), "training_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
    "training_manifest_sha256": hashlib.sha256(training_bytes).hexdigest(),
    "prepared_manifest_sha256": prepared_hash, "training_protocol_id": report["config_training_hash"],
    "config_identity": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
    "selector_sha256": hashlib.sha256((root / "scripts/select_checkpoint.py").read_bytes()).hexdigest(),
    "selection_validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "dev_root": str(dev_root.relative_to(root)), "selection_split": "dev",
    "selection_rule": "maximum completed_correct/requested; ties choose smallest optimizer step",
    "candidates": candidates, "selected_step": selected["step"],
    "adapter_path": selected["adapter_path"], "adapter_identity": selected["adapter_identity"],
    "adapter_files": selected["adapter_files"], "test_outcomes_used": False}
selection["selection_id"] = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
if stored is not None and stored != selection:
    raise ValueError("Frozen development selection or any bound artifact changed.")
SELECTION_RESULT = selection
