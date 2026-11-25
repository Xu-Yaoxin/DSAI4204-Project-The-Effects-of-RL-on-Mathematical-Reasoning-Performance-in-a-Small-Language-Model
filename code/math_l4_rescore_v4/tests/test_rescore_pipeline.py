"""Exercise posthoc rescoring with real scorers and complete temporary CPU fixtures."""
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

package = Path(__file__).resolve().parents[1]
frozen = package.parent / "math_l4_v1"
score_fields = ("correct", "reward", "extracted", "status", "format_compliant", "truncated")
checks = []
with tempfile.TemporaryDirectory(prefix="math-rescore-fixture-") as temporary:
    workspace = Path(temporary).resolve()
    root = workspace / "experiment"
    run_relative = "outputs/math_l4_v1_fix1"
    run_root = root / run_relative
    for name in ("scripts", "configs", "data/processed", run_relative + "/formal", run_relative + "/dev"):
        (root / name).mkdir(parents=True)
    for path in (frozen / "scripts").glob("*.py"):
        shutil.copy2(path, root / "scripts" / path.name)
    cfg = json.loads((frozen / "configs/math_l4.json").read_text())
    model_files = {"fixture.config": "a" * 64, "fixture.weights": "b" * 64}
    cfg["model"]["expected_identity"] = hashlib.sha256(json.dumps(model_files, sort_keys=True).encode()).hexdigest()
    cfg["training"]["epochs"] = 2
    cfg["data"].update(train_size=8, pilot_size=4, engineering_size=4, dev_size=2, test_size=2)
    cfg["evaluation"].update(dev_samples=4, n_samples=16, bootstrap_replicates=100)
    (root / "configs/math_l4.json").write_text(json.dumps(cfg))
    datasets = {}
    for split, size in (("train", 8), ("dev", 2), ("test", 2)):
        datasets[split] = [{"id": f"{split}-{index}", "question": f"Compute fixture {split} {index}.",
            "answer": str(731 + index), "source_answer": str(731 + index), "split": split,
            "family_id": f"{split}-family-{index}", "group_id": "math_l4", "group_label": "MATH Level4",
            "difficulty_label": "Level 4", "source": {"subject": "algebra", "split": "test" if split == "test" else "train",
            "dataset": "fixture", "revision": "pinned"}} for index in range(size)]
    for split in ("pilot", "engineering"):
        datasets[split] = [dict(row, split=split) for row in datasets["train"][:4]]
    data_manifest = {"schema_version": 1, "kind": "math_level4_splits", "purpose": "formal_base_rl_comparison", "files": {}}
    for split, rows in datasets.items():
        content = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (root / f"data/processed/{split}.jsonl").write_bytes(content)
        data_manifest["files"][split + ".jsonl"] = {"sha256": hashlib.sha256(content).hexdigest(), "rows": len(rows)}
    (root / "data/processed/manifest.json").write_text(json.dumps(data_manifest))
    data_manifest_hash = hashlib.sha256((root / "data/processed/manifest.json").read_bytes()).hexdigest()
    settings = {key: cfg["evaluation"][key] for key in ("temperature", "top_p", "top_k", "max_new_tokens", "max_prompt_length",
        "seed", "context_limit", "length_policy", "generation_guard")}
    settings.update(prompt_template=cfg["prompt"]["template"], dtype="bfloat16", backend="transformers-sdpa", batch_size=1,
        world_size=1, add_special_tokens=True, stopping_rule="tokenizer-eos-or-native-context-or-guard", repetition_penalty=1.0,
        eos_token_id=cfg["model"]["expected_eos_token_id"], package_versions={"torch": "fixture"})
    code_hashes = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() for name in (
        "evaluate.py", "verify_answers.py", "load_model.py", "fingerprint_model.py", "repetition_guard.py",
        "validate_guarded_sample.py", "doctor.py", "validate_evaluation.py", "validate_selection.py")}
    training_protocol = {"training": cfg["training"]}
    for field, name in (("verifier_sha256", "verify_answers.py"), ("training_script_sha256", "train_grpo.py"),
        ("loader_sha256", "load_model.py"), ("fingerprint_sha256", "fingerprint_model.py"), ("doctor_sha256", "doctor.py"),
        ("resume_helper_sha256", "training_resume.py"), ("checkpoint_event_sha256", "training_checkpoint_event.py")):
        training_protocol[field] = hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest()
    training_protocol_hash = hashlib.sha256(json.dumps(training_protocol, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    training_manifest = {"config": cfg, "mode": "formal", "fresh_adapter_from_base": True,
        "model_identity": cfg["model"]["expected_identity"], "config_training_hash": training_protocol_hash,
        "training_protocol": training_protocol, "resume_binding": {"fixture": "sealed"}}
    (run_root / "formal/manifest.json").write_text(json.dumps(training_manifest))
    report = {"mode": "formal", "status": "completed", "gates": {"all_passed": True, "adapter_updated": True},
        "model_identity": cfg["model"]["expected_identity"], "config_training_hash": training_protocol_hash,
        "prepared_manifest_sha256": data_manifest_hash, "package_versions": settings["package_versions"],
        "split_sha256": {split: data_manifest["files"][split + ".jsonl"]["sha256"] for split in datasets},
        "checkpoints": [{"step": step, "path": f"{run_relative}/formal/checkpoint-{step}"} for step in (2, 4)],
        "optimizer_steps": 4, "completion_files": {"manifest.json": hashlib.sha256((run_root / "formal/manifest.json").read_bytes()).hexdigest()}}
    (run_root / "formal/report.json").write_text(json.dumps(report))
    adapters = {}
    for step in (2, 4):
        path = run_root / f"formal/checkpoint-{step}"
        path.mkdir()
        (path / "adapter_config.json").write_text("{}")
        (path / "adapter_model.safetensors").write_bytes(f"fixture adapter {step}".encode())
        files = {}
        digest = hashlib.sha256()
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            content = (path / name).read_bytes()
            files[name] = hashlib.sha256(content).hexdigest()
            digest.update(name.encode())
            digest.update(content)
        adapters[step] = {"path": str(path.relative_to(root)), "identity": digest.hexdigest(), "files": files}
        receipt = {"step": step, "binding": training_manifest["resume_binding"], "checkpoint_name": path.name,
            "files": {name: {"sha256": digest} for name, digest in files.items()}}
        (path / "resume_state.json").write_text(json.dumps(receipt))
    interpreter = [sys.executable, "-S"]
    selection = None
    originals = {}
    manifests = {}
    for relative, label, split, count, step in (("dev/checkpoint-2", "rl", "dev", 4, 2),
        ("dev/checkpoint-4", "rl", "dev", 4, 4), ("test_base", "base", "test", 16, 2),
        ("test_rl_split_merged", "rl", "test", 16, 2)):
        if split == "test" and selection is None:
            selected = subprocess.run(interpreter + [str(root / "scripts/select_checkpoint.py"), "--training-report",
                run_relative + "/formal/report.json", "--dev-root", run_relative + "/dev", "--output", run_relative + "/selection.json"],
                capture_output=True, text=True, timeout=30)
            assert selected.returncode == 0, selected.stderr
            selection = json.loads((run_root / "selection.json").read_text())
            assert selection["selected_step"] == 2 and selection["test_outcomes_used"] is False
        destination = run_root / relative
        destination.mkdir(parents=True)
        rows = []
        for question in datasets[split]:
            for index in range(count):
                answer = question["answer"]
                text = f"#### {answer}"
                token_ids = [int(answer), settings["eos_token_id"]]
                reason, seconds, evidence, timing = "eos", 1.25, None, None
                if split == "test" and index == 0:
                    token_ids, reason, seconds = [7] * 1024, "repetition_guard", 40.5
                    evidence = runpy.run_path(str(root / "scripts/repetition_guard.py"), init_globals={"TOKEN_IDS": token_ids,
                        "GUARD_POLICY": settings["generation_guard"]})["GUARD_RESULT"]["evidence"]
                    score = {"correct": None, "reward": None, "extracted": None, "status": "unscored", "format_compliant": None, "truncated": False}
                elif split == "test" and index == 1:
                    token_ids, reason, seconds = [int(answer)], "answer_timeout", 300.2
                    timing = {"timeout_seconds": 300, "elapsed_seconds": 300.1, "last_token_age_seconds": 3.5, "observed_tokens_at_timeout": 1}
                    score = {"correct": False, "reward": 0.0, "extracted": None, "status": "timeout", "format_compliant": None, "truncated": False}
                else:
                    if split == "test" and index == 2:
                        text = ("#### Term 1: $2+3$\nIntermediate calculation.\n#### 731" if answer == "731" else
                            "Final Answer:\nThe maximum area of the triangle is 732 square units.\n#### 732")
                    elif split == "test" and index == 4:
                        text = "#### Term 1: $2+3$\nIntermediate calculation.\n#### 0"
                    score = runpy.run_path(str(root / "scripts/verify_answers.py"), init_globals={"SCORE_INPUT": [
                        {"text": text, "answer": answer, "truncated": False}]})["SCORE_OUTPUT"][0]
                rows.append({**question, **score, "sample_index": index,
                    "sample_seed": int.from_bytes(hashlib.sha256(f"{settings['seed']}:{question['id']}:{index}".encode()).digest()[:4], "big"),
                    "text": text, "raw_text": "Raw transcript:\n" + text, "output_token_ids": token_ids, "output_tokens": len(token_ids),
                    "prompt_tokens": 5, "generation_limit": settings["context_limit"] - 5, "seconds": seconds,
                    "termination_reason": reason, "scoring_evaluable": reason == "eos", "guard_evidence": evidence, "timeout_evidence": timing,
                    "fixture_extra": {"generation_worker": "fixture", "keep_verbatim": [False, None, "untouched"]}})
        if split == "test" and label == "rl":
            rows.reverse()
        content = "".join(json.dumps(row) + "\n" for row in rows).encode()
        (destination / "samples.jsonl").write_bytes(content)
        manifest = {"schema_version": 1, "experiment_type": "math_l4_base_rl", "synthetic": False, "model_label": label,
            "split": split, "path_base": "code_root", "complete": True,
            "adapter_path": adapters[step]["path"] if label == "rl" else None,
            "adapter_identity": adapters[step]["identity"] if label == "rl" else None,
            "adapter_files": adapters[step]["files"] if label == "rl" else None,
            "selection_identity": selection["selection_id"] if label == "rl" and split == "test" else None,
            "selection_path": run_relative + "/selection.json" if label == "rl" and split == "test" else None,
            "dataset_path": f"data/processed/{split}.jsonl", "dataset_sha256": data_manifest["files"][split + ".jsonl"]["sha256"],
            "dataset_manifest_path": "data/processed/manifest.json", "dataset_manifest_sha256": data_manifest_hash,
            "n_samples": count, "question_count": 2, "total_samples": len(rows), "config": cfg, "settings": settings,
            "code_hashes": code_hashes, "model_identity": cfg["model"]["expected_identity"], "model_files": model_files,
            "samples_sha256": hashlib.sha256(content).hexdigest()}
        protocol = {key: manifest[key] for key in ("dataset_sha256", "dataset_manifest_sha256", "n_samples", "settings", "code_hashes", "experiment_type")}
        manifest["protocol_id"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        (destination / "manifest.json").write_text(json.dumps(manifest))
        if split == "test":
            originals[label] = rows
            manifests[label] = manifest
    source_before = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    output = workspace / "rescore"
    command = interpreter + [str(package / "rescore.py"), "--experiment-root", str(root), "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, "Real posthoc subprocess failed:\n" + result.stdout + result.stderr
    checks.append("complete_real_subprocess")
    rescored = {}
    for label in ("base", "rl"):
        rows = [json.loads(line) for line in (output / label / "samples.jsonl").read_text().splitlines()]
        original_by_key = {(row["id"], row["sample_index"]): row for row in originals[label]}
        rescored[label] = {(row["id"], row["sample_index"]): row for row in rows}
        assert len(rows) == 32 and set(rescored[label]) == set(original_by_key)
        for key, row in rescored[label].items():
            original = original_by_key[key]
            assert row["old_score"] == {field: original[field] for field in score_fields}
            assert {field: value for field, value in row.items() if field not in (*score_fields, "old_score", "rescore")} == {
                field: value for field, value in original.items() if field not in score_fields}, (label, key, "generation changed")
            assert row["rescore"]["scorer_version"] == "4.0.0" and row["rescore"]["method"] == "posthoc"
            if key[1] == 2:
                assert original["status"] == "extraction_failed" and original["correct"] is False
                assert row["status"] == "correct" and row["correct"] is True and row["reward"] == 1.0
                assert row["extracted"] == original["answer"] and row["format_compliant"] is True
            elif key[1] == 4:
                assert original["status"] == "extraction_failed" and row["status"] == "incorrect"
                assert original["correct"] is False and row["correct"] is False
                assert original["reward"] == row["reward"] == 0.0
                assert original["extracted"] is None and row["extracted"] == "0" and row["format_compliant"] is True
            else:
                assert {field: row[field] for field in score_fields} == row["old_score"], (label, key)
        manifest = json.loads((output / label / "manifest.json").read_text())
        assert manifest["schema_version"] == "math-l4-posthoc-rescore-v1" and manifest["posthoc"] is True
        assert manifest["scorer_version"] == "4.0.0"
        assert manifest["generation_provenance"]["source_manifest"] == manifests[label]
        assert "protocol_id" not in manifest and "code_hashes" not in manifest
    checks.extend(["all_generation_fields_preserved", "eos_rescored", "timeout_and_repetition_not_rescored", "frozen_generation_provenance"])
    changes = [json.loads(line) for line in (output / "changes.jsonl").read_text().splitlines()]
    expected_changes = {(label, question_id, index) for label in ("base", "rl") for question_id in ("test-0", "test-1") for index in (2, 4)}
    assert len(changes) == 8 and {(row["model_label"], row["id"], row["sample_index"]) for row in changes} == expected_changes
    for change in changes:
        label = change["model_label"]
        key = (change["id"], change["sample_index"])
        new = rescored[label][key]
        assert change["old_score"] == new["old_score"]
        assert change["new_score"] == {field: new[field] for field in score_fields}
        assert set(change["changed_fields"]) == ({"correct", "reward", "extracted", "status", "format_compliant"}
            if key[1] == 2 else {"extracted", "status", "format_compliant"})
        assert change["sample_seed"] == new["sample_seed"]
        source_key = change["source_key"]
        assert source_key["model_label"] == label and source_key["id"] == key[0] and source_key["sample_index"] == key[1]
        assert source_key["sample_seed"] == new["sample_seed"]
        original_line = originals[label][source_key["line_number"] - 1]
        assert (original_line["id"], original_line["sample_index"]) == key
    checks.extend(["exact_changed_rows_audit", "audit_nonboolean_score_changes"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["question_count"] == 2 and summary["n_samples"] == 16
    for label in ("base", "rl"):
        assert summary["metrics"][label + "_pass_at_1"]["estimate"] == 26 / 32
        assert summary["metrics"][label + "_pass_at_1"]["unknown_bounds"] == [26 / 32, 28 / 32]
        assert summary["original_statistics"]["metrics"][label + "_pass_at_1"]["estimate"] == 24 / 32
        assert summary["diagnostics"][label]["correct_count"] == 26
        assert summary["diagnostics"][label]["requested_count"] == 32
        assert summary["diagnostics"][label]["termination_counts"] == {"eos": 28, "answer_timeout": 2, "repetition_guard": 2}
    assert summary["metrics"]["delta_pass_at_1"]["estimate"] == 0
    per_question = [json.loads(line) for line in (output / "per_question.jsonl").read_text().splitlines()]
    assert len(per_question) == 2 and {row["id"] for row in per_question} == {"test-0", "test-1"}
    for question in per_question:
        for label in ("base", "rl"):
            assert question[label]["correct_count"] == 13 and question["original_scores"][label]["correct_count"] == 12
            assert question[label]["unknown_count"] == 1 and question[label]["timeout_count"] == 1
    checks.append("paired_statistics_ignore_source_row_order")
    ready = json.loads((output / "ready.json").read_text())
    assert ready["complete"] is True and ready["identity"]
    for relative, expected_hash in ready["artifacts_sha256"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected_hash, relative
    saved_output = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    output_timestamps = {str(path.relative_to(output)): path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0 and "CACHED" in result.stdout, result.stdout + result.stderr
    assert saved_output == {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    assert output_timestamps == {str(path.relative_to(output)): path.stat().st_mtime_ns for path in output.rglob("*") if path.is_file()}
    checks.append("deterministic_verified_reuse")
    source_after = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    assert source_after == source_before
    checks.append("all_frozen_source_bytes_preserved")

    # Every rejected grid is rehashed so the structural check must catch the defect.
    sample_path = run_root / "test_rl_split_merged/samples.jsonl"
    manifest_path = sample_path.with_name("manifest.json")
    sample_bytes, manifest_bytes = sample_path.read_bytes(), manifest_path.read_bytes()
    for mutation in ("unbound_hash", "duplicate", "missing", "wrong_gold", "sample_seed", "eos_score", "interrupted_score",
        "selection", "config", "split", "model_label", "protocol", "adapter"):
        rows = [json.loads(line) for line in sample_bytes.decode().splitlines()]
        manifest = json.loads(manifest_bytes)
        if mutation == "unbound_hash":
            rows[0]["raw_text"] += " changed"
        elif mutation == "duplicate":
            rows[-1] = rows[0]
        elif mutation == "missing":
            rows.pop()
            manifest["total_samples"] = len(rows)
        elif mutation == "wrong_gold":
            rows[0]["answer"] = "999"
        elif mutation == "sample_seed":
            rows[0]["sample_seed"] += 1
        elif mutation == "eos_score":
            rows[0]["correct"] = False
        elif mutation == "interrupted_score":
            rows[-1]["correct"] = False
            rows[-1]["reward"] = 0.0
        elif mutation == "selection":
            manifest["selection_identity"] = "0" * 64
        elif mutation == "config":
            manifest["config"]["evaluation"]["temperature"] = 0.7
        elif mutation == "split":
            manifest["split"] = "dev"
        elif mutation == "model_label":
            manifest["model_label"] = "base"
        elif mutation == "protocol":
            manifest["protocol_id"] = "0" * 64
        else:
            manifest["adapter_path"] = run_relative + "/formal/checkpoint-4"
        content = "".join(json.dumps(row) + "\n" for row in rows).encode()
        if mutation != "unbound_hash":
            manifest["samples_sha256"] = hashlib.sha256(content).hexdigest()
        sample_path.write_bytes(content)
        manifest_path.write_text(json.dumps(manifest))
        rejected_output = workspace / ("reject-" + mutation)
        rejected_command = command[:-1] + [str(rejected_output)]
        result = subprocess.run(rejected_command, capture_output=True, text=True, timeout=90)
        assert result.returncode != 0, (mutation, result.stdout, result.stderr)
        assert not (rejected_output / "ready.json").exists(), mutation
        sample_path.write_bytes(sample_bytes)
        manifest_path.write_bytes(manifest_bytes)
        checks.append("reject_" + mutation)

    # A valid, rehashed source revision cannot reuse output from different source bytes.
    changed_rows = [json.loads(line) for line in sample_bytes.decode().splitlines()]
    changed_rows[0]["raw_text"] += " additional original-generation metadata"
    changed_content = "".join(json.dumps(row) + "\n" for row in changed_rows).encode()
    changed_manifest = json.loads(manifest_bytes)
    changed_manifest["samples_sha256"] = hashlib.sha256(changed_content).hexdigest()
    sample_path.write_bytes(changed_content)
    manifest_path.write_text(json.dumps(changed_manifest))
    validated_change = runpy.run_path(str(root / "scripts/validate_evaluation.py"), init_globals={"EVALUATION_PATH": sample_path.parent})
    assert len(validated_change["EVALUATION_RESULT"]["rows"]) == 32
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode != 0 and saved_output == {
        str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    sample_path.write_bytes(sample_bytes)
    manifest_path.write_bytes(manifest_bytes)
    checks.append("reject_reuse_changed_valid_source")

    # Pinning must reject changed validator code before executing that code.
    validator = root / "scripts/validate_evaluation.py"
    validator_bytes = validator.read_bytes()
    validator.write_text('raise RuntimeError("MUTATED VALIDATOR EXECUTED")\n')
    result = subprocess.run(command[:-1] + [str(workspace / "reject-source-code")], capture_output=True, text=True, timeout=90)
    assert result.returncode != 0 and "MUTATED VALIDATOR EXECUTED" not in result.stderr
    validator.write_bytes(validator_bytes)
    checks.append("reject_source_code_before_execution")

    # A copied CLI and scorer exercise byte-identity cache guards without editing the release.
    copied_package = workspace / "runner-copy"
    copied_package.mkdir()
    for path in package.glob("*.py"):
        shutil.copy2(path, copied_package / path.name)
    copied_output = workspace / "copied-output"
    copied_command = interpreter + [str(copied_package / "rescore.py"), "--experiment-root", str(root), "--output", str(copied_output)]
    result = subprocess.run(copied_command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    copied_artifacts = {str(path.relative_to(copied_output)): path.read_bytes() for path in copied_output.rglob("*") if path.is_file()}
    for filename in ("scorer.py", "rescore.py"):
        changed = copied_package / filename
        original_bytes = changed.read_bytes()
        changed.write_bytes(original_bytes + b"\n# Changed cache binding fixture.\n")
        result = subprocess.run(copied_command, capture_output=True, text=True, timeout=90)
        assert result.returncode != 0, (filename, result.stdout, result.stderr)
        assert copied_artifacts == {str(path.relative_to(copied_output)): path.read_bytes() for path in copied_output.rglob("*") if path.is_file()}
        changed.write_bytes(original_bytes)
        checks.append("reject_changed_" + filename)

    for mutation in ("tampered_samples", "missing_summary", "incomplete_ready"):
        if mutation == "tampered_samples":
            target = output / "base/samples.jsonl"
            target.write_bytes(saved_output["base/samples.jsonl"] + b" ")
        elif mutation == "missing_summary":
            target = output / "summary.json"
            target.unlink()
        else:
            target = output / "ready.json"
            incomplete = json.loads(saved_output["ready.json"])
            incomplete["complete"] = False
            target.write_text(json.dumps(incomplete))
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        assert result.returncode != 0, (mutation, result.stdout, result.stderr)
        target.write_bytes(saved_output[str(target.relative_to(output))])
        checks.append("reject_reuse_" + mutation)
    partial = workspace / "partial"
    partial.mkdir()
    (partial / "samples.jsonl").write_bytes(b"partial output must not be overwritten\n")
    result = subprocess.run(command[:-1] + [str(partial)], capture_output=True, text=True, timeout=90)
    assert result.returncode != 0 and not (partial / "ready.json").exists()
    assert (partial / "samples.jsonl").read_bytes() == b"partial output must not be overwritten\n"
    checks.append("reject_partial_output")
    linked_output = workspace / "linked-output"
    linked_output.symlink_to(output, target_is_directory=True)
    for name, unsafe_output in (("source", root), ("source_descendant", run_root / "test_base"),
        ("source_ancestor", workspace), ("symlink", linked_output)):
        result = subprocess.run(command[:-1] + [str(unsafe_output)], capture_output=True, text=True, timeout=90)
        assert result.returncode != 0, (name, result.stdout, result.stderr)
        checks.append("reject_output_" + name)
    assert source_before == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}
    assert saved_output == {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}

print(f"PASS [POSTHOC PIPELINE] {len(checks)} checks: {', '.join(checks)}")
