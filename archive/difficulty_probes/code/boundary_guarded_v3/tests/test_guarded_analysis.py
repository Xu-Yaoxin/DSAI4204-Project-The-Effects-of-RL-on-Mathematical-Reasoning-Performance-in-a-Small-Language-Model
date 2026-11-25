"""Exercise unknown-answer bounds, timeout policy and guarded merge integrity."""

import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
checks = 0
with tempfile.TemporaryDirectory(prefix="guarded-analysis-") as temporary:
    fixture = Path(temporary).resolve() / "code"
    (fixture / "scripts").mkdir(parents=True)
    for name in ("analyze_boundary.py", "merge_boundary.py", "verify_answers.py", "validate_guarded_sample.py", "repetition_guard.py"):
        source = root / "scripts" / name
        shutil.copyfile(source, fixture / "scripts" / name)
        tree = ast.parse(source.read_text())
        assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) for node in ast.walk(tree))
    checks += 1
    guard = json.loads((root / "configs/guarded_v3.json").read_text())["evaluation"]["generation_guard"]
    settings = {"seed": 17, "temperature": 0.6, "top_p": 0.95, "top_k": 0, "max_new_tokens": 262144,
        "max_prompt_length": 2048, "context_limit": 262144, "length_policy": "native_context_minus_actual_prompt",
        "generation_guard": guard, "eos_token_id": 248044, "prompt_template": "Solve {question}",
        "dtype": "bfloat16", "backend": "synthetic", "batch_size": 1, "world_size": 1}
    dataset = [{"id": f"q{i}", "question": f"Fixture {i}: one plus one?", "answer": "2", "source_answer": "2",
        "group_id": "mixed" if i < 4 else "all_unknown", "group_label": "Mixed" if i < 4 else "All unknown",
        "family_id": f"family-{i}", "difficulty_label": "synthetic", "source": {"dataset": "synthetic", "row_index": i}} for i in range(6)]
    groups = [{"id": "mixed", "label": "Mixed", "count": 4}, {"id": "all_unknown", "label": "All unknown", "count": 2}]
    data_dir = fixture / "data"
    data_dir.mkdir()
    data_bytes = "".join(json.dumps(row) + "\n" for row in dataset).encode()
    (data_dir / "questions.jsonl").write_bytes(data_bytes)
    data_manifest = {"schema_version": 1, "purpose": "exploratory_baseline_calibration", "groups": groups,
        "files": {"questions.jsonl": {"sha256": hashlib.sha256(data_bytes).hexdigest(), "rows": 6}}}
    (data_dir / "manifest.json").write_text(json.dumps(data_manifest))
    model_files = {"fixture.safetensors": {"bytes": 4, "sha256": "f" * 64}}
    model_identity = hashlib.sha256(json.dumps(model_files, sort_keys=True).encode()).hexdigest()
    manifest = {"schema_version": 1, "experiment_type": "exploratory_baseline_calibration", "synthetic": True,
        "model_label": "base", "model_identity": model_identity, "model_files": model_files,
        "adapter_path": None, "adapter_identity": None, "path_base": "code_root", "complete": True,
        "question_count": 6, "total_samples": 24, "n_samples": 4, "groups": groups,
        "dataset_path": "data/questions.jsonl", "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "dataset_manifest_path": "data/manifest.json", "dataset_manifest_sha256": hashlib.sha256((data_dir / "manifest.json").read_bytes()).hexdigest(),
        "config": {"model": {"expected_identity": model_identity}, "evaluation": {**settings, "n_samples": 4}}, "settings": settings,
        "code_hashes": {name: hashlib.sha256((fixture / "scripts" / name).read_bytes()).hexdigest() for name in ("verify_answers.py", "validate_guarded_sample.py", "repetition_guard.py")}}
    protocol = {key: manifest[key] for key in ("dataset_sha256", "dataset_manifest_sha256", "n_samples", "settings", "code_hashes", "experiment_type")}
    manifest["protocol_id"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    rows = []
    for i, item in enumerate(dataset):
        for index in range(4):
            reason = "no_token_timeout"
            if i == 0 and index < 2 or i == 2 and index < 2 or i == 3:
                reason = "eos"
            elif i == 2:
                reason = "answer_timeout"
            correct = i == 0 and index == 0 or i == 2 and index < 2
            row = {**item, "sample_index": index, "sample_seed": int.from_bytes(hashlib.sha256(f"17:{item['id']}:{index}".encode()).digest()[:4], "big"),
                "prompt_tokens": 20, "generation_limit": 262124, "seconds": 0.5,
                "termination_reason": reason, "scoring_evaluable": reason == "eos", "guard_evidence": None, "timeout_evidence": None}
            if reason == "eos":
                row.update(text="#### 2" if correct else "#### 3", output_token_ids=[12, 34, 248044], output_tokens=3)
                score = runpy.run_path(str(fixture / "scripts/verify_answers.py"), init_globals={"SCORE_INPUT": [row]})["SCORE_OUTPUT"][0]
                row.update(score)
            else:
                row.update(text="", output_token_ids=[], output_tokens=0, status="unscored", correct=None, reward=None, extracted=None,
                    format_compliant=None, truncated=False)
                duration = guard["answer_timeout_seconds"] if reason == "answer_timeout" else guard["no_token_timeout_seconds"]
                row.update(seconds=duration, timeout_evidence={"timeout_seconds": duration, "elapsed_seconds": duration, "last_token_age_seconds": 1 if reason == "answer_timeout" else duration, "observed_tokens_at_timeout": 0})
                if reason == "answer_timeout":
                    row.update(status="timeout", correct=False, reward=0.0)
            if i == 0 and index == 2:
                token_ids = [17, 18] * 512
                evidence = runpy.run_path(str(fixture / "scripts/repetition_guard.py"), init_globals={"TOKEN_IDS": token_ids, "GUARD_POLICY": guard})["GUARD_RESULT"]["evidence"]
                row.update(termination_reason="repetition_guard", text="#### 2\n" + "loop " * 1024,
                    output_token_ids=token_ids, output_tokens=len(token_ids), timeout_evidence=None, guard_evidence=evidence)
            rows.append(row)
    input_dir = fixture / "outputs/base"
    input_dir.mkdir(parents=True)
    raw = "".join(json.dumps(row) + "\n" for row in rows).encode()
    manifest["samples_sha256"] = hashlib.sha256(raw).hexdigest()
    (input_dir / "samples.jsonl").write_bytes(raw)
    (input_dir / "manifest.json").write_text(json.dumps(manifest))
    command = [sys.executable, "-S", str(fixture / "scripts/analyze_boundary.py"), "--input", "outputs/base", "--output", "outputs/analysis", "--bootstrap", "100"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.stdout, result.stderr)
    summary = json.loads((fixture / "outputs/analysis/summary.json").read_text())
    by_group = {group["group_id"]: group for group in summary["groups"]}
    mixed, unknown = by_group["mixed"], by_group["all_unknown"]
    assert mixed["correct_count"] == 3 and mixed["evaluable_count"] == 8 and mixed["unknown_count"] == 6
    assert mixed["timeout_count"] == 2 and mixed["timeout_rate"] == 2 / 16 and mixed["completion_rate"] == 0.5
    assert mixed["completed_only_accuracy"] == 3 / 8 and mixed["observed_success_per_request"]["estimate"] == 3 / 16
    assert mixed["incomplete_answer_bounds"] == {"lower": 3 / 16, "upper": 9 / 16, "is_confidence_interval": False}
    assert mixed["pass_at_1"] is None and mixed["point_in_target"] is None and mixed["target_assessment"] == "unresolved_incomplete_answers"
    assert mixed["pass_at_4_bounds"] == {"lower": 0.5, "upper": 0.75, "is_confidence_interval": False}
    checks += 1
    assert unknown["completed_only_accuracy"] is None and unknown["completion_rate"] == 0
    assert unknown["observed_success_per_request"]["estimate"] == 0 and unknown["incomplete_answer_bounds"]["upper"] == 1
    assert unknown["pass_at_4_bounds"] == {"lower": 0, "upper": 1, "is_confidence_interval": False}
    assert unknown["format_compliance_rate_among_evaluable"] is None
    assert summary["unknown_count"] == 14 and summary["timeout_count"] == 2 and summary["correct_count"] == 3
    checks += 1
    questions = {row["id"]: row for row in [json.loads(line) for line in (fixture / "outputs/analysis/per_question.jsonl").read_text().split("\n") if line.strip()]}
    assert questions["q0"]["termination_counts"].get("repetition_guard") == 1
    assert questions["q0"]["correct_count"] == 1
    assert questions["q0"]["pass_at_4_bounds"]["lower"] == questions["q0"]["pass_at_4_bounds"]["upper"] == 1
    assert questions["q2"]["unknown_count"] == 0 and questions["q2"]["pass_at_1"] == 0.5
    assert questions["q2"]["completed_only_accuracy"] == 1 and questions["q2"]["timeout_count"] == 2
    candidates = json.loads((fixture / "outputs/analysis/calibration_candidates.json").read_text())
    assert candidates["metric"] == "observed_success_per_request" and candidates["automatic_selection_performed"] is False
    assert "not confidence intervals" in (fixture / "outputs/analysis/report.md").read_text()
    checks += 1
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and "CACHED" in result.stdout
    checks += 1
    for shard in (0, 1):
        destination = fixture / f"outputs/shard-{shard}"
        destination.mkdir()
        shard_rows = [row for row in rows if int(row["id"][1:]) % 2 == shard]
        shard_raw = "".join(json.dumps(row) + "\n" for row in reversed(shard_rows)).encode()
        shard_manifest = {**manifest, "total_samples": len(shard_rows), "sharding": {"count": 2, "index": shard, "assignment": "question_index_modulo"},
            "samples_sha256": hashlib.sha256(shard_raw).hexdigest()}
        (destination / "samples.jsonl").write_bytes(shard_raw)
        (destination / "manifest.json").write_text(json.dumps(shard_manifest))
    merge_command = [sys.executable, "-S", str(fixture / "scripts/merge_boundary.py"), "--inputs", "outputs/shard-1", "outputs/shard-0", "--output", "outputs/merged"]
    result = subprocess.run(merge_command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.stdout, result.stderr)
    merged = [json.loads(line) for line in (fixture / "outputs/merged/samples.jsonl").read_text().split("\n") if line.strip()]
    assert merged == rows and sum(row["correct"] is None for row in merged) == 14
    checks += 1
    # Tokens arriving during shutdown cannot change a deadline already recorded.
    late_rows = copy.deepcopy(rows)
    late_rows[10].update(output_token_ids=[17, 18] * 512, output_tokens=1024,
        text="Tokens received after the deadline: " + "loop " * 1024, seconds=305)
    assert late_rows[10]["timeout_evidence"]["observed_tokens_at_timeout"] == 0
    late_raw = "".join(json.dumps(row) + "\n" for row in late_rows).encode()
    late_input = fixture / "outputs/late-base"
    late_input.mkdir()
    (late_input / "samples.jsonl").write_bytes(late_raw)
    (late_input / "manifest.json").write_text(json.dumps({**manifest, "samples_sha256": hashlib.sha256(late_raw).hexdigest()}))
    result = subprocess.run([sys.executable, "-S", str(fixture / "scripts/analyze_boundary.py"), "--input", "outputs/late-base",
        "--output", "outputs/late-analysis", "--bootstrap", "100"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.stdout, result.stderr)
    late_summary = json.loads((fixture / "outputs/late-analysis/summary.json").read_text())
    assert late_summary["timeout_count"] == 2 and late_summary["unknown_count"] == 14
    assert late_summary["incomplete_answer_bounds"] == summary["incomplete_answer_bounds"]
    checks += 1
    shard0_path = fixture / "outputs/shard-0"
    late_shard_raw = "".join(json.dumps(row) + "\n" for row in late_rows if int(row["id"][1:]) % 2 == 0).encode()
    shard0_manifest = json.loads((shard0_path / "manifest.json").read_text())
    (shard0_path / "samples.jsonl").write_bytes(late_shard_raw)
    (shard0_path / "manifest.json").write_text(json.dumps({**shard0_manifest, "samples_sha256": hashlib.sha256(late_shard_raw).hexdigest()}))
    result = subprocess.run(merge_command[:-1] + ["outputs/late-merged"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, (result.stdout, result.stderr)
    late_merged = [json.loads(line) for line in (fixture / "outputs/late-merged/samples.jsonl").read_text().split("\n") if line.strip()]
    assert late_merged[10]["termination_reason"] == "answer_timeout" and late_merged[10]["correct"] is False
    checks += 1
    mutations = ("unknown_false", "timeout_null", "evaluable", "missing_eos", "early_eos", "wrong_length", "bad_seed", "bad_limit", "bad_duration", "wrong_timeout", "fake_repeat", "timeout_prior_repeat", "timeout_missing_prefix")
    for mutation in mutations:
        changed = copy.deepcopy(rows)
        if mutation == "unknown_false":
            changed[2]["correct"] = False
        elif mutation == "timeout_null":
            changed[10]["correct"] = None
        elif mutation == "evaluable":
            changed[2]["scoring_evaluable"] = True
        elif mutation == "missing_eos":
            changed[0]["output_token_ids"][-1] = 99
        elif mutation == "early_eos":
            changed[0]["output_token_ids"][0] = 248044
        elif mutation == "wrong_length":
            changed[0]["output_tokens"] = 4
        elif mutation == "bad_seed":
            changed[0]["sample_seed"] += 1
        elif mutation == "bad_limit":
            changed[0]["generation_limit"] += 1
        elif mutation == "bad_duration":
            changed[0]["seconds"] = -1
        elif mutation == "wrong_timeout":
            changed[10]["timeout_evidence"]["elapsed_seconds"] = 299
        elif mutation == "timeout_prior_repeat":
            changed[10].update(output_token_ids=[17, 18] * 512, output_tokens=1024)
            changed[10]["timeout_evidence"]["observed_tokens_at_timeout"] = 1024
        elif mutation == "timeout_missing_prefix":
            del changed[10]["timeout_evidence"]["observed_tokens_at_timeout"]
        elif mutation == "fake_repeat":
            changed[2].update(termination_reason="repetition_guard", timeout_evidence=None, guard_evidence={})
        changed_raw = "".join(json.dumps(row) + "\n" for row in changed).encode()
        (input_dir / "samples.jsonl").write_bytes(changed_raw)
        (input_dir / "manifest.json").write_text(json.dumps({**manifest, "samples_sha256": hashlib.sha256(changed_raw).hexdigest()}))
        result = subprocess.run(command[:-4] + ["--output", f"outputs/reject-{mutation}", "--bootstrap", "100"], capture_output=True, text=True, timeout=30)
        assert result.returncode != 0 and not (fixture / f"outputs/reject-{mutation}").exists(), (mutation, result.stdout, result.stderr)
        shard0_rows = [row for row in changed if int(row["id"][1:]) % 2 == 0]
        shard0_raw = "".join(json.dumps(row) + "\n" for row in shard0_rows).encode()
        shard0_path = fixture / "outputs/shard-0"
        shard0_manifest = json.loads((shard0_path / "manifest.json").read_text())
        (shard0_path / "samples.jsonl").write_bytes(shard0_raw)
        (shard0_path / "manifest.json").write_text(json.dumps({**shard0_manifest, "samples_sha256": hashlib.sha256(shard0_raw).hexdigest()}))
        result = subprocess.run(merge_command[:-1] + [f"outputs/reject-merge-{mutation}"], capture_output=True, text=True, timeout=30)
        assert result.returncode != 0 and not (fixture / f"outputs/reject-merge-{mutation}").exists(), (mutation, result.stdout, result.stderr)
        checks += 2
    (input_dir / "samples.jsonl").write_bytes(raw)
    (input_dir / "manifest.json").write_text(json.dumps(manifest))
    for name in ("validate_guarded_sample.py", "repetition_guard.py"):
        helper = fixture / "scripts" / name
        original_helper = helper.read_bytes()
        helper.write_bytes(original_helper + b"\n# changed helper identity\n")
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0 and "identity changed" in result.stderr
        helper.write_bytes(original_helper)
        checks += 1
print(f"[PASS] {checks} guarded analysis checks | unknown bounds, 300-second policy failures, all-unknown safety, shared validation and guarded merge | no model/GPU/network")
