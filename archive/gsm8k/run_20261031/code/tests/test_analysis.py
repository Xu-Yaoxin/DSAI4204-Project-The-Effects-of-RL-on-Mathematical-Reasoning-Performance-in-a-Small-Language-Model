#!/usr/bin/env python3
"""Run analytical regression checks without model or GPU dependencies."""

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


script = Path(__file__).resolve().parents[1] / "scripts" / "analyze.py"
settings = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 64, "seed": 42, "prompt_template": "Solve: {question}", "dtype": "float32", "backend": "synthetic"}
manifest_template = {"schema_version": 1, "n_samples": 16, "split": "test", "dataset_sha256": "fixture-dataset-sha256", "protocol_id": "fixture-protocol", "model_identity": "fixture-base-weights", "synthetic": True, "complete": True, "settings": settings}
correct_counts = {"base": [1, 16, 0, 0], "rl": [8, 0, 1, 0]}
fixture_rows = {}
for label in ("base", "rl"):
    fixture_rows[label] = []
    for question_index, count in enumerate(correct_counts[label]):
        for sample_index in range(16):
            correct = sample_index < count
            fixture_rows[label].append({"id": f"q{question_index}", "sample_index": sample_index, "correct": correct, "status": "correct" if correct else "incorrect", "format_compliant": True, "truncated": False, "output_tokens": 12, "family_id": None, "variant_type": "original", "question": f"What is {question_index} + 1?", "answer": str(question_index + 1), "text": f"Final answer: {question_index + 1 if correct else 99}"})

test_count = 0
with tempfile.TemporaryDirectory(prefix="rlvr-analysis-test-") as temporary:
    root = Path(temporary)
    for label in ("base", "rl"):
        (root / label).mkdir()
        manifest = dict(manifest_template, model_label=label, adapter_identity=None if label == "base" else "fixture-adapter")
        (root / label / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / label / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in fixture_rows[label]), encoding="utf-8")
    command = [sys.executable, str(script), "--base", str(root / "base"), "--rl", str(root / "rl"), "--output", str(root / "analysis"), "--bootstrap", "100", "--seed", "9"]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads((root / "analysis" / "summary.json").read_text())
    per_question = [json.loads(line) for line in (root / "analysis" / "per_question.jsonl").read_text().splitlines()]
    assert math.isclose(summary["metrics"]["base"]["pass_at_k"]["1"], 17 / 64)
    assert math.isclose(summary["metrics"]["rl"]["pass_at_k"]["1"], 9 / 64)
    assert math.isclose(per_question[0]["base"]["pass_at_k"]["2"], 1 / 8)
    assert math.isclose(per_question[0]["rl"]["pass_at_k"]["2"], 1 - math.comb(8, 2) / math.comb(16, 2))
    assert summary["coverage"] == {"both": 1, "base_only": 1, "rl_only": 1, "neither": 1, "total": 4}
    assert summary["comparisons"]["delta1_minus_delta16"]["estimate"] == -0.125
    assert summary["sampling"]["bootstrap_unit"] == "question"
    assert summary["synthetic"] is True
    assert "SYNTHETIC" in (root / "analysis" / "report.md").read_text()
    assert (root / "analysis" / "manual_inspection.csv").exists()
    first_summary = (root / "analysis" / "summary.json").read_text()
    repeated = subprocess.run(command, text=True, capture_output=True)
    assert repeated.returncode == 0, repeated.stderr
    assert (root / "analysis" / "summary.json").read_text() == first_summary
    test_count += 1

    bad_manifest_cases = [("protocol_id", "changed-protocol"), ("model_identity", "different-base-weights"), ("n_samples", 8), ("complete", False), ("synthetic", False), ("settings", dict(settings, temperature=0.7)), ("samples_sha256", "changed-sample-hash"), ("question_count", 5)]
    for field, value in bad_manifest_cases:
        manifest = dict(manifest_template, model_label="rl", adapter_identity="fixture-adapter")
        manifest[field] = value
        (root / "rl" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode != 0, f"Invalid {field} was accepted"
        test_count += 1
    (root / "rl" / "manifest.json").write_text(json.dumps(dict(manifest_template, model_label="rl", adapter_identity="fixture-adapter")), encoding="utf-8")

    bad_rows_cases = []
    bad_rows_cases.append(("duplicate", fixture_rows["rl"] + [fixture_rows["rl"][0]]))
    bad_rows_cases.append(("incomplete", fixture_rows["rl"][:-1]))
    for status in ("infrastructure_error", "invalid_gold"):
        rows = copy.deepcopy(fixture_rows["rl"])
        rows[0]["status"] = status
        bad_rows_cases.append((status, rows))
    for field, value in (("question", "Changed question"), ("answer", "999"), ("sample_index", 100), ("correct", "true"), ("status", "unrecognized_status")):
        rows = copy.deepcopy(fixture_rows["rl"])
        rows[0][field] = value
        bad_rows_cases.append((field, rows))
    for name, rows in bad_rows_cases:
        (root / "rl" / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        result = subprocess.run(command, text=True, capture_output=True)
        assert result.returncode != 0, f"Invalid {name} was accepted"
        test_count += 1

    rows = copy.deepcopy(fixture_rows["rl"])
    for row in rows[:16]:
        row["question"] = "Consistent within RL, but different from Base"
    (root / "rl" / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and "Paired runs have changed" in result.stderr, result.stderr
    test_count += 1

    rows = copy.deepcopy(fixture_rows["rl"])
    rows[-1].update(status="extraction_failed", format_compliant=False, truncated=True, output_tokens=64)
    (root / "rl" / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads((root / "analysis" / "summary.json").read_text())
    assert summary["metrics"]["rl"]["pass_at_k"]["1"] == 9 / 64
    assert summary["diagnostics"]["rl"]["extraction_failure_rate"] == 1 / 64
    assert summary["diagnostics"]["rl"]["truncation_rate"] == 1 / 64
    assert summary["diagnostics"]["rl"]["format_compliance_rate"] == 63 / 64
    test_count += 1

    for label in ("base", "rl"):
        manifest = dict(manifest_template, model_label=label, adapter_identity=None if label == "base" else "fixture-adapter", split="structural", n_samples=4)
        (root / label / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        rows = []
        for family_index in range(3):
            for variant in ("numerical", "relational"):
                count = 2 if label == "base" else family_index + 2 if variant == "numerical" else family_index + 1
                for sample_index in range(4):
                    correct = sample_index < count
                    rows.append({"id": f"f{family_index}-{variant}", "sample_index": sample_index, "correct": correct, "status": "correct" if correct else "incorrect", "format_compliant": True, "truncated": False, "output_tokens": 12, "family_id": f"f{family_index}", "variant_type": variant, "question": f"{variant} question {family_index}", "answer": "1", "text": "Final answer: 1" if correct else "Final answer: 0"})
        (root / label / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    summary = json.loads((root / "analysis" / "summary.json").read_text())
    assert summary["sampling"]["bootstrap_unit"] == "family"
    assert summary["sampling"]["resampling_unit_count"] == 3
    assert summary["sampling"]["k_values"] == [1, 2, 4]
    assert summary["comparisons"]["delta1_minus_delta16"] is None
    assert summary["structural"]["paired_family_count"] == 3
    assert summary["structural"]["relational_minus_numerical"] == {"estimate": -0.25, "ci_low": -0.25, "ci_high": -0.25}
    assert summary["structural"]["variants"]["numerical"]["delta_pass_at_1"]["estimate"] == 0.25
    test_count += 1

    real_manifests = {}
    for label in ("base", "rl"):
        sample_bytes = "".join(json.dumps(row) + "\n" for row in fixture_rows[label]).encode()
        (root / label / "samples.jsonl").write_bytes(sample_bytes)
        manifest = dict(manifest_template, model_label=label, adapter_identity=None if label == "base" else "fixture-adapter",
            synthetic=False, question_count=4, total_samples=64, samples_sha256=hashlib.sha256(sample_bytes).hexdigest(),
            code_hashes={"verify_answers.py": "fixture-source-identity"})
        protocol = {field: manifest[field] for field in ["dataset_sha256", "n_samples", "settings", "code_hashes"]}
        manifest["protocol_id"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        real_manifests[label] = manifest
        (root / label / "manifest.json").write_text(json.dumps(manifest))
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    test_count += 1
    invalid = copy.deepcopy(real_manifests["rl"])
    invalid["settings"]["temperature"] = 0.9
    (root / "rl" / "manifest.json").write_text(json.dumps(invalid))
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and "protocol_id does not match" in result.stderr, result.stderr
    test_count += 1
    invalid = copy.deepcopy(real_manifests["rl"])
    invalid.pop("samples_sha256")
    (root / "rl" / "manifest.json").write_text(json.dumps(invalid))
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and "require source and sample-content hashes" in result.stderr, result.stderr
    test_count += 1

print(f"[TEST 100%] {test_count} analysis regression scenarios passed")
