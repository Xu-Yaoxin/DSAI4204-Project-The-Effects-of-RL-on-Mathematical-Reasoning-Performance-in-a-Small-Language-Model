#!/usr/bin/env python3
"""Check evaluation planning and frozen-data gates without loading any model."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


root = Path(__file__).resolve().parents[1]
script = root / "scripts" / "evaluate.py"
configuration = json.loads((root / "configs" / "v1.json").read_text())
test_count = 0
with tempfile.TemporaryDirectory(prefix="rlvr-evaluation-test-") as temporary:
    work = Path(temporary)
    config = copy.deepcopy(configuration)
    config["data"]["output_dir"] = str(work / "data")
    (work / "data").mkdir()
    for split in ("engineering", "test"):
        row = {"id": f"{split}-0", "split": split, "question": "What is 1 + 1?", "answer": "2", "family_id": None, "variant_type": "original"}
        (work / "data" / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    config_path = work / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = work / "evaluation-output"
    base_command = [sys.executable, str(script), "--config", str(config_path), "--model", "base", "--output", str(output), "--plan"]
    result = subprocess.run(base_command + ["--split", "engineering"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "[EVAL PLAN]" in result.stdout
    assert not output.exists(), "Planning must not create an evaluation run"
    test_count += 1

    result = subprocess.run(base_command + ["--split", "test"], text=True, capture_output=True)
    assert result.returncode != 0 and "selection" in result.stderr.lower(), result.stderr
    test_count += 1

    test_hash = hashlib.sha256((work / "data" / "test.jsonl").read_bytes()).hexdigest()
    selection = {"schema_version": 1, "evaluation_config": config["evaluation"], "prompt": config["prompt"], "model_identity": "unused-for-plan", "adapter_identity": "unused-for-plan", "test_sha256": test_hash}
    selection_path = work / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    formal_command = base_command + ["--split", "test", "--selection", str(selection_path)]
    result = subprocess.run(formal_command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not output.exists()
    test_count += 1

    bad_selection_cases = [("test_sha256", "changed-data-hash"), ("prompt", {"template": "Different {question}"}), ("evaluation_config", dict(config["evaluation"], temperature=0.9))]
    for field, value in bad_selection_cases:
        invalid = copy.deepcopy(selection)
        invalid[field] = value
        selection_path.write_text(json.dumps(invalid), encoding="utf-8")
        result = subprocess.run(formal_command, text=True, capture_output=True)
        assert result.returncode != 0, f"Frozen selection mismatch {field} was accepted"
        test_count += 1
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    result = subprocess.run(formal_command + ["--samples", "8"], text=True, capture_output=True)
    assert result.returncode != 0, "Formal sample budget override was accepted"
    test_count += 1

    result = subprocess.run(base_command + ["--split", "engineering", "--adapter", "unused"], text=True, capture_output=True)
    assert result.returncode != 0, "Base model accepted an adapter"
    test_count += 1

    rl_command = [sys.executable, str(script), "--config", str(config_path), "--model", "rl", "--output", str(output), "--plan", "--split", "engineering"]
    result = subprocess.run(rl_command, text=True, capture_output=True)
    assert result.returncode != 0, "RL model accepted a missing adapter"
    test_count += 1

    row["split"] = "test"
    row["id"] = "test-0"
    row["answer"] = "not a number"
    (work / "data" / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = subprocess.run(formal_command, text=True, capture_output=True)
    assert result.returncode != 0, "Invalid canonical gold was accepted"
    test_count += 1

    structural_row = {"id": "family-0-numerical", "split": "structural", "question": "What is 1 + 1?", "answer": "2", "family_id": "family-0", "variant_type": "numerical", "verified": False}
    structural_path = work / "data" / "structural.jsonl"
    structural_path.write_text(json.dumps(structural_row) + "\n", encoding="utf-8")
    structural_hash = hashlib.sha256(structural_path.read_bytes()).hexdigest()
    structural_selection = dict(selection, structural_sha256=structural_hash)
    selection_path.write_text(json.dumps(structural_selection))
    structural_command = base_command + ["--split", "structural", "--selection", str(selection_path)]
    review = {"approved": True, "reviewer": "test fixture reviewer", "dataset_sha256": structural_hash,
        "records": [{"id": structural_row["id"], "verified": True, "reviewed_answer": "4/2",
                     "question_sha256": hashlib.sha256(structural_row["question"].encode()).hexdigest()}]}
    review_path = work / "data" / "structural_review.json"
    review_path.write_text(json.dumps(review))
    result = subprocess.run(structural_command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    test_count += 1
    for mutation in ["duplicate", "missing", "extra", "answer", "blank_answer", "noncanonical_answer", "reviewer", "approved", "verified", "hash"]:
        invalid_review = copy.deepcopy(review)
        if mutation == "duplicate":
            invalid_review["records"].append(copy.deepcopy(invalid_review["records"][0]))
        elif mutation == "missing":
            invalid_review["records"] = []
        elif mutation == "extra":
            invalid_review["records"].append(dict(invalid_review["records"][0], id="extra-question"))
        elif mutation == "answer":
            invalid_review["records"][0]["reviewed_answer"] = "3"
        elif mutation == "blank_answer":
            invalid_review["records"][0]["reviewed_answer"] = ""
        elif mutation == "noncanonical_answer":
            invalid_review["records"][0]["reviewed_answer"] = "2\nThis is extra prose."
        elif mutation == "reviewer":
            invalid_review["reviewer"] = " "
        elif mutation == "approved":
            invalid_review["approved"] = 1
        elif mutation == "verified":
            invalid_review["records"][0]["verified"] = 1
        elif mutation == "hash":
            invalid_review["records"][0]["question_sha256"] = "changed"
        review_path.write_text(json.dumps(invalid_review))
        result = subprocess.run(structural_command, text=True, capture_output=True)
        assert result.returncode != 0, f"Invalid structural review {mutation} was accepted"
        test_count += 1

    unicode_row = {"id": "engineering-0", "split": "engineering", "question": "What is 1 + 1?\u2028Give a numerical answer.", "answer": "2", "family_id": None, "variant_type": "original"}
    (work / "data" / "engineering.jsonl").write_text(json.dumps(unicode_row, ensure_ascii=False) + "\n", encoding="utf-8")
    result = subprocess.run(base_command + ["--split", "engineering"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    test_count += 1
    leaking_config = copy.deepcopy(config)
    leaking_config["prompt"]["template"] = "Question: {question}; gold: {answer}"
    config_path.write_text(json.dumps(leaking_config))
    result = subprocess.run(base_command + ["--split", "engineering"], text=True, capture_output=True)
    assert result.returncode != 0 and "no other fields" in result.stderr, result.stderr
    test_count += 1

    for extra_environment in [{"WORLD_SIZE": "2"}, {"LOCAL_WORLD_SIZE": "4"}, {"RANK": "1"}, {"LOCAL_RANK": "1"}]:
        environment = dict(os.environ, WORLD_SIZE="1", LOCAL_WORLD_SIZE="1", RANK="0", LOCAL_RANK="0")
        environment.update(extra_environment)
        result = subprocess.run(base_command + ["--split", "engineering"], env=environment, text=True, capture_output=True)
        assert result.returncode != 0 and "Evaluation uses one process" in result.stderr, result.stderr
        selection_command = [sys.executable, str(root / "scripts/select_checkpoint.py"), "--run", "unused", "--output", "unused"]
        result = subprocess.run(selection_command, env=environment, text=True, capture_output=True)
        assert result.returncode != 0 and "Checkpoint selection uses one process" in result.stderr, result.stderr
        test_count += 1

print(f"[TEST 100%] {test_count} evaluation planning and freeze scenarios passed")
