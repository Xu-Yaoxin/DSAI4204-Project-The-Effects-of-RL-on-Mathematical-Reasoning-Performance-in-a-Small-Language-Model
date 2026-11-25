"""Offline behavioral checks for exact scoring, split isolation, and variants."""

import hashlib
import json
import random
import runpy
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path


code_root = Path(__file__).resolve().parents[1]
cases = [
    ({"text": "Reasoning: 6 times 7.\n#### 42", "answer": "42"}, "correct", True, "42"),
    ({"text": r"\boxed{\frac{2}{4}}", "answer": "0.5"}, "correct", False, "1/2"),
    ({"text": "#### 1,000.00", "answer": "1000"}, "correct", True, "1000"),
    ({"text": "The final answer is $42$.", "answer": "42"}, "correct", False, "42"),
    ({"text": "Thus, the answer is -3/2.", "answer": "-1.5"}, "correct", False, "-3/2"),
    ({"text": "#### 0.5\n#### 1/2", "answer": "1/2"}, "correct", True, "1/2"),
    ({"text": "#### 42 #### 42", "answer": "42"}, "correct", False, "42"),
    ({"text": "#### 42\n" + r"\boxed{42}", "answer": "42"}, "correct", True, "42"),
    ({"text": "Reasoning ends here.\n42", "answer": "42"}, "correct", False, "42"),
    ({"text": "#### 4.2e1", "answer": "42"}, "correct", True, "42"),
    ({"text": "#### −42", "answer": "-42"}, "correct", True, "-42"),
    ({"text": r"\boxed{-\frac{1}{2}}", "answer": "-0.5"}, "correct", False, "-1/2"),
    ({"text": "#### 42", "answer": "42", "truncated": True}, "correct", True, "42"),
    ({"text": "#### 41", "answer": "42"}, "incorrect", True, "41"),
    ({"text": "#### 42\n#### 43", "answer": "42"}, "ambiguous", False, None),
    ({"text": r"\boxed{42} and \boxed{43}", "answer": "42"}, "ambiguous", False, None),
    ({"text": "We have 42 apples so far.", "answer": "42"}, "extraction_failed", False, None),
    ({"text": "3 + 39 = 42", "answer": "42"}, "extraction_failed", False, None),
    ({"text": "If the answer is 42, check it.", "answer": "42"}, "extraction_failed", False, None),
    ({"text": "#### 42 + 1", "answer": "42"}, "extraction_failed", False, None),
    ({"text": "#### 1/0", "answer": "1"}, "extraction_failed", False, None),
    ({"text": "#### 1,00", "answer": "100"}, "extraction_failed", False, None),
    ({"text": "#### 42..", "answer": "42"}, "extraction_failed", False, None),
    ({"text": "#### 50%", "answer": "50"}, "extraction_failed", False, None),
    ({"text": r"\boxed{42", "answer": "42"}, "extraction_failed", False, None),
    ({"text": r"\boxed{\frac{1}{0}}", "answer": "1"}, "extraction_failed", False, None),
    ({"text": "#### __import__('os').system('false')", "answer": "0"}, "extraction_failed", False, None),
    ({"text": "#### 1e999999", "answer": "1"}, "extraction_failed", False, None),
    ({"text": "#### 42", "answer": "4 + 2"}, "invalid_gold", False, None),
    ({"text": "#### 42", "answer": "42", "infrastructure_error": "GPU failure"}, "infrastructure_error", False, None),
    ({"text": "", "answer": "42", "truncated": True}, "extraction_failed", False, None),
    ({"text": "#### 42\n#### not a number", "answer": "42"}, "extraction_failed", False, None),
    ({"text": "#### 5\n6", "answer": "5"}, "ambiguous", False, None),
    ({"text": "#### 5\nFinal answer: 6", "answer": "5"}, "ambiguous", False, None),
    ({"text": "#### 5\n5", "answer": "5"}, "correct", True, "5"),
    ({"text": "Reasoning ends: #### 42", "answer": "42"}, "correct", False, "42"),
    ({"text": "####42", "answer": "42"}, "correct", False, "42"),
]
scores = runpy.run_path(str(code_root / "scripts/verify_answers.py"), init_globals={"SCORE_INPUT": [case[0] for case in cases]})["SCORE_OUTPUT"]
assert len(scores) == len(cases)
for index, ((record, status, compliant, extracted), score) in enumerate(zip(cases, scores)):
    assert score["status"] == status, (index, record, score, status)
    assert score["format_compliant"] is compliant, (index, score)
    assert score["extracted"] == extracted, (index, score)
    assert score["correct"] is (status == "correct"), (index, score)
    assert score["reward"] == float(status == "correct"), (index, score)
    assert score["truncated"] is bool(record.get("truncated", False)), (index, score)
print(f"[PASS] {len(cases)} exact-verifier behavior cases")

with tempfile.TemporaryDirectory(prefix="dsai4207-data-test-") as temporary_directory:
    temporary_root = Path(temporary_directory)
    raw_train = temporary_root / "train.jsonl"
    raw_test = temporary_root / "test.jsonl"
    train_rows = [{"question": f"Training toy {index}:\u2028Sam owns {index} beads and receives one. How many?", "answer": f"Add one.\n#### {index + 1}"} for index in range(20)]
    test_rows = [{"question": f"Testing toy {index}:\u0085Jo owns {index} beads and receives two. How many?", "answer": f"Add two.\n#### {index + 2}"} for index in range(7)]
    raw_train.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_rows), encoding="utf-8")
    raw_test.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in test_rows), encoding="utf-8")
    output_dir = temporary_root / "processed"
    config = {"seed": 17, "data": {"raw_train": str(raw_train), "raw_test": str(raw_test), "output_dir": str(output_dir),
                                  "train_size": 8, "pilot_size": 4, "engineering_size": 2, "dev_size": 4, "test_size": 4}}
    config_path = temporary_root / "v1.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    command = [sys.executable, str(code_root / "scripts/prepare_data.py"), "--config", str(config_path), "--allow-unverified-source"]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    initial_bytes = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    manifest = json.loads(initial_bytes["manifest.json"])
    assert manifest["source_provenance"]["status"] == "unverified_explicit_override"
    splits = {name: [json.loads(line) for line in initial_bytes[name + ".jsonl"].decode().split("\n") if line.strip()] for name in ["train", "pilot", "engineering", "dev", "test"]}
    assert all("\u2028" in row["question"] for row in splits["train"] + splits["dev"])
    assert all("\u0085" in row["question"] for row in splits["test"])
    split_ids = {name: {row["id"] for row in rows} for name, rows in splits.items()}
    assert split_ids["engineering"] <= split_ids["pilot"] <= split_ids["train"]
    assert split_ids["train"].isdisjoint(split_ids["dev"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["dev"].isdisjoint(split_ids["test"])
    assert all(row["source"]["official_split"] == "train" for row in splits["train"] + splits["dev"])
    assert all(row["source"]["official_split"] == "test" for row in splits["test"])
    for filename, metadata in manifest["files"].items():
        assert hashlib.sha256(initial_bytes[filename]).hexdigest() == metadata["sha256"]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert initial_bytes == {path.name: path.read_bytes() for path in output_dir.iterdir()}

    config["seed"] = 18
    config_path.write_text(json.dumps(config), encoding="utf-8")
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0 and "Immutable data conflict" in completed.stderr, completed.stderr
    assert initial_bytes == {path.name: path.read_bytes() for path in output_dir.iterdir()}
    config["seed"] = 17
    config_path.write_text(json.dumps(config), encoding="utf-8")
    print("[PASS] deterministic official-split isolation, nested subsets, hashes, and immutable conflict")

    variant_command = [sys.executable, str(code_root / "scripts/build_variants.py"), "--config", str(config_path), "--families", "40"]
    completed = subprocess.run(variant_command, text=True, capture_output=True)
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    structural_bytes = (output_dir / "structural.jsonl").read_bytes()
    variants = [json.loads(line) for line in structural_bytes.decode().split("\n") if line.strip()]
    structural_manifest = json.loads((output_dir / "structural_manifest.json").read_text())
    review = json.loads((output_dir / "structural_review.template.json").read_text())
    assert len(variants) == 160
    pairs = {}
    template_families = set()
    for row in variants:
        pairs.setdefault(row["pair_id"], []).append(row)
        template_families.add(row["family_id"])
        assert row["verified"] is False and row["review_required"] is True
        assert row["oracle"]["operation_count"] == 2
        numbers = row["oracle"]["parameters"]
        template = row["source"]["template"]
        variant_type = row["variant_type"]
        if template == "item_discount":
            if variant_type in ["original", "numerical"]:
                expected = numbers["items"] * numbers["price"] - numbers["items"] * numbers["discount"]
            elif variant_type == "relational":
                expected = numbers["items"] * numbers["price"] - numbers["discount"]
            else:
                expected = numbers["items"] * numbers["price"] + numbers["extra_items"] * numbers["price"]
        elif template == "boxed_stock":
            if variant_type in ["original", "numerical"]:
                expected = numbers["boxes"] * numbers["per_box"] + numbers["loose"]
            elif variant_type == "relational":
                expected = numbers["boxes"] * numbers["regular_per_box"] + numbers["boxes"] * numbers["colored_per_box"]
            else:
                expected = numbers["boxes"] * numbers["per_box"] - numbers["damaged"]
        elif template == "shared_fund":
            if variant_type in ["original", "numerical"]:
                expected = Fraction(numbers["fund"], numbers["people"]) - Fraction(numbers["fee"], numbers["people"])
            elif variant_type == "relational":
                expected = Fraction(numbers["fund"] - numbers["personal_fee"] * numbers["people"], numbers["people"])
            else:
                expected = Fraction(numbers["fund"] + numbers["personal_gift"] * numbers["people"], numbers["people"])
        else:
            if variant_type in ["original", "numerical"]:
                expected = numbers["speed"] * numbers["hours"] + numbers["extra_km"]
            elif variant_type == "relational":
                expected = numbers["speed"] * numbers["hours"] + numbers["speed"] * numbers["extra_hours"]
            else:
                expected = numbers["speed"] * numbers["hours"] - numbers["return_km"]
        assert Fraction(row["answer"]) == expected, row
    assert len(pairs) == 40
    assert len(template_families) == structural_manifest["template_families"] == 4
    assert structural_manifest["pairs"] == 40
    for family in pairs.values():
        assert {row["variant_type"] for row in family} == {"original", "numerical", "relational", "recombined"}
        by_type = {row["variant_type"]: row for row in family}
        assert by_type["original"]["answer"] != by_type["relational"]["answer"], family
    assert structural_manifest["dataset_sha256"] == hashlib.sha256(structural_bytes).hexdigest() == review["dataset_sha256"]
    assert review["approved"] is False and all(row["verified"] is False for row in review["records"])
    assert all(row["reviewed_answer"] == "" for row in review["records"])
    completed = subprocess.run(variant_command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert structural_bytes == (output_dir / "structural.jsonl").read_bytes()
    print("[PASS] 40 controlled pairs in 4 template families, independent arithmetic, distinct relational answers, pending manual review")
    (output_dir / "train.jsonl").write_bytes(initial_bytes["train.jsonl"] + b"\n")
    completed = subprocess.run(variant_command, text=True, capture_output=True)
    assert completed.returncode != 0 and "Split hash mismatch" in completed.stderr, completed.stderr
    assert structural_bytes == (output_dir / "structural.jsonl").read_bytes()
    (output_dir / "train.jsonl").write_bytes(initial_bytes["train.jsonl"])
    print("[PASS] structural generation rejects prepared-data tampering")

    leakage_output = temporary_root / "leakage"
    config["data"]["output_dir"] = str(leakage_output)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    test_order = list(range(7))
    random.Random(18).shuffle(test_order)
    test_rows[test_order[0]]["question"] = "  " + splits["train"][0]["question"].upper().replace(" ", "  ") + "  "
    raw_test.write_text("".join(json.dumps(row) + "\n" for row in test_rows), encoding="utf-8")
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0 and "Normalized question overlap" in completed.stderr, completed.stderr
    assert not leakage_output.exists()
    print("[PASS] cross-official-split leakage rejected before any output")

print("[PASS] All data/verifier tests passed without model dependencies.")
