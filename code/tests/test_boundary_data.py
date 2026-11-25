"""Exercise deterministic baseline calibration preparation with isolated source fixtures."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
production = root / "scripts/prepare_boundary_data.py"
count = 0
with tempfile.TemporaryDirectory(prefix="rlvr-boundary-data-") as temporary:
    fixture = Path(temporary) / "code"
    (fixture / "scripts").mkdir(parents=True)
    (fixture / "scripts/verify_answers.py").write_bytes((root / "scripts/verify_answers.py").read_bytes())
    history = fixture / "data/processed"
    history.mkdir(parents=True)
    for split in ("train", "pilot", "engineering", "dev", "test"):
        row = {"id": "gsm8k-test-000100" if split == "test" else "train-fixture", "question": "main family101 instance0" if split == "train" else f"History {split}", "answer": "5", "source": {"official_split": "test" if split == "test" else "train", "row_index": 100 if split == "test" else 999}}
        (history / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    output = fixture / "data/boundary_v1"
    specifications = []
    for subset in ("main", "p1", "p2", "math500"):
        records = []
        if subset == "math500":
            relative = "math500/test.jsonl"
            for level in range(1, 6):
                for index in range(20):
                    records.append({"problem": f"MATH level{level} item{index}" + (" [asy] diagram" if index == 0 else ""), "answer": "1/2" if index != 1 else "\\sqrt{2}", "level": level, "subject": "Algebra", "unique_id": f"test/algebra/{level}-{index}.json"})
        else:
            relative = f"gsm_symbolic/{subset}.jsonl"
            for family in range(100, 130):
                for instance in (0, 1):
                    records.append({"question": f"{subset} family{family} instance{instance}", "answer": "Calculation\n#### " + ("NaN" if subset == "p2" and family == 102 else "5"), "original_id": family, "instance": instance, "id": family - 100, "canary": "fixture-canary"})
        raw = "".join(json.dumps(row) + "\n" for row in records).encode()
        target = output / "raw" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        specifications.append({"dataset": "fixture", "revision": "fixture-revision", "remote": relative, "local": relative, "git_blob_sha1": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest(), "license": "fixture-only"})
    tree = ast.parse(production.read_text())
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "source_specs" for target in node.targets):
            tree.body[index] = ast.parse("source_specs = " + repr(specifications)).body[0]
    fixture_script = fixture / "scripts/prepare_boundary_data.py"
    fixture_script.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    command = [sys.executable, "-S", str(fixture_script), "--offline"]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    rows = [json.loads(line) for line in (output / "questions.jsonl").read_text().split("\n") if line.strip()]
    assert len(rows) == 100 and len({row["id"] for row in rows}) == 100
    assert manifest["purpose"] == "exploratory_baseline_calibration"
    assert manifest["model_scores_used_for_selection"] is False
    assert all(sum(row["group_id"] == group["id"] for row in rows) == 20 for group in manifest["groups"])
    assert 100 not in manifest["selected_gsm_original_ids"] and 102 not in manifest["selected_gsm_original_ids"]
    assert manifest["eligible_shared_gsm_families"] == 28
    assert manifest["exclusions"]["main"]["exact_normalized_history_question"] == 1
    assert manifest["exclusions"]["math500"] == {"requires_diagram": 5, "unsupported_non_scalar_gold": 5}
    assert {level: sum(row["source"].get("level") == level for row in rows) for level in range(1, 6)} == {1: 6, 2: 7, 3: 7, 4: 10, 5: 10}
    assert all(row["answer"] == "1/2" for row in rows if row["group_id"].startswith("math_"))
    assert all(row["answer"] == "5" and row["source_answer"].endswith("#### 5") for row in rows if row["group_id"].startswith("gsm_"))
    assert len({(row["source"]["original_id"], row["source"]["instance"]) for row in rows if row["group_id"].startswith("gsm_")}) == 20
    assert all("[asy]" not in row["question"] for row in rows)
    assert (output / "questions.md").is_file() and (output / "README.md").is_file()
    count += 14
    before = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in output.rglob("*") if path.is_file()}
    result = subprocess.run(command + ["--verify"], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    after = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in output.rglob("*") if path.is_file()}
    assert before == after
    count += 2
    raw_path = output / "raw/gsm_symbolic/main.jsonl"
    original = raw_path.read_bytes()
    raw_path.write_bytes(original + b" ")
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode != 0 and "Git blob mismatch" in result.stderr
    raw_path.write_bytes(original)
    count += 1
    question_path = output / "questions.jsonl"
    original = question_path.read_bytes()
    question_path.write_bytes(original.replace(b"main family", b"changed family", 1))
    result = subprocess.run(command + ["--verify"], text=True, capture_output=True)
    assert result.returncode != 0 and "provenance changed" in result.stderr
    question_path.write_bytes(original)
    count += 1
    history_path = history / "train.jsonl"
    original = history_path.read_bytes()
    history_path.write_bytes(original.replace(b"main family101", b"other family101"))
    result = subprocess.run(command + ["--verify"], text=True, capture_output=True)
    assert result.returncode != 0 and "provenance changed" in result.stderr
    history_path.write_bytes(original)
    count += 1
    verifier_path = fixture / "scripts/verify_answers.py"
    original = verifier_path.read_bytes()
    verifier_path.write_bytes(original + b"\n# fixture mutation\n")
    result = subprocess.run(command + ["--verify"], text=True, capture_output=True)
    assert result.returncode != 0 and "provenance changed" in result.stderr
    verifier_path.write_bytes(original)
    count += 1
    original = raw_path.read_bytes()
    raw_path.unlink()
    result = subprocess.run(command + ["--verify"], text=True, capture_output=True)
    assert result.returncode != 0 and "Missing pinned source" in result.stderr
    raw_path.write_bytes(original)
    count += 1
print(f"[PASS] {count} boundary-data scenarios | fixed sampling, shared families, historical exclusions, scalar gold, quotas and immutable provenance | no model/network")
