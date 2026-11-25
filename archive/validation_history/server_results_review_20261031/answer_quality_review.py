"""Post-hoc answer audit; preserve the archived protocol and all primary outputs."""
from collections import Counter
import hashlib
import json
from math import comb
from pathlib import Path
import random

review_root = Path(__file__).resolve().parent
archive = review_root.parents[3] / "server_results/run_20261031/code"
destination = review_root / "nested_marker_diagnostic"
destination.mkdir(parents=True, exist_ok=True)
source_path = archive / "scripts/verify_answers.py"
original_source = source_path.read_text()
old = "        _candidates.append(_match.group(1).strip())\n\n    _lines ="
new = "        if not re.match(r\"(?<!#)####(?!#)\", _match.group(1).strip()):\n            _candidates.append(_match.group(1).strip())\n\n    _lines ="
assert original_source.count(old) == 1
patched_source = original_source.replace(old, new)
(destination / "verify_answers.original.py").write_text(original_source)
(destination / "verify_answers.nested_marker_only.py").write_text(patched_source)

fixtures = [
    {"text": "Answer: #### 5", "answer": "5"},
    {"text": "Final answer: #### 6", "answer": "5"},
    {"text": "Answer: #### 5\n#### 6", "answer": "5"},
    {"text": "Answer: #### 5 apples", "answer": "5"},
    {"text": "The answer is: 5\n#### 5", "answer": "5"},
    {"text": "Answer: 5\n#### 5", "answer": "5"},
]
fixture_namespace = {"SCORE_INPUT": fixtures}
exec(compile(patched_source, "diagnostic_nested_marker_verifier", "exec"), fixture_namespace)
assert [row["status"] for row in fixture_namespace["SCORE_OUTPUT"]] == [
    "correct", "incorrect", "ambiguous", "extraction_failed", "extraction_failed", "correct"]

records = {}
summary = {"scope": "Post-hoc diagnostic only, not a replacement of the original primary analysis",
    "rule": "Skip only a redundant explicit-answer candidate beginning with literal ####; the existing #### extractor still parses and validates it, including malformed or contradictory candidates",
    "gold_used_to_define_rule": False,
    "source_verifier_sha256": hashlib.sha256(original_source.encode()).hexdigest(),
    "diagnostic_verifier_sha256": hashlib.sha256(patched_source.encode()).hexdigest(),
    "training_reward_unchanged": True, "original_files_unchanged": True,
    "trained_policy_was_optimized_with_original_verifier": True,
    "fixture_checks": len(fixtures), "models": {}}
for model in ("base", "rl"):
    path = archive / f"outputs/v1_2/test_{model}/samples.jsonl"
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    records[model] = rows
    original_namespace = {"SCORE_INPUT": rows}
    exec(compile(original_source, str(source_path), "exec"), original_namespace)
    for row, scored in zip(rows, original_namespace["SCORE_OUTPUT"]):
        assert all(row[key] == scored[key] for key in scored)
    namespace = {"SCORE_INPUT": rows}
    exec(compile(patched_source, "diagnostic_nested_marker_verifier", "exec"), namespace)
    scores = namespace["SCORE_OUTPUT"]
    diagnostic_rows = []
    changes = []
    transitions = Counter()
    question_scores = {}
    for row, score in zip(rows, scores):
        transitions[f"{row['status']}->{score['status']}"] += 1
        diagnostic_rows.append({**row, **score,
            "primary_score": {key: row[key] for key in score},
            "diagnostic_rule": "duplicate_explicit_hash_marker_only"})
        question_scores.setdefault(row["id"], []).append(score["correct"])
        if (row["status"], row["correct"]) != (score["status"], score["correct"]):
            changes.append({"id": row["id"], "sample_index": row["sample_index"],
                "primary_status": row["status"], "diagnostic_status": score["status"],
                "diagnostic_correct": score["correct"], "answer": row["answer"], "text": row["text"]})
    assert len(rows) == 3200 and len(question_scores) == 200
    assert all(len(values) == 16 for values in question_scores.values())
    metrics = {}
    for k in (1, 2, 4, 8, 16):
        values = []
        for question in sorted(question_scores):
            correct = sum(question_scores[question])
            values.append(1.0 - (comb(16 - correct, k) / comb(16, k) if 16 - correct >= k else 0.0))
        metrics[str(k)] = sum(values) / len(values)
    (destination / f"{model}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in diagnostic_rows))
    (destination / f"{model}.changed.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in changes))
    summary["models"][model] = {"source_samples_sha256": hashlib.sha256(raw).hexdigest(),
        "sample_count": len(rows), "question_count": len(question_scores),
        "primary_correct": sum(row["correct"] for row in rows),
        "diagnostic_correct": sum(row["correct"] for row in scores),
        "changed_status_count": len(changes), "status_transitions": dict(transitions),
        "pass_at_k": metrics}
summary["delta_pass_at_k"] = {key: summary["models"]["rl"]["pass_at_k"][key] - summary["models"]["base"]["pass_at_k"][key] for key in metrics}
summary["delta1_minus_delta16"] = summary["delta_pass_at_k"]["1"] - summary["delta_pass_at_k"]["16"]
(destination / "score_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

rng = random.Random(20261031)
selected = []
for model, count in (("base", 24), ("rl", 12)):
    pool = [row for row in records[model] if row["status"] == "extraction_failed"]
    for row in rng.sample(pool, count):
        selected.append({"review_group": f"{model}_extraction_failed", "model": model, **row})
by_key = {model: {(row["id"], row["sample_index"]): row for row in rows} for model, rows in records.items()}
for key in rng.sample(sorted(by_key["base"]), 8):
    for model in ("base", "rl"):
        selected.append({"review_group": "random_paired_outputs", "model": model, **by_key[model][key]})
(review_root / "answer_quality_review.sample.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
print(json.dumps(summary, indent=2))
