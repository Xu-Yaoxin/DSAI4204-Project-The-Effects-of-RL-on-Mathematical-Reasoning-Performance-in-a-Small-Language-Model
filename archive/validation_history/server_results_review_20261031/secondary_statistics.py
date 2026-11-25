"""Paired bootstrap for a separately labelled post-hoc scoring sensitivity audit."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

out = Path(__file__).resolve().parent
directory = out / "nested_marker_diagnostic"
data = {}
sources = {}
for label in ("base", "rl"):
    path = directory / f"{label}.jsonl"
    raw = path.read_bytes()
    sources[label] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == 3200
    data[label] = {}
    for row in rows:
        assert type(row["correct"]) is bool
        item = data[label].setdefault(row["id"], {})
        assert row["sample_index"] not in item
        item[row["sample_index"]] = row
    assert len(data[label]) == 200
    assert all(set(group) == set(range(16)) for group in data[label].values())
assert data["base"].keys() == data["rl"].keys()
ks = [1, 2, 4, 8, 16]
questions = []
coverage = Counter({"both": 0, "base_only": 0, "rl_only": 0, "neither": 0})
for identifier in sorted(data["base"]):
    item = {"id": identifier}
    for label in ("base", "rl"):
        group = data[label][identifier]
        count = sum(row["correct"] for row in group.values())
        item[f"{label}_count"] = count
        for k in ks:
            item[f"{label}_{k}"] = 1.0 if 16 - count < k else 1 - math.comb(16 - count, k) / math.comb(16, k)
    for k in ks:
        item[f"delta_{k}"] = item[f"rl_{k}"] - item[f"base_{k}"]
    item["delta1_minus_delta16"] = item["delta_1"] - item["delta_16"]
    b, r = item["base_count"] > 0, item["rl_count"] > 0
    coverage["both" if b and r else "base_only" if b else "rl_only" if r else "neither"] += 1
    questions.append(item)
keys = [f"{label}_{k}" for label in ("base", "rl", "delta") for k in ks] + ["delta1_minus_delta16"]
point = {key: statistics.fmean(row[key] for row in questions) for key in keys}
rng = random.Random(42)
draws = {key: [] for key in keys}
for iteration in range(2000):
    sample = rng.choices(questions, k=200)
    for key in keys:
        draws[key].append(statistics.fmean(row[key] for row in sample))
intervals = {}
for key in keys:
    ordered = sorted(draws[key])
    bounds = []
    for q in (0.025, 0.975):
        position = q * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        bounds.append(ordered[low] + (ordered[high] - ordered[low]) * (position - low))
    intervals[key] = {"estimate": point[key], "ci_low": bounds[0], "ci_high": bounds[1]}
summary = {"status": "completed", "analysis_type": "post_hoc_nested_marker_scoring_sensitivity",
    "primary_results_replaced": False, "training_repeated": False, "checkpoint_reselected": False,
    "scope": "Same saved generated text and originally frozen checkpoint; both models rescored by the same narrowly changed verifier.",
    "sample_count_per_model": 3200, "question_count": 200, "n_samples": 16,
    "bootstrap": {"replicates": 2000, "seed": 42, "unit": "question", "method": "paired percentile, linear quantiles"},
    "metrics": intervals, "coverage": dict(coverage), "sources": sources,
    "caveats": ["Post-hoc diagnostic, not the originally specified primary endpoint.",
        "Original training and development checkpoint selection used the original verifier.",
        "This narrow parser change does not resolve all semantically valid answer formats or verify reasoning.",
        "Intervals describe question uncertainty for one fixed training run, not training-seed uncertainty."]}
(out / "secondary_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
(out / "secondary_per_question.jsonl").write_text("".join(json.dumps(row) + "\n" for row in questions))
print(json.dumps({"metrics": {key: intervals[key] for key in ["base_1", "rl_1", "delta_1", "base_16", "rl_16", "delta_16", "delta1_minus_delta16"]}, "coverage": dict(coverage)}, indent=2))
