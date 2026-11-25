"""Prepare 100 provenance-locked baseline calibration questions without model calls."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import runpy
import ssl
import urllib.request

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("data/boundary_v1"))
parser.add_argument("--offline", action="store_true")
parser.add_argument("--verify", action="store_true")
args = parser.parse_args()
output = (root / args.output).resolve()
if output == root or output in root.parents:
    raise SystemExit("Use a dedicated dataset directory")
seed = 20261031
apple_revision = "93b5b3758d9d9841ffe81d6cd2ae2b030685b078"
math_revision = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
source_specs = [
    {"dataset": "apple/GSM-Symbolic", "revision": apple_revision, "remote": "main/test.jsonl", "local": "gsm_symbolic/main.jsonl", "git_blob_sha1": "8644707d8edb3179cc53876357b4d0f847f5335f", "license": "CC-BY-NC-ND-4.0"},
    {"dataset": "apple/GSM-Symbolic", "revision": apple_revision, "remote": "p1/test.jsonl", "local": "gsm_symbolic/p1.jsonl", "git_blob_sha1": "97d3ee573a68f5ca1e761f32994b5e1315910678", "license": "CC-BY-NC-ND-4.0"},
    {"dataset": "apple/GSM-Symbolic", "revision": apple_revision, "remote": "p2/test.jsonl", "local": "gsm_symbolic/p2.jsonl", "git_blob_sha1": "b5cdd79707018991723e255e9232015623b26521", "license": "CC-BY-NC-ND-4.0"},
    {"dataset": "apple/GSM-Symbolic", "revision": apple_revision, "remote": "README.md", "local": "gsm_symbolic/README.md", "git_blob_sha1": "f5a7fa6b1fe8c4f4da817c60edbc5e9364e8e628", "license": "CC-BY-NC-ND-4.0"},
    {"dataset": "apple/GSM-Symbolic", "revision": apple_revision, "remote": "LICENSE-CC-BY-NC-ND-4.0.md", "local": "gsm_symbolic/LICENSE.md", "git_blob_sha1": "eece3f3d351c27555403d89a3700b68c56bfffd6", "license": "CC-BY-NC-ND-4.0"},
    {"dataset": "HuggingFaceH4/MATH-500", "revision": math_revision, "remote": "test.jsonl", "local": "math500/test.jsonl", "git_blob_sha1": "2376b9a194b46c0790e197c91b7249e5f88ac09b", "license": "Original MATH repository: MIT; H4 card does not declare a separate license"},
    {"dataset": "HuggingFaceH4/MATH-500", "revision": math_revision, "remote": "README.md", "local": "math500/README.md", "git_blob_sha1": "7bcd88fdd6d84875d4599f8893aa5ed5938dbca8", "license": "Original MATH repository: MIT; H4 card does not declare a separate license"},
    {"dataset": "hendrycks/math", "revision": "985bdc1696e88e8643f081a0ff4719da39f2ae2a", "remote": "LICENSE", "local": "math500/ORIGINAL_MATH_LICENSE", "git_blob_sha1": None, "license": "MIT", "url": "https://raw.githubusercontent.com/hendrycks/math/985bdc1696e88e8643f081a0ff4719da39f2ae2a/LICENSE"},
]
print("[BOUNDARY DATA 1/4] 0% | PINNED SOURCES | models=0", flush=True)
previous = json.loads((output / "manifest.json").read_text()) if (output / "manifest.json").is_file() else None
if args.verify and previous is None:
    raise SystemExit("No completed data manifest to verify")
sources = []
for index, specification in enumerate(source_specs, 1):
    url = specification.get("url", f"https://huggingface.co/datasets/{specification['dataset']}/resolve/{specification['revision']}/{specification['remote']}")
    target = output / "raw" / specification["local"]
    if any(path.is_symlink() for path in (target, *target.parents)):
        raise SystemExit(f"Unsafe source symlink: {target}")
    if not target.is_file():
        if args.offline or args.verify or previous:
            raise SystemExit(f"Missing pinned source: {target}")
        context = ssl.create_default_context()
        try:
            import certifi
        except ImportError:
            pass
        else:
            context = ssl.create_default_context(cafile=certifi.where())
        print(f"[DOWNLOAD {index}/{len(source_specs)}] {specification['dataset']}/{specification['remote']}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "DSAI4207-baseline-calibration/1"})
        with urllib.request.urlopen(request, context=context, timeout=90) as response:
            content = response.read()
    else:
        content = target.read_bytes()
    blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if specification["git_blob_sha1"] and blob != specification["git_blob_sha1"]:
        raise SystemExit(f"Pinned Git blob mismatch: {target}")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.with_suffix(target.suffix + ".part").write_bytes(content)
        target.with_suffix(target.suffix + ".part").replace(target)
    sources.append({**specification, "url": url, "path": str(target.relative_to(output)), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    print(f"[SOURCE {index}/{len(source_specs)}] verified {len(content):,} bytes", flush=True)

history_hashes = {}
old_ids = set()
old_questions = set()
for split in ("train", "pilot", "engineering", "dev", "test"):
    path = root / "data/processed" / f"{split}.jsonl"
    content = path.read_bytes()
    history_hashes[str(path.relative_to(root))] = hashlib.sha256(content).hexdigest()
    for row in [json.loads(line) for line in content.decode().split("\n") if line.strip()]:
        old_questions.add(" ".join(row["question"].casefold().split()))
        if row.get("source", {}).get("official_split") == "test":
            old_ids.add(row["source"]["row_index"])
        elif str(row.get("id", "")).startswith("gsm8k-test-"):
            old_ids.add(int(row["id"].rsplit("-", 1)[-1]))

print("[BOUNDARY DATA 2/4] 25% | GOLD VALIDATION AND HISTORY EXCLUSION", flush=True)
verifier = root / "scripts/verify_answers.py"
verifier_hash = hashlib.sha256(verifier.read_bytes()).hexdigest()
pools = {}
exclusions = {}
source_counts = {}
for subset in ("main", "p1", "p2", "math500"):
    path = output / "raw" / (f"gsm_symbolic/{subset}.jsonl" if subset != "math500" else "math500/test.jsonl")
    rows = [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
    source_counts[subset] = len(rows)
    records = []
    exclusion_counts = Counter()
    for index, row in enumerate(rows):
        question = row["problem"] if subset == "math500" else row["question"]
        source_answer = row["answer"]
        if subset != "math500" and row["original_id"] in old_ids:
            exclusion_counts["previously_evaluated_gsm8k_family"] += 1
            continue
        if " ".join(question.casefold().split()) in old_questions:
            exclusion_counts["exact_normalized_history_question"] += 1
            continue
        if "[asy]" in question.lower() or "\\includegraphics" in question.lower() or "<img" in question.lower():
            exclusion_counts["requires_diagram"] += 1
            continue
        gold = source_answer.rsplit("####", 1)[-1].strip() if subset != "math500" else source_answer
        records.append({"row_index": index, "raw": row, "question": question, "gold": gold})
    scores = runpy.run_path(str(verifier), init_globals={"SCORE_INPUT": [{"text": f"#### {item['gold']}", "answer": item["gold"]} for item in records]})["SCORE_OUTPUT"]
    usable = []
    for item, score in zip(records, scores):
        if not score["correct"]:
            exclusion_counts["unsupported_non_scalar_gold"] += 1
            continue
        if subset == "math500" and item["raw"].get("level") not in (1, 2, 3, 4, 5):
            exclusion_counts["unsupported_level"] += 1
            continue
        item["canonical_answer"] = score["extracted"]
        usable.append(item)
    pools[subset] = usable
    exclusions[subset] = dict(exclusion_counts)

families = set.intersection(*[{item["raw"]["original_id"] for item in pools[subset]} for subset in ("main", "p1", "p2")])
eligible_families = {}
for original_id in sorted(families):
    instances = set.intersection(*[{item["raw"]["instance"] for item in pools[subset] if item["raw"]["original_id"] == original_id} for subset in ("main", "p1", "p2")])
    if instances:
        eligible_families[original_id] = sorted(instances)
if len(eligible_families) < 20:
    raise SystemExit(f"Only {len(eligible_families)} eligible shared families; need20")
rng = random.Random(seed)
chosen_families = sorted(rng.sample(sorted(eligible_families), 20))
chosen_instances = {original_id: rng.choice(eligible_families[original_id]) for original_id in chosen_families}
groups = [{"id": "gsm_symbolic", "label": "GSM-Symbolic main", "count": 20}, {"id": "gsm_p1", "label": "GSM-Symbolic P1", "count": 20}, {"id": "gsm_p2", "label": "GSM-Symbolic P2", "count": 20}, {"id": "math_l1_3", "label": "MATH-500 scalar Level1-3", "count": 20}, {"id": "math_l4_5", "label": "MATH-500 scalar Level4-5", "count": 20}]
selected = []
for subset, group in zip(("main", "p1", "p2"), groups[:3]):
    for original_id in chosen_families:
        matches = [item for item in pools[subset] if item["raw"]["original_id"] == original_id and item["raw"]["instance"] == chosen_instances[original_id]]
        if len(matches) != 1:
            raise SystemExit("Duplicate source family/instance")
        item = matches[0]
        selected.append({"id": f"{group['id']}-{original_id:04d}-{chosen_instances[original_id]:02d}", "question": item["question"], "answer": item["canonical_answer"], "source_answer": item["raw"]["answer"], "group_id": group["id"], "group_label": group["label"], "family_id": f"gsm8k-test-{original_id:06d}", "difficulty_label": subset, "source": {"dataset": "apple/GSM-Symbolic", "revision": apple_revision, "split": "test", "subset": subset, "row_index": item["row_index"], "original_id": original_id, "instance": chosen_instances[original_id], "canary": item["raw"].get("canary")}})
for level, quota in ((1, 6), (2, 7), (3, 7), (4, 10), (5, 10)):
    candidates = [item for item in pools["math500"] if item["raw"]["level"] == level]
    if len(candidates) < quota:
        raise SystemExit(f"Level{level} only has{len(candidates)} usable scalar problems; need{quota}")
    group = groups[3 if level <= 3 else 4]
    for item in sorted(rng.sample(candidates, quota), key=lambda item: item["row_index"]):
        selected.append({"id": f"math500-{item['row_index']:04d}", "question": item["question"], "answer": item["canonical_answer"], "source_answer": item["raw"]["answer"], "group_id": group["id"], "group_label": group["label"], "family_id": item["raw"]["unique_id"], "difficulty_label": f"Level {level}", "source": {"dataset": "HuggingFaceH4/MATH-500", "revision": math_revision, "split": "test", "row_index": item["row_index"], "original_id": item["raw"]["unique_id"], "level": level, "subject": item["raw"]["subject"]}})
if len(selected) != 100 or len({item["id"] for item in selected}) != 100 or len({" ".join(item["question"].casefold().split()) for item in selected}) != 100:
    raise SystemExit("Selected pool must have100 unique questions and ids")
content = "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in selected).encode()
manifest = {"schema_version": 1, "purpose": "exploratory_baseline_calibration", "seed": seed, "model_scores_used_for_selection": False,
    "files": {"questions.jsonl": {"sha256": hashlib.sha256(content).hexdigest(), "rows": len(selected)}}, "groups": groups, "sources": sources,
    "verifier_sha256": verifier_hash, "history_files_sha256": history_hashes, "source_rows": source_counts, "exclusions": exclusions,
    "eligible_scalar_rows": {key: len(value) for key, value in pools.items()}, "eligible_shared_gsm_families": len(eligible_families),
    "selected_gsm_original_ids": chosen_families, "selected_gsm_instances": {str(key): value for key, value in chosen_instances.items()},
    "math_level_quotas": {"1": 6, "2": 7, "3": 7, "4": 10, "5": 10},
    "limitations": ["100-question exploratory screen, not an independent confirmation set.", "GSM variants share20 templates; do not treat60 rows as60 independent families.", "MATH subset accepts scalar rational gold and excludes diagrams; scores do not represent full MATH-500.", "Fixed random selection does not guarantee40-50% baseline accuracy; generation is required.", "Historical exact-question and known-family exclusion cannot rule out pretraining contamination.", "Raw solutions and canaries are provenance only and must never enter model prompts."]}
print("[BOUNDARY DATA 3/4] 75% | FIVE GROUPS OF20 QUESTIONS", flush=True)
manifest_path = output / "manifest.json"
if previous:
    if previous != manifest or (output / "questions.jsonl").read_bytes() != content:
        raise SystemExit("Existing boundary data or provenance changed; preserve it and use a new output")
else:
    (output / "questions.jsonl").write_bytes(content)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    readable = ["# Baseline boundary screen:100 questions", "", "Exploratory calibration only. Questions are verbatim source text; source_answer and numeric answers remain in questions.jsonl for scoring, never in model prompts.", ""]
    for group in groups:
        readable.extend([f"## {group['label']}", ""])
        for item in selected:
            if item["group_id"] == group["id"]:
                readable.extend([f"### {item['id']}", "", item["question"], ""])
    (output / "questions.md").write_text("\n".join(readable), encoding="utf-8")
    (output / "README.md").write_text("# Baseline calibration data v1\n\n100 verbatim questions in five groups of20. Only baseline inference is intended. Seed20261031; selection did not use model outputs.\n\nGSM-Symbolic main/P1/P2 use the same20 source families and matching instance indices, excluding known previously evaluated GSM8K test families. MATH-500 uses scalar, text-only questions with level quotas6/7/7/10/10. Non-scalar answers and diagrams are excluded before random selection. No40-50% accuracy is promised before measurement.\n\nQuestions: questions.md. Exact scoring gold/source answers: questions.jsonl. Provenance, exclusions, hashes and limitations: manifest.json. Original files and source licenses: raw/.\n\nSources: https://huggingface.co/datasets/apple/GSM-Symbolic (CC-BY-NC-ND-4.0); https://huggingface.co/datasets/HuggingFaceH4/MATH-500 (derived from OpenAI PRM800K MATH split; original MATH repository MIT license included separately). The MATH source card does not declare a separate license. Questions are preserved unchanged; only metadata and gold canonicalization are added.\n\nRun from code: python3 scripts/prepare_boundary_data.py --verify. Verification is offline and reruns deterministic selection. Do not pass source solutions, gold answers or canary metadata into the model prompt.\n", encoding="utf-8")
print(f"✅ [BOUNDARY DATA 4/4] 100% | {'VERIFIED' if previous else 'SAVED'} |100questions | {output / 'questions.jsonl'}", flush=True)
