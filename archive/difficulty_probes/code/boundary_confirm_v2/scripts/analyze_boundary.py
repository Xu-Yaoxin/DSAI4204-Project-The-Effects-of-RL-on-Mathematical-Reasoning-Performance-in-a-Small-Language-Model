"""Analyze a complete baseline-only difficulty calibration without loading a model."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import runpy
import statistics


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=Path("outputs/confirmation_v2/base"))
parser.add_argument("--output", type=Path, default=Path("outputs/confirmation_v2/analysis"))
parser.add_argument("--bootstrap", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
if args.bootstrap < 100:
    parser.error("--bootstrap must be at least 100")
source = (root / args.input).absolute()
output = (root / args.output).absolute()
for path in (source, output):
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise SystemExit(f"Symlink paths are not accepted: {path}")
source, output = source.resolve(), output.resolve()
if source == output or source in output.parents or output in source.parents or output == root or output in root.parents:
    raise SystemExit("Analysis output must be separate from its input and the code root")
if root in output.parents and root / "outputs" not in output.parents:
    raise SystemExit("Analysis outputs inside code must be under outputs/")
source_paths = [source / "manifest.json", source / "samples.jsonl"]
if any(path.is_symlink() or not path.is_file() for path in source_paths):
    raise SystemExit("A completed manifest.json and samples.jsonl are required")
manifest_bytes, sample_bytes = [path.read_bytes() for path in source_paths]
manifest = json.loads(manifest_bytes)
if (manifest.get("schema_version") != 1 or manifest.get("experiment_type") != "exploratory_baseline_calibration"
        or manifest.get("model_label") != "base" or manifest.get("adapter_identity") is not None
        or manifest.get("adapter_path") is not None or manifest.get("complete") is not True
        or type(manifest.get("synthetic")) is not bool):
    raise SystemExit("Only complete, explicitly labeled baseline-only calibration runs can be analyzed")
if not isinstance(manifest.get("model_identity"), str) or not manifest["model_identity"].strip():
    raise SystemExit("A recorded baseline model identity is required")
n_samples = manifest.get("n_samples")
if type(n_samples) is not int or not 1 <= n_samples <= 100:
    raise SystemExit("Invalid fixed sample count")
if hashlib.sha256(sample_bytes).hexdigest() != manifest.get("samples_sha256"):
    raise SystemExit("Saved samples differ from their completed content hash")
settings = manifest.get("settings")
if not isinstance(settings, dict) or any(type(settings.get(key)) is not int for key in ("max_new_tokens", "max_prompt_length", "seed")):
    raise SystemExit("Missing generation limits or seed")
if settings["max_new_tokens"] < 1 or settings["max_prompt_length"] < 1 or settings["seed"] < 0:
    raise SystemExit("Invalid generation limits or seed")
code_hashes = manifest.get("code_hashes")
if not isinstance(code_hashes, dict) or "verify_answers.py" not in code_hashes:
    raise SystemExit("Original scorer identity is required")
verifier_path = root / "scripts/verify_answers.py"
verifier_hash = hashlib.sha256(verifier_path.read_bytes()).hexdigest()
if code_hashes["verify_answers.py"] != verifier_hash:
    raise SystemExit("The scorer changed since this calibration; preserve results and use a separate rescoring protocol")
for filename, expected in code_hashes.items():
    if Path(filename).name != filename or not isinstance(expected, str) or len(expected) != 64:
        raise SystemExit("Invalid recorded code identity")
if (settings.get("context_limit") != 262144 or settings["max_new_tokens"] != 262144
        or settings.get("length_policy") != "native_context_minus_actual_prompt"):
    raise ValueError("Invalid native-context confirmation protocol")
protocol = {key: manifest[key] for key in ("dataset_sha256", "dataset_manifest_sha256", "n_samples", "settings", "code_hashes", "experiment_type")}
if hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() != manifest.get("protocol_id"):
    raise SystemExit("Generation protocol does not match its recorded components")
for key, digest_key in (("dataset_path", "dataset_sha256"), ("dataset_manifest_path", "dataset_manifest_sha256")):
    relative = Path(manifest[key])
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit("Recorded dataset paths must stay relative to the code root")
    path = root / relative
    if any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file():
        raise SystemExit(f"Unsafe or missing dataset input: {path}")
    source_paths.append(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[digest_key]:
        raise SystemExit(f"Dataset content changed: {key}")
dataset = [json.loads(line) for line in source_paths[2].read_text(encoding="utf-8").split("\n") if line.strip()]
dataset_manifest = json.loads(source_paths[3].read_text(encoding="utf-8"))
if not dataset or len({row["id"] for row in dataset}) != len(dataset):
    raise SystemExit("Dataset must have unique nonempty question identities")
declared_data = dataset_manifest.get("files", {}).get(source_paths[2].name, {})
if declared_data.get("sha256") != manifest["dataset_sha256"] or declared_data.get("rows") != len(dataset):
    raise SystemExit("Prepared manifest does not bind the evaluated dataset bytes and row count")
data_by_id = {row["id"]: row for row in dataset}
group_specs = dataset_manifest.get("groups")
if not isinstance(group_specs, list) or not group_specs or len({item["id"] for item in group_specs}) != len(group_specs):
    raise SystemExit("Prepared dataset must declare distinct calibration groups")
group_specs = {item["id"]: item for item in group_specs}
actual_group_counts = Counter(row["group_id"] for row in dataset)
if set(actual_group_counts) != set(group_specs) or any(group_specs[key].get("count") != count for key, count in actual_group_counts.items()):
    raise SystemExit("Prepared group counts do not match the dataset")
if type(manifest.get("question_count")) is not int or manifest["question_count"] != len(dataset):
    raise SystemExit("Recorded question count differs from prepared data")
rows = [json.loads(line) for line in sample_bytes.decode("utf-8").split("\n") if line.strip()]
if type(manifest.get("total_samples")) is not int or manifest["total_samples"] != len(dataset) * n_samples or len(rows) != manifest["total_samples"]:
    raise SystemExit("Saved sample count does not match the complete fixed grid")
score_fields = ("correct", "reward", "extracted", "status", "format_compliant", "truncated")
records = {}
for row in rows:
    if (not isinstance(row, dict) or row.get("id") not in data_by_id or type(row.get("sample_index")) is not int
            or not 0 <= row["sample_index"] < n_samples):
        raise SystemExit("Unexpected question identity or sample index")
    original = data_by_id[row["id"]]
    if any(row.get(key) != value for key, value in original.items()):
        raise SystemExit(f"Question or source metadata changed: {row['id']}")
    key = (row["id"], row["sample_index"])
    if key in records:
        raise SystemExit("Duplicate saved sample")
    if (any(type(row.get(field)) is not bool for field in ("correct", "format_compliant", "truncated"))
            or type(row.get("reward")) not in (int, float) or not math.isfinite(row["reward"])
            or not isinstance(row.get("text"), str)
            or row.get("status") not in {"correct", "incorrect", "extraction_failed", "ambiguous"}):
        raise SystemExit("Invalid scoring record; infrastructure errors and invalid gold cannot be counted as wrong answers")
    for field, limit in (("output_tokens", settings["max_new_tokens"]), ("prompt_tokens", settings["max_prompt_length"])):
        if type(row.get(field)) is not int or not (1 if field == "prompt_tokens" else 0) <= row[field] <= limit:
            raise SystemExit("Invalid generation token counts")
    if (type(row.get("generation_limit")) is not int
            or type(row.get("prompt_tokens")) is not int
            or row["generation_limit"] != settings["context_limit"] - row["prompt_tokens"]
            or type(row.get("output_tokens")) is not int or row["output_tokens"] > row["generation_limit"]):
        raise ValueError("Invalid per-answer native context budget")
    expected_seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{row['id']}:{row['sample_index']}".encode()).digest()[:4], "big")
    if type(row.get("sample_seed")) is not int or row["sample_seed"] != expected_seed:
        raise SystemExit("Sample seed differs from its deterministic question/sample identity")
    if row["truncated"] and row["output_tokens"] != row.get("generation_limit"):
        raise SystemExit("Truncation flag contradicts the token count")
    records[key] = row
expected_grid = {(question_id, index) for question_id in data_by_id for index in range(n_samples)}
if set(records) != expected_grid:
    raise SystemExit("Saved answers do not form the complete question-by-sample grid")
replayed = runpy.run_path(str(verifier_path), init_globals={"SCORE_INPUT": rows})["SCORE_OUTPUT"]
if len(replayed) != len(rows) or any(any(row.get(field) != score[field] for field in score_fields) for row, score in zip(rows, replayed)):
    raise SystemExit("Saved scoring fields differ from the original scorer replay")

identity = {"input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "samples_sha256": manifest["samples_sha256"],
            "dataset_sha256": manifest["dataset_sha256"], "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "verifier_sha256": verifier_hash,
            "bootstrap": args.bootstrap, "seed": args.seed}
if output.exists():
    marker = output / "ready.json"
    if marker.is_symlink() or not marker.is_file():
        raise SystemExit("Analysis output is partial; preserve it and choose a new --output")
    ready = json.loads(marker.read_text())
    if ready.get("complete") is not True or ready.get("identity") != identity:
        raise SystemExit("Existing analysis belongs to different inputs, code or options; choose a new --output")
    inventory = {}
    for path in output.rglob("*"):
        if path.is_symlink():
            raise SystemExit("Analysis output contains a symlink")
        if path.is_file() and path != marker:
            inventory[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if inventory != ready.get("artifacts_sha256"):
        raise SystemExit("Completed analysis artifacts were changed")
    print(f"✅ [BOUNDARY ANALYSIS CACHED] 100% | {output}")
    raise SystemExit(0)

print(f"[BOUNDARY ANALYSIS 1/3] 33% | validated {len(dataset)} questions / {len(rows)} answers | baseline only", flush=True)
per_question = []
for question_id in sorted(data_by_id):
    original = data_by_id[question_id]
    question_rows = [records[(question_id, index)] for index in range(n_samples)]
    correct_count = sum(row["correct"] for row in question_rows)
    per_question.append({"id": question_id, "group_id": original["group_id"], "family_id": original["family_id"],
        "correct_count": correct_count, "n_samples": n_samples, "pass_at_1": correct_count / n_samples,
        "pass_at_4": (1 - math.comb(n_samples - correct_count, 4) / math.comb(n_samples, 4)) if n_samples >= 4 else None,
        "status_counts": dict(Counter(row["status"] for row in question_rows)),
        "truncated_count": sum(row["truncated"] for row in question_rows)})
groups = []
for group_id, spec in group_specs.items():
    questions = [row for row in per_question if row["group_id"] == group_id]
    question_ids = {row["id"] for row in questions}
    group_rows = [row for row in rows if row["id"] in question_ids]
    families = {}
    for question in questions:
        if not isinstance(question["family_id"], str) or not question["family_id"].strip():
            raise SystemExit("A family identity is required for honest uncertainty estimates")
        families.setdefault(question["family_id"], []).append(question["pass_at_1"])
    point = statistics.fmean(question["pass_at_1"] for question in questions)
    unit_count = len(families)
    if n_samples == 1 and unit_count == len(questions):
        z = 1.959963984540054
        denominator = 1 + z * z / len(questions)
        center = (point + z * z / (2 * len(questions))) / denominator
        half_width = z * math.sqrt(point * (1 - point) / len(questions) + z * z / (4 * len(questions) ** 2)) / denominator
        low, high = max(0.0, center - half_width), min(1.0, center + half_width)
        method = "Wilson score interval, 95%, one draw per distinct question/family"
    else:
        group_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{group_id}".encode()).digest()[:8], "big")
        rng = random.Random(group_seed)
        units = [families[key] for key in sorted(families)]
        draws = []
        for _ in range(args.bootstrap):
            selected = rng.choices(units, k=unit_count)
            draws.append(sum(sum(unit) for unit in selected) / sum(len(unit) for unit in selected))
        ordered = sorted(draws)
        bounds = []
        for probability in (0.025, 0.975):
            position = probability * (len(ordered) - 1)
            lower, upper = math.floor(position), math.ceil(position)
            bounds.append(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))
        low, high = bounds
        method = "Family-cluster percentile bootstrap, 95%, linear quantiles"
    status_counts = dict(Counter(row["status"] for row in group_rows))
    truncation_rate = statistics.fmean(row["truncated"] for row in group_rows)
    groups.append({"group_id": group_id, "group_label": spec["label"], "question_count": len(questions),
        "family_count": unit_count, "sample_count": len(group_rows), "n_samples": n_samples,
        "pass_at_1": {"estimate": point, "ci_low": low, "ci_high": high, "interval_method": method},
        "degenerate_interval": low == high,
        "pass_at_4": statistics.fmean(question["pass_at_4"] for question in questions) if n_samples >= 4 else None,
        "point_in_target": 0.4 <= point <= 0.5, "ci_entirely_in_target": 0.4 <= low <= high <= 0.5,
        "ci_overlaps_target": low <= 0.5 and high >= 0.4, "distance_to_target_center": abs(point - 0.45),
        "low_precision": len(questions) < 100 or unit_count < 100, "status_counts": status_counts,
        "format_compliance_rate": statistics.fmean(row["format_compliant"] for row in group_rows),
        "extraction_failure_rate": status_counts.get("extraction_failed", 0) / len(group_rows),
        "ambiguous_rate": status_counts.get("ambiguous", 0) / len(group_rows), "truncation_rate": truncation_rate,
        "length_budget_concern": truncation_rate >= 0.1,
        "mean_output_tokens": statistics.fmean(row["output_tokens"] for row in group_rows)})
ranking = sorted(groups, key=lambda item: (item["distance_to_target_center"], item["group_id"]))
synthetic = manifest["synthetic"]
summary = {"schema_version": 1, "experiment_type": "exploratory_baseline_calibration", "synthetic": synthetic,
    "model_label": "base", "adapter_identity": None, "model_identity": manifest.get("model_identity"),
    "protocol_id": manifest["protocol_id"], "settings": settings, "provenance": identity,
    "question_count": len(dataset), "total_samples": len(rows), "n_samples": n_samples,
    "target": {"metric": "pass_at_1", "lower": 0.4, "upper": 0.5, "center": 0.45}, "groups": groups,
    "limitations": ["SYNTHETIC FIXTURE: no measured model capability." if synthetic else "Exploratory difficulty screening; no claim that a selected group is confirmed to have 40-50% success.",
        "Repeated answers and instances from the same template family are dependent. Intervals resample families and are marginal, not simultaneous.",
        "A degenerate bootstrap interval reflects zero empirical variation in the collected question scores, not certainty about future performance.",
        "The target applies to pass@1, not success within four attempts. No individual questions are selected using model outcomes.",
        "Dataset/level labels are source-defined strata, not a demonstrated total ordering of difficulty for this model.",
        "MATH rows are text-only scalar-answer subsets, not full MATH benchmark scores. Final-value correctness does not establish reasoning validity.",
        "Review extraction failures and truncation before interpreting low scores as model inability. Results are conditional on the recorded generation budget.",
        "Choose future source strata using baseline calibration, then evaluate Base/RL on fresh questions. For symbolic variants, keep entire original-template families out of confirmation data."]}
candidates = {"schema_version": 1, "kind": "exploratory_group_ranking_only", "synthetic": synthetic,
    "metric": "pass_at_1", "target_range": [0.4, 0.5], "automatic_selection_performed": False,
    "fresh_confirmation_questions_required": True, "exclude_calibration_families_from_confirmation": True,
    "ranked_groups": [{key: item[key] for key in ("group_id", "group_label", "pass_at_1", "point_in_target", "ci_entirely_in_target", "ci_overlaps_target", "low_precision", "length_budget_concern")} for item in ranking]}
report = ["# Baseline difficulty calibration / 基线难度摸底", "",
    "**SYNTHETIC FIXTURE — no model performance was measured.**" if synthetic else "**Exploratory baseline-only screening. These are not formal held-out comparison results.**", "",
    f"Questions: {len(dataset)}. Samples per question: {n_samples}. Saved answers: {len(rows)}.",
    "Target: 40–50% single-attempt success (pass@1). Dataset labels describe sources and levels; measured difficulty is reported below.", "",
    "| Group | Questions | pass@1 | 95% interval | pass@4 | Target point? | Truncated | Extraction failure |",
    "|---|---:|---:|---:|---:|---|---:|---:|"]
for item in groups:
    metric = item["pass_at_1"]
    pass4 = f"{item['pass_at_4']:.1%}" if item["pass_at_4"] is not None else "not collected"
    report.append(f"| {item['group_label']} | {item['question_count']} | {metric['estimate']:.1%} | [{metric['ci_low']:.1%}, {metric['ci_high']:.1%}] | {pass4} | {'yes' if item['point_in_target'] else 'no'} | {item['truncation_rate']:.1%} | {item['extraction_failure_rate']:.1%} |")
report += ["", "## Group ranking / 题组排序", "",
    "Groups are ranked by distance of pass@1 from 45%. A point inside 40–50% is a candidate for fresh confirmation, not an established ability boundary. No questions or models are automatically selected.", ""]
for index, item in enumerate(ranking, 1):
    report.append(f"{index}. {item['group_label']}: pass@1 {item['pass_at_1']['estimate']:.1%}; interval wholly in target: {item['ci_entirely_in_target']}; length-budget concern: {item['length_budget_concern']}.")
report += ["", "## Interpretation limits / 解释边界", ""] + [f"- {item}" for item in summary["limitations"]]
report += ["", "With one response per question, intervals use Wilson when families are unique. With repeated responses or shared families, intervals bootstrap whole families, retaining all responses within each question. Twenty families give coarse uncertainty; more draws from the same questions cannot replace fresh independent questions.",
    "", "`summary.json` retains interval methods, diagnostics and protocol identities. `per_question.jsonl` retains all question scores, without selecting favorable examples. `calibration_candidates.json` ranks only source groups for a subsequent independent confirmation.", ""]

print("[BOUNDARY ANALYSIS 2/3] 67% | computed group metrics and uncertainty | no automatic selection", flush=True)
for path, expected in zip(source_paths, (identity["input_manifest_sha256"], identity["samples_sha256"], identity["dataset_sha256"], identity["dataset_manifest_sha256"])):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Input changed during analysis: {path}")
if hashlib.sha256(verifier_path.read_bytes()).hexdigest() != verifier_hash or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != identity["analyzer_sha256"]:
    raise SystemExit("Analysis or scorer code changed during analysis")
output.mkdir(parents=True, exist_ok=False)
for filename, value in (("summary.json", summary), ("calibration_candidates.json", candidates)):
    (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
(output / "per_question.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in per_question), encoding="utf-8")
(output / "report.md").write_text("\n".join(report), encoding="utf-8")
artifacts = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir()) if path.is_file()}
ready = {"complete": True, "identity": identity, "artifacts_sha256": artifacts, "models_loaded": 0, "network_required": False}
(output / "ready.json.tmp").write_text(json.dumps(ready, indent=2) + "\n")
(output / "ready.json.tmp").replace(output / "ready.json")
print(f"✅ [BOUNDARY ANALYSIS 3/3] 100% | [SAVED] {output / 'report.md'}", flush=True)
