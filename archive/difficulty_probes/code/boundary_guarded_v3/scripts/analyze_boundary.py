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
parser.add_argument("--input", type=Path, default=Path("outputs/guarded_v3_serial/base"))
parser.add_argument("--output", type=Path, default=Path("outputs/guarded_v3_serial/analysis"))
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
if manifest.get("path_base") != "code_root" or not isinstance(manifest.get("model_files"), dict) or not manifest["model_files"]:
    raise SystemExit("Recorded model inventory and code-root-relative paths are required")
if hashlib.sha256(json.dumps(manifest["model_files"], sort_keys=True).encode()).hexdigest() != manifest["model_identity"]:
    raise SystemExit("Recorded model inventory identity is invalid")
expected_model_identity = manifest.get("config", {}).get("model", {}).get("expected_identity")
if expected_model_identity is not None and expected_model_identity != manifest["model_identity"]:
    raise SystemExit("Baseline identity differs from the configured original model")
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
guard_paths = [root / "scripts" / name for name in ("validate_guarded_sample.py", "repetition_guard.py")]
guard_hashes = {}
for path in guard_paths:
    if path.is_symlink() or not path.is_file():
        raise SystemExit("Missing or unsafe guarded scoring helper")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if code_hashes.get(path.name) != digest:
        raise SystemExit("Guard/helper identity changed; preserve results under the original protocol")
    guard_hashes[path.name] = digest
validator_path = guard_paths[0]
if not isinstance(settings.get("generation_guard"), dict) or type(settings.get("eos_token_id")) is not int:
    raise SystemExit("Guarded generation settings and EOS identity are required")
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
    validated = runpy.run_path(str(validator_path), init_globals={"SAMPLE": row, "SETTINGS": settings})
    if validated.get("VALIDATED") is not True:
        raise SystemExit("Guarded answer validation did not complete")
    if type(row.get("seconds")) not in (int, float) or not math.isfinite(row["seconds"]) or row["seconds"] < 0:
        raise SystemExit("Invalid saved duration")
    expected_seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{row['id']}:{row['sample_index']}".encode()).digest()[:4], "big")
    if type(row.get("sample_seed")) is not int or row["sample_seed"] != expected_seed:
        raise SystemExit("Sample seed differs from its deterministic question/sample identity")
    if row["truncated"] and row["output_tokens"] != row.get("generation_limit"):
        raise SystemExit("Truncation flag contradicts the token count")
    records[key] = row
expected_grid = {(question_id, index) for question_id in data_by_id for index in range(n_samples)}
if set(records) != expected_grid:
    raise SystemExit("Saved answers do not form the complete question-by-sample grid")

identity = {"input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "samples_sha256": manifest["samples_sha256"],
            "dataset_sha256": manifest["dataset_sha256"], "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "verifier_sha256": verifier_hash, "guard_helpers_sha256": guard_hashes,
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

print(f"[BOUNDARY ANALYSIS 1/3] 33% | validated {len(dataset)} questions / {len(rows)} terminal attempts | baseline only", flush=True)
per_question = []
for question_id in sorted(data_by_id):
    original = data_by_id[question_id]
    question_rows = [records[(question_id, index)] for index in range(n_samples)]
    correct_count = sum(row["correct"] is True for row in question_rows)
    evaluable_count = sum(row["scoring_evaluable"] for row in question_rows)
    timeout_count = sum(row["termination_reason"] == "answer_timeout" for row in question_rows)
    unknown_count = n_samples - evaluable_count - timeout_count
    lower, upper = correct_count / n_samples, (correct_count + unknown_count) / n_samples
    pass4_lower = (1 - math.comb(n_samples - correct_count, 4) / math.comb(n_samples, 4)) if n_samples >= 4 else None
    pass4_upper = (1 - math.comb(n_samples - correct_count - unknown_count, 4) / math.comb(n_samples, 4)) if n_samples >= 4 else None
    per_question.append({"id": question_id, "group_id": original["group_id"], "family_id": original["family_id"],
        "correct_count": correct_count, "evaluable_count": evaluable_count, "completed_count": evaluable_count,
        "unknown_count": unknown_count, "policy_failure_count": timeout_count, "timeout_count": timeout_count, "timeout_rate": timeout_count / n_samples, "n_samples": n_samples, "completion_rate": evaluable_count / n_samples,
        "completed_only_accuracy": correct_count / evaluable_count if evaluable_count else None,
        "observed_success_per_request": lower, "incomplete_answer_bounds": {"lower": lower, "upper": upper, "is_confidence_interval": False},
        "pass_at_1": lower if unknown_count == 0 else None,
        "pass_at_4": pass4_lower if unknown_count == 0 else None,
        "pass_at_4_bounds": {"lower": pass4_lower, "upper": pass4_upper, "is_confidence_interval": False} if n_samples >= 4 else None,
        "status_counts": dict(Counter(row["status"] for row in question_rows)),
        "termination_counts": dict(Counter(row["termination_reason"] for row in question_rows)),
        "truncated_count": sum(row["truncated"] for row in question_rows)})
groups = []
for group_id, spec in group_specs.items():
    questions = [row for row in per_question if row["group_id"] == group_id]
    question_ids = {row["id"] for row in questions}
    group_rows = [row for row in rows if row["id"] in question_ids]
    evaluable_rows = [row for row in group_rows if row["scoring_evaluable"]]
    correct_count = sum(row["correct"] is True for row in group_rows)
    evaluable_count = len(evaluable_rows)
    timeout_count = sum(row["termination_reason"] == "answer_timeout" for row in group_rows)
    unknown_count = len(group_rows) - evaluable_count - timeout_count
    families = {}
    for question in questions:
        if not isinstance(question["family_id"], str) or not question["family_id"].strip():
            raise SystemExit("A family identity is required for uncertainty estimates")
        families.setdefault(question["family_id"], []).append(question["observed_success_per_request"])
    point = statistics.fmean(question["observed_success_per_request"] for question in questions)
    upper_bound = statistics.fmean(question["incomplete_answer_bounds"]["upper"] for question in questions)
    unit_count = len(families)
    if n_samples == 1 and unit_count == len(questions):
        z = 1.959963984540054
        denominator = 1 + z * z / len(questions)
        center = (point + z * z / (2 * len(questions))) / denominator
        half_width = z * math.sqrt(point * (1 - point) / len(questions) + z * z / (4 * len(questions) ** 2)) / denominator
        low, high = max(0.0, center - half_width), min(1.0, center + half_width)
        method = "Wilson score interval, 95%, observed successful request, one draw per distinct question/family"
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
        method = "Family-cluster percentile bootstrap, 95%, observed successful request, linear quantiles"
    observed = {"estimate": point, "ci_low": low, "ci_high": high, "interval_method": method,
        "metric": "completed_correct_answers_per_requested_attempt", "aborted_answers_are_mathematical_errors": False}
    status_counts = dict(Counter(row["status"] for row in group_rows))
    truncation_rate = statistics.fmean(row["truncated"] for row in group_rows)
    pass4_bounds = {"lower": statistics.fmean(question["pass_at_4_bounds"]["lower"] for question in questions),
        "upper": statistics.fmean(question["pass_at_4_bounds"]["upper"] for question in questions), "is_confidence_interval": False} if n_samples >= 4 else None
    groups.append({"group_id": group_id, "group_label": spec["label"], "question_count": len(questions),
        "family_count": unit_count, "sample_count": len(group_rows), "n_samples": n_samples,
        "correct_count": correct_count, "evaluable_count": evaluable_count, "completed_count": evaluable_count,
        "unknown_count": unknown_count, "policy_failure_count": timeout_count, "timeout_count": timeout_count, "timeout_rate": timeout_count / len(group_rows), "completion_rate": evaluable_count / len(group_rows),
        "completed_only_accuracy": correct_count / evaluable_count if evaluable_count else None,
        "observed_success_per_request": observed,
        "incomplete_answer_bounds": {"lower": point, "upper": upper_bound, "is_confidence_interval": False},
        "pass_at_1": observed if unknown_count == 0 else None,
        "pass_at_4": pass4_bounds["lower"] if n_samples >= 4 and unknown_count == 0 else None,
        "pass_at_4_bounds": pass4_bounds, "degenerate_interval": low == high,
        "observed_success_point_in_target": 0.4 <= point <= 0.5,
        "point_in_target": (0.4 <= point <= 0.5) if unknown_count == 0 else None,
        "ci_entirely_in_target": (0.4 <= low <= high <= 0.5) if unknown_count == 0 else None,
        "ci_overlaps_target": (low <= 0.5 and high >= 0.4) if unknown_count == 0 else None,
        "target_assessment": "unresolved_incomplete_answers" if unknown_count else "all_attempts_classified_under_guarded_policy",
        "distance_to_target_center": abs(point - 0.45), "low_precision": len(questions) < 100 or unit_count < 100,
        "status_counts": status_counts, "termination_counts": dict(Counter(row["termination_reason"] for row in group_rows)),
        "format_compliance_rate_among_evaluable": statistics.fmean(row["format_compliant"] for row in evaluable_rows) if evaluable_rows else None,
        "extraction_failure_rate_among_evaluable": status_counts.get("extraction_failed", 0) / evaluable_count if evaluable_count else None,
        "ambiguous_rate_among_evaluable": status_counts.get("ambiguous", 0) / evaluable_count if evaluable_count else None,
        "truncation_rate": truncation_rate, "length_budget_concern": truncation_rate >= 0.1,
        "mean_output_tokens": statistics.fmean(row["output_tokens"] for row in group_rows)})
ranking = sorted(groups, key=lambda item: (item["distance_to_target_center"], item["group_id"]))
synthetic = manifest["synthetic"]
all_correct = sum(row["correct"] is True for row in rows)
all_evaluable = sum(row["scoring_evaluable"] for row in rows)
all_timeouts = sum(row["termination_reason"] == "answer_timeout" for row in rows)
all_unknown = len(rows) - all_evaluable - all_timeouts
summary = {"schema_version": 2, "experiment_type": "exploratory_baseline_calibration", "synthetic": synthetic,
    "model_label": "base", "adapter_identity": None, "model_identity": manifest.get("model_identity"),
    "protocol_id": manifest["protocol_id"], "settings": settings, "provenance": identity,
    "question_count": len(dataset), "total_samples": len(rows), "n_samples": n_samples,
    "evaluable_count": all_evaluable, "completed_count": all_evaluable, "unknown_count": all_unknown, "policy_failure_count": all_timeouts, "timeout_count": all_timeouts, "timeout_rate": all_timeouts / len(rows),
    "completion_rate": all_evaluable / len(rows), "correct_count": all_correct,
    "completed_only_accuracy": all_correct / all_evaluable if all_evaluable else None,
    "observed_success_per_request": all_correct / len(rows),
    "incomplete_answer_bounds": {"lower": all_correct / len(rows), "upper": (all_correct + all_unknown) / len(rows), "is_confidence_interval": False},
    "target": {"metric": "pass_at_1", "lower": 0.4, "upper": 0.5, "center": 0.45, "unknown_answers_prevent_exact_target_assessment": True},
    "groups": groups,
    "limitations": ["SYNTHETIC FIXTURE: no measured model capability." if synthetic else "Exploratory baseline screening under a new guarded stopping protocol; no formal Base/RL comparison.",
        "Only natural-EOS answers are scoring-evaluable. User-defined answer_timeout at the recorded deadline is a policy failure scored zero, not evidence of a mathematical error. Repetition, no-token timeout and context-limit stops remain unscored unknowns.",
        "Observed success per request is completed-and-correct/n. Its 95% interval concerns operational success under this guard, not the unknown mathematical correctness of aborted answers.",
        "Incomplete-answer bounds [c/n,(c+u)/n] are worst/best-case assignments for unknown answers, not confidence intervals or guarantees about generation without the guard.",
        "Completed-only accuracy excludes answer timeouts and every other non-EOS stop; it conditions on finishing and can be selection-biased. It is not the target metric or ranking criterion.",
        "Pass@4 bounds assign unknown answers as failures/successes; they are diagnostic bounds, not confidence intervals. Success within four attempts is not pass@1.",
        "Repeated answers and template instances are dependent. Intervals resample entire families, are marginal rather than simultaneous, and do not remove missing-answer uncertainty.",
        "A degenerate bootstrap interval reflects zero empirical variation, not certainty about future performance. Small family counts give coarse precision.",
        "Source/level labels do not establish a global model difficulty order. MATH rows are scalar/text-only subsets; answer correctness does not prove reasoning validity.",
        "No question is selected using its observed correctness. Fresh confirmation/comparison must exclude all calibration template families."]}
candidates = {"schema_version": 2, "kind": "exploratory_group_ranking_only", "synthetic": synthetic,
    "metric": "observed_success_per_request", "target_range": [0.4, 0.5], "automatic_selection_performed": False,
    "ranking_is_not_mathematical_accuracy_when_unknown": True,
    "fresh_confirmation_questions_required": True, "exclude_calibration_families_from_confirmation": True,
    "ranked_groups": [{key: item[key] for key in ("group_id", "group_label", "observed_success_per_request", "unknown_count", "completion_rate", "incomplete_answer_bounds", "target_assessment", "point_in_target", "low_precision")} for item in ranking]}
report = ["# Guarded baseline calibration / 带重复防护的基线评估", "",
    "**SYNTHETIC FIXTURE — no model performance was measured.**" if synthetic else "**Exploratory baseline-only results under the recorded guard protocol.**", "",
    f"Questions: {len(dataset)}; attempts/question: {n_samples}; handled attempts: {len(rows)}; evaluable: {all_evaluable}; unknown: {all_unknown}; answer-timeout policy failures: {all_timeouts}.",
    f"EOS-completed answers receive verifier scores. Answer timeout at {settings['generation_guard']['answer_timeout_seconds']} seconds is a user-defined zero-score policy failure; other guard-aborted answers retain null scores.", "",
    "| Group | Questions | Completion | Correct/request | 95% interval for observed success | Incomplete-answer bounds (not CI) | Completed-only accuracy | pass@4 bounds (not CI) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|"]
for item in groups:
    metric = item["observed_success_per_request"]
    bounds = item["incomplete_answer_bounds"]
    conditional = f"{item['completed_only_accuracy']:.1%}" if item["completed_only_accuracy"] is not None else "undefined (0 completed)"
    p4 = item["pass_at_4_bounds"]
    p4_text = f"[{p4['lower']:.1%}, {p4['upper']:.1%}]" if p4 is not None else "not collected"
    report.append(f"| {item['group_label']} | {item['question_count']} | {item['completion_rate']:.1%} | {metric['estimate']:.1%} | [{metric['ci_low']:.1%}, {metric['ci_high']:.1%}] | [{bounds['lower']:.1%}, {bounds['upper']:.1%}] | {conditional} | {p4_text} |")
report += ["", "40–50% target decisions remain unresolved for groups with any unknown answer. Completed-only accuracy must not be used to select easier-looking data.",
    "", "## Stop diagnostics / 终止原因", ""]
for item in groups:
    report.append(f"- {item['group_label']}: {json.dumps(item['termination_counts'], sort_keys=True)}; unknown={item['unknown_count']}; answer-timeout failures={item['timeout_count']}; target assessment={item['target_assessment']}.")
report += ["", "## Interpretation limits / 解释边界", ""] + [f"- {item}" for item in summary["limitations"]]
report += ["", "`per_question.jsonl` preserves counts and unknown bounds for every question. `calibration_candidates.json` ranks groups descriptively by observed success per request, never completed-only accuracy; it does not select questions or claim the unknown outcomes are errors.", ""]

print("[BOUNDARY ANALYSIS 2/3] 67% | computed group metrics and uncertainty | no automatic selection", flush=True)
for path in guard_paths:
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != guard_hashes[path.name]:
        raise SystemExit("Guard/helper code changed while processing answers")
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
