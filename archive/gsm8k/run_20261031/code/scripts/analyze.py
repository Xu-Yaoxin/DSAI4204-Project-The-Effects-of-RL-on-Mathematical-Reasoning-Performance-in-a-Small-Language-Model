#!/usr/bin/env python3
"""Validate paired generation records and estimate finite-sample pass@k."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", required=True, type=Path)
parser.add_argument("--rl", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--bootstrap", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
args.base = (root / args.base).resolve()
args.rl = (root / args.rl).resolve()
args.output = (root / args.output).resolve()
if args.bootstrap < 100:
    parser.error("--bootstrap must be at least 100; 2000 or more is recommended")

manifests = {}
records = {}
required_settings = {"temperature", "top_p", "max_new_tokens", "seed", "prompt_template", "dtype", "backend"}
required_fields = {"id", "sample_index", "correct", "status", "format_compliant", "truncated", "output_tokens", "family_id", "variant_type", "question", "answer", "text"}
for label, directory in (("base", args.base), ("rl", args.rl)):
    with (directory / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for field in ("schema_version", "model_label", "n_samples", "split", "dataset_sha256", "protocol_id", "model_identity", "adapter_identity", "synthetic", "complete", "settings"):
        if field not in manifest:
            raise SystemExit(f"{label}: missing manifest field {field}")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["model_label"] != label:
        raise SystemExit(f"{label}: unsupported schema or incorrect model_label")
    if manifest["complete"] is not True or not isinstance(manifest["synthetic"], bool):
        raise SystemExit(f"{label}: incomplete run or invalid synthetic flag")
    if type(manifest["n_samples"]) is not int or manifest["n_samples"] < 1:
        raise SystemExit(f"{label}: n_samples must be a positive integer")
    for field in ("split", "dataset_sha256", "protocol_id", "model_identity"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise SystemExit(f"{label}: invalid manifest field {field}")
    if not isinstance(manifest["settings"], dict) or not required_settings.issubset(manifest["settings"]):
        raise SystemExit(f"{label}: missing generation settings")
    if not manifest["synthetic"]:
        if not isinstance(manifest.get("code_hashes"), dict) or not isinstance(manifest.get("samples_sha256"), str):
            raise SystemExit(f"{label}: real evaluations require source and sample-content hashes")
        if type(manifest.get("question_count")) is not int or type(manifest.get("total_samples")) is not int:
            raise SystemExit(f"{label}: real evaluations require recorded question and sample counts")
        protocol = {"dataset_sha256": manifest["dataset_sha256"], "n_samples": manifest["n_samples"], "settings": manifest["settings"], "code_hashes": manifest["code_hashes"]}
        if hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() != manifest["protocol_id"]:
            raise SystemExit(f"{label}: protocol_id does not match its recorded components")
    if label == "base" and manifest["adapter_identity"] is not None:
        raise SystemExit("base: the unchanged baseline must not load an adapter")
    if label == "rl" and (not isinstance(manifest["adapter_identity"], str) or not manifest["adapter_identity"].strip()):
        raise SystemExit("rl: a nonempty adapter_identity is required")
    manifests[label] = manifest
    records[label] = {}
    if "samples_sha256" in manifest and hashlib.sha256((directory / "samples.jsonl").read_bytes()).hexdigest() != manifest["samples_sha256"]:
        raise SystemExit(f"{label}: samples file differs from its completed manifest hash")
    with (directory / "samples.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not required_fields.issubset(row):
                raise SystemExit(f"{label}: missing sample fields at line {line_number}")
            if not isinstance(row["id"], str) or not row["id"].strip():
                raise SystemExit(f"{label}: invalid question id at line {line_number}")
            if type(row["sample_index"]) is not int or not 0 <= row["sample_index"] < manifest["n_samples"]:
                raise SystemExit(f"{label}: invalid sample_index at line {line_number}")
            if any(type(row[field]) is not bool for field in ("correct", "format_compliant", "truncated")):
                raise SystemExit(f"{label}: diagnostic flags must be booleans at line {line_number}")
            if type(row["output_tokens"]) is not int or row["output_tokens"] < 0:
                raise SystemExit(f"{label}: invalid output_tokens at line {line_number}")
            if any(not isinstance(row[field], str) for field in ("question", "answer", "text", "status", "variant_type")):
                raise SystemExit(f"{label}: invalid text field at line {line_number}")
            if row["family_id"] is not None and not isinstance(row["family_id"], str):
                raise SystemExit(f"{label}: family_id must be a string or null")
            if row["status"] in {"infrastructure_error", "invalid_gold"}:
                raise SystemExit(f"{label}: {row['status']} must be repaired before analysis")
            if row["status"] not in {"correct", "incorrect", "extraction_failed", "ambiguous", "ok"}:
                raise SystemExit(f"{label}: unknown scoring status {row['status']!r}")
            if row["status"] == "correct" and not row["correct"]:
                raise SystemExit(f"{label}: scoring status contradicts correct flag")
            if row["status"] in {"incorrect", "extraction_failed", "ambiguous"} and row["correct"]:
                raise SystemExit(f"{label}: scoring status contradicts correct flag")
            question_rows = records[label].setdefault(row["id"], {})
            if row["sample_index"] in question_rows:
                raise SystemExit(f"{label}: duplicate sample for {row['id']}")
            if question_rows:
                reference = next(iter(question_rows.values()))
                if any(row[field] != reference[field] for field in ("question", "answer", "family_id", "variant_type")):
                    raise SystemExit(f"{label}: changed question/answer/family metadata within {row['id']}")
            question_rows[row["sample_index"]] = row
    if not records[label]:
        raise SystemExit(f"{label}: empty samples file")
    if "question_count" in manifest and manifest["question_count"] != len(records[label]):
        raise SystemExit(f"{label}: sample question count differs from manifest")
    if "total_samples" in manifest and manifest["total_samples"] != sum(len(group) for group in records[label].values()):
        raise SystemExit(f"{label}: sample total differs from manifest")
    for question_id, question_rows in records[label].items():
        if set(question_rows) != set(range(manifest["n_samples"])):
            raise SystemExit(f"{label}: incomplete sample grid for {question_id}")

for field in ("schema_version", "n_samples", "split", "dataset_sha256", "protocol_id", "model_identity", "synthetic", "settings"):
    if manifests["base"][field] != manifests["rl"][field]:
        raise SystemExit(f"Paired runs have mismatched {field}")
if manifests["base"].get("code_hashes") != manifests["rl"].get("code_hashes"):
    raise SystemExit("Paired runs have mismatched code_hashes")
if set(records["base"]) != set(records["rl"]):
    raise SystemExit("Paired runs have mismatched question ids")

n_samples = manifests["base"]["n_samples"]
k_values = [k for k in (1, 2, 4, 8, 16) if k <= n_samples]
questions = []
coverage = {"both": 0, "base_only": 0, "rl_only": 0, "neither": 0, "total": len(records["base"])}
diagnostics = {}
for label in ("base", "rl"):
    all_rows = [row for group in records[label].values() for row in group.values()]
    status_counts = {}
    for row in all_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    diagnostics[label] = {
        "sample_count": len(all_rows),
        "format_compliance_rate": statistics.fmean(row["format_compliant"] for row in all_rows),
        "truncation_rate": statistics.fmean(row["truncated"] for row in all_rows),
        "extraction_failure_rate": statistics.fmean(row["status"] == "extraction_failed" for row in all_rows),
        "ambiguous_answer_rate": statistics.fmean(row["status"] == "ambiguous" for row in all_rows),
        "mean_output_tokens": statistics.fmean(row["output_tokens"] for row in all_rows),
        "status_counts": status_counts,
        "infrastructure_error_count": 0,
        "invalid_gold_count": 0,
    }

for question_id in sorted(records["base"]):
    base_reference = records["base"][question_id][0]
    rl_reference = records["rl"][question_id][0]
    if any(base_reference[field] != rl_reference[field] for field in ("question", "answer", "family_id", "variant_type")):
        raise SystemExit(f"Paired runs have changed question/answer/family metadata for {question_id}")
    question = {"id": question_id, "family_id": base_reference["family_id"], "variant_type": base_reference["variant_type"], "question": base_reference["question"], "answer": base_reference["answer"], "n_samples": n_samples}
    for label in ("base", "rl"):
        correct_count = sum(row["correct"] for row in records[label][question_id].values())
        pass_at_k = {}
        for k in k_values:
            pass_at_k[str(k)] = 1.0 if n_samples - correct_count < k else 1.0 - math.comb(n_samples - correct_count, k) / math.comb(n_samples, k)
        question[label] = {"correct_count": correct_count, "pass_at_k": pass_at_k}
    question["delta_pass_at_k"] = {str(k): question["rl"]["pass_at_k"][str(k)] - question["base"]["pass_at_k"][str(k)] for k in k_values}
    base_solved = question["base"]["correct_count"] > 0
    rl_solved = question["rl"]["correct_count"] > 0
    category = "both" if base_solved and rl_solved else "base_only" if base_solved else "rl_only" if rl_solved else "neither"
    question["observed_coverage"] = category
    coverage[category] += 1
    questions.append(question)

structural_types = {"numerical", "relational", "recombined"}
is_structural = manifests["base"]["split"].startswith("structural") or any(question["variant_type"] in structural_types for question in questions)
families = {}
for question in questions:
    if is_structural and not question["family_id"]:
        raise SystemExit("Structural evaluation requires nonempty family_id for every question")
    family_id = question["family_id"] or question["id"]
    families.setdefault(family_id, []).append(question)

units = list(families.values()) if is_structural else [[question] for question in questions]
rng = random.Random(args.seed)
bootstrap_statistics = {f"{label}_{k}": [] for label in ("base", "rl", "delta") for k in k_values}
if 16 in k_values:
    bootstrap_statistics["delta1_minus_delta16"] = []
for replicate in range(args.bootstrap):
    sample = [question for unit in rng.choices(units, k=len(units)) for question in unit]
    for k in k_values:
        base_mean = statistics.fmean(question["base"]["pass_at_k"][str(k)] for question in sample)
        rl_mean = statistics.fmean(question["rl"]["pass_at_k"][str(k)] for question in sample)
        bootstrap_statistics[f"base_{k}"].append(base_mean)
        bootstrap_statistics[f"rl_{k}"].append(rl_mean)
        bootstrap_statistics[f"delta_{k}"].append(rl_mean - base_mean)
    if 16 in k_values:
        bootstrap_statistics["delta1_minus_delta16"].append(bootstrap_statistics["delta_1"][-1] - bootstrap_statistics["delta_16"][-1])

point_estimates = {}
for k in k_values:
    point_estimates[f"base_{k}"] = statistics.fmean(question["base"]["pass_at_k"][str(k)] for question in questions)
    point_estimates[f"rl_{k}"] = statistics.fmean(question["rl"]["pass_at_k"][str(k)] for question in questions)
    point_estimates[f"delta_{k}"] = point_estimates[f"rl_{k}"] - point_estimates[f"base_{k}"]
if 16 in k_values:
    point_estimates["delta1_minus_delta16"] = point_estimates["delta_1"] - point_estimates["delta_16"]

structural_variants = {}
paired_family_contrasts = []
if is_structural:
    for variant_type in sorted({question["variant_type"] for question in questions}):
        variant_questions = [question for question in questions if question["variant_type"] == variant_type]
        variant_families = [[question for question in family if question["variant_type"] == variant_type] for family in families.values()]
        variant_families = [family for family in variant_families if family]
        structural_variants[variant_type] = {"question_count": len(variant_questions), "family_count": len(variant_families), "base_pass_at_1": statistics.fmean(question["base"]["pass_at_k"]["1"] for question in variant_questions), "rl_pass_at_1": statistics.fmean(question["rl"]["pass_at_k"]["1"] for question in variant_questions)}
        statistic_key = f"variant_{variant_type}_delta1"
        point_estimates[statistic_key] = statistics.fmean(question["delta_pass_at_k"]["1"] for question in variant_questions)
        bootstrap_statistics[statistic_key] = []
        for replicate in range(args.bootstrap):
            sample = [question for family in rng.choices(variant_families, k=len(variant_families)) for question in family]
            bootstrap_statistics[statistic_key].append(statistics.fmean(question["delta_pass_at_k"]["1"] for question in sample))
    for family in families.values():
        numerical = [question["delta_pass_at_k"]["1"] for question in family if question["variant_type"] == "numerical"]
        relational = [question["delta_pass_at_k"]["1"] for question in family if question["variant_type"] == "relational"]
        if numerical and relational:
            paired_family_contrasts.append(statistics.fmean(relational) - statistics.fmean(numerical))
    if paired_family_contrasts:
        point_estimates["relational_minus_numerical"] = statistics.fmean(paired_family_contrasts)
        bootstrap_statistics["relational_minus_numerical"] = [statistics.fmean(rng.choices(paired_family_contrasts, k=len(paired_family_contrasts))) for _ in range(args.bootstrap)]

intervals = {}
for statistic_key, values in bootstrap_statistics.items():
    ordered = sorted(values)
    bounds = []
    for probability in (0.025, 0.975):
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        bounds.append(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))
    intervals[statistic_key] = {"estimate": point_estimates[statistic_key], "ci_low": bounds[0], "ci_high": bounds[1]}
for variant_type in structural_variants:
    structural_variants[variant_type]["delta_pass_at_1"] = intervals[f"variant_{variant_type}_delta1"]

interpretation = ["pass@k estimates success within k independent draws; it does not measure whether a model can select a correct answer without a verifier.", "Coverage categories use all n collected samples; with n greater than 16 they describe observed pass@n, not estimated pass@16.", "Intervals are marginal 95% paired percentile bootstrap intervals; they quantify evaluation-question uncertainty conditional on this training run, not variation across training seeds.", "An interval crossing zero is inconclusive evidence about a difference, not proof of no change.", "Correct final values do not establish valid reasoning; inspect retained outputs manually."]
if manifests["base"]["synthetic"]:
    interpretation.insert(0, "SYNTHETIC FIXTURE ONLY: these values test the pipeline and are not measurements of model capability.")
if len(units) < 2:
    interpretation.append("Only one resampling unit is available: bootstrap intervals are degenerate and do not support population inference.")
if 16 not in k_values:
    hypothesis_assessment = "Not assessed: fewer than 16 samples per question were collected."
elif intervals["delta_1"]["ci_low"] > 0 and intervals["delta1_minus_delta16"]["ci_low"] > 0:
    hypothesis_assessment = "Both directional contrasts have positive marginal 95% intervals in this run; this supports the finite-sample hypothesis conditional on this protocol."
elif intervals["delta_1"]["ci_high"] < 0 or intervals["delta1_minus_delta16"]["ci_high"] < 0:
    hypothesis_assessment = "At least one directional contrast has a negative marginal 95% interval; the result challenges the stated finite-sample hypothesis."
else:
    hypothesis_assessment = "Evidence is inconclusive: uncertainty does not establish both delta1 > 0 and delta1 - delta16 > 0."
if manifests["base"]["synthetic"]:
    hypothesis_assessment = "Synthetic example; these fixtures cannot support or challenge the scientific hypothesis."

summary = {
    "schema_version": 1,
    "synthetic": manifests["base"]["synthetic"],
    "provenance": {"path_base": "code_root", "base_directory": str(args.base.relative_to(root)) if args.base.is_relative_to(root) else str(args.base), "rl_directory": str(args.rl.relative_to(root)) if args.rl.is_relative_to(root) else str(args.rl), "split": manifests["base"]["split"], "dataset_sha256": manifests["base"]["dataset_sha256"], "protocol_id": manifests["base"]["protocol_id"], "model_identity": manifests["base"]["model_identity"], "adapter_identity": manifests["rl"]["adapter_identity"], "settings": manifests["base"]["settings"]},
    "sampling": {"n_samples": n_samples, "question_count": len(questions), "k_values": k_values, "bootstrap_replicates": args.bootstrap, "seed": args.seed, "bootstrap_unit": "family" if is_structural else "question", "resampling_unit_count": len(units), "interval_method": "paired percentile bootstrap, 95%, linear quantiles"},
    "metrics": {label: {"pass_at_k": {str(k): point_estimates[f"{label}_{k}"] for k in k_values}, "pass_at_k_ci": {str(k): intervals[f"{label}_{k}"] for k in k_values}} for label in ("base", "rl")},
    "comparisons": {"delta_pass_at_k": {str(k): intervals[f"delta_{k}"] for k in k_values}, "delta1_minus_delta16": intervals.get("delta1_minus_delta16")},
    "coverage": coverage,
    "diagnostics": diagnostics,
    "structural": {"enabled": is_structural, "variants": structural_variants, "relational_minus_numerical": intervals.get("relational_minus_numerical"), "paired_family_count": len(paired_family_contrasts), "contrast_definition": "Within each family, mean relational delta pass@1 minus mean numerical delta pass@1; then equal-weight mean across families containing both types.", "claim_scope": "Controlled structural variation only; unseen-combination claims require a separate training-structure audit."},
    "hypothesis_assessment": hypothesis_assessment,
    "interpretation": interpretation,
}

args.output.mkdir(parents=True, exist_ok=True)
with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    handle.write("\n")
with (args.output / "per_question.jsonl").open("w", encoding="utf-8") as handle:
    for question in questions:
        handle.write(json.dumps(question, ensure_ascii=False, allow_nan=False) + "\n")

report = ["# Paired RLVR evaluation", "", "**SYNTHETIC PIPELINE TEST — not model results.**" if summary["synthetic"] else "Recorded model evaluation; interpretation is conditional on this training run.", "", f"Split: `{summary['provenance']['split']}`. Questions: {len(questions)}. Independent responses per question and model: {n_samples}.", f"95% intervals: {args.bootstrap} paired bootstrap replicates, resampling {summary['sampling']['bootstrap_unit']}s; seed {args.seed}.", "", "| k | Base pass@k | RL pass@k | RL − Base (95% CI) |", "|---:|---:|---:|---:|"]
for k in k_values:
    delta = intervals[f"delta_{k}"]
    report.append(f"| {k} | {point_estimates[f'base_{k}']:.4f} | {point_estimates[f'rl_{k}']:.4f} | {delta['estimate']:+.4f} [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] |")
report.extend(["", "## Primary contrasts", "", hypothesis_assessment])
if 16 in k_values:
    difference = intervals["delta1_minus_delta16"]
    report.append(f"\nD = delta1 − delta16: {difference['estimate']:+.4f}; 95% CI [{difference['ci_low']:+.4f}, {difference['ci_high']:+.4f}].")
report.extend(["", f"Observed coverage across all {n_samples} samples: both {coverage['both']}; Base only {coverage['base_only']}; RL only {coverage['rl_only']}; neither {coverage['neither']}.", "", "## Diagnostics", "", "| Model | Format compliance | Extraction failure | Ambiguous answer | Truncation | Mean output tokens |", "|---|---:|---:|---:|---:|---:|"])
for label in ("base", "rl"):
    diagnostic = diagnostics[label]
    report.append(f"| {label} | {diagnostic['format_compliance_rate']:.2%} | {diagnostic['extraction_failure_rate']:.2%} | {diagnostic['ambiguous_answer_rate']:.2%} | {diagnostic['truncation_rate']:.2%} | {diagnostic['mean_output_tokens']:.1f} |")
if is_structural:
    report.extend(["", "## Controlled structural variation", "", "| Variant | Questions | Families | Base pass@1 | RL pass@1 | Delta pass@1 (95% CI) |", "|---|---:|---:|---:|---:|---:|"])
    for variant_type, variant in structural_variants.items():
        delta = variant["delta_pass_at_1"]
        report.append(f"| {variant_type} | {variant['question_count']} | {variant['family_count']} | {variant['base_pass_at_1']:.4f} | {variant['rl_pass_at_1']:.4f} | {delta['estimate']:+.4f} [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}] |")
    contrast = intervals.get("relational_minus_numerical")
    if contrast:
        report.extend(["", f"Paired-family gain contrast (relational − numerical): {contrast['estimate']:+.4f}; 95% CI [{contrast['ci_low']:+.4f}, {contrast['ci_high']:+.4f}], using {len(paired_family_contrasts)} families containing both types.", "Negative values indicate weaker RL gains for relational variants. An interval crossing zero is inconclusive."])
    else:
        report.extend(["", "Relational-versus-numerical contrast unavailable: no families contain both types."])
    report.extend(["", summary["structural"]["contrast_definition"], summary["structural"]["claim_scope"]])
report.extend(["", "## Interpretation limits", ""] + [f"- {item}" for item in interpretation])
report.extend(["", "## Manual inspection", "", "`manual_inspection.csv` includes up to 200 deterministic candidate outputs with diagnostic failures, model coverage disagreements, differing correctness, or a correct response among the first two draws. This convenience sample is not a random audit and cannot estimate invalid-reasoning prevalence. Review `text` against `question` and `answer`, and record a reasoning judgment separately.", "", "The observed score uses every valid generated sample in the denominator; format failures and ambiguous extractions do not disappear from the analysis. Infrastructure errors and invalid gold answers are rejected before computing scores.", ""])
(args.output / "report.md").write_text("\n".join(report), encoding="utf-8")

inspection_fields = ["id", "model", "sample_index", "family_id", "variant_type", "correct", "status", "format_compliant", "truncated", "output_tokens", "question", "answer", "text", "reasoning_valid_manual", "reviewer_note"]
inspection_count = 0
with (args.output / "manual_inspection.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=inspection_fields)
    writer.writeheader()
    for question in questions:
        for sample_index in range(n_samples):
            paired_correctness_differs = records["base"][question["id"]][sample_index]["correct"] != records["rl"][question["id"]][sample_index]["correct"]
            for label in ("base", "rl"):
                row = records[label][question["id"]][sample_index]
                if inspection_count < 200 and (paired_correctness_differs or question["observed_coverage"] in {"base_only", "rl_only"} or not row["format_compliant"] or row["truncated"] or row["status"] in {"extraction_failed", "ambiguous"} or (row["correct"] and sample_index < 2)):
                    inspection = {field: row[field] for field in inspection_fields if field in row}
                    inspection.update({"model": label, "reasoning_valid_manual": "", "reviewer_note": ""})
                    writer.writerow(inspection)
                    inspection_count += 1
print(f"[ANALYSIS 100%] questions={len(questions)} n={n_samples} unit={summary['sampling']['bootstrap_unit']} synthetic={summary['synthetic']}")
print(f"[SAVED] {args.output.resolve() / 'summary.json'}")
print(f"[SAVED] {args.output.resolve() / 'report.md'}")
