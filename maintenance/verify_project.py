#!/usr/bin/env python3
"""Verify retained project evidence without scoring, model execution, or writes."""

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time


root = Path(__file__).resolve().parents[1]
started = time.monotonic()
stage = 0
stages = 10
hashes_checked = 0
bytes_hashed = 0
score_fields = ("correct", "reward", "extracted", "status", "format_compliant", "truncated")
score_keys = set(score_fields)
result = {"read_only": True, "scoring_executed": False, "models_loaded": 0,
          "tests_executed": False, "bootstrap_recomputed": False}

try:
    stage = 1
    print(f"[VERIFY {stage}/{stages}] 0% | relocation and final result identity", flush=True)
    relocation = json.loads((root / "maintenance/relocation.json").read_text())
    if relocation.get("schema") != "project-relocation-v1":
        raise ValueError("Unsupported relocation schema.")
    mappings = []
    for entry in relocation["mappings"]:
        old = Path(entry["old_absolute"])
        new = Path(entry["new_relative"])
        if not old.is_absolute() or new.is_absolute() or ".." in new.parts or not new.parts:
            raise ValueError(f"Unsafe relocation entry: {entry}")
        target = root / new
        if any(p.is_symlink() for p in (target, *target.parents)) or not target.exists():
            raise ValueError(f"Missing or unsafe relocated target: {new}")
        mappings.append((len(old.parts), old, target))
    if len({entry[1] for entry in mappings}) != len(mappings):
        raise ValueError("Duplicate historical relocation roots.")
    final_root = root / "results/final"
    ready_bytes = (final_root / "ready.json").read_bytes()
    if hashlib.sha256(ready_bytes).hexdigest() != relocation["original_math_ready_sha256"]:
        raise ValueError("The original MATH ready.json bytes changed.")
    ready = json.loads(ready_bytes)
    identity = ready["identity"]
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (ready.get("complete") is not True or ready.get("posthoc") is not True
            or ready.get("scorer_version") != "4.0.0" or ready["identity_sha256"] != identity_hash
            or ready.get("question_count") != 200 or ready.get("answers_per_model") != 3200):
        raise ValueError("Invalid final result identity or declared evaluation size.")
    historical = Path(identity["experiment_root"])
    matches = [(length, old, new) for length, old, new in mappings if historical == old or old in historical.parents]
    if not matches:
        raise ValueError("No relocation mapping for the frozen experiment.")
    _, old, new = max(matches)
    experiment = new / historical.relative_to(old)
    if experiment != root / "archive/math_l4/experiment":
        raise ValueError("The frozen experiment resolves to an unexpected destination.")
    expected_artifacts = {"summary.json", "changes.jsonl", "per_question.jsonl",
                          "base/manifest.json", "base/samples.jsonl", "rl/manifest.json", "rl/samples.jsonl"}
    observed_artifacts = set()
    for path in final_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in final results: {path}")
        if path.is_file() and path.name != ".DS_Store" and path != final_root / "ready.json":
            observed_artifacts.add(path.relative_to(final_root).as_posix())
    if set(ready["artifacts_sha256"]) != expected_artifacts or observed_artifacts != expected_artifacts:
        raise ValueError("Final results must contain exactly the seven frozen artifacts and ready.json.")
    for relative, expected_hash in ready["artifacts_sha256"].items():
        path = final_root / relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                bytes_hashed += len(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"Final artifact hash mismatch: {relative}")
        hashes_checked += 1
    summary = json.loads((final_root / "summary.json").read_text())
    if summary["identity_sha256"] != identity_hash or summary["scorer_sha256"] != identity["scorer_sha256"]:
        raise ValueError("Summary identity differs from the ready record.")

    stage = 2
    print(f"[VERIFY {stage}/{stages}] 10% | 124 frozen sources and two parent audit records", flush=True)
    sources = identity["source_files"]
    if len(sources) != 124 or set(identity["source_audit_sha256"]) != {"source_inventory.json", "integrity_audit.json"}:
        raise ValueError("Unexpected frozen source or parent audit inventory.")
    observed_sources = set()
    for path in experiment.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in frozen source: {path}")
        if path.is_file() and path.name != ".DS_Store":
            observed_sources.add(path.relative_to(experiment).as_posix())
    if observed_sources != set(sources):
        raise ValueError("Frozen experiment file inventory changed.")
    hash_jobs = [(experiment, relative, entry["sha256"], entry["bytes"]) for relative, entry in sources.items()]
    hash_jobs += [(experiment.parent, relative, expected_hash, None)
                  for relative, expected_hash in identity["source_audit_sha256"].items()]
    for base, relative, expected_hash, expected_size in hash_jobs:
        relative_path = Path(relative)
        path = base / relative_path
        if relative_path.is_absolute() or ".." in relative_path.parts or not relative_path.parts:
            raise ValueError(f"Unsafe source path: {relative}")
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"Missing or unsafe frozen source: {path}")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f"Source size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                bytes_hashed += len(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"Frozen source hash mismatch: {path}")
        hashes_checked += 1
    inventory = json.loads((experiment.parent / "source_inventory.json").read_text())
    audit = json.loads((experiment.parent / "integrity_audit.json").read_text())
    if inventory.get("source_modified") is not False or audit.get("complete") is not True:
        raise ValueError("The source inventory or audit is not a completed immutable snapshot.")
    if len(inventory["files"]) != 117 or any(sources.get(k) != v for k, v in inventory["files"].items()):
        raise ValueError("The original 117-file inventory differs from the frozen sources.")
    if audit["source_inventory_sha256"] != identity["source_audit_sha256"]["source_inventory.json"]:
        raise ValueError("The integrity audit does not bind the source inventory.")
    for key, relative in (("analysis_ready_sha256", "outputs/math_l4_v1_fix1/main_analysis/ready.json"),
                          ("merge_provenance_sha256", "outputs/math_l4_v1_fix1/test_rl_split_merged/merge_provenance.json")):
        if audit[key] != sources[relative]["sha256"]:
            raise ValueError(f"Audit artifact binding differs: {key}")

    stage = 3
    print(f"[VERIFY {stage}/{stages}] 20% | training bundle and frozen scorer/statistics source hashes", flush=True)
    training = root / "code/math_l4_v1"
    bundle_bytes = (training / "bundle_manifest.json").read_bytes()
    if hashlib.sha256(bundle_bytes).hexdigest() != sources["bundle_manifest.json"]["sha256"]:
        raise ValueError("The active training bundle differs from the frozen original.")
    bundle = json.loads(bundle_bytes)
    if len(bundle["files_sha256"]) != 81:
        raise ValueError("The training bundle must bind 81 files.")
    code_jobs = [(training, relative, expected_hash) for relative, expected_hash in bundle["files_sha256"].items()]
    code_jobs += [(root / "code/math_l4_rescore_v4", "scorer.py", identity["scorer_sha256"]),
                  (root / "code/math_l4_rescore_v4", "rescore.py", identity["runner_sha256"])]
    for base, relative, expected_hash in code_jobs:
        rel = Path(relative)
        path = base / rel
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError(f"Unsafe code path: {relative}")
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"Missing or unsafe bound code/data file: {path}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"Training/scoring file hash mismatch: {path}")
        hashes_checked += 1
        bytes_hashed += len(payload)
    statistics_text = (experiment / "scripts/analyze.py").read_text()
    statistics_block = statistics_text[statistics_text.index("records = {label:"):statistics_text.index("limitations = [")]
    if (sources["scripts/analyze.py"]["sha256"] != identity["statistics_source_sha256"]
            or hashlib.sha256(statistics_block.encode()).hexdigest() != identity["statistics_block_sha256"]):
        raise ValueError("The frozen statistics source or extraction block changed.")
    if bundle["scorer_sha256"] != sources["scripts/verify_answers.py"]["sha256"]:
        raise ValueError("The original training scorer binding changed.")

    stage = 4
    print(f"[VERIFY {stage}/{stages}] 30% | four raw-answer and selected-adapter exports", flush=True)
    exports = relocation["exports"]
    required_exports = {"results/raw/baseline.jsonl", "results/raw/rl.jsonl",
                        "models/rl_checkpoint384/adapter_config.json", "models/rl_checkpoint384/adapter_model.safetensors"}
    if len(exports) != 4 or {entry["export_relative"] for entry in exports} != required_exports:
        raise ValueError("Expected exactly four documented exports.")
    for entry in exports:
        for field in ("source_relative", "export_relative"):
            relative = Path(entry[field])
            path = root / relative
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError(f"Unsafe export path: {relative}")
            if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError(f"Missing or unsafe export/source: {relative}")
            if path.stat().st_size != entry["bytes"]:
                raise ValueError(f"Export/source size mismatch: {relative}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    digest.update(chunk)
                    bytes_hashed += len(chunk)
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"Export/source hash mismatch: {relative}")
            hashes_checked += 1

    stage = 5
    print(f"[VERIFY {stage}/{stages}] 40% | 6,400 paired records and unchanged generation fields", flush=True)
    dataset = [json.loads(line) for line in (experiment / "data/processed/test.jsonl").read_text().splitlines() if line.strip()]
    dataset_by_id = {row["id"]: row for row in dataset}
    if len(dataset) != 200 or len(dataset_by_id) != 200:
        raise ValueError("The frozen test set does not contain 200 distinct questions.")
    expected_grid = {(qid, index) for qid in dataset_by_id for index in range(16)}
    observations = {version: {label: {} for label in ("base", "rl")} for version in ("original", "final")}
    diagnostics = {version: {label: {"status_counts": Counter(), "termination_counts": Counter(), "correct_count": 0,
                    "unknown_count": 0, "completed_count": 0, "requested_count": 0} for label in ("base", "rl")}
                   for version in ("original", "final")}
    calculated_changes = {}
    change_counts = {label: Counter() for label in ("base", "rl")}
    seed_grids = {}
    for label, source_relative in (("base", "outputs/math_l4_v1_fix1/test_base"),
                                    ("rl", "outputs/math_l4_v1_fix1/test_rl_split_merged")):
        source_dir = experiment / source_relative
        original_manifest = json.loads((source_dir / "manifest.json").read_text())
        final_manifest = json.loads((final_root / label / "manifest.json").read_text())
        source_samples_hash = sources[source_relative + "/samples.jsonl"]["sha256"]
        if (final_manifest.get("complete") is not True or final_manifest.get("model_label") != label
                or final_manifest["identity_sha256"] != identity_hash or final_manifest["score_fields"] != list(score_fields)
                or final_manifest["samples_sha256"] != ready["artifacts_sha256"][label + "/samples.jsonl"]
                or final_manifest["generation_provenance"]["source_manifest"] != original_manifest
                or final_manifest["generation_provenance"]["source_manifest_sha256"] != sources[source_relative + "/manifest.json"]["sha256"]
                or final_manifest["generation_provenance"]["source_samples_sha256"] != source_samples_hash):
            raise ValueError(f"Final manifest generation binding changed: {label}")
        if identity["evaluations"][label] != {"manifest_sha256": sources[source_relative + "/manifest.json"]["sha256"],
                                              "samples_sha256": source_samples_hash}:
            raise ValueError(f"Frozen evaluation identity mismatch: {label}")
        raw_records = {}
        for version, path in (("original", source_dir / "samples.jsonl"), ("final", final_root / label / "samples.jsonl")):
            seen = set()
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        raise ValueError(f"Unexpected blank JSONL record: {path}:{line_number}")
                    row = json.loads(line)
                    key = (row["id"], row["sample_index"])
                    if type(row["sample_index"]) is not int or key not in expected_grid or key in seen:
                        raise ValueError(f"Invalid/duplicate sample key: {label}/{version}/{key}")
                    seen.add(key)
                    scores = {field: row[field] for field in score_fields}
                    if row["correct"] is not None and type(row["correct"]) is not bool:
                        raise ValueError(f"Correctness is not boolean/unknown: {label}/{key}")
                    excluded = score_keys if version == "original" else score_keys | {"old_score", "rescore"}
                    payload = {k: v for k, v in row.items() if k not in excluded}
                    payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    source_key = {"model_label": label, "id": row["id"], "sample_index": row["sample_index"],
                                  "sample_seed": row["sample_seed"], "line_number": line_number}
                    if version == "original":
                        if "old_score" in row or "rescore" in row:
                            raise ValueError("Original generation records already contain post-hoc metadata.")
                        if any(row.get(k) != v for k, v in dataset_by_id[row["id"]].items()):
                            raise ValueError(f"Original generation metadata differs from its dataset: {label}/{key}")
                        raw_records[key] = {"payload_hash": payload_hash, "score": scores, "source_key": source_key}
                    else:
                        original = raw_records[key]
                        if payload_hash != original["payload_hash"] or json.dumps(row["old_score"], sort_keys=True) != json.dumps(original["score"], sort_keys=True):
                            raise ValueError(f"Generation fields or original score changed: {label}/{key}")
                        rescore = row["rescore"]
                        if (rescore["source_key"] != original["source_key"] or rescore["source_samples_sha256"] != source_samples_hash
                                or rescore["scorer_sha256"] != identity["scorer_sha256"] or rescore["scorer_version"] != "4.0.0"
                                or rescore["scoring_applied"] is not (row["termination_reason"] == "eos")):
                            raise ValueError(f"Post-hoc record provenance mismatch: {label}/{key}")
                        changed_fields = [field for field in score_fields if original["score"][field] != scores[field]]
                        if row["termination_reason"] != "eos" and changed_fields:
                            raise ValueError(f"Non-EOS score fields changed: {label}/{key}")
                        change_counts[label]["eos_rescored" if row["termination_reason"] == "eos" else "stopped_preserved"] += 1
                        if changed_fields:
                            calculated_changes[(label, *key)] = {**original["source_key"], "source_key": original["source_key"],
                                "source_samples_sha256": source_samples_hash, "termination_reason": row["termination_reason"],
                                "old_score": original["score"], "new_score": scores, "changed_fields": changed_fields}
                            change_counts[label]["changed_samples"] += 1
                            for field in changed_fields:
                                change_counts[label]["field_" + field] += 1
                            if "correct" in changed_fields:
                                change_counts[label]["correctness_changes"] += 1
                                change_counts[label]["false_to_true"] += int(original["score"]["correct"] is False and scores["correct"] is True)
                                change_counts[label]["true_to_false"] += int(original["score"]["correct"] is True and scores["correct"] is False)
                    group = observations[version][label].setdefault(row["id"], {"correct_count": 0, "unknown_count": 0,
                        "completed_count": 0, "timeout_count": 0, "termination_counts": Counter(), "count": 0})
                    group["count"] += 1
                    group["correct_count"] += int(row["correct"] is True)
                    group["unknown_count"] += int(row["correct"] is None)
                    group["completed_count"] += int(row["termination_reason"] == "eos")
                    group["timeout_count"] += int(row["termination_reason"] == "answer_timeout")
                    group["termination_counts"][row["termination_reason"]] += 1
                    diagnostic = diagnostics[version][label]
                    diagnostic["requested_count"] += 1
                    diagnostic["correct_count"] += int(row["correct"] is True)
                    diagnostic["unknown_count"] += int(row["correct"] is None)
                    diagnostic["completed_count"] += int(row["termination_reason"] == "eos")
                    diagnostic["termination_counts"][row["termination_reason"]] += 1
                    diagnostic["status_counts"][row["status"]] += 1
            if seen != expected_grid:
                raise ValueError(f"Incomplete 200 x 16 sample grid: {label}/{version}")
        seed_grids[label] = {key: value["source_key"]["sample_seed"] for key, value in raw_records.items()}
        if dict(change_counts[label]) != summary["changes"][label] or final_manifest["changes"] != summary["changes"][label]:
            raise ValueError(f"Recorded change counters differ: {label}")
        print(f"[VERIFY {stage}/{stages}] {45 if label == 'base' else 50}% | {label}: all 3,200 record pairs preserved", flush=True)
    if seed_grids["base"] != seed_grids["rl"]:
        raise ValueError("Base and RL sample seeds are not paired.")

    stage = 6
    print(f"[VERIFY {stage}/{stages}] 50% | exact pass@1/4/16 arithmetic and 25 change records", flush=True)
    per_question_rows = [json.loads(line) for line in (final_root / "per_question.jsonl").read_text().splitlines() if line.strip()]
    per_question = {row["id"]: row for row in per_question_rows}
    if len(per_question_rows) != 200 or set(per_question) != set(dataset_by_id):
        raise ValueError("Per-question output does not contain the complete unique test grid.")
    aggregate = {version: {} for version in observations}
    per_question_values = {version: {label: {} for label in ("base", "rl")} for version in observations}
    for version in ("original", "final"):
        recorded_summary = summary["original_statistics"] if version == "original" else summary
        for label in ("base", "rl"):
            for field, value in diagnostics[version][label].items():
                if recorded_summary["diagnostics"][label][field] != value:
                    raise ValueError(f"Diagnostic mismatch: {version}/{label}/{field}")
            expected_correct = {("original", "base"): 1393, ("original", "rl"): 1950,
                                ("final", "base"): 1393, ("final", "rl"): 1968}[(version, label)]
            if diagnostics[version][label]["correct_count"] != expected_correct:
                raise ValueError(f"Unexpected correct-answer count: {version}/{label}")
            for qid, counts in observations[version][label].items():
                record = per_question[qid]
                if record["n_samples"] != 16 or record["family_id"] != dataset_by_id[qid]["family_id"] or counts["count"] != 16:
                    raise ValueError(f"Per-question identity/count mismatch: {qid}")
                expected_record = record["original_scores"][label] if version == "original" else record[label]
                for field in ("correct_count", "unknown_count", "completed_count", "timeout_count", "termination_counts"):
                    if expected_record[field] != counts[field]:
                        raise ValueError(f"Per-question count mismatch: {version}/{label}/{qid}/{field}")
                values = {}
                for k in (1, 4, 16):
                    lower = 1 - Fraction(math.comb(16 - counts["correct_count"], k), math.comb(16, k))
                    upper = 1 - Fraction(math.comb(16 - counts["correct_count"] - counts["unknown_count"], k), math.comb(16, k))
                    values[k] = (lower, upper)
                    saved = expected_record[f"pass_at_{k}"]
                    if (not math.isclose(saved["estimate"], float(lower), rel_tol=0.0, abs_tol=1e-12)
                            or any(not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12)
                                   for actual, expected in zip(saved["unknown_bounds"], (lower, upper)))):
                        raise ValueError(f"Per-question pass@{k} mismatch: {version}/{label}/{qid}")
                per_question_values[version][label][qid] = values
            for k in (1, 4, 16):
                bounds = tuple(sum((item[k][index] for item in per_question_values[version][label].values()), Fraction()) / 200
                               for index in (0, 1))
                aggregate[version][f"{label}_pass_at_{k}"] = bounds
        for k in (1, 4, 16):
            base_bounds = aggregate[version][f"base_pass_at_{k}"]
            rl_bounds = aggregate[version][f"rl_pass_at_{k}"]
            estimate = rl_bounds[0] - base_bounds[0]
            aggregate[version][f"delta_pass_at_{k}"] = (estimate,)
            saved = recorded_summary["metrics"][f"delta_pass_at_{k}"]
            if any(not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12)
                   for actual, expected in zip(saved["unknown_bounds"], (rl_bounds[0] - base_bounds[1], rl_bounds[1] - base_bounds[0]))):
                raise ValueError(f"Aggregate delta bounds mismatch: {version}/pass@{k}")
            if version == "final":
                for qid in dataset_by_id:
                    b = per_question_values[version]["base"][qid][k]
                    r = per_question_values[version]["rl"][qid][k]
                    record = per_question[qid]
                    if (not math.isclose(record[f"delta_pass_at_{k}"], float(r[0] - b[0]), rel_tol=0.0, abs_tol=1e-12)
                            or any(not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12)
                                   for actual, expected in zip(record[f"delta_pass_at_{k}_unknown_bounds"], (r[0] - b[1], r[1] - b[0])))):
                        raise ValueError(f"Per-question delta mismatch: {qid}/pass@{k}")
        aggregate[version]["delta_pass_at_1_minus_delta_pass_at_16"] = (
            aggregate[version]["delta_pass_at_1"][0] - aggregate[version]["delta_pass_at_16"][0],)
        for metric, bounds in aggregate[version].items():
            saved = recorded_summary["metrics"][metric]
            if not math.isclose(saved["estimate"], float(bounds[0]), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Aggregate metric mismatch: {version}/{metric}")
            if len(bounds) == 2 and any(not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12)
                                        for actual, expected in zip(saved["unknown_bounds"], bounds)):
                raise ValueError(f"Aggregate metric bounds mismatch: {version}/{metric}")
        coverage = Counter()
        for qid in dataset_by_id:
            base_success = observations[version]["base"][qid]["correct_count"] > 0
            rl_success = observations[version]["rl"][qid]["correct_count"] > 0
            coverage["both" if base_success and rl_success else "base_only" if base_success else "rl_only" if rl_success else "neither"] += 1
        if coverage != recorded_summary["observed_coverage_counts"]:
            raise ValueError(f"Observed coverage mismatch: {version}")
    changes = [json.loads(line) for line in (final_root / "changes.jsonl").read_text().splitlines() if line.strip()]
    recorded_changes = {(row["model_label"], row["id"], row["sample_index"]): row for row in changes}
    if len(changes) != 25 or len(recorded_changes) != 25 or recorded_changes != calculated_changes or summary["changed_samples"] != 25:
        raise ValueError("The 25 recorded changes do not exactly match the raw/final score differences.")
    result["math_l4"] = {"source_files": 124, "training_bundle_files": 81, "answers_per_model": 3200,
        "correct_answers": {label: diagnostics["final"][label]["correct_count"] for label in ("base", "rl")},
        "changed_records": 25, "exact_metrics": {key: str(value[0]) for key, value in aggregate["final"].items()}}

    stage = 7
    print(f"[VERIFY {stage}/{stages}] 60% | GSM8K identity, 33 source files, and 20 result artifacts", flush=True)
    gsm_results = root / "archive/gsm8k/rescore_v1_4"
    gsm_ready = json.loads((gsm_results / "ready.json").read_text())
    gsm_identity = gsm_ready["identity"]
    if (gsm_ready.get("complete") is not True
            or hashlib.sha256(json.dumps(gsm_identity, sort_keys=True).encode()).hexdigest() != gsm_ready["identity_sha256"]
            or len(gsm_identity["source_files_sha256"]) != 33 or len(gsm_ready["artifacts_sha256"]) != 20):
        raise ValueError("Invalid GSM8K ready identity or inventory size.")
    gsm_provenance = json.loads((gsm_results / "provenance.json").read_text())
    historical = Path(gsm_provenance["source_archive"])
    matches = [(length, old, new) for length, old, new in mappings if historical == old or old in historical.parents]
    if not matches:
        raise ValueError("No relocation mapping for the original GSM8K archive.")
    _, old, new = max(matches)
    gsm_archive = new / historical.relative_to(old)
    if gsm_archive != root / "archive/gsm8k/run_20261031/code":
        raise ValueError("The GSM8K archive resolves to an unexpected destination.")
    observed = set()
    for path in gsm_results.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in GSM8K results: {path}")
        if path.is_file() and path.name != ".DS_Store" and path != gsm_results / "ready.json":
            observed.add(path.relative_to(gsm_results).as_posix())
    if observed != set(gsm_ready["artifacts_sha256"]):
        raise ValueError("The GSM8K result artifact inventory changed.")
    gsm_jobs = [(gsm_archive, relative, expected_hash) for relative, expected_hash in gsm_identity["source_files_sha256"].items()]
    gsm_jobs += [(gsm_results, relative, expected_hash) for relative, expected_hash in gsm_ready["artifacts_sha256"].items()]
    gsm_jobs += [(root / "code/scripts", relative, expected_hash) for relative, expected_hash in gsm_identity["scoring_code_sha256"].items()]
    for base, relative, expected_hash in gsm_jobs:
        rel = Path(relative)
        path = base / rel
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError(f"Unsafe GSM8K source path: {relative}")
        if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"Missing or unsafe GSM8K file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                bytes_hashed += len(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"GSM8K file hash mismatch: {path}")
        hashes_checked += 1
    result["gsm8k"] = {"source_files": 33, "result_artifacts": 20, "scoring_source_files": 3}

    stage = 8
    print(f"[VERIFY {stage}/{stages}] 70% | cleanup inventory and removed continuation state", flush=True)
    cleanup = json.loads((root / "maintenance/cleanup.json").read_text())
    deleted = cleanup["files"]
    if (cleanup.get("status") != "completed" or cleanup.get("all_deleted_paths_absent") is not True
            or cleanup["file_count"] != len(deleted) or cleanup["total_bytes"] != sum(entry["bytes"] for entry in deleted)
            or len({entry["path"] for entry in deleted}) != len(deleted)):
        raise ValueError("Cleanup metadata is incomplete or internally inconsistent.")
    removed_state_count = 0
    absent_deleted_count = 0
    strict_deleted_count = 0
    regenerated_cache_paths = []
    for entry in deleted:
        rel = Path(entry["path"])
        path = root / rel
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            raise ValueError(f"Unsafe cleanup path: {rel}")
        old_gsm_state = (rel.parts[:3] == ("archive", "gsm8k", "run_20261031")
                         and (rel.name in {"optimizer.pt", "scheduler.pt"} or rel.name.startswith("rng_state")))
        regenerable = (not old_gsm_state and (rel.name == ".DS_Store" or "__pycache__" in rel.parts
            or rel.suffix == ".pyc" or rel.is_relative_to("code/packages/linux-cp311-cu128/wheels")
            or rel.is_relative_to("code/math_l4_rl_split_v1/payload")))
        strict_deleted_count += int(not regenerable)
        if path.exists() or path.is_symlink():
            if not regenerable:
                raise ValueError(f"A removed obsolete path exists: {rel}")
            if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError(f"A regenerated cache path is not a safe regular file: {rel}")
            regenerated_cache_paths.append(rel.as_posix())
        else:
            absent_deleted_count += 1
        if old_gsm_state:
            removed_state_count += 1
    if removed_state_count != 24:
        raise ValueError("Expected 24 documented GSM8K optimizer/scheduler/RNG removals.")
    for path in (gsm_archive / "outputs/v1_2").rglob("*"):
        if path.is_file() and (path.name in {"optimizer.pt", "scheduler.pt"} or path.name.startswith("rng_state")):
            raise ValueError(f"Obsolete GSM8K continuation state remains: {path}")
    result["cleanup"] = {"deleted_files_absent": absent_deleted_count,
        "strict_deleted_paths_absent": strict_deleted_count, "removed_continuation_state_files": removed_state_count,
        "regenerated_cache_count": len(regenerated_cache_paths), "regenerated_cache_paths": sorted(regenerated_cache_paths)}

    stage = 9
    print(f"[VERIFY {stage}/{stages}] 80% | dependency lock and complete imported checkpoint/evidence", flush=True)
    package_manifest = json.loads((root / "code/packages/linux-cp311-cu128/manifest.json").read_text())
    lock_bytes = (root / "code/packages/linux-cp311-cu128/requirements-lock.txt").read_bytes()
    requirement_bytes = (root / "requirement.txt").read_bytes()
    if (package_manifest["package_count"] != 84 or len(package_manifest["packages"]) != 84
            or hashlib.sha256(lock_bytes).hexdigest() != package_manifest["lock_sha256"]
            or hashlib.sha256(requirement_bytes).hexdigest() != package_manifest["lock_sha256"]
            or requirement_bytes != lock_bytes):
        raise ValueError("requirement.txt must exactly preserve the original 84-package hash lock.")
    imports = json.loads((root / "maintenance/downloads_import.json").read_text())
    if (imports.get("complete") is not True or imports["selected_step"] != 384
            or imports.get("downloads_runtime_dependency") is not False or imports.get("base_weights_copied") is not False
            or len(imports["checkpoint_files"]) != 13 or len(imports["supplementary_files"]) != 37):
        raise ValueError("The imported checkpoint/supplementary evidence inventory is incomplete.")
    imported_paths = set()
    for entry in imports["checkpoint_files"] + imports["supplementary_files"]:
        rel = Path(entry["path"])
        path = root / rel
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel in imported_paths:
            raise ValueError(f"Unsafe or duplicate imported destination: {rel}")
        imported_paths.add(rel)
        if (not path.is_file() or any(p.is_symlink() for p in (path, *path.parents))
                or path.stat().st_size != entry["bytes"]):
            raise ValueError(f"Imported evidence file missing, unsafe, or changed in size: {rel}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                bytes_hashed += len(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Imported evidence hash mismatch: {rel}")
        hashes_checked += 1
    checkpoint = root / "models/rl_checkpoint384"
    seal_bytes = (checkpoint / "resume_state.json").read_bytes()
    seal = json.loads(seal_bytes)
    formal_manifest = json.loads((experiment / "outputs/math_l4_v1_fix1/formal/manifest.json").read_text())
    frozen_seal_hash = sources["outputs/math_l4_v1_fix1/formal/checkpoint-384/resume_state.json"]["sha256"]
    if (hashlib.sha256(seal_bytes).hexdigest() != frozen_seal_hash
            or seal["step"] != 384 or seal["checkpoint_name"] != "checkpoint-384"
            or seal["binding"] != formal_manifest["resume_binding"] or len(seal["files"]) != 12):
        raise ValueError("The complete selected checkpoint differs from its original seal/binding.")
    observed_checkpoint = set()
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in selected checkpoint: {path}")
        if path.is_file() and path.name != ".DS_Store":
            observed_checkpoint.add(path.relative_to(checkpoint).as_posix())
    if observed_checkpoint != set(seal["files"]) | {"resume_state.json"}:
        raise ValueError("The selected checkpoint does not contain the complete sealed inventory.")
    for relative, entry in seal["files"].items():
        rel = Path(relative)
        path = checkpoint / rel
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or path.stat().st_size != entry["size"]:
            raise ValueError(f"Invalid sealed checkpoint path/size: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                bytes_hashed += len(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Sealed checkpoint hash mismatch: {relative}")
        hashes_checked += 1
    for relative in ("code/scripts/download_model.py", "code/models/qwen3.5-2b-base-manifest.json"):
        if not (root / relative).is_file():
            raise ValueError(f"Required public Base download support is missing: {relative}")
    result["environment_and_imports"] = {"locked_packages": 84, "checkpoint_files": 13,
        "sealed_state_files": 12, "supplementary_files": 37, "downloads_paths_read": 0,
        "checkpoint_state_loaded": False, "base_weights_loaded": False}

    stage = 10
    print(f"[VERIFY {stage}/{stages}] 90% | paper metadata and unchanged PDF bytes", flush=True)
    paper_metadata = root / "docs/papers/source.json"
    if paper_metadata.exists():
        paper = json.loads(paper_metadata.read_text())
        required = {"title", "authors", "arxiv_id", "version", "abstract_url", "pdf_url", "doi",
                    "local_file", "bytes", "sha256", "page_count", "retrieved_at_utc", "verification", "project_scope"}
        if not required <= set(paper) or paper["arxiv_id"] != "2504.13837" or paper["version"] != "v5":
            raise ValueError("Paper source metadata is incomplete or identifies an unexpected version.")
        if (paper["abstract_url"] != "https://arxiv.org/abs/2504.13837v5"
                or paper["pdf_url"] != "https://arxiv.org/pdf/2504.13837v5"
                or paper["doi"] != "10.48550/arXiv.2504.13837"
                or paper["page_count"] != 31 or not paper["authors"] or not paper["title"]):
            raise ValueError("Paper identifier metadata is inconsistent.")
        rel = Path(paper["local_file"])
        path = paper_metadata.parent / rel
        if rel.is_absolute() or len(rel.parts) != 1 or not path.is_file() or path.is_symlink():
            raise ValueError("Paper local_file must name a regular PDF beside source.json.")
        payload = path.read_bytes()
        if (len(payload) != paper["bytes"] or hashlib.sha256(payload).hexdigest() != paper["sha256"]
                or not payload.startswith(b"%PDF-") or b"%%EOF" not in payload[-1024:]):
            raise ValueError("Paper PDF bytes, size, or file markers differ from the source record.")
        hashes_checked += 1
        bytes_hashed += len(payload)
        result["paper"] = {"bytes_verified": True, "version": "2504.13837v5", "online_metadata_rechecked": False,
                           "page_count_recomputed": False}
    else:
        result["paper"] = {"checked": False, "reason": "Optional docs/papers/source.json is absent."}
    result.update(passed=True, file_hash_checks=hashes_checked, bytes_hashed=bytes_hashed,
                  elapsed_seconds=round(time.monotonic() - started, 3))
    print("[VERIFY 10/10] 100% | PASS | retained evidence and record arithmetic verified", flush=True)
    print(json.dumps(result, indent=2), flush=True)
except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
    result.update(passed=False, failed_stage=stage, error=f"{type(error).__name__}: {error}",
                  elapsed_seconds=round(time.monotonic() - started, 3))
    print(f"[VERIFY {stage}/{stages}] FAIL | {error}", file=sys.stderr, flush=True)
    print(json.dumps(result, indent=2), file=sys.stderr, flush=True)
    raise SystemExit(1)
