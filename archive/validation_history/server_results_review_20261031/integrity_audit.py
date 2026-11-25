"""Audit archived run identity and recompute its published metrics without a GPU."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys

out = Path(__file__).resolve().parent
project = out.parents[3]
archive = project / "server_results/run_20261031/code"
run = archive / "outputs/v1_2"
config_bytes = (archive / "configs/v1.json").read_bytes()
config = json.loads(config_bytes)
prepared_path = archive / "data/processed/manifest.json"
prepared = json.loads(prepared_path.read_bytes())
selection = json.loads((run / "selection/selection.json").read_bytes())
summary = json.loads((run / "main_analysis/summary.json").read_bytes())
assert config["version"] == "1.2.2"
checks = []
source_rows = {}
for label, info in prepared["sources"].items():
    raw = (archive / info["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == info["sha256"]
    source_rows[label] = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(source_rows[label]) == info["rows"]
checks.append("raw_source_hashes_and_row_counts")
splits = {}
for filename, info in prepared["files"].items():
    raw = (archive / "data/processed" / filename).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == info["sha256"]
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == info["rows"] and [row["id"] for row in rows] == info["ids"]
    for row in rows:
        official = source_rows[row["source"]["official_split"]][row["source"]["row_index"]]
        assert row["question"] == official["question"]
        assert Fraction(row["answer"]) == Fraction(official["answer"].rsplit("####", 1)[1].strip().replace(",", ""))
    splits[Path(filename).stem] = {row["id"]: row for row in rows}
for left, right in [("train", "dev"), ("train", "test"), ("dev", "test")]:
    assert not (set(splits[left]) & set(splits[right]))
    assert not ({" ".join(row["question"].casefold().split()) for row in splits[left].values()}
                & {" ".join(row["question"].casefold().split()) for row in splits[right].values()})
assert set(splits["engineering"]) <= set(splits["pilot"]) <= set(splits["train"])
checks.append("all_prepared_hashes_ids_official_questions_gold_and_split_isolation")
stage_reports = {}
checkpoint_count = 0
for stage, expected_steps in [("engineering_train", 2), ("pilot", 96), ("formal", 384)]:
    report = json.loads((run / stage / "report.json").read_bytes())
    manifest = json.loads((run / stage / "manifest.json").read_bytes())
    assert report["status"] == "completed" and all(report["gates"].values())
    assert report["optimizer_steps"] == report["expected_optimizer_steps"] == expected_steps
    assert report["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert report["prepared_manifest_sha256"] == hashlib.sha256(prepared_path.read_bytes()).hexdigest()
    protocol = manifest["training_protocol"]
    assert hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == report["config_training_hash"]
    for field, filename in [("verifier_sha256", "verify_answers.py"), ("training_script_sha256", "train_grpo.py"),
                            ("loader_sha256", "load_model.py"), ("fingerprint_sha256", "fingerprint_model.py"),
                            ("doctor_sha256", "doctor.py")]:
        assert hashlib.sha256((archive / "scripts" / filename).read_bytes()).hexdigest() == protocol[field]
    assert manifest["fresh_adapter_from_base"] is True
    assert report["model_identity"] == selection["model_identity"]
    for label, expected in report["split_sha256"].items():
        assert hashlib.sha256((archive / "data/processed" / f"{label}.jsonl").read_bytes()).hexdigest() == expected
    for rank in report["rank_audits"]:
        assert rank["base_sha256_before"] == rank["base_sha256_after"]
        assert rank["adapter_updated"] and rank["all_adapters_finite"]
        assert hashlib.sha256((run / stage / f"reward_audit.rank-{rank['rank']}.jsonl").read_bytes()).hexdigest() == rank["reward_audit_sha256"]
    for checkpoint in report["checkpoints"]:
        path = archive / checkpoint["path"]
        for name in ["adapter_config.json", "adapter_model.safetensors", "trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]:
            assert (path / name).is_file() and (path / name).stat().st_size > 0
        state = json.loads((path / "trainer_state.json").read_bytes())
        assert state["global_step"] == checkpoint["step"]
        checkpoint_count += 1
    stage_reports[stage] = {key: report[key] for key in ["status", "optimizer_steps", "gates", "training_seconds",
        "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "effective_update_steps", "changed_adapter_parameters"]}
checks.append("training_gates_protocol_source_hashes_rank_reward_hashes_and_checkpoint_inventory")
assert hashlib.sha256((run / "formal/report.json").read_bytes()).hexdigest() == selection["training_report_sha256"]
assert hashlib.sha256((run / "formal/manifest.json").read_bytes()).hexdigest() == selection["training_manifest_sha256"]
candidate_scores = []
for candidate in selection["candidates"]:
    adapter = archive / candidate["adapter_path"]
    digest = hashlib.sha256()
    for filename in ["adapter_config.json", "adapter_model.safetensors"]:
        digest.update(filename.encode())
        with (adapter / filename).open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
    assert digest.hexdigest() == candidate["adapter_identity"]
    evaluation = archive / candidate["evaluation_path"]
    manifest = json.loads((evaluation / "manifest.json").read_bytes())
    raw = (evaluation / "samples.jsonl").read_bytes()
    assert manifest["complete"] and not manifest["synthetic"]
    assert hashlib.sha256(raw).hexdigest() == manifest["samples_sha256"]
    assert manifest["adapter_identity"] == candidate["adapter_identity"]
    assert manifest["dataset_sha256"] == selection["dev_sha256"]
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == 800 and {(row["id"], row["sample_index"]) for row in rows} == {(key, index) for key in splits["dev"] for index in range(4)}
    assert all(row["question"] == splits["dev"][row["id"]]["question"] and row["answer"] == splits["dev"][row["id"]]["answer"] for row in rows)
    score = sum(row["correct"] for row in rows) / len(rows)
    assert score == candidate["pass_at_1"]
    candidate_scores.append({"step": candidate["step"], "pass_at_1": score})
best = sorted(candidate_scores, key=lambda item: (-item["pass_at_1"], item["step"]))[0]
assert best["step"] == selection["step"] and best["pass_at_1"] == selection["pass_at_1"]
checks.append("all_candidate_adapter_hashes_dev_grids_scores_and_frozen_selection")
manifests = {}
all_rows = {}
correct_counts = {}
recomputed = {}
for label in ["base", "rl"]:
    directory = run / f"test_{label}"
    manifest = json.loads((directory / "manifest.json").read_bytes())
    raw = (directory / "samples.jsonl").read_bytes()
    assert manifest["complete"] and not manifest["synthetic"]
    assert manifest["config"] == config
    assert manifest["selection"] == selection
    assert manifest["dataset_sha256"] == selection["test_sha256"]
    assert manifest["model_identity"] == selection["model_identity"]
    assert hashlib.sha256(json.dumps(manifest["model_files"], sort_keys=True).encode()).hexdigest() == manifest["model_identity"]
    assert hashlib.sha256(raw).hexdigest() == manifest["samples_sha256"]
    for filename, expected in manifest["code_hashes"].items():
        assert hashlib.sha256((archive / "scripts" / filename).read_bytes()).hexdigest() == expected
    protocol = {key: manifest[key] for key in ["dataset_sha256", "n_samples", "settings", "code_hashes"]}
    assert hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == manifest["protocol_id"]
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == 3200
    assert {(row["id"], row["sample_index"]) for row in rows} == {(key, index) for key in splits["test"] for index in range(16)}
    assert all(row["question"] == splits["test"][row["id"]]["question"] and row["answer"] == splits["test"][row["id"]]["answer"] for row in rows)
    scores = runpy.run_path(str(archive / "scripts/verify_answers.py"), init_globals={"SCORE_INPUT": rows})["SCORE_OUTPUT"]
    assert len(scores) == len(rows)
    for row, score in zip(rows, scores):
        assert all(row[key] == value for key, value in score.items())
    counts = Counter({key: 0 for key in splits["test"]})
    for row in rows:
        counts[row["id"]] += int(row["correct"])
    values = {}
    for k in [1, 2, 4, 8, 16]:
        values[str(k)] = math.fsum(1 - math.comb(16 - count, k) / math.comb(16, k) if 16 - count >= k else 1.0 for count in counts.values()) / len(counts)
        assert abs(values[str(k)] - summary["metrics"][label]["pass_at_k"][str(k)]) < 1e-12
    recomputed[label] = {"pass_at_k": values, "correct_samples": sum(counts.values()), "status_counts": dict(Counter(row["status"] for row in rows))}
    manifests[label] = manifest
    all_rows[label] = rows
    correct_counts[label] = counts
assert manifests["base"]["adapter_identity"] is None
assert manifests["rl"]["adapter_identity"] == selection["adapter_identity"]
for field in ["protocol_id", "settings", "code_hashes", "model_identity", "model_files", "dataset_sha256"]:
    assert manifests["base"][field] == manifests["rl"][field]
assert [(row["id"], row["sample_index"], row["sample_seed"]) for row in all_rows["base"]] == [(row["id"], row["sample_index"], row["sample_seed"]) for row in all_rows["rl"]]
checks.append("6400_sample_hashes_complete_grids_metadata_same_protocol_seeds_and_exact_verifier_replay")
coverage = {"both": 0, "base_only": 0, "rl_only": 0, "neither": 0, "total": len(splits["test"])}
for key in splits["test"]:
    b, r = bool(correct_counts["base"][key]), bool(correct_counts["rl"][key])
    category = "both" if b and r else "base_only" if b else "rl_only" if r else "neither"
    coverage[category] += 1
assert coverage == summary["coverage"]
checks.append("independent_math_comb_pass_at_k_and_coverage_recomputation")
replay = subprocess.run([sys.executable, str(archive / "scripts/analyze.py"), "--base", "outputs/v1_2/test_base",
    "--rl", "outputs/v1_2/test_rl", "--output", str(out / "recomputed"), "--bootstrap", "2000", "--seed", "42"],
    cwd=archive, text=True, capture_output=True, check=True)
(out / "recomputed_analysis.log").write_text(replay.stdout + replay.stderr)
assert json.loads((out / "recomputed/summary.json").read_bytes()) == summary
for filename in ["report.md", "per_question.jsonl", "manual_inspection.csv"]:
    assert (out / "recomputed" / filename).read_bytes() == (run / "main_analysis" / filename).read_bytes()
checks.append("archived_analysis_replay_2000_bootstraps_exact_summary_report_per_question_inspection_match")
result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "archive": str(archive),
    "archive_config_version": config["version"], "training_resume_v13_present": False,
    "checks": checks, "checkpoint_inventory_count": checkpoint_count, "stage_reports": stage_reports,
    "selection": {"step": selection["step"], "pass_at_1": selection["pass_at_1"], "candidates": candidate_scores},
    "independent_metrics": recomputed, "coverage": coverage, "published_summary_sha256": hashlib.sha256((run / "main_analysis/summary.json").read_bytes()).hexdigest(),
    "gpu_execution": False, "archived_files_modified": False,
    "limitations": ["Base weights were intentionally excluded from backup; their bytes were not rehashed locally.",
        "Optimizer and RNG pickle files were checked for presence only, never deserialized.",
        "Engineering training gate claims were checked for internal consistency; no training was rerun.",
        "Verifier replay reproduces the original scoring rules and does not validate semantic answer extraction or reasoning.",
        "No test answers were rescored under changed rules; format sensitivity requires a separate audit."]}
(out / "integrity_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": result["status"], "checks": checks, "selection": result["selection"], "independent_metrics": recomputed, "coverage": coverage}, indent=2))
