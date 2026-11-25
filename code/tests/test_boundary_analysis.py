"""Check baseline difficulty statistics, provenance and immutable outputs offline."""

import copy
import hashlib
import json
import math
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile


root = Path(__file__).resolve().parents[1]
checked = 0
with tempfile.TemporaryDirectory(prefix="boundary-analysis-test-") as temporary:
    fixture = Path(temporary).resolve() / "code"
    (fixture / "scripts").mkdir(parents=True)
    for filename in ("analyze_boundary.py", "verify_answers.py"):
        shutil.copyfile(root / "scripts" / filename, fixture / "scripts" / filename)
    data_dir = fixture / "data/boundary_v1"
    data_dir.mkdir(parents=True)
    source = fixture / "outputs/boundary_v1/base"
    source.mkdir(parents=True)
    data = []
    for group in ("gsm_symbolic", "math_l4_5"):
        for index in range(4):
            data.append({"id": f"{group}-{index}", "question": f"A fixture\u2028question {group}/{index}: what is one plus one?", "answer": "2",
                "source_answer": "2", "group_id": group, "group_label": group, "family_id": f"family-{group}-{index}",
                "source": {"dataset": "synthetic", "revision": "fixture", "split": "test", "row_index": index, "original_id": index},
                "difficulty_label": "fixture only"})
    data_raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data).encode()
    data_path = data_dir / "questions.jsonl"
    data_path.write_bytes(data_raw)
    data_manifest = {"schema_version": 1, "groups": [{"id": group, "label": group, "count": 4} for group in ("gsm_symbolic", "math_l4_5")],
        "files": {"questions.jsonl": {"sha256": hashlib.sha256(data_raw).hexdigest(), "rows": len(data)}}}
    data_manifest_path = data_dir / "manifest.json"
    data_manifest_path.write_text(json.dumps(data_manifest))
    settings = {"temperature": 0.6, "top_p": 0.95, "top_k": 0, "max_new_tokens": 32, "max_prompt_length": 128, "seed": 271828,
        "prompt_template": "Solve {question}", "dtype": "bfloat16", "backend": "synthetic", "batch_size": 1, "world_size": 1}
    rows = []
    for item in data:
        count = [0, 1, 2, 4][item["source"]["row_index"]] if item["group_id"] == "gsm_symbolic" else 4
        for index in range(4):
            text = "#### 2" if index < count else "#### 3"
            score = runpy.run_path(str(fixture / "scripts/verify_answers.py"), init_globals={"SCORE_INPUT": [{"text": text, "answer": "2"}]})["SCORE_OUTPUT"][0]
            rows.append({**item, **score, "sample_index": index, "text": text,
                "sample_seed": int.from_bytes(hashlib.sha256(f"{settings['seed']}:{item['id']}:{index}".encode()).digest()[:4], "big"),
                "output_tokens": 4, "prompt_tokens": 20, "seconds": 0.01})
    original_rows = copy.deepcopy(rows)
    samples_path = source / "samples.jsonl"
    sample_raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    samples_path.write_bytes(sample_raw)
    manifest = {"schema_version": 1, "experiment_type": "exploratory_baseline_calibration", "synthetic": True, "model_label": "base",
        "model_identity": "synthetic-no-model", "adapter_identity": None, "adapter_path": None, "n_samples": 4,
        "question_count": 8, "total_samples": 32, "complete": True, "samples_sha256": hashlib.sha256(sample_raw).hexdigest(),
        "dataset_path": "data/boundary_v1/questions.jsonl", "dataset_sha256": hashlib.sha256(data_raw).hexdigest(),
        "dataset_manifest_path": "data/boundary_v1/manifest.json", "dataset_manifest_sha256": hashlib.sha256(data_manifest_path.read_bytes()).hexdigest(),
        "settings": settings, "code_hashes": {"verify_answers.py": hashlib.sha256((fixture / "scripts/verify_answers.py").read_bytes()).hexdigest()}}
    protocol = {key: manifest[key] for key in ("dataset_sha256", "dataset_manifest_sha256", "n_samples", "settings", "code_hashes", "experiment_type")}
    manifest["protocol_id"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    original_manifest = copy.deepcopy(manifest)
    manifest_raw = manifest_path.read_bytes()
    command = [sys.executable, "-S", str(fixture / "scripts/analyze_boundary.py"), "--bootstrap", "100"]
    completed = subprocess.run(command, cwd=Path(temporary), text=True, capture_output=True, timeout=20)
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    output = fixture / "outputs/boundary_v1/analysis"
    summary = json.loads((output / "summary.json").read_text())
    groups = {item["group_id"]: item for item in summary["groups"]}
    assert groups["gsm_symbolic"]["pass_at_1"]["estimate"] == 7 / 16
    assert groups["gsm_symbolic"]["pass_at_4"] == 3 / 4
    assert groups["gsm_symbolic"]["point_in_target"] is True
    assert groups["gsm_symbolic"]["ci_entirely_in_target"] is False
    assert groups["gsm_symbolic"]["low_precision"] is True
    assert groups["math_l4_5"]["pass_at_1"]["estimate"] == 1
    assert groups["math_l4_5"]["degenerate_interval"] is True
    assert summary["synthetic"] is True
    assert "SYNTHETIC FIXTURE" in (output / "report.md").read_text()
    ranking = json.loads((output / "calibration_candidates.json").read_text())
    assert ranking["ranked_groups"][0]["group_id"] == "gsm_symbolic"
    assert ranking["automatic_selection_performed"] is False
    assert ranking["fresh_confirmation_questions_required"] is True
    assert len((output / "per_question.jsonl").read_text().split("\n")) == 9
    assert samples_path.read_bytes() == sample_raw and manifest_path.read_bytes() == manifest_raw
    checked += 1
    first_summary = (output / "summary.json").read_bytes()
    completed = subprocess.run(command, cwd=Path(temporary), text=True, capture_output=True, timeout=20)
    assert completed.returncode == 0 and "CACHED" in completed.stdout, (completed.stdout, completed.stderr)
    assert (output / "summary.json").read_bytes() == first_summary
    checked += 1
    completed = subprocess.run(command + ["--seed", "43"], text=True, capture_output=True)
    assert completed.returncode != 0 and "different inputs" in completed.stderr
    checked += 1
    (output / "summary.json").write_text("{}")
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0 and "artifacts were changed" in completed.stderr
    (output / "summary.json").write_bytes(first_summary)
    checked += 1

    for label, key, value in (("incomplete", "complete", False), ("adapter", "adapter_identity", "wrong"),
                             ("rl_label", "model_label", "rl"), ("samples_hash", "samples_sha256", "0" * 64),
                             ("sample_count", "total_samples", 31), ("question_count", "question_count", 7),
                             ("sample_boolean", "n_samples", True), ("purpose", "experiment_type", "formal_test"),
                             ("no_model_identity", "model_identity", None)):
        changed = copy.deepcopy(original_manifest)
        changed[key] = value
        manifest_path.write_text(json.dumps(changed))
        completed = subprocess.run(command + ["--output", f"outputs/reject-{label}"], text=True, capture_output=True)
        assert completed.returncode != 0, (label, completed.stdout, completed.stderr)
        assert not (fixture / f"outputs/reject-{label}").exists()
        checked += 1
    manifest_path.write_bytes(manifest_raw)

    row_mutations = ["duplicate", "missing", "question", "gold", "correct", "infrastructure", "seed", "token_count", "truncation", "family", "boolean_flag"]
    for label in row_mutations:
        bad_rows = copy.deepcopy(original_rows)
        if label == "duplicate":
            bad_rows[-1] = copy.deepcopy(bad_rows[0])
        elif label == "missing":
            bad_rows = bad_rows[:-1]
        elif label == "question":
            bad_rows[0]["question"] = "Changed question"
        elif label == "gold":
            bad_rows[0]["answer"] = "3"
        elif label == "correct":
            bad_rows[0]["correct"] = True
        elif label == "infrastructure":
            bad_rows[0]["status"] = "infrastructure_error"
        elif label == "seed":
            bad_rows[0]["sample_seed"] = 1
        elif label == "token_count":
            bad_rows[0]["output_tokens"] = 999
        elif label == "truncation":
            bad_rows[0]["truncated"] = True
        elif label == "family":
            bad_rows[0]["family_id"] = "changed-family"
        elif label == "boolean_flag":
            bad_rows[0]["correct"] = "False"
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bad_rows).encode()
        samples_path.write_bytes(raw)
        changed = copy.deepcopy(original_manifest)
        changed["samples_sha256"] = hashlib.sha256(raw).hexdigest()
        manifest_path.write_text(json.dumps(changed))
        completed = subprocess.run(command + ["--output", f"outputs/reject-{label}"], text=True, capture_output=True)
        assert completed.returncode != 0, (label, completed.stdout, completed.stderr)
        assert not (fixture / f"outputs/reject-{label}").exists()
        checked += 1
    samples_path.write_bytes(sample_raw)
    manifest_path.write_bytes(manifest_raw)
    data_path.write_bytes(data_raw + b"\n")
    completed = subprocess.run(command + ["--output", "outputs/reject-data"], text=True, capture_output=True)
    assert completed.returncode != 0 and "Dataset content changed" in completed.stderr
    data_path.write_bytes(data_raw)
    checked += 1

    one_rows = [row for row in original_rows if row["sample_index"] == 0]
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in one_rows).encode()
    samples_path.write_bytes(raw)
    changed = copy.deepcopy(original_manifest)
    changed.update(n_samples=1, total_samples=8, samples_sha256=hashlib.sha256(raw).hexdigest())
    protocol = {key: changed[key] for key in ("dataset_sha256", "dataset_manifest_sha256", "n_samples", "settings", "code_hashes", "experiment_type")}
    changed["protocol_id"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest_path.write_text(json.dumps(changed))
    completed = subprocess.run(command + ["--output", "outputs/wilson"], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    single = json.loads((fixture / "outputs/wilson/summary.json").read_text())["groups"][0]
    assert single["pass_at_4"] is None
    assert "Wilson" in single["pass_at_1"]["interval_method"]
    assert single["pass_at_1"]["estimate"] == 0.75
    assert math.isclose(single["pass_at_1"]["ci_low"], 0.30064184258240184, abs_tol=1e-10)
    assert math.isclose(single["pass_at_1"]["ci_high"], 0.9544127391902995, abs_tol=1e-10)
    checked += 1

    clustered_data = copy.deepcopy(data)
    for item in clustered_data:
        item["family_id"] = f"{item['group_id']}-cluster-{item['source']['row_index'] // 2}"
    clustered_rows = copy.deepcopy(original_rows)
    for row in clustered_rows:
        row["family_id"] = f"{row['group_id']}-cluster-{row['source']['row_index'] // 2}"
    clustered_raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clustered_data).encode()
    data_path.write_bytes(clustered_raw)
    clustered_manifest = copy.deepcopy(data_manifest)
    clustered_manifest["files"]["questions.jsonl"]["sha256"] = hashlib.sha256(clustered_raw).hexdigest()
    data_manifest_path.write_text(json.dumps(clustered_manifest))
    clustered_sample_raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clustered_rows).encode()
    samples_path.write_bytes(clustered_sample_raw)
    changed = copy.deepcopy(original_manifest)
    changed.update(dataset_sha256=hashlib.sha256(clustered_raw).hexdigest(), dataset_manifest_sha256=hashlib.sha256(data_manifest_path.read_bytes()).hexdigest(),
                   samples_sha256=hashlib.sha256(clustered_sample_raw).hexdigest())
    protocol = {key: changed[key] for key in ("dataset_sha256", "dataset_manifest_sha256", "n_samples", "settings", "code_hashes", "experiment_type")}
    changed["protocol_id"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest_path.write_text(json.dumps(changed))
    completed = subprocess.run(command + ["--output", "outputs/clustered"], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    clustered = json.loads((fixture / "outputs/clustered/summary.json").read_text())["groups"][0]
    assert clustered["family_count"] == 2 and clustered["question_count"] == 4
    assert clustered["pass_at_1"]["estimate"] == 7 / 16
    assert clustered["pass_at_1"]["ci_low"] == 0.125 and clustered["pass_at_1"]["ci_high"] == 0.75
    assert "Family-cluster" in clustered["pass_at_1"]["interval_method"]
    checked += 1

    for label, requested in (("overlap", str(source)), ("parent", str(source.parent)), ("code", str(fixture))):
        completed = subprocess.run(command + ["--output", requested], text=True, capture_output=True)
        assert completed.returncode != 0 and "separate" in completed.stderr
        checked += 1
    linked = fixture / "outputs/linked"
    linked.symlink_to(source, target_is_directory=True)
    completed = subprocess.run(command + ["--input", str(linked)], text=True, capture_output=True)
    assert completed.returncode != 0 and "Symlink" in completed.stderr
    checked += 1

print(f"[PASS] {checked} baseline analysis scenarios | exact metrics, uncertainty, target groups, provenance, immutable outputs and Unicode | no model/GPU/network")
