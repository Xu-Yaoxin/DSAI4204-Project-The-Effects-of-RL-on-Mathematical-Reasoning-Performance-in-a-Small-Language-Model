"""Verify two-worker merges against frozen data and the actual baseline analyzer."""

import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


root = Path(__file__).resolve().parents[1]
checked = 0
with tempfile.TemporaryDirectory(prefix="boundary-merge-test-") as temporary:
    fixture = Path(temporary).resolve() / "code"
    (fixture / "scripts").mkdir(parents=True)
    for filename in ("merge_boundary.py", "analyze_boundary.py", "verify_answers.py", "validate_guarded_sample.py", "repetition_guard.py"):
        shutil.copyfile(root / "scripts" / filename, fixture / "scripts" / filename)
    data_dir = fixture / "data/boundary_v1"
    data_dir.mkdir(parents=True)
    dataset = []
    for group in ("gsm_symbolic", "math_l4_5"):
        for index in range(4):
            dataset.append({"id": f"{group}-{index}", "question": f"Fixture\u2028question {group}/{index}: one plus one?", "answer": "2",
                "source_answer": "2", "group_id": group, "group_label": group, "family_id": f"family-{group}-{index}",
                "source": {"dataset": "synthetic", "revision": "fixture", "split": "test", "row_index": index, "original_id": index},
                "difficulty_label": "fixture only"})
    data_raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dataset).encode()
    data_path = data_dir / "questions.jsonl"
    data_path.write_bytes(data_raw)
    groups = [{"id": group, "label": group, "count": 4} for group in ("gsm_symbolic", "math_l4_5")]
    data_manifest = {"schema_version": 1, "purpose": "exploratory_baseline_calibration", "groups": groups,
        "files": {"questions.jsonl": {"sha256": hashlib.sha256(data_raw).hexdigest(), "rows": len(dataset)}}}
    data_manifest_path = data_dir / "manifest.json"
    data_manifest_path.write_text(json.dumps(data_manifest))
    settings = {"temperature": 0.6, "top_p": 0.95, "top_k": 0, "max_new_tokens": 262144, "context_limit": 262144, "length_policy": "native_context_minus_actual_prompt", "max_prompt_length": 128, "seed": 271828,
        "prompt_template": "Solve {question}", "dtype": "bfloat16", "backend": "synthetic", "batch_size": 1, "world_size": 1}
    settings["generation_guard"] = json.loads((root / "configs/guarded_v3.json").read_text())["evaluation"]["generation_guard"]
    settings["eos_token_id"] = 248044
    model_files = {"model.safetensors": {"bytes": 4, "sha256": "f" * 64}}
    model_identity = hashlib.sha256(json.dumps(model_files, sort_keys=True).encode()).hexdigest()
    config = {"model": {"repo_id": "Qwen/Qwen3.5-2B-Base", "path": "models/fixture", "dtype": "bfloat16", "expected_identity": model_identity},
        "data": {"path": "data/boundary_v1/questions.jsonl", "manifest": "data/boundary_v1/manifest.json"},
        "prompt": {"template": "Solve {question}"}, "evaluation": {**settings, "n_samples": 4}}
    template = {"schema_version": 1, "experiment_type": "exploratory_baseline_calibration", "synthetic": True, "path_base": "code_root", "model_label": "base",
        "model_identity": model_identity, "model_files": model_files, "adapter_identity": None, "adapter_path": None, "n_samples": 4,
        "question_count": 8, "total_samples": 16, "complete": True, "groups": groups, "config": config,
        "dataset_path": "data/boundary_v1/questions.jsonl", "dataset_sha256": hashlib.sha256(data_raw).hexdigest(),
        "dataset_manifest_path": "data/boundary_v1/manifest.json", "dataset_manifest_sha256": hashlib.sha256(data_manifest_path.read_bytes()).hexdigest(),
        "settings": settings, "code_hashes": {name: hashlib.sha256((fixture / "scripts" / name).read_bytes()).hexdigest() for name in ("verify_answers.py", "validate_guarded_sample.py", "repetition_guard.py")}}
    protocol = {key: template[key] for key in ("dataset_sha256", "dataset_manifest_sha256", "n_samples", "settings", "code_hashes", "experiment_type")}
    template["protocol_id"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    all_rows = []
    for item in dataset:
        count = [0, 1, 2, 4][item["source"]["row_index"]] if item["group_id"] == "gsm_symbolic" else 4
        for index in range(4):
            text = "#### 2" if index < count else "#### 3"
            score = runpy.run_path(str(fixture / "scripts/verify_answers.py"), init_globals={"SCORE_INPUT": [{"text": text, "answer": "2"}]})["SCORE_OUTPUT"][0]
            all_rows.append({**item, **score, "sample_index": index, "text": text,
                "sample_seed": int.from_bytes(hashlib.sha256(f"{settings['seed']}:{item['id']}:{index}".encode()).digest()[:4], "big"),
                "output_tokens": 4, "output_token_ids": [12, 34, 56, 248044], "termination_reason": "eos", "scoring_evaluable": True, "guard_evidence": None, "timeout_evidence": None, "prompt_tokens": 20, "generation_limit": 262124, "seconds": 0.01})
    paths = []
    original_manifests = []
    original_rows = []
    original_manifest_bytes = []
    original_sample_bytes = []
    for shard in (0, 1):
        directory = fixture / f"outputs/workers/shard-{shard}"
        directory.mkdir(parents=True)
        paths.append(directory)
        assigned = {row["id"] for index, row in enumerate(dataset) if index % 2 == shard}
        rows = [row for row in all_rows if row["id"] in assigned]
        rows.reverse()
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        manifest = copy.deepcopy(template)
        manifest.update(sharding={"count": 2, "index": shard, "assignment": "question_index_modulo"}, samples_sha256=hashlib.sha256(raw).hexdigest(),
            invocation_seconds=1 + shard, invocation_output_tokens=len(rows) * 4, gpu="synthetic-no-GPU", peak_allocated_gib=0, execution_cpu_threads=2)
        (directory / "samples.jsonl").write_bytes(raw)
        (directory / "manifest.json").write_text(json.dumps(manifest))
        original_manifests.append(manifest)
        original_rows.append(rows)
        original_manifest_bytes.append((directory / "manifest.json").read_bytes())
        original_sample_bytes.append(raw)
    output = fixture / "outputs/guarded_v3/base"
    command = [sys.executable, "-S", str(fixture / "scripts/merge_boundary.py"), "--inputs", str(paths[1]), str(paths[0])]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=20)
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    manifest = json.loads((output / "manifest.json").read_text())
    merged = [json.loads(line) for line in (output / "samples.jsonl").read_text().split("\n") if line.strip()]
    assert merged == all_rows
    assert "sharding" not in manifest and manifest["total_samples"] == 32 and manifest["question_count"] == 8
    assert manifest["protocol_id"] == template["protocol_id"] and manifest["settings"] == settings
    assert manifest["execution_provenance"]["workers"] == 2 and manifest["execution_provenance"]["generation_protocol_unchanged"] is True
    assert "invocation_seconds" not in manifest and "peak_allocated_gib" not in manifest
    assert not list(output.parent.glob(".base.merge-*/"))
    for index, directory in enumerate(paths):
        assert (directory / "manifest.json").read_bytes() == original_manifest_bytes[index]
        assert (directory / "samples.jsonl").read_bytes() == original_sample_bytes[index]
    checked += 1
    analyzed = subprocess.run([sys.executable, "-S", str(fixture / "scripts/analyze_boundary.py"), "--input", str(output),
        "--output", "outputs/analysis", "--bootstrap", "100"], text=True, capture_output=True, timeout=20)
    assert analyzed.returncode == 0, (analyzed.stdout, analyzed.stderr)
    summary = json.loads((fixture / "outputs/analysis/summary.json").read_text())
    assert summary["groups"][0]["pass_at_1"]["estimate"] == 7 / 16
    assert summary["groups"][1]["pass_at_1"]["estimate"] == 1
    assert summary["synthetic"] is True
    checked += 1
    snapshot = {path.name: path.read_bytes() for path in output.iterdir()}
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0 and "CACHED" in completed.stdout
    assert {path.name: path.read_bytes() for path in output.iterdir()} == snapshot
    checked += 1
    reverse_command = command[:-2] + [str(paths[0]), str(paths[1])]
    completed = subprocess.run(reverse_command, text=True, capture_output=True)
    assert completed.returncode == 0 and "CACHED" in completed.stdout
    checked += 1
    (output / "samples.jsonl").write_text("{}\n")
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0 and "artifacts were changed" in completed.stderr
    (output / "samples.jsonl").write_bytes(snapshot["samples.jsonl"])
    checked += 1

    mutations = [("incomplete", "complete", False), ("adapter", "adapter_identity", "trained-adapter"), ("not_base", "model_label", "rl"),
        ("model", "model_identity", "0" * 64), ("count", "total_samples", 15), ("questions", "question_count", 4),
        ("hash", "samples_sha256", "0" * 64), ("protocol", "protocol_id", "0" * 64), ("duplicate_shard", "sharding", {"count": 2, "index": 0, "assignment": "question_index_modulo"}),
        ("wrong_assignment", "sharding", {"count": 2, "index": 1, "assignment": "random"}), ("not_sharded", "sharding", None),
        ("different_config", "config", {**config, "extra": 1}), ("different_settings", "settings", {**settings, "temperature": 1.0})]
    for name, key, value in mutations:
        changed = copy.deepcopy(original_manifests[1])
        changed[key] = value
        (paths[1] / "manifest.json").write_text(json.dumps(changed))
        target = f"outputs/reject-{name}"
        completed = subprocess.run(command + ["--output", target], text=True, capture_output=True)
        assert completed.returncode != 0, (name, completed.stdout, completed.stderr)
        assert not (fixture / target).exists()
        checked += 1
    (paths[1] / "manifest.json").write_bytes(original_manifest_bytes[1])

    for name in ("duplicate", "missing", "wrong_shard", "gold", "status", "correct", "seed", "seconds", "tokens", "truncated", "generation_limit", "partial_line"):
        rows = copy.deepcopy(original_rows[1])
        if name == "duplicate":
            rows[-1] = copy.deepcopy(rows[0])
        elif name == "missing":
            rows.pop()
        elif name == "wrong_shard":
            rows[0] = copy.deepcopy(original_rows[0][0])
        elif name == "gold":
            rows[0]["answer"] = "999"
        elif name == "status":
            rows[0]["status"] = "infrastructure_error"
        elif name == "correct":
            rows[0]["correct"] = not rows[0]["correct"]
        elif name == "seed":
            rows[0]["sample_seed"] = 3
        elif name == "seconds":
            rows[0]["seconds"] = -1
        elif name == "tokens":
            rows[0]["output_tokens"] = 262144
        elif name == "generation_limit":
            rows[0]["generation_limit"] += 1
        elif name == "truncated":
            rows[0]["truncated"] = True
        raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        if name == "partial_line":
            raw = raw[:-1]
        (paths[1] / "samples.jsonl").write_bytes(raw)
        changed = copy.deepcopy(original_manifests[1])
        changed["samples_sha256"] = hashlib.sha256(raw).hexdigest()
        (paths[1] / "manifest.json").write_text(json.dumps(changed))
        target = f"outputs/reject-row-{name}"
        completed = subprocess.run(command + ["--output", target], text=True, capture_output=True)
        assert completed.returncode != 0, (name, completed.stdout, completed.stderr)
        assert not (fixture / target).exists()
        checked += 1
    (paths[1] / "manifest.json").write_bytes(original_manifest_bytes[1])
    (paths[1] / "samples.jsonl").write_bytes(original_sample_bytes[1])

    changed = copy.deepcopy(original_manifests[1])
    changed["invocation_seconds"] = 100
    (paths[1] / "manifest.json").write_text(json.dumps(changed))
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0 and "different sources" in completed.stderr
    (paths[1] / "manifest.json").write_bytes(original_manifest_bytes[1])
    checked += 1
    data_path.write_bytes(data_raw + b"\n")
    completed = subprocess.run(command + ["--output", "outputs/reject-data"], text=True, capture_output=True)
    assert completed.returncode != 0 and "Frozen dataset input changed" in completed.stderr
    data_path.write_bytes(data_raw)
    checked += 1
    verifier = fixture / "scripts/verify_answers.py"
    original_verifier = verifier.read_bytes()
    verifier.write_bytes(original_verifier + b"\n# Changed scoring source\n")
    completed = subprocess.run(command + ["--output", "outputs/reject-verifier"], text=True, capture_output=True)
    assert completed.returncode != 0 and "Scorer identity changed" in completed.stderr
    verifier.write_bytes(original_verifier)
    checked += 1

    partial = fixture / "outputs/partial"
    partial.mkdir()
    (partial / "preserve.txt").write_text("keep original partial output")
    completed = subprocess.run(command + ["--output", str(partial)], text=True, capture_output=True)
    assert completed.returncode != 0 and "partial" in completed.stderr
    assert (partial / "preserve.txt").read_text() == "keep original partial output"
    checked += 1
    for invalid_output in (paths[0], paths[0].parent, fixture):
        completed = subprocess.run(command + ["--output", str(invalid_output)], text=True, capture_output=True)
        assert completed.returncode != 0 and "separate" in completed.stderr
        checked += 1
    linked = fixture / "outputs/linked"
    linked.symlink_to(paths[0], target_is_directory=True)
    bad_command = command[:-2] + [str(linked), str(paths[1])]
    completed = subprocess.run(bad_command, text=True, capture_output=True)
    assert completed.returncode != 0 and "Symlink" in completed.stderr
    checked += 1
    bad_command = command[:-2] + [str(paths[0]), str(paths[0])]
    completed = subprocess.run(bad_command, text=True, capture_output=True)
    assert completed.returncode != 0 and "distinct" in completed.stderr
    checked += 1

print(f"[PASS] {checked} guarded merge integrity scenarios | identities, exact EOS scores, atomic publication, cached reuse and analyzer integration | no model/GPU/network")
