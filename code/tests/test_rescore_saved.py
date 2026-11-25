"""Exercise offline rescoring with isolated archive fixtures and blocked model/network imports."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile


root = Path(__file__).resolve().parents[1]
checks = 0
with tempfile.TemporaryDirectory(prefix="rlvr-rescore-") as temporary:
    sandbox = Path(temporary).resolve()
    code = sandbox / "code"
    archive = sandbox / "archive"
    (code / "scripts").mkdir(parents=True)
    (archive / "scripts").mkdir(parents=True)
    for name in ("verify_answers.py", "analyze.py", "rescore_saved.py"):
        shutil.copy2(root / "scripts" / name, code / "scripts" / name)
    old_verifier = (root / "scripts/verify_answers.py").read_text(encoding="utf-8")
    old_verifier += '\nfor _saved_row, _saved_score in zip(SCORE_INPUT, SCORE_OUTPUT):\n    if "Answer: ####" in _saved_row["text"]:\n        _saved_score.update(correct=False, reward=0.0, extracted=None, status="extraction_failed", format_compliant=False)\n'
    (archive / "scripts/verify_answers.py").write_text(old_verifier, encoding="utf-8")
    (archive / "scripts/evaluate.py").write_text('raise RuntimeError("Generation must never run")\n')
    code_hashes = {name: hashlib.sha256((archive / "scripts" / name).read_bytes()).hexdigest() for name in ("verify_answers.py", "evaluate.py")}
    model_files = {"config.json": {"bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}}
    model_identity = hashlib.sha256(json.dumps(model_files, sort_keys=True).encode()).hexdigest()
    config = {"version": "fixture", "evaluation": {"n_samples": 2, "dev_samples": 2}}
    (archive / "configs").mkdir()
    (archive / "configs/v1.json").write_text(json.dumps(config))
    prepared = {"sources": {}, "files": {}}
    datasets = {}
    raw_splits = {"train": [{"question": "Training one plus one?", "answer": "1+1=2\n#### 2"}, {"question": "Dev two plus three?", "answer": "2+3=5\n#### 5"}, {"question": "Dev three plus four?", "answer": "3+4=7\n#### 7"}], "test": [{"question": "Test one plus four?", "answer": "1+4=5\n#### 5"}, {"question": "Test two plus five?", "answer": "2+5=7\n#### 7"}]}
    for split, rows in raw_splits.items():
        relative = f"data/raw/{split}.jsonl"
        path = archive / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        prepared["sources"][split] = {"path": relative, "rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    for split, official_split, indexes in (("train", "train", [0]), ("dev", "train", [1, 2]), ("test", "test", [0, 1])):
        datasets[split] = []
        for index in indexes:
            raw = raw_splits[official_split][index]
            datasets[split].append({"id": f"{official_split}-{index}", "question": raw["question"], "answer": raw["answer"].split("#### ")[1], "family_id": None, "variant_type": "original", "split": split, "source": {"dataset": "gsm8k", "official_split": official_split, "row_index": index}})
        path = archive / f"data/processed/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in datasets[split]))
        prepared["files"][path.name] = {"ids": [row["id"] for row in datasets[split]], "rows": len(datasets[split]), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (archive / "data/processed/manifest.json").write_text(json.dumps(prepared))
    settings = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 512, "seed": 42, "prompt_template": "Solve: {question}", "dtype": "bfloat16", "backend": "fixture"}
    protocols = {}
    for split in ("dev", "test"):
        protocol = {"dataset_sha256": prepared["files"][f"{split}.jsonl"]["sha256"], "n_samples": 2, "settings": settings, "code_hashes": code_hashes}
        protocols[split] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    training = archive / "outputs/v1_2/formal"
    training.mkdir(parents=True)
    (training / "report.json").write_text(json.dumps({"status": "completed", "model_identity": model_identity}))
    (training / "manifest.json").write_text(json.dumps({"fixture": True}))
    candidates = []
    for step, score in ((128, 0.25), (256, 0.5), (384, 0.75)):
        relative = f"outputs/v1_2/formal/checkpoint-{step}"
        adapter = archive / relative
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(json.dumps({"step": step}))
        (adapter / "adapter_model.safetensors").write_bytes(f"hash-only-fixture-{step}".encode())
        digest = hashlib.sha256()
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            digest.update(name.encode())
            digest.update((adapter / name).read_bytes())
        candidates.append({"step": step, "adapter_path": relative, "adapter_identity": digest.hexdigest(), "pass_at_1": score, "protocol_id": protocols["dev"], "evaluation_path": f"outputs/v1_2/selection/dev/checkpoint-{step}"})
    selection = {"schema_version": 1, "path_base": "code_root", **candidates[-1], "candidates": candidates, "model_identity": model_identity, "settings": settings, "code_hashes": code_hashes,
        "training_run": "outputs/v1_2/formal", "training_report_sha256": hashlib.sha256((training / "report.json").read_bytes()).hexdigest(), "training_manifest_sha256": hashlib.sha256((training / "manifest.json").read_bytes()).hexdigest(),
        "dev_sha256": prepared["files"]["dev.jsonl"]["sha256"], "test_sha256": prepared["files"]["test.jsonl"]["sha256"]}
    (archive / "outputs/v1_2/selection").mkdir()
    (archive / "outputs/v1_2/selection/selection.json").write_text(json.dumps(selection))
    plans = [("outputs/v1_2/test_base", "test", "base", None), ("outputs/v1_2/test_rl", "test", "rl", candidates[-1])]
    plans += [(item["evaluation_path"], "dev", "rl", item) for item in candidates]
    for relative, split, label, candidate in plans:
        rows = []
        for index, gold in enumerate(datasets[split]):
            for sample_index in range(2):
                ordinal = index * 2 + sample_index
                nested = label == "base" or split == "dev" and candidate["step"] == 128 and ordinal < 3
                correct = label == "base" or split == "dev" and candidate["step"] == 128 or ordinal < (2 if split == "dev" and candidate["step"] == 256 else 3)
                answer = gold["answer"] if correct else "99"
                text = f"Reasoning\u2028line separator\n{'Answer: #### ' if nested else '#### '}{answer}"
                rows.append({**gold, "sample_index": sample_index, "sample_seed": 100 + ordinal, "output_tokens": 20, "prompt_tokens": 10, "seconds": 0.1, "text": text, "truncated": False})
        scores = runpy.run_path(str(archive / "scripts/verify_answers.py"), init_globals={"SCORE_INPUT": rows})["SCORE_OUTPUT"]
        rows = [{**row, **score} for row, score in zip(rows, scores)]
        directory = archive / relative
        directory.mkdir(parents=True)
        (directory / "samples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        manifest = {"schema_version": 1, "complete": True, "synthetic": False, "model_label": label, "n_samples": 2, "split": split, "dataset_sha256": prepared["files"][f"{split}.jsonl"]["sha256"], "protocol_id": protocols[split], "model_identity": model_identity,
            "adapter_identity": candidate["adapter_identity"] if candidate else None, "adapter_path": candidate["adapter_path"] if candidate else None, "settings": settings, "code_hashes": code_hashes, "config": config, "model_files": model_files,
            "selection": selection if split == "test" else None, "question_count": 2, "total_samples": 4, "samples_sha256": hashlib.sha256((directory / "samples.jsonl").read_bytes()).hexdigest()}
        (directory / "manifest.json").write_text(json.dumps(manifest))

    blocker = sandbox / "blocker"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text('import sys\nfor name in ("socket", "torch", "transformers", "trl", "peft", "datasets", "accelerate", "huggingface_hub", "requests", "httpx"):\n    sys.modules[name] = None\n')
    env = {**os.environ, "PYTHONPATH": str(blocker), "PYTHONDONTWRITEBYTECODE": "1"}
    blocked = subprocess.run([sys.executable, "-c", "import socket"], env=env, capture_output=True, text=True)
    assert blocked.returncode and "None in sys.modules" in blocked.stderr
    checks += 1
    source_before = {str(path.relative_to(archive)): hashlib.sha256(path.read_bytes()).hexdigest() for path in archive.rglob("*") if path.is_file()}
    command = [sys.executable, str(code / "scripts/rescore_saved.py"), "--archive", str(archive), "--bootstrap", "100"]
    result = subprocess.run(command, cwd=sandbox, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    destination = code / "outputs/rescore_v1_4"
    ready = json.loads((destination / "ready.json").read_bytes())
    assert ready["complete"] is True and ready["models_loaded"] == ready["new_generations"] == 0
    checks += 1
    diagnostics = json.loads((destination / "diagnostics.json").read_bytes())
    assert diagnostics["test_base"]["original_correct"] == 0 and diagnostics["test_base"]["revised_correct"] == 4
    assert diagnostics["test_base"]["incorrect_to_correct"] == 4 and diagnostics["test_rl"]["revised_correct"] == 3
    checks += 1
    sensitivity = json.loads((destination / "selection_sensitivity.json").read_bytes())
    assert sensitivity["would_select_step_under_revised_dev_scores"] == 128 and sensitivity["test_adapter_step_retained"] == 384
    assert sensitivity["selection_would_change"] is True and sensitivity["automatic_reselection_performed"] is False
    checks += 1
    rescored = [json.loads(line) for line in (destination / "test_base/samples.jsonl").read_text().split("\n") if line.strip()]
    assert len(rescored) == 4 and all(row["correct"] and not row["original_score"]["correct"] for row in rescored)
    assert "\u2028" in rescored[0]["text"]
    checks += 1
    manifest = json.loads((destination / "test_base/manifest.json").read_bytes())
    assert manifest["scoring_revision"]["kind"] == "posthoc_offline_rescore" and manifest["protocol_id"] == protocols["test"]
    assert manifest["code_hashes"] == code_hashes and manifest["original_generation_manifest"]["samples_sha256"] != manifest["samples_sha256"]
    assert "posthoc" in (destination / "README.md").read_text().lower()
    checks += 1
    source_after = {str(path.relative_to(archive)): hashlib.sha256(path.read_bytes()).hexdigest() for path in archive.rglob("*") if path.is_file()}
    assert source_before == source_after
    checks += 1
    artifact_before = {str(path.relative_to(destination)): path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    result = subprocess.run(command, cwd=sandbox, env=env, capture_output=True, text=True)
    assert result.returncode == 0 and "RESCORE CACHED" in result.stdout, result.stdout + result.stderr
    assert artifact_before == {str(path.relative_to(destination)): path.read_bytes() for path in destination.rglob("*") if path.is_file()}
    checks += 1

    for scenario in ("output_tamper", "extra_output", "output_symlink", "source_tamper", "verifier_changed", "options_changed", "partial_output", "archive_overlap", "source_symlink", "candidate_adapter_tamper", "unsafe_candidate_path", "sample_gold_tamper", "sample_duplicate", "paired_seed_tamper", "dev_seed_tamper", "settings_tamper", "selection_tamper", "original_score_tamper"):
        restores = {}
        remove = []
        attempt = list(command)
        expected = ""
        if scenario == "output_tamper":
            path = destination / "diagnostics.json"
            restores[path] = path.read_bytes()
            path.write_text("{}")
            expected = "artifact hash mismatch"
        elif scenario == "extra_output":
            path = destination / "extra.txt"
            path.write_text("extra")
            remove.append(path)
            expected = "inventory differs"
        elif scenario == "output_symlink":
            path = destination / "linked"
            path.symlink_to(archive / "configs/v1.json")
            remove.append(path)
            expected = "output symlink"
        elif scenario == "source_tamper":
            path = archive / "outputs/v1_2/test_base/samples.jsonl"
            restores[path] = path.read_bytes()
            path.write_bytes(restores[path] + b" \n")
            expected = "sample hash mismatch"
        elif scenario == "verifier_changed":
            path = code / "scripts/verify_answers.py"
            restores[path] = path.read_bytes()
            path.write_bytes(restores[path] + b"\n# Changed policy revision\n")
            expected = "different source/scoring code/options"
        elif scenario == "options_changed":
            attempt += ["--seed", "101"]
            expected = "different source/scoring code/options"
        elif scenario == "partial_output":
            partial = code / "outputs/partial"
            partial.mkdir()
            attempt += ["--output", str(partial)]
            expected = "Output is partial"
        elif scenario == "archive_overlap":
            attempt += ["--output", str(archive / "outputs/new")]
            expected = "must not overlap"
        elif scenario == "source_symlink":
            path = archive / "data/processed/test.jsonl"
            restores[path] = path.read_bytes()
            other = sandbox / "external_data.jsonl"
            other.write_bytes(restores[path])
            path.unlink()
            path.symlink_to(other)
            expected = "unsafe archive input"
        elif scenario == "candidate_adapter_tamper":
            path = archive / candidates[0]["adapter_path"] / "adapter_model.safetensors"
            restores[path] = path.read_bytes()
            path.write_bytes(b"changed")
            expected = "Adapter identity mismatch"
        elif scenario in ("unsafe_candidate_path", "selection_tamper"):
            path = archive / "outputs/v1_2/selection/selection.json"
            restores[path] = path.read_bytes()
            modified = json.loads(restores[path])
            if scenario == "unsafe_candidate_path":
                modified["candidates"][0]["adapter_path"] = "../outside"
                expected = "Unsafe archive-relative"
            else:
                modified["step"] = 128
                expected = "selected checkpoint does not match"
            path.write_text(json.dumps(modified))
        else:
            directory = archive / (candidates[0]["evaluation_path"] if scenario == "dev_seed_tamper" else "outputs/v1_2/test_base")
            path = directory / "samples.jsonl"
            manifest_path = directory / "manifest.json"
            restores[path] = path.read_bytes()
            restores[manifest_path] = manifest_path.read_bytes()
            modified = [json.loads(line) for line in restores[path].decode().split("\n") if line.strip()]
            modified_manifest = json.loads(restores[manifest_path])
            if scenario == "sample_gold_tamper":
                modified[0]["answer"] = "99"
                expected = "differs from archived gold"
            elif scenario == "sample_duplicate":
                modified[1] = dict(modified[0])
                expected = "Incomplete/duplicate sample grid"
            elif scenario == "paired_seed_tamper":
                modified[0]["sample_seed"] += 1
                expected = "sample seeds differ"
            elif scenario == "dev_seed_tamper":
                modified[0]["sample_seed"] += 1
                expected = "Development candidate sample seeds differ"
            elif scenario == "settings_tamper":
                modified_manifest["settings"]["temperature"] = 0.8
                expected = "settings/code/config/model mismatch"
            elif scenario == "original_score_tamper":
                modified[0]["reward"] = 1.0
                expected = "Original verifier replay differs"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in modified), encoding="utf-8")
            modified_manifest["samples_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(modified_manifest))
        result = subprocess.run(attempt, cwd=sandbox, env=env, capture_output=True, text=True)
        assert result.returncode != 0 and expected in result.stdout + result.stderr, (scenario, result.stdout, result.stderr)
        for path, data in restores.items():
            if path.is_symlink():
                path.unlink()
            path.write_bytes(data)
        for path in remove:
            path.unlink()
        checks += 1
    shell = (root / "rescore_local.sh").read_text()
    assert 'RLVR_PYTHON:-python3' in shell and '.venv/bin/python' not in shell and 'scripts/rescore_saved.py "$@"' in shell
    checks += 1
print(f"✅ [OFFLINE RESCORE TEST 100%] {checks} provenance, Unicode, frozen-selection, immutable-output and tamper checks; socket/model imports blocked; isolated fixtures only")
