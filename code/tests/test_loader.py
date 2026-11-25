"""Validate checkpoint metadata, data readiness, and portable plans without model packages."""
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
fingerprint_script = root / "scripts/fingerprint_model.py"
test_count = 0
with tempfile.TemporaryDirectory(prefix="rlvr-loader-test-") as directory:
    temporary = Path(directory)
    model_dir = temporary / "model"
    model_dir.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (model_dir / name).write_text("{}")
    header = json.dumps({"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
    payload = len(header).to_bytes(8, "little") + header + b"\x00\x00\x01\x00"
    weight_path = model_dir / "model.safetensors"
    weight_path.write_bytes(payload)
    fingerprint = runpy.run_path(str(fingerprint_script), init_globals={"MODEL_PATH": model_dir})
    assert fingerprint["model_layout"]["tensor_count"] == 1
    assert fingerprint["model_identity"]
    relocated = temporary / "relocated-model"
    shutil.copytree(model_dir, relocated)
    moved = runpy.run_path(str(fingerprint_script), init_globals={"MODEL_PATH": relocated})
    assert fingerprint["model_identity"] == moved["model_identity"]
    metadata_only = runpy.run_path(str(fingerprint_script), init_globals={"MODEL_PATH": model_dir, "HASH_MODEL": False})
    assert metadata_only["model_identity"] is None and metadata_only["model_layout"] == fingerprint["model_layout"]
    test_count += 1
    weight_path.write_bytes(payload[:-1] + b"\x02")
    changed = runpy.run_path(str(fingerprint_script), init_globals={"MODEL_PATH": model_dir})
    assert changed["model_identity"] != fingerprint["model_identity"]
    test_count += 1
    for damaged in (b"", b"12345678", payload[:-1], payload + b"\x00"):
        weight_path.write_bytes(damaged)
        try:
            runpy.run_path(str(fingerprint_script), init_globals={"MODEL_PATH": model_dir})
        except (ValueError, UnicodeDecodeError):
            pass
        else:
            raise AssertionError("Damaged checkpoint accepted")
        test_count += 1
    weight_path.write_bytes(payload)
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
    try:
        runpy.run_path(str(fingerprint_script), init_globals={"MODEL_PATH": model_dir})
    except ValueError as error:
        assert "shards differ" in str(error)
    else:
        raise AssertionError("Missing shard accepted")
    test_count += 1
    try:
        runpy.run_path(str(root / "scripts/load_model.py"), init_globals={"MODEL_PATH": temporary / "absent"})
    except FileNotFoundError as error:
        assert "Missing local model" in str(error)
    else:
        raise AssertionError("Missing model must fail before importing GPU packages")
    test_count += 1

    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {
        "model_type": "qwen3_5_text", "hidden_size": 2048, "num_hidden_layers": 24}}))
    for accelerate_version in ("1.11.0", None):
        mocked_versions = ["2.10.0+cu128", "5.3.0", "0.18.1",
            accelerate_version if accelerate_version is not None else importlib.metadata.PackageNotFoundError("accelerate"),
            "4.7.0", "1.16.0", "0.6.2", None, None, None]
        with patch("importlib.metadata.version", side_effect=mocked_versions):
            try:
                runpy.run_path(str(root / "scripts/load_model.py"), init_globals={"MODEL_PATH": model_dir})
            except RuntimeError as error:
                assert "Expected accelerate==1.10.1" in str(error), str(error)
            else:
                raise AssertionError("A missing or mismatched Accelerate version must fail before GPU imports")
        test_count += 1

    # Small local fixtures exercise provenance bindings; they are never used for GPU training.
    config = json.loads((root / "configs/v1.json").read_text())
    raw_dir, processed_dir = temporary / "raw", temporary / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    config["data"].update({"raw_train": str(raw_dir / "train.jsonl"), "raw_test": str(raw_dir / "test.jsonl"),
                           "output_dir": str(processed_dir), "engineering_size": 4, "pilot_size": 8,
                           "train_size": 16, "dev_size": 4, "test_size": 4})
    config["training"].update({"num_generations": 8, "gradient_accumulation_steps": 32,
                               "steps_per_generation": 8, "per_device_train_batch_size": 1, "epochs": 3})
    config_path = temporary / "config.json"
    config_path.write_text(json.dumps(config))
    source_bytes, license_bytes = b'{"fixture": true}\n', b"MIT License\nTest fixture only\n"
    (raw_dir / "source_manifest.json").write_bytes(source_bytes)
    (raw_dir / "LICENSE").write_bytes(license_bytes)
    manifest = {"kind": "gsm8k_splits", "seed": config["data"]["seed"], "files": {}, "sources": {},
                "source_provenance": {"status": "verified", "repository": "https://github.com/openai/grade-school-math",
                    "license": "MIT", "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "license_sha256": hashlib.sha256(license_bytes).hexdigest()},
                "verifier_sha256": hashlib.sha256((root / "scripts/verify_answers.py").read_bytes()).hexdigest()}
    for split in ("train", "test"):
        raw_payload = (json.dumps({"fixture": split}) + "\n").encode()
        (raw_dir / (split + ".jsonl")).write_bytes(raw_payload)
        manifest["sources"][split] = {"sha256": hashlib.sha256(raw_payload).hexdigest()}
    training_rows = [{"id": f"train-{index}", "question": f"Fixture\u2028question {index}: 1 plus 1?", "answer": "2"} for index in range(16)]
    for split in ("engineering", "pilot", "train", "dev", "test"):
        if split in ("engineering", "pilot", "train"):
            rows = [dict(row, split=split) for row in training_rows[:config["data"][split + "_size"]]]
        else:
            rows = [{"id": f"{split}-{index}", "question": f"{split} fixture {index}: 2 plus 2?", "answer": "4", "split": split} for index in range(4)]
        split_payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        filename = split + ".jsonl"
        (processed_dir / filename).write_bytes(split_payload)
        manifest["files"][filename] = {"sha256": hashlib.sha256(split_payload).hexdigest(), "rows": len(rows), "ids": [row["id"] for row in rows]}
    (processed_dir / "manifest.json").write_text(json.dumps(manifest))
    data_check = runpy.run_path(str(root / "scripts/doctor.py"), init_globals={"DOCTOR_CONFIG": config})["report"]
    assert data_check["ready"] and data_check["data_ready"] and data_check["versions"] == {}
    assert data_check["gpu"]["checked"] is False
    test_count += 1
    command = [sys.executable, str(root / "scripts/train_grpo.py"), "--config", str(config_path),
               "--mode", "pilot", "--output", "outputs/no_model_plan_fixture", "--plan"]
    result = subprocess.run(command, cwd=temporary, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["effective_batch_size"] == 32 and plan["unique_prompts_per_update"] == 4
    assert plan["generation_batch_size"] == 8 and plan["generation_batches_per_update"] == 4
    assert plan["expected_optimizer_steps"] == 6 and plan["expected_reward_groups"] == 24
    assert plan["adapter_path"] == "outputs/no_model_plan_fixture/adapter" and plan["gpu_checked"] is False
    assert not (root / "outputs/no_model_plan_fixture").exists()
    test_count += 1
    for planned_world_size in (1, 2, 4):
        result = subprocess.run(command + ["--world-size", str(planned_world_size)], cwd=temporary, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        distributed_plan = json.loads(result.stdout)
        topology = distributed_plan["topology"]
        assert topology["world_size"] == planned_world_size
        assert topology["gradient_accumulation_steps_per_rank"] == 32 // planned_world_size
        assert topology["global_effective_batch_size"] == 32
        assert topology["global_unique_prompts_per_update"] == 4
        assert topology["per_rank_generation_batch_size"] == 8
        assert topology["global_generation_batch_size"] == 8 * planned_world_size
        assert distributed_plan["expected_optimizer_steps"] == 6
        assert distributed_plan["generation_batches_per_update"] == 4 // planned_world_size
        test_count += 1
    result = subprocess.run(command + ["--world-size", "3"], cwd=temporary, text=True, capture_output=True)
    assert result.returncode != 0 and "invalid choice" in result.stderr
    test_count += 1
    for key, value, fragment in (("steps_per_generation", 3, "complete num_generations groups"),
                                 ("steps_per_generation", 24, "divide gradient_accumulation_steps"),
                                 ("gradient_accumulation_steps", 30, "divisible by num_generations"),
                                 ("num_generations", 1, "at least two"),
                                 ("num_iterations", 2, "one optimizer pass"),
                                 ("per_device_train_batch_size", 0, "positive integer")):
        invalid = copy.deepcopy(config)
        invalid["training"][key] = value
        config_path.write_text(json.dumps(invalid))
        result = subprocess.run(command, cwd=temporary, text=True, capture_output=True)
        assert result.returncode != 0 and fragment in result.stderr, result.stderr
        test_count += 1
    config_path.write_text(json.dumps(config))
    (processed_dir / "pilot.jsonl").write_text((processed_dir / "pilot.jsonl").read_text().replace('"answer": "2"', '"answer": "3"', 1))
    tampered = runpy.run_path(str(root / "scripts/doctor.py"), init_globals={"DOCTOR_CONFIG": config})["report"]
    assert not tampered["data_ready"] and not tampered["checks"]["pilot_content_hash"]
    assert not tampered["checks"]["consistent_training_subsets"]
    test_count += 1

print(f"[TEST 100%] {test_count} loader/data-readiness/batch-plan scenarios passed; no model packages or GPU execution")
