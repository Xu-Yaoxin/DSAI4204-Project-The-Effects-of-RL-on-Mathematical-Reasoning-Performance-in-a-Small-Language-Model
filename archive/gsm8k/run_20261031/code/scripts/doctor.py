"""Read-only data and GPU readiness checks; no downloads or model execution."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import runpy
import sys

internal_check = "DOCTOR_CONFIG" in globals()
root = Path(__file__).resolve().parents[1]
if internal_check:
    cfg = DOCTOR_CONFIG
    args = argparse.Namespace(data_only=True, gpu=False, hash_model=False, output=None, world_size=1)
else:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v1.json")
    parser.add_argument("--data-only", action="store_true", help="Verify local dataset provenance and splits without GPU/model packages")
    parser.add_argument("--gpu", action="store_true", help="Also import torch and query CUDA; never load model weights")
    parser.add_argument("--world-size", type=int, choices=(1, 2, 4), default=int(os.environ.get("WORLD_SIZE", "1")),
                        help="Expected selected GPU count for --gpu (default: WORLD_SIZE or 1)")
    parser.add_argument("--hash-model", action="store_true", help="Hash all checkpoint bytes after inspecting shard metadata")
    parser.add_argument("--output", help="New JSON report path; existing files are never overwritten")
    args = parser.parse_args()
    if args.data_only and (args.gpu or args.hash_model):
        parser.error("--data-only cannot be combined with --gpu or --hash-model")
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

checks = {"python_3_11_or_later": sys.version_info >= (3, 11)}
errors = {}
versions = {}
data_details = {}
data_dir = Path(cfg["data"]["output_dir"]).expanduser()
if not data_dir.is_absolute():
    data_dir = root / data_dir
manifest_path = data_dir / "manifest.json"
checks["prepared_manifest"] = manifest_path.is_file()
prepared_manifest = {}
if manifest_path.is_file():
    try:
        manifest_bytes = manifest_path.read_bytes()
        prepared_manifest = json.loads(manifest_bytes)
        data_details["prepared_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    except (OSError, ValueError) as error:
        checks["prepared_manifest"] = False
        errors["prepared_manifest"] = str(error)
provenance = prepared_manifest.get("source_provenance", {})
checks["prepared_split_protocol"] = (prepared_manifest.get("kind") == "gsm8k_splits"
    and prepared_manifest.get("seed") == cfg["data"]["seed"])
checks["verified_data_source"] = (provenance.get("status") == "verified"
    and provenance.get("repository") == "https://github.com/openai/grade-school-math"
    and provenance.get("license") == "MIT")
data_details["source_provenance"] = provenance
raw_paths = {}
for split in ("train", "test"):
    raw_path = Path(cfg["data"]["raw_" + split]).expanduser()
    if not raw_path.is_absolute():
        raw_path = root / raw_path
    raw_paths[split] = raw_path
    checks["raw_" + split] = raw_path.is_file()
    if raw_path.is_file():
        raw_digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        checks["raw_" + split + "_matches_prepared"] = prepared_manifest.get("sources", {}).get(split, {}).get("sha256") == raw_digest
for filename, field in (("source_manifest.json", "source_manifest_sha256"), ("LICENSE", "license_sha256")):
    path = raw_paths["train"].parent / filename
    checks[field] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == provenance.get(field)

split_records = {}
for split in ("engineering", "pilot", "train", "dev", "test"):
    path = data_dir / (split + ".jsonl")
    checks["prepared_" + split] = path.is_file()
    if not path.is_file():
        continue
    try:
        payload = path.read_bytes()
        rows = [json.loads(line) for line in payload.decode("utf-8").split("\n") if line.strip()]
        if not rows or any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not isinstance(row.get("question"), str) or not row["question"].strip()
                or not isinstance(row.get("answer"), str) or row.get("split") != split for row in rows):
            raise ValueError("Expected nonempty id/question/answer records with the correct split label.")
        ids = [row["id"] for row in rows]
        expected = prepared_manifest.get("files", {}).get(path.name, {})
        checks[split + "_content_hash"] = expected.get("sha256") == hashlib.sha256(payload).hexdigest()
        checks[split + "_rows_and_ids"] = (expected.get("rows") == len(rows)
            and expected.get("ids") == ids and len(set(ids)) == len(ids)
            and cfg["data"][split + "_size"] == len(rows))
        scores = runpy.run_path(str(root / "scripts/verify_answers.py"), init_globals={"SCORE_INPUT": [
            {"text": "#### " + row["answer"], "answer": row["answer"]} for row in rows]})["SCORE_OUTPUT"]
        checks[split + "_valid_gold"] = all(score["correct"] for score in scores)
        split_records[split] = rows
        data_details[split] = {"rows": len(rows), "sha256": hashlib.sha256(payload).hexdigest()}
    except (OSError, ValueError, KeyError, TypeError) as error:
        checks["prepared_" + split] = False
        errors[split] = str(error)
checks["verifier_matches_preparation"] = prepared_manifest.get("verifier_sha256") == hashlib.sha256(
    (root / "scripts/verify_answers.py").read_bytes()).hexdigest()
if len(split_records) == 5:
    split_ids = {split: {row["id"] for row in rows} for split, rows in split_records.items()}
    checks["nested_training_subsets"] = split_ids["engineering"] <= split_ids["pilot"] <= split_ids["train"]
    training_rows = {row["id"]: (row["question"], row["answer"]) for row in split_records["train"]}
    checks["consistent_training_subsets"] = all(training_rows.get(row["id"]) == (row["question"], row["answer"])
        for split in ("engineering", "pilot") for row in split_records[split])
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        left_questions = {" ".join(row["question"].casefold().split()) for row in split_records[left]}
        right_questions = {" ".join(row["question"].casefold().split()) for row in split_records[right]}
        checks[left + "_" + right + "_disjoint"] = not (split_ids[left] & split_ids[right] or left_questions & right_questions)
data_ready = all(checks.values())

gpu = {"checked": bool(args.gpu), "expected_world_size": args.world_size}
model_details = {}
if not args.data_only:
    model_path = Path(cfg["model"]["path"]).expanduser()
    if not model_path.is_absolute():
        model_path = root / model_path
    checks["local_model_config"] = (model_path / "config.json").is_file()
    checks["local_model_weights"] = bool(list(model_path.glob("*.safetensors")))
    checks["local_tokenizer"] = (model_path / "tokenizer_config.json").is_file() and (model_path / "tokenizer.json").is_file()
    if checks["local_model_config"] and checks["local_model_weights"] and checks["local_tokenizer"]:
        try:
            model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
            text_config = model_config.get("text_config", {})
            checks["official_model_architecture"] = (model_config.get("model_type") == "qwen3_5"
                and text_config.get("model_type") == "qwen3_5_text" and text_config.get("hidden_size") == 2048
                and text_config.get("num_hidden_layers") == 24 and "quantization_config" not in model_config
                and "quantization_config" not in text_config)
            fingerprint = runpy.run_path(str(root / "scripts/fingerprint_model.py"), init_globals={
                "MODEL_PATH": model_path, "HASH_MODEL": args.hash_model})
            model_details = {key: fingerprint[key] for key in ("model_identity", "model_files", "model_layout")}
            checks["checkpoint_layout"] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            checks["checkpoint_layout"] = False
            errors["checkpoint_layout"] = str(error)
    for package, expected in {"torch": "2.10.0", "transformers": "5.3.0", "peft": "0.18.1", "trl": "1.16.0", "accelerate": "1.10.1"}.items():
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
        checks[package + "_version"] = bool(versions[package] and versions[package].split("+")[0] == expected)
    for package in ("datasets", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
        checks[package + "_installed"] = versions[package] is not None
    if args.gpu:
        try:
            import torch
            gpu.update(available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
                       visible_devices=torch.cuda.device_count())
            checks["selected_gpu_count"] = torch.cuda.device_count() == args.world_size
            selected_devices = []
            for device_index in range(min(args.world_size, torch.cuda.device_count())):
                properties = torch.cuda.get_device_properties(device_index)
                selected_devices.append({"index": device_index, "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": properties.total_memory, "bf16_hardware": properties.major >= 8})
            gpu["devices"] = selected_devices
            checks["cuda_bf16"] = (torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                                   and len(selected_devices) == args.world_size
                                   and all(device["bf16_hardware"] for device in selected_devices))
            if torch.cuda.is_available():
                gpu["name"] = torch.cuda.get_device_name()
        except (ImportError, RuntimeError, OSError) as error:
            gpu["error"] = str(error)
            checks["cuda_bf16"] = False
report = {"platform": platform.platform(), "python": sys.version, "scope": "data" if args.data_only else "training",
          "checks": checks, "errors": errors, "data": data_details, "data_ready": data_ready,
          "versions": versions, "gpu": gpu, "ready": data_ready if args.data_only else all(checks.values()) and bool(args.gpu),
          **model_details, "note": "Local integrity/readiness checks only; model weights and GPU training were not executed."}
if not internal_check:
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    raise SystemExit(0 if report["ready"] else 2)
