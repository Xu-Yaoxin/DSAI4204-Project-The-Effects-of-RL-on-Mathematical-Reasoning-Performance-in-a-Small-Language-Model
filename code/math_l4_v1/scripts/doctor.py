"""Verify frozen MATH split provenance and optional local GPU readiness."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import runpy
import sys

internal_check = "DOCTOR_CONFIG" in globals()
root = Path(__file__).resolve().parents[1]
if internal_check:
    cfg = DOCTOR_CONFIG
    args = argparse.Namespace(data_only=True, gpu=False, hash_model=False, output=None, world_size=1)
else:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/math_l4.json")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--world-size", type=int, choices=(1,), default=1)
    parser.add_argument("--hash-model", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.data_only and (args.gpu or args.hash_model):
        parser.error("--data-only cannot be combined with --gpu or --hash-model")
    cfg = json.loads((root / args.config).read_text())
checks = {"python_3_11_or_later": sys.version_info >= (3, 11)}
errors = {}
versions = {}
data_details = {}
data_dir = (root / cfg["data"]["output_dir"]).absolute()
checks["safe_data_paths"] = not any(p.is_symlink() for p in (data_dir, *data_dir.parents))
manifest_path = data_dir / "manifest.json"
checks["prepared_manifest"] = manifest_path.is_file() and not manifest_path.is_symlink()
try:
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    checks["prepared_split_protocol"] = (manifest.get("kind") == "math_level4_splits" and manifest.get("purpose") == "formal_base_rl_comparison"
        and manifest.get("seed") == cfg["data"]["seed"] and manifest.get("model_scores_used_for_selection") is False)
    data_details["prepared_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    if cfg["data"].get("manifest_sha256"):
        checks["pinned_manifest_hash"] = data_details["prepared_manifest_sha256"] == cfg["data"]["manifest_sha256"]
    provenance = manifest["source_provenance"]
    data_details["source_provenance"] = provenance
    revision = "21a5633873b6a120296cce3e2df9d5550074f4a3"
    checks["verified_data_source"] = (provenance.get("status") == "verified"
        and provenance.get("repository") == "https://huggingface.co/datasets/EleutherAI/hendrycks_math"
        and provenance.get("revision") == revision and provenance.get("license") == "Original MATH repository MIT"
        and provenance.get("official_train_rows") == 7500 and provenance.get("official_test_rows") == 5000)
    fixed = {"provenance/math_source_tree.json": "2214dac8812eb1702f205b91272e91359e98a3f7224ea577514d6e06745bfa8f",
        "provenance/gold_parser.py": "0e0121cb6994f1a5206e4d1d85fa560dbf350e4fbb9b5904f218ef2d13475ae1",
        "data/raw/math_full/README.md": "2372e22a154ca3ed5865378e270b32d8ed5b615c4592fad42630a731a950bcc0",
        "data/raw/math_full/ORIGINAL_MATH_LICENSE": "8f6401fb0b1fbfcac17aac8d61e4e1b0d2de9aa190d8aa655fedc5fd783b2ace"}
    checks["fixed_provenance_manifest"] = manifest.get("fixed_provenance_sha256") == fixed
    for relative,digest in fixed.items():
        path = root / relative
        checks["provenance_"+relative] = path.is_file() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==digest
    tree = json.loads((root / "provenance/math_source_tree.json").read_text())
    expected_sources = {"data/raw/math_full/"+r["path"]: r for r in tree if r["path"].endswith(".parquet")}
    recorded_sources = {r["path"]: r for r in provenance["source_files"]}
    checks["source_inventory"] = set(expected_sources)==set(recorded_sources) and len(recorded_sources)==14
    for relative,record in expected_sources.items():
        path = root / relative
        checks["source_"+relative] = (path.is_file() and not path.is_symlink()
            and path.stat().st_size==record["size"] and hashlib.sha256(path.read_bytes()).hexdigest()==record["lfs"]["oid"]
            and recorded_sources.get(relative,{}).get("sha256")==record["lfs"]["oid"])
    for key,expected_path in (("history_exclusions","provenance/history_exclusions.json"),("eligible_pool","provenance/eligible_pool.jsonl")):
        relative = manifest[key+"_path"]
        if relative != expected_path:
            raise ValueError("Unexpected provenance path")
        path = root / relative
        checks[key+"_hash"] = path.is_file() and not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()==manifest[key+"_sha256"]
    history = json.loads((root / manifest["history_exclusions_path"]).read_text())
    excluded = set(history["excluded_normalized_question_sha256"])
    checks["history_ledger"] = (history.get("selection_uses_observed_scores") is False
        and len(excluded)==history.get("known_prior_question_count")
        and all(len(s)==64 and all(c in "0123456789abcdef" for c in s) for s in excluded))
    pool = [json.loads(line) for line in (root / manifest["eligible_pool_path"]).read_text().split("\n") if line.strip()]
    pool_by_id = {r["id"]:r for r in pool}
    normalized = [hashlib.sha256(" ".join(r["question"].casefold().split()).encode()).hexdigest() for r in pool]
    checks["eligible_unique_rows"] = len(pool)==manifest["eligible_pool_rows"]==len(pool_by_id)==len(set(normalized))
    checks["eligible_history_exclusion"] = not (set(normalized)&excluded)
    checks["eligible_identity"] = all(r["normalized_question_sha256"]==h and r["family_id"]=="math-problem-"+h
        and r["difficulty_label"]=="Level 4" and r["source"]["dataset"]=="EleutherAI/hendrycks_math"
        and r["source"]["revision"]==revision and r["source"]["level"]==4
        and r["source"]["split"] in ("train","test") and type(r["source"]["row_index"]) is int
        and 0 <= r["source"]["row_index"] < manifest["source_rows_by_subject_split"][r["source"]["subject"]+"/"+r["source"]["split"]]
        and r["source"]["source_file"]==r["source"]["subject"]+"/"+r["source"]["split"]+"-00000-of-00001.parquet"
        and r["source"]["original_id"]==r["source"]["source_file"]+"#row="+str(r["source"]["row_index"])
        and r["id"]==f"math-l4-{r['source']['split']}-{r['source']['subject']}-{r['source']['row_index']:04d}"
        for r,h in zip(pool,normalized))
    checks["eligible_counts"] = dict(Counter(r["source"]["split"] for r in pool))==manifest["eligible_by_official_split"]
    checks["preparation_code"] = manifest["preparation_script_sha256"]==hashlib.sha256((root/"scripts/prepare_data.py").read_bytes()).hexdigest()
    split_records = {}
    for split in ("engineering","pilot","train","dev","test"):
        path = data_dir / (split+".jsonl")
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing or unsafe prepared split: "+split)
        payload = path.read_bytes()
        rows = [json.loads(line) for line in payload.decode().split("\n") if line.strip()]
        ids = [r["id"] for r in rows]
        expected = manifest["files"][path.name]
        checks[split+"_content_hash"] = hashlib.sha256(payload).hexdigest()==expected["sha256"]
        checks[split+"_rows_and_ids"] = (len(rows)==cfg["data"][split+"_size"]==expected["rows"]==manifest["sizes"][split]
            and ids==expected["ids"] and len(ids)==len(set(ids)))
        checks[split+"_source_binding"] = all(r.get("split")==split and r["id"] in pool_by_id
            and {k:v for k,v in r.items() if k!="split"}==pool_by_id[r["id"]] for r in rows)
        checks[split+"_official_split"] = all(r["source"]["split"]==("test" if split=="test" else "train") for r in rows)
        checks[split+"_subject_counts"] = dict(Counter(r["source"]["subject"] for r in rows))=={k:v for k,v in expected["subject_counts"].items() if v}
        scores = runpy.run_path(str(root/"scripts/verify_answers.py"),init_globals={"SCORE_INPUT":[{"text":"#### "+r["answer"],"answer":r["answer"]} for r in rows]})["SCORE_OUTPUT"]
        checks[split+"_valid_gold"] = all(r["correct"] is True and r["reward"]==1 for r in scores)
        split_records[split] = rows
        data_details[split] = {"rows":len(rows),"sha256":hashlib.sha256(payload).hexdigest(),"subject_counts":expected["subject_counts"]}
    split_ids = {split:{r["id"] for r in rows} for split,rows in split_records.items()}
    checks["nested_training_subsets"] = split_ids["engineering"] <= split_ids["pilot"] <= split_ids["train"]
    for left,right in (("train","dev"),("train","test"),("dev","test")):
        lh = {r["normalized_question_sha256"] for r in split_records[left]}
        rh = {r["normalized_question_sha256"] for r in split_records[right]}
        checks[left+"_"+right+"_disjoint"] = not (split_ids[left]&split_ids[right] or lh&rh)
    subjects = sorted({r["source"]["subject"] for r in pool})
    remaining = {split:[r for r in pool if r["source"]["split"]==split] for split in ("train","test")}
    expected_selected = {}
    for split in ("train","dev","test","pilot","engineering"):
        available = expected_selected["train"] if split=="pilot" else expected_selected["pilot"] if split=="engineering" else remaining["test" if split=="test" else "train"]
        count = cfg["data"][split+"_size"]
        by_subject = {subject:[r for r in available if r["source"]["subject"]==subject] for subject in subjects}
        quotas = {subject:count*len(by_subject[subject])//len(available) for subject in subjects}
        order = sorted(subjects,key=lambda subject:(-(count*len(by_subject[subject])%len(available)),subject))
        for subject in order[:count-sum(quotas.values())]:
            quotas[subject]+=1
        chosen = []
        for subject in subjects:
            rng_seed = int.from_bytes(hashlib.sha256(f"{cfg['data']['seed']}:{split}:{subject}".encode()).digest()[:8],"big")
            chosen.extend(random.Random(rng_seed).sample(sorted(by_subject[subject],key=lambda r:r["id"]),quotas[subject]))
        random.Random(int.from_bytes(hashlib.sha256(f"{cfg['data']['seed']}:{split}:order".encode()).digest()[:8],"big")).shuffle(chosen)
        expected_selected[split]=chosen
        checks[split+"_deterministic_stratification"] = [r["id"] for r in chosen]==[r["id"] for r in split_records[split]] and quotas==manifest["files"][split+".jsonl"]["subject_counts"]
        if split in ("train","dev","test"):
            used = {r["id"] for r in chosen}
            official = "test" if split=="test" else "train"
            remaining[official]=[r for r in remaining[official] if r["id"] not in used]
except (OSError,ValueError,KeyError,TypeError,IndexError,ZeroDivisionError) as error:
    checks["data_validation"] = False
    errors["data_validation"] = f"{type(error).__name__}: {error}"
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
