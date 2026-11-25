"""Evaluate the unchanged base model on a frozen exploratory difficulty set."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import runpy
import string
import time


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/confirmation_v2.json")
parser.add_argument("--output", default="outputs/confirmation_v2/base")
parser.add_argument("--samples", type=int, help="Override repeats; use a new output directory when changing this")
parser.add_argument("--num-shards", type=int, choices=(1, 2), default=1, help="Independent processes; all repeats of a question stay on one shard")
parser.add_argument("--shard-index", type=int, default=0)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--cache-only", action="store_true", help="Validate existing answers without loading a GPU model; exit 3 when valid but incomplete")
parser.add_argument("--plan", action="store_true", help="Validate the frozen data without loading model packages")
args = parser.parse_args()
if args.cache_only and (not args.resume or args.plan):
    parser.error("--cache-only requires --resume and cannot be combined with --plan.")
if not 0 <= args.shard_index < args.num_shards:
    parser.error("--shard-index must be in [0, --num-shards).")
root = Path(__file__).resolve().parents[1]
if any(int(os.environ.get(name, default)) != expected for name, default, expected in (
        ("WORLD_SIZE", "1", 1), ("LOCAL_WORLD_SIZE", "1", 1), ("RANK", "0", 0), ("LOCAL_RANK", "0", 0))):
    raise ValueError("Boundary evaluation is a single-process baseline run; do not use torchrun.")
config_path = (root / args.config).resolve()
output = (root / args.output).resolve()
if not config_path.is_relative_to(root) or not output.is_relative_to(root / "outputs") or output == root / "outputs":
    raise ValueError("Use configuration within code/ and a dedicated output directory within outputs/.")
if any(path.is_symlink() for path in (root / args.output, *(root / args.output).parents)):
    raise ValueError("Output path must not contain symlinks.")
cfg = json.loads(config_path.read_text())
if cfg["model"].get("dtype") != "bfloat16" or cfg["model"].get("repo_id") != "Qwen/Qwen3.5-2B-Base":
    raise ValueError("This calibration measures the original BF16 Qwen3.5-2B-Base only.")
if any(cfg["model"].get(key) is not None for key in ("adapter", "adapter_path", "adapter_identity")):
    raise ValueError("Baseline calibration cannot load a trained adapter.")
expected_model_identity = cfg["model"].get("expected_identity")
if expected_model_identity is not None and (not isinstance(expected_model_identity, str) or len(expected_model_identity) != 64
        or any(character not in "0123456789abcdef" for character in expected_model_identity)):
    raise ValueError("Expected baseline identity must be a lowercase SHA-256 digest.")
data_path = (root / cfg["data"]["path"]).resolve()
data_manifest_path = (root / cfg["data"]["manifest"]).resolve()
if any(not path.is_relative_to(root) for path in (data_path, data_manifest_path)):
    raise ValueError("Dataset paths must remain within code/.")
dataset_bytes = data_path.read_bytes()
dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
dataset_manifest_bytes = data_manifest_path.read_bytes()
dataset_manifest_sha256 = hashlib.sha256(dataset_manifest_bytes).hexdigest()
data_manifest = json.loads(dataset_manifest_bytes)
rows = [json.loads(line) for line in dataset_bytes.decode("utf-8").split("\n") if line.strip()]
if data_manifest.get("schema_version") != 1 or data_manifest.get("purpose") != "exploratory_baseline_calibration":
    raise ValueError("Dataset must declare exploratory baseline calibration, schema version 1.")
recorded_file = data_manifest.get("files", {}).get(data_path.name, {})
if recorded_file.get("sha256") != dataset_sha256 or recorded_file.get("rows") != len(rows) or not rows:
    raise ValueError("Dataset content differs from its frozen hash or count.")
groups = data_manifest.get("groups")
if (not isinstance(groups, list) or not groups or any(not isinstance(group, dict) for group in groups)
        or len({group.get("id") for group in groups}) != len(groups)):
    raise ValueError("Dataset groups must have distinct IDs.")
group_by_id = {group["id"]: group for group in groups}
for group in groups:
    if not isinstance(group["id"], str) or not group["id"] or not isinstance(group.get("label"), str) or type(group.get("count")) is not int or group["count"] < 1:
        raise ValueError("Invalid group label/count.")
for row in rows:
    if (not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("id", "question", "answer", "group_id", "group_label", "family_id", "difficulty_label"))
            or not isinstance(row.get("source"), dict) or "source_answer" not in row):
        raise ValueError("Invalid calibration question schema.")
    if row["group_id"] not in group_by_id or row["group_label"] != group_by_id[row["group_id"]]["label"]:
        raise ValueError("Question group differs from its frozen manifest.")
if len({row["id"] for row in rows}) != len(rows) or Counter(row["group_id"] for row in rows) != Counter({group["id"]: group["count"] for group in groups}):
    raise ValueError("Duplicate question IDs or mismatched group counts.")
if len({" ".join(row["question"].casefold().split()) for row in rows}) != len(rows):
    raise ValueError("Duplicate question text within calibration set.")
work_rows = [row for index, row in enumerate(rows) if index % args.num_shards == args.shard_index]
if not work_rows:
    raise ValueError("The requested shard has no questions; reduce --num-shards.")
verifier_path = root / "scripts/verify_answers.py"
verifier_code = compile(verifier_path.read_text(), str(verifier_path), "exec")
score_state = {"SCORE_INPUT": [{"text": f"#### {row['answer']}", "answer": row["answer"]} for row in rows]}
exec(verifier_code, score_state)
if not all(score["correct"] for score in score_state["SCORE_OUTPUT"]):
    raise ValueError("Every gold answer must be an exact scalar supported by the fixed verifier.")
evaluation = cfg["evaluation"]
n_samples = args.samples if args.samples is not None else evaluation["n_samples"]
if type(n_samples) is not int or not 1 <= n_samples <= 16:
    raise ValueError("n_samples must be an integer between 1 and 16.")
settings = {key: evaluation[key] for key in ("temperature", "top_p", "top_k", "max_new_tokens", "max_prompt_length", "seed", "context_limit", "length_policy")}
settings.update(prompt_template=cfg["prompt"]["template"], dtype="bfloat16", backend="transformers-sdpa",
                batch_size=1, world_size=1, add_special_tokens=True,
                stopping_rule="tokenizer-eos-or-native-context", repetition_penalty=1.0)
if (type(settings["temperature"]) not in (int, float) or not math.isfinite(settings["temperature"]) or settings["temperature"] <= 0
        or type(settings["top_p"]) not in (int, float) or not math.isfinite(settings["top_p"]) or not 0 < settings["top_p"] <= 1):
    raise ValueError("Use finite positive temperature and top_p in (0, 1].")
for field in ("max_new_tokens", "max_prompt_length", "top_k", "seed"):
    if type(settings[field]) is not int or settings[field] < (1 if field.startswith("max_") else 0):
        raise ValueError(f"Invalid generation setting: {field}")
if (type(settings["context_limit"]) is not int or settings["context_limit"] != 262144
        or settings["max_new_tokens"] != settings["context_limit"]
        or settings["length_policy"] != "native_context_minus_actual_prompt"
        or settings["max_prompt_length"] >= settings["context_limit"]):
    raise ValueError("Confirmation uses native context 262144 minus actual prompt tokens; no silent shorter budget.")
if (not isinstance(settings["prompt_template"], str) or [(name, spec, conversion)
        for _, name, spec, conversion in string.Formatter().parse(settings["prompt_template"]) if name is not None] != [("question", "", None)]):
    raise ValueError("Prompt must contain only one plain {question} field; no answer or solution fields.")
print("=" * 72, flush=True)
print(f"[BOUNDARY 1/4] DATA | base only | questions={len(rows)} | groups={len(groups)} | samples/question={n_samples}", flush=True)
print(f"[SHARD] index={args.shard_index}/{args.num_shards} | assigned_questions={len(work_rows)}/{len(rows)} | assignment=question_index_modulo", flush=True)
print(f"[BUDGET] responses={len(work_rows)*n_samples} | max_output_tokens={len(work_rows)*n_samples*settings['max_new_tokens']} | exploratory pass@1 target=40%-50%", flush=True)
print("[LENGTH POLICY] each response limit = 262144 - actual prompt tokens; EOS may end earlier", flush=True)
if expected_model_identity is not None:
    print(f"[BASE IDENTITY] required={expected_model_identity} | --plan does not read model weights", flush=True)
if args.plan:
    print("✅ [PLAN COMPLETE] Dataset hashes and scalar gold verified; no model/GPU imported.", flush=True)
    raise SystemExit(0)
if output.exists() and not args.resume:
    raise FileExistsError(f"Output exists: {output}; use --resume or a new directory.")
lock_stream = None
if output.exists():
    import fcntl
    if not (output / "manifest.json").is_file() or (output / ".evaluation.lock").is_symlink():
        raise ValueError("Output is incomplete or unsafe; preserve it and use a new directory.")
    if args.cache_only and not (output / ".evaluation.lock").is_file():
        raise ValueError("Cached evaluation lock is missing; preserve the output and inspect its integrity.")
    lock_stream = (output / ".evaluation.lock").open("r" if args.cache_only else "a")
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
package_versions = {}
for package in ("torch", "transformers", "peft", "accelerate", "datasets", "trl", "safetensors", "flash-linear-attention", "fla-core", "causal-conv1d"):
    try:
        package_versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        package_versions[package] = None
settings["package_versions"] = package_versions
code_hashes = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest()
               for name in ("evaluate_boundary.py", "verify_answers.py", "load_model.py", "fingerprint_model.py")}
experiment_type = "exploratory_baseline_calibration"
protocol = {"dataset_sha256": dataset_sha256, "dataset_manifest_sha256": dataset_manifest_sha256,
            "n_samples": n_samples, "settings": settings, "code_hashes": code_hashes, "experiment_type": experiment_type}
protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
expected_metadata = {"schema_version": 1, "experiment_type": experiment_type, "synthetic": False,
                     "model_label": "base", "adapter_path": None, "adapter_identity": None,
                     "dataset_path": cfg["data"]["path"], "dataset_sha256": dataset_sha256,
                     "dataset_manifest_path": cfg["data"]["manifest"], "dataset_manifest_sha256": dataset_manifest_sha256,
                     "n_samples": n_samples, "question_count": len(rows), "config": cfg,
                     "settings": settings, "code_hashes": code_hashes, "protocol_id": protocol_id}
if args.num_shards > 1:
    expected_metadata["sharding"] = {"count": args.num_shards, "index": args.shard_index, "assignment": "question_index_modulo"}
existing = {}
previous_manifest = None
expected_keys = {(row["id"], index) for row in work_rows for index in range(n_samples)}
row_by_id = {row["id"]: row for row in work_rows}
score_fields = ("correct", "reward", "extracted", "status", "format_compliant", "truncated")
if output.exists():
    previous_manifest = json.loads((output / "manifest.json").read_text())
    if type(previous_manifest.get("complete")) is not bool:
        raise ValueError("Invalid completion flag in cached manifest.")
    if previous_manifest.get("sharding") != expected_metadata.get("sharding"):
        raise ValueError("Cannot resume: sharding changed.")
    for field, expected in expected_metadata.items():
        if previous_manifest.get(field) != expected:
            raise ValueError(f"Cannot resume: {field} changed.")
    samples_path = output / "samples.jsonl"
    if samples_path.is_symlink():
        raise ValueError("Cached samples must not be a symlink.")
    sample_bytes = samples_path.read_bytes() if samples_path.exists() else b""
    if sample_bytes and not sample_bytes.endswith(b"\n"):
        raise ValueError("Interrupted final JSONL line: preserve the file, remove only its incomplete last line, then resume. No automatic overwrite.")
    if previous_manifest["complete"] and hashlib.sha256(sample_bytes).hexdigest() != previous_manifest.get("samples_sha256"):
        raise ValueError("Completed samples differ from their recorded hash.")
    for line in sample_bytes.decode("utf-8").split("\n"):
        if not line.strip():
            continue
        saved = json.loads(line)
        if (not isinstance(saved, dict) or saved.get("id") not in row_by_id or type(saved.get("sample_index")) is not int
                or not 0 <= saved["sample_index"] < n_samples):
            raise ValueError("Cached sample ID/index is invalid.")
        key = (saved["id"], saved["sample_index"])
        if key in existing:
            raise ValueError("Duplicate cached sample.")
        row = row_by_id[saved["id"]]
        if any(saved.get(field) != value for field, value in row.items()):
            raise ValueError("Cached question metadata differs from frozen data.")
        sample_seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{saved['id']}:{saved['sample_index']}".encode()).digest()[:4], "big")
        if (not isinstance(saved.get("text"), str) or any(type(saved.get(field)) is not bool for field in ("correct", "format_compliant", "truncated"))
                or type(saved.get("reward")) not in (int, float) or type(saved.get("sample_seed")) is not int or saved["sample_seed"] != sample_seed
                or type(saved.get("output_tokens")) is not int or not 1 <= saved["output_tokens"] <= settings["max_new_tokens"]
                or type(saved.get("prompt_tokens")) is not int or not 1 <= saved["prompt_tokens"] <= settings["max_prompt_length"]
                or type(saved.get("seconds")) not in (int, float) or not math.isfinite(saved["seconds"]) or saved["seconds"] < 0
                or (saved["truncated"] and saved["output_tokens"] != saved.get("generation_limit"))):
            raise ValueError("Cached generation or score metadata is invalid.")
        if (type(saved.get("generation_limit")) is not int
                or saved["generation_limit"] != settings["context_limit"] - saved["prompt_tokens"]
                or saved["output_tokens"] > saved["generation_limit"]):
            raise ValueError("Cached per-answer native context budget is invalid.")
        score_state = {"SCORE_INPUT": [saved]}
        exec(verifier_code, score_state)
        if any(saved.get(field) != score_state["SCORE_OUTPUT"][0][field] for field in score_fields):
            raise ValueError("Cached score differs from verifier replay.")
        existing[key] = saved
    if previous_manifest["complete"] and (set(existing) != expected_keys or previous_manifest.get("total_samples") != len(expected_keys)):
        raise ValueError("Completed evaluation has an incomplete sample grid.")
if previous_manifest:
    identity = runpy.run_path(str(root / "scripts/fingerprint_model.py"), init_globals={"MODEL_PATH": root / cfg["model"]["path"]})
    if expected_model_identity is not None and identity["model_identity"] != expected_model_identity:
        raise ValueError("Base identity differs from the expected original baseline in configuration.")
    if previous_manifest.get("model_identity") != identity["model_identity"] or previous_manifest.get("model_files") != identity["model_files"]:
        raise ValueError("Cannot resume: original base model files changed.")
if set(existing) == expected_keys:
    if not previous_manifest["complete"]:
        previous_manifest.update(complete=True, total_samples=len(existing), recovered_complete_on_resume=True,
                                 samples_sha256=hashlib.sha256((output / "samples.jsonl").read_bytes()).hexdigest())
        (output / "manifest.json.tmp").write_text(json.dumps(previous_manifest, indent=2) + "\n")
        (output / "manifest.json.tmp").replace(output / "manifest.json")
    print(f"✅ [BOUNDARY CACHED] 100% | {len(existing)} answers verified; no GPU model loaded | {output}", flush=True)
    if lock_stream is not None:
        lock_stream.close()
    raise SystemExit(0)

if args.cache_only:
    print(f"[CACHE INCOMPLETE] verified={len(existing)}/{len(expected_keys)} | no GPU model loaded | {output}", flush=True)
    if lock_stream is not None:
        lock_stream.close()
    raise SystemExit(3)

print(f"[BOUNDARY 2/4] LOAD | unchanged base weights | remaining={len(expected_keys)-len(existing)} | adapter=None", flush=True)
execution_cpu_threads = None
if "RLVR_CPU_THREADS" in os.environ:
    try:
        execution_cpu_threads = int(os.environ["RLVR_CPU_THREADS"])
    except ValueError as error:
        raise ValueError("RLVR_CPU_THREADS must be a positive integer.") from error
    if execution_cpu_threads < 1:
        raise ValueError("RLVR_CPU_THREADS must be a positive integer.")
    import torch
    torch.set_num_threads(execution_cpu_threads)
    torch.set_num_interop_threads(1)
loaded = runpy.run_path(str(root / "scripts/load_model.py"), init_globals={
    "MODEL_PATH": root / cfg["model"]["path"], "ADAPTER_PATH": None, "TRAINING": False})
if loaded["adapter_identity"] is not None or loaded["package_versions"] != package_versions:
    raise ValueError("Baseline loader returned an adapter or a changed software environment.")
if expected_model_identity is not None and loaded["model_identity"] != expected_model_identity:
    raise ValueError("Base identity differs from the expected original baseline in configuration.")
if previous_manifest and previous_manifest["model_identity"] != loaded["model_identity"]:
    raise ValueError("Base model changed after resume validation.")
model, tokenizer = loaded["model"], loaded["tokenizer"]
if getattr(model.config, "max_position_embeddings", None) != settings["context_limit"]:
    raise ValueError("Loaded model context differs from pinned native context; no extension or silent shortening.")
if tokenizer.eos_token_id is None:
    raise ValueError("Tokenizer EOS must be defined.")
import torch
from transformers import set_seed

prompt_lengths = {}
for row in work_rows:
    prompt = settings["prompt_template"].format(question=row["question"])
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    length = tokens["input_ids"].shape[1]
    if not 1 <= length <= settings["max_prompt_length"]:
        raise ValueError(f"Prompt too long: {row['id']} has {length} tokens; no questions are silently truncated.")
    prompt_lengths[row["id"]] = length
print(f"[PREFLIGHT] all {len(work_rows)} assigned prompts fit | maximum={max(prompt_lengths.values())} tokens", flush=True)
if not output.exists():
    output.mkdir(parents=True, exist_ok=False)
    import fcntl
    lock_stream = (output / ".evaluation.lock").open("a")
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
manifest = {**expected_metadata, "path_base": "code_root", "complete": False,
            "model_identity": loaded["model_identity"], "model_files": loaded["model_files"],
            "gpu": torch.cuda.get_device_name(), "groups": groups}
if execution_cpu_threads is not None:
    manifest["execution_cpu_threads"] = {"intraop": execution_cpu_threads, "interop": 1}
(output / "manifest.json.tmp").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
(output / "manifest.json.tmp").replace(output / "manifest.json")
start = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
completed = len(existing)
initial_completed = completed
new_tokens = 0
print("[BOUNDARY 3/4] GENERATE | responses are independently seeded; gold never enters the prompt", flush=True)
with (output / "samples.jsonl").open("a", encoding="utf-8") as stream:
    for row in work_rows:
        if all((row["id"], sample_index) in existing for sample_index in range(n_samples)):
            continue
        prompt = settings["prompt_template"].format(question=row["question"])
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length != prompt_lengths[row["id"]]:
            raise ValueError("Tokenizer output changed after prompt preflight.")
        generation_limit = settings["context_limit"] - prompt_length
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        for sample_index in range(n_samples):
            if (row["id"], sample_index) in existing:
                continue
            sample_seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{row['id']}:{sample_index}".encode()).digest()[:4], "big")
            set_seed(sample_seed)
            sample_start = time.perf_counter()
            print(f"[ANSWER START] id={row['id']} sample={sample_index+1}/{n_samples} | prompt_tokens={prompt_length} | generation_limit={generation_limit}", flush=True)
            try:
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=True, temperature=settings["temperature"], top_p=settings["top_p"],
                        top_k=settings["top_k"], max_new_tokens=generation_limit, num_return_sequences=1,
                        repetition_penalty=1.0, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                continuation = generated[0, prompt_length:].tolist()
                if not 1 <= len(continuation) <= generation_limit:
                    raise ValueError("Generation returned an invalid number of continuation tokens.")
                ended = continuation[-1] == tokenizer.eos_token_id
                truncated = len(continuation) >= generation_limit and not ended
                text = tokenizer.decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                score_state = {"SCORE_INPUT": [{"text": text, "answer": row["answer"], "truncated": truncated}]}
                exec(verifier_code, score_state)
                score = score_state["SCORE_OUTPUT"][0]
            except Exception as error:
                with (output / "infrastructure_errors.jsonl").open("a", encoding="utf-8") as errors:
                    errors.write(json.dumps({"id": row["id"], "sample_index": sample_index, "sample_seed": sample_seed,
                                            "status": "infrastructure_error", "error": repr(error)}) + "\n")
                if lock_stream is not None:
                    lock_stream.close()
                raise
            sample = {**row, **score, "sample_index": sample_index, "sample_seed": sample_seed, "text": text,
                      "output_tokens": len(continuation), "prompt_tokens": prompt_length, "generation_limit": generation_limit, "seconds": time.perf_counter() - sample_start}
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            completed += 1
            new_tokens += len(continuation)
            elapsed = time.perf_counter() - start
            eta = elapsed / (completed - initial_completed) * (len(expected_keys) - completed)
            print(f"[BOUNDARY PROGRESS] {100*completed/len(expected_keys):6.2f}% | {completed}/{len(expected_keys)} | group={row['group_id']} | status={score['status']} | tokens={len(continuation)} | elapsed={elapsed:.0f}s | ETA={eta/60:.1f}min | [SAVED] {output/'samples.jsonl'}", flush=True)
elapsed = time.perf_counter() - start
manifest.update(complete=True, total_samples=completed, invocation_seconds=elapsed, invocation_output_tokens=new_tokens,
                invocation_tokens_per_second=new_tokens/max(elapsed, 1e-9), peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                samples_sha256=hashlib.sha256((output / "samples.jsonl").read_bytes()).hexdigest())
(output / "manifest.json.tmp").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
(output / "manifest.json.tmp").replace(output / "manifest.json")
if lock_stream is not None:
    lock_stream.close()
print(f"✅ [BOUNDARY 4/4] 100% | COMPLETE | {completed} baseline answers | [SAVED] {output/'manifest.json'}", flush=True)
