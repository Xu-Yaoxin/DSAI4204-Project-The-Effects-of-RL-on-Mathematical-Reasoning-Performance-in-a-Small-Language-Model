"""Sample questions independently and validate cached runs before loading a GPU model."""
import argparse
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
parser.add_argument("--config", default="configs/v1.json")
parser.add_argument("--model", choices=["base", "rl"], required=True)
parser.add_argument("--adapter")
parser.add_argument("--split", choices=["engineering", "pilot", "dev", "test", "structural"], required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--samples", type=int)
parser.add_argument("--selection", help="Frozen development selection JSON, required for test/structural")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--plan", action="store_true", help="Validate data and show budget without loading a model")
args = parser.parse_args()
if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1
        or int(os.environ.get("RANK", "0")) not in {-1, 0} or int(os.environ.get("LOCAL_RANK", "0")) not in {-1, 0}):
    raise ValueError("Evaluation uses one process and one selected GPU; do not launch evaluate.py with multi-process torchrun.")
root = Path(__file__).resolve().parents[1]
config_path = (root / args.config).resolve()
cfg = json.loads(config_path.read_text())
output = (root / args.output).resolve()
if output.exists() and not args.resume and not args.plan:
    raise FileExistsError(f"Output exists: {output}; use a new path or --resume.")
data_path = (root / cfg["data"]["output_dir"] / f"{args.split}.jsonl").resolve()
dataset_bytes = data_path.read_bytes()
dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
rows = [json.loads(line) for line in dataset_bytes.decode().split("\n") if line.strip()]
if not rows:
    raise ValueError("Dataset must be nonempty with unique IDs.")
for row in rows:
    if (not isinstance(row.get("id"), str) or not row["id"].strip() or row.get("split") != args.split
            or not isinstance(row.get("question"), str) or not row["question"].strip()
            or not isinstance(row.get("answer"), str) or not isinstance(row.get("variant_type"), str)
            or (row.get("family_id") is not None and not isinstance(row["family_id"], str))):
        raise ValueError("Unexpected data schema/split.")
if len({row["id"] for row in rows}) != len(rows):
    raise ValueError("Dataset must be nonempty with unique IDs.")
verifier_source = (root / "scripts/verify_answers.py").read_text()
verifier_code = compile(verifier_source, str(root / "scripts/verify_answers.py"), "exec")
score_state = {"SCORE_INPUT": [{"text": f"#### {row['answer']}", "answer": row["answer"]} for row in rows]}
exec(verifier_code, score_state)
if not all(item["correct"] for item in score_state["SCORE_OUTPUT"]):
    raise ValueError("Dataset contains an invalid canonical answer.")
if args.model == "rl" and not args.adapter:
    raise ValueError("RL evaluation requires --adapter.")
if args.model == "base" and args.adapter:
    raise ValueError("Base evaluation cannot load an adapter.")
review_hash = None
if args.split == "structural":
    review_bytes = data_path.with_name("structural_review.json").read_bytes()
    review_hash = hashlib.sha256(review_bytes).hexdigest()
    review = json.loads(review_bytes)
    if (not isinstance(review.get("records"), list) or not isinstance(review.get("reviewer"), str)
            or review.get("approved") is not True or not review["reviewer"].strip()
            or review.get("dataset_sha256") != dataset_hash):
        raise ValueError("Structural data requires a complete human review bound to its content hash.")
    reviewed = {}
    for item in review["records"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] in reviewed:
            raise ValueError("Structural review contains invalid or duplicate record IDs.")
        reviewed[item["id"]] = item
    if set(reviewed) != {row["id"] for row in rows}:
        raise ValueError("Structural review must cover exactly the evaluated question IDs.")
    for row in rows:
        item = reviewed[row["id"]]
        if (not row.get("family_id") or item.get("verified") is not True
                or not isinstance(item.get("reviewed_answer"), str) or not item["reviewed_answer"].strip()
                or item.get("question_sha256") != hashlib.sha256(row["question"].encode()).hexdigest()):
            raise ValueError(f"Unreviewed structural question: {row['id']}")
    score_state = {"SCORE_INPUT": [{"text": f"#### {row['answer']}", "answer": reviewed[row["id"]]["reviewed_answer"]} for row in rows]}
    exec(verifier_code, score_state)
    if not all(item["correct"] for item in score_state["SCORE_OUTPUT"]):
        raise ValueError("A human-reviewed structural answer differs from the dataset gold.")
evaluation = cfg["evaluation"]
n_samples = args.samples if args.samples is not None else (evaluation["dev_samples"] if args.split == "dev" else evaluation["n_samples"])
if type(n_samples) is not int or not 1 <= n_samples <= 100:
    raise ValueError("Samples must be an integer between 1 and 100.")
settings = {
    "temperature": evaluation["temperature"], "top_p": evaluation["top_p"], "top_k": evaluation["top_k"],
    "max_new_tokens": evaluation["max_new_tokens"], "max_prompt_length": evaluation["max_prompt_length"],
    "seed": evaluation["seed"], "prompt_template": cfg["prompt"]["template"],
    "dtype": cfg["model"]["dtype"], "backend": "transformers-sdpa", "batch_size": 1, "world_size": 1,
    "add_special_tokens": True, "stopping_rule": "tokenizer-eos-or-max_new_tokens", "repetition_penalty": 1.0,
}
if (settings["dtype"] != "bfloat16" or type(settings["temperature"]) not in {int, float}
        or not math.isfinite(settings["temperature"]) or settings["temperature"] <= 0
        or type(settings["top_p"]) not in {int, float} or not 0 < settings["top_p"] <= 1):
    raise ValueError("v1 requires BF16 and finite positive sampling temperature/top_p.")
for field in ["max_new_tokens", "max_prompt_length", "top_k", "seed"]:
    if type(settings[field]) is not int or settings[field] < (1 if field.startswith("max_") else 0):
        raise ValueError(f"Invalid integer generation setting: {field}")
if (not isinstance(settings["prompt_template"], str)
        or [(name, spec, conversion) for _, name, spec, conversion in string.Formatter().parse(settings["prompt_template"]) if name is not None] != [("question", "", None)]):
    raise ValueError("Prompt must contain exactly one plain {question} field and no other fields.")
selection = None
if args.split in {"test", "structural"}:
    if not args.selection:
        raise ValueError("Freeze the development checkpoint selection before formal evaluation; supply --selection.")
    selection = json.loads((root / args.selection).read_text())
    if selection.get("schema_version") != 1 or selection["evaluation_config"] != evaluation or selection["prompt"] != cfg["prompt"]:
        raise ValueError("Evaluation configuration changed after development selection.")
    if n_samples != evaluation["n_samples"]:
        raise ValueError("Change sample budget in a new frozen configuration before testing.")
    if selection.get(f"{args.split}_sha256") != dataset_hash:
        raise ValueError(f"{args.split} data differ from the frozen development selection.")
print(f"[EVAL PLAN] split={args.split} questions={len(rows)} samples={n_samples} answers={len(rows)*n_samples} max_output_tokens={len(rows)*n_samples*settings['max_new_tokens']}", flush=True)
if args.plan:
    raise SystemExit(0)

package_versions = {}
for package in ["torch", "transformers", "peft", "accelerate", "datasets", "trl", "safetensors", "flash-linear-attention", "fla-core", "causal-conv1d"]:
    try:
        package_versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        package_versions[package] = None
settings["package_versions"] = package_versions
code_hashes = {name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest()
               for name in ["verify_answers.py", "load_model.py", "fingerprint_model.py", "evaluate.py", "select_checkpoint.py"]}
if selection and (selection.get("settings") != settings or selection.get("code_hashes") != code_hashes):
    raise ValueError("Software, verifier or generation protocol changed after development selection.")
if selection:
    if not isinstance(selection.get("training_run"), str) or not selection["training_run"].strip():
        raise ValueError("Frozen selection is missing its training-run provenance.")
    selected_training_run = (root / selection["training_run"]).resolve()
    for filename, field in [("report.json", "training_report_sha256"), ("manifest.json", "training_manifest_sha256")]:
        if hashlib.sha256((selected_training_run / filename).read_bytes()).hexdigest() != selection.get(field):
            raise ValueError(f"Training dependency {filename} changed after development selection.")
protocol = {"dataset_sha256": dataset_hash, "n_samples": n_samples, "settings": settings, "code_hashes": code_hashes}
protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()

adapter_path = (root / args.adapter).resolve() if args.adapter else None
adapter_identity = None
train_manifest = None
if adapter_path:
    train_manifest_path = adapter_path.parent / "manifest.json"
    if not train_manifest_path.is_file():
        raise ValueError("Adapter must belong to a v1 training run with manifest.json.")
    train_manifest = json.loads(train_manifest_path.read_text())
    adapter_digest = hashlib.sha256()
    for filename in ["adapter_config.json", "adapter_model.safetensors"]:
        adapter_digest.update(filename.encode())
        with (adapter_path / filename).open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                adapter_digest.update(chunk)
    adapter_identity = adapter_digest.hexdigest()
    if selection and selection["adapter_identity"] != adapter_identity:
        raise ValueError("RL checkpoint is not the selected development checkpoint.")

existing = {}
previous_manifest = None
expected_keys = {(row["id"], i) for row in rows for i in range(n_samples)}
if output.exists():
    previous_manifest = json.loads((output / "manifest.json").read_text())
    if (type(previous_manifest.get("schema_version")) is not int or previous_manifest["schema_version"] != 1
            or any(type(previous_manifest.get(field)) is not int for field in ["n_samples", "question_count"])
            or previous_manifest.get("synthetic") is not False
            or type(previous_manifest.get("complete")) is not bool):
        raise ValueError("Cannot resume an invalid or synthetic evaluation manifest.")
    expected_metadata = {"protocol_id": protocol_id, "adapter_identity": adapter_identity,
        "model_label": args.model, "dataset_sha256": dataset_hash, "n_samples": n_samples,
        "split": args.split, "question_count": len(rows), "settings": settings, "code_hashes": code_hashes,
        "training_topology": train_manifest.get("topology") if train_manifest else None,
        "selection": selection, "structural_review_sha256": review_hash}
    for field, expected_value in expected_metadata.items():
        if previous_manifest.get(field) != expected_value:
            raise ValueError(f"Cannot resume: {field} changed.")
    samples_path = output / "samples.jsonl"
    if previous_manifest["complete"] and (not samples_path.is_file()
            or hashlib.sha256(samples_path.read_bytes()).hexdigest() != previous_manifest.get("samples_sha256")):
        raise ValueError("Completed cached samples differ from their manifest hash.")
    if samples_path.exists():
        for line in samples_path.read_text().split("\n"):
            if not line.strip():
                continue
            try:
                saved = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Incomplete JSONL line: preserve the file, remove only its interrupted last line, then resume.") from error
            if (not isinstance(saved.get("id"), str) or type(saved.get("sample_index")) is not int
                    or not isinstance(saved.get("text"), str)
                    or any(type(saved.get(field)) is not bool for field in ["correct", "format_compliant", "truncated"])
                    or type(saved.get("reward")) not in {int, float}):
                raise ValueError("Invalid sample schema in resume file.")
            key = (saved["id"], saved["sample_index"])
            if key in existing:
                raise ValueError("Duplicate saved sample; refusing to resume.")
            existing[key] = saved
    if not set(existing).issubset(expected_keys):
        raise ValueError("Unexpected sample IDs in resume file.")
    if previous_manifest["complete"] and (set(existing) != expected_keys
            or type(previous_manifest.get("total_samples")) is not int or previous_manifest["total_samples"] != len(expected_keys)):
        raise ValueError("Completed cached evaluation has an incomplete sample grid.")

cached_samples = []
for row in rows:
    for sample_index in range(n_samples):
        saved = existing.get((row["id"], sample_index))
        if saved is not None:
            if any(saved.get(field) != row.get(field) for field in ["question", "answer", "family_id", "variant_type", "split"]):
                raise ValueError("Saved sample content differs from dataset.")
            expected_seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{row['id']}:{sample_index}".encode()).digest()[:4], "big")
            if (type(saved.get("sample_seed")) is not int or saved["sample_seed"] != expected_seed
                    or type(saved.get("output_tokens")) is not int or not 0 <= saved["output_tokens"] <= settings["max_new_tokens"]
                    or type(saved.get("prompt_tokens")) is not int or not 1 <= saved["prompt_tokens"] <= settings["max_prompt_length"]
                    or (saved["truncated"] and saved["output_tokens"] != settings["max_new_tokens"])):
                raise ValueError("Invalid saved sample seed/token count.")
            cached_samples.append(saved)
if cached_samples:
    score_state = {"SCORE_INPUT": [{"text": saved["text"], "answer": saved["answer"], "truncated": saved["truncated"]} for saved in cached_samples]}
    exec(verifier_code, score_state)
    for saved, rescored in zip(cached_samples, score_state["SCORE_OUTPUT"]):
        if any(saved.get(field) != rescored.get(field) for field in ["correct", "reward", "status", "extracted", "format_compliant", "truncated"]):
            raise ValueError("Saved score differs from the verifier; preserve and inspect the run.")

if previous_manifest and set(existing) == expected_keys:
    identity = runpy.run_path(str(root / "scripts/fingerprint_model.py"), init_globals={"MODEL_PATH": root / cfg["model"]["path"]})
    if previous_manifest.get("model_identity") != identity["model_identity"]:
        raise ValueError("Cannot reuse completed evaluation: base model files changed.")
    if train_manifest and train_manifest["model_identity"] != identity["model_identity"]:
        raise ValueError("Adapter was trained on different base weights.")
    if selection and selection["model_identity"] != identity["model_identity"]:
        raise ValueError("Model changed after development selection.")
    if not previous_manifest["complete"]:
        previous_manifest.update(complete=True, total_samples=len(existing), recovered_complete_on_resume=True,
            samples_sha256=hashlib.sha256((output / "samples.jsonl").read_bytes()).hexdigest())
        (output / "manifest.json.tmp").write_text(json.dumps(previous_manifest, indent=2, ensure_ascii=False) + "\n")
        (output / "manifest.json.tmp").replace(output / "manifest.json")
    print(f"[EVAL CACHED] Validated {len(existing)} samples; no GPU model loaded; {output}", flush=True)
    raise SystemExit(0)

loaded = runpy.run_path(str(root / "scripts/load_model.py"), init_globals={
    "MODEL_PATH": root / cfg["model"]["path"], "ADAPTER_PATH": adapter_path, "TRAINING": False})
model, tokenizer = loaded["model"], loaded["tokenizer"]
if loaded["package_versions"] != package_versions or loaded["adapter_identity"] != adapter_identity:
    raise ValueError("Model environment or adapter changed during preflight.")
if train_manifest and train_manifest["model_identity"] != loaded["model_identity"]:
    raise ValueError("Adapter was trained on different base weights.")
if previous_manifest and previous_manifest["model_identity"] != loaded["model_identity"]:
    raise ValueError("Cannot resume: base model files changed.")
if selection and selection["model_identity"] != loaded["model_identity"]:
    raise ValueError("Model changed after development selection.")
import torch
from transformers import set_seed

output.mkdir(parents=True, exist_ok=True)
manifest = {
    "schema_version": 1, "path_base": "code_root", "model_label": args.model, "n_samples": n_samples, "split": args.split,
    "dataset_sha256": dataset_hash, "protocol_id": protocol_id, "model_identity": loaded["model_identity"],
    "adapter_identity": adapter_identity, "adapter_path": (str(adapter_path.relative_to(root)) if adapter_path.is_relative_to(root) else str(adapter_path)) if adapter_path else None,
    "training_topology": train_manifest.get("topology") if train_manifest else None,
    "synthetic": False, "complete": False, "settings": settings, "code_hashes": code_hashes,
    "question_count": len(rows), "model_files": loaded["model_files"], "config": cfg,
    "gpu": torch.cuda.get_device_name(), "selection": selection, "structural_review_sha256": review_hash,
}
(output / "manifest.json.tmp").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
(output / "manifest.json.tmp").replace(output / "manifest.json")
start = time.perf_counter()
torch.cuda.reset_peak_memory_stats()
completed = len(existing)
new_tokens = 0
with (output / "samples.jsonl").open("a", encoding="utf-8") as stream:
    for row in rows:
        if all((row["id"], sample_index) in existing for sample_index in range(n_samples)):
            continue
        prompt = settings["prompt_template"].format(question=row["question"])
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length > settings["max_prompt_length"]:
            raise ValueError(f"Prompt too long for {row['id']}: {prompt_length}; do not silently truncate questions.")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        for sample_index in range(n_samples):
            if (row["id"], sample_index) in existing:
                continue
            seed_material = f"{settings['seed']}:{row['id']}:{sample_index}".encode()
            sample_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
            set_seed(sample_seed)
            sample_start = time.perf_counter()
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, do_sample=True, temperature=settings["temperature"], top_p=settings["top_p"],
                        top_k=settings["top_k"], max_new_tokens=settings["max_new_tokens"], num_return_sequences=1,
                        repetition_penalty=1.0, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                continuation = generated[0, prompt_length:].tolist()
                ended = bool(continuation and continuation[-1] == tokenizer.eos_token_id)
                truncated = len(continuation) >= settings["max_new_tokens"] and not ended
                answer_text = tokenizer.decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                score_state = {"SCORE_INPUT": [{"text": answer_text, "answer": row["answer"], "truncated": truncated}]}
                exec(verifier_code, score_state)
                score = score_state["SCORE_OUTPUT"][0]
            except Exception as error:
                failure = {"id": row["id"], "sample_index": sample_index, "status": "infrastructure_error",
                           "error": repr(error), "seed": sample_seed}
                with (output / "infrastructure_errors.jsonl").open("a") as errors:
                    errors.write(json.dumps(failure) + "\n")
                raise
            sample = {**row, **score, "sample_index": sample_index, "text": answer_text,
                      "sample_seed": sample_seed, "output_tokens": len(continuation),
                      "prompt_tokens": prompt_length, "seconds": time.perf_counter() - sample_start}
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            completed += 1
            new_tokens += len(continuation)
        print(f"[EVAL] {100*completed/(len(rows)*n_samples):6.2f}% | {completed}/{len(rows)*n_samples} | saved={output/'samples.jsonl'}", flush=True)
elapsed = time.perf_counter() - start
manifest.update(complete=True, total_samples=completed, invocation_seconds=elapsed,
    invocation_output_tokens=new_tokens, invocation_tokens_per_second=new_tokens/max(elapsed, 1e-9),
    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
    samples_sha256=hashlib.sha256((output / "samples.jsonl").read_bytes()).hexdigest())
(output / "manifest.json.tmp").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
(output / "manifest.json.tmp").replace(output / "manifest.json")
print(f"✅ [EVAL COMPLETE] {completed} samples | {output}")
