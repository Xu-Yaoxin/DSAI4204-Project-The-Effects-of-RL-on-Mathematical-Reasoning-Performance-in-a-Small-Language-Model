"""Validate and atomically merge two independently seeded baseline evaluation shards."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import tempfile


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inputs", nargs="+", required=True, type=Path)
parser.add_argument("--output", type=Path, default=Path("outputs/guarded_v3/base"))
args = parser.parse_args()
if len(args.inputs) != 2:
    parser.error("Exactly two complete shard directories are required")
directories = [(root / path).absolute() for path in args.inputs]
output = (root / args.output).absolute()
for path in (*directories, output):
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise SystemExit(f"Symlink paths are not accepted: {path}")
directories = [path.resolve() for path in directories]
output = output.resolve()
if len(set(directories)) != 2:
    raise SystemExit("Shard inputs must be distinct")
if output == root or output in root.parents or any(output == path or output in path.parents or path in output.parents for path in directories):
    raise SystemExit("Merged output must be separate from input directories and the code root")
if root in output.parents and root / "outputs" not in output.parents:
    raise SystemExit("Merged outputs inside code must remain under outputs/")
print("=" * 72, flush=True)
print("[BOUNDARY MERGE 1/3] 0% | validate complete baseline shards", flush=True)
inputs = {}
source_hashes = {}
for directory in directories:
    manifest_path, samples_path = directory / "manifest.json", directory / "samples.jsonl"
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, samples_path)):
        raise SystemExit("Both shard manifests and sample files must exist without symlinks")
    manifest_bytes, samples_bytes = manifest_path.read_bytes(), samples_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    sharding = manifest.get("sharding")
    if (not isinstance(sharding, dict) or type(sharding.get("count")) is not int or sharding["count"] != 2
            or type(sharding.get("index")) is not int or sharding["index"] not in (0, 1)
            or sharding.get("assignment") != "question_index_modulo"):
        raise SystemExit("Each input must declare an exact two-worker question-index-modulo shard")
    shard_index = sharding["index"]
    if shard_index in inputs:
        raise SystemExit("Duplicate shard index")
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("experiment_type") != "exploratory_baseline_calibration" or manifest.get("model_label") != "base"
            or manifest.get("adapter_identity") is not None or manifest.get("adapter_path") is not None
            or manifest.get("complete") is not True or type(manifest.get("synthetic")) is not bool):
        raise SystemExit("Merge requires complete baseline-only calibration manifests")
    if not samples_bytes.endswith(b"\n") or hashlib.sha256(samples_bytes).hexdigest() != manifest.get("samples_sha256"):
        raise SystemExit("Shard sample bytes are incomplete or differ from their completed hash")
    source_hashes[manifest_path] = hashlib.sha256(manifest_bytes).hexdigest()
    source_hashes[samples_path] = hashlib.sha256(samples_bytes).hexdigest()
    rows = [json.loads(line) for line in samples_bytes.decode("utf-8").split("\n") if line.strip()]
    inputs[shard_index] = {"directory": directory, "manifest": manifest, "rows": rows,
        "manifest_sha256": source_hashes[manifest_path], "samples_sha256": source_hashes[samples_path]}
reference = inputs[0]["manifest"]
common_fields = ("schema_version", "experiment_type", "synthetic", "model_label", "adapter_path", "adapter_identity", "model_identity", "model_files",
                 "dataset_path", "dataset_sha256", "dataset_manifest_path", "dataset_manifest_sha256", "n_samples", "question_count",
                 "config", "settings", "code_hashes", "protocol_id", "groups", "path_base")
for field in common_fields:
    if field not in reference or inputs[1]["manifest"].get(field) != reference[field]:
        raise SystemExit(f"Shards have missing or mismatched common metadata: {field}")
if reference["path_base"] != "code_root":
    raise SystemExit("Dataset paths must be recorded relative to the code root")
n_samples = reference["n_samples"]
if type(n_samples) is not int or not 1 <= n_samples <= 16:
    raise SystemExit("Invalid fixed sample count")
settings = reference["settings"]
if (not isinstance(settings, dict) or any(type(settings.get(key)) is not int for key in ("max_new_tokens", "max_prompt_length", "seed"))
        or settings["max_new_tokens"] < 1 or settings["max_prompt_length"] < 1 or settings["seed"] < 0):
    raise SystemExit("Invalid generation settings")
code_hashes = reference["code_hashes"]
verifier_path = root / "scripts/verify_answers.py"
verifier_hash = hashlib.sha256(verifier_path.read_bytes()).hexdigest()
if not isinstance(code_hashes, dict) or code_hashes.get("verify_answers.py") != verifier_hash:
    raise SystemExit("Scorer identity changed; preserve these results for a separate rescoring protocol")
for name, value in code_hashes.items():
    if Path(name).name != name or not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SystemExit("Invalid recorded source-code hash")
if (settings.get("context_limit") != 262144 or settings["max_new_tokens"] != 262144
        or settings.get("length_policy") != "native_context_minus_actual_prompt"):
    raise ValueError("Invalid native-context confirmation protocol")
guard_paths = [root / "scripts" / name for name in ("validate_guarded_sample.py", "repetition_guard.py")]
guard_hashes = {}
for path in guard_paths:
    if path.is_symlink() or not path.is_file():
        raise SystemExit("Missing or unsafe guarded scoring helper")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if code_hashes.get(path.name) != digest:
        raise SystemExit("Guard/helper identity changed; preserve results under the original protocol")
    guard_hashes[path.name] = digest
validator_path = guard_paths[0]
if not isinstance(settings.get("generation_guard"), dict) or type(settings.get("eos_token_id")) is not int:
    raise SystemExit("Guarded generation settings and EOS identity are required")
protocol = {key: reference[key] for key in ("dataset_sha256", "dataset_manifest_sha256", "n_samples", "settings", "code_hashes", "experiment_type")}
if hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() != reference["protocol_id"]:
    raise SystemExit("Recorded generation protocol differs from its components")
if not isinstance(reference["model_files"], dict) or not reference["model_files"]:
    raise SystemExit("A recorded base-model file inventory is required")
if hashlib.sha256(json.dumps(reference["model_files"], sort_keys=True).encode()).hexdigest() != reference["model_identity"]:
    raise SystemExit("Recorded model inventory identity is invalid")
expected_identity = reference["config"].get("model", {}).get("expected_identity")
if expected_identity is not None and expected_identity != reference["model_identity"]:
    raise SystemExit("Recorded baseline differs from the configured original model identity")
data_paths = []
for field, hash_field in (("dataset_path", "dataset_sha256"), ("dataset_manifest_path", "dataset_manifest_sha256")):
    relative = Path(reference[field])
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit("Dataset paths must stay relative to the code root")
    path = root / relative
    if any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file():
        raise SystemExit("Unsafe or missing dataset input")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != reference[hash_field]:
        raise SystemExit("Frozen dataset input changed")
    source_hashes[path] = digest
    data_paths.append(path)
dataset = [json.loads(line) for line in data_paths[0].read_text(encoding="utf-8").split("\n") if line.strip()]
data_manifest = json.loads(data_paths[1].read_text(encoding="utf-8"))
declared = data_manifest.get("files", {}).get(data_paths[0].name, {})
if (not dataset or len({row["id"] for row in dataset}) != len(dataset) or type(reference["question_count"]) is not int
        or reference["question_count"] != len(dataset) or declared.get("rows") != len(dataset)
        or declared.get("sha256") != reference["dataset_sha256"] or reference["groups"] != data_manifest.get("groups")):
    raise SystemExit("Frozen dataset identity, count or groups are inconsistent")
if Counter(row["group_id"] for row in dataset) != Counter({group["id"]: group["count"] for group in reference["groups"]}):
    raise SystemExit("Frozen group counts do not match the question set")
dataset_by_id = {row["id"]: row for row in dataset}
positions = {row["id"]: index for index, row in enumerate(dataset)}
score_fields = ("correct", "reward", "extracted", "status", "format_compliant", "truncated")
merged_rows = {}
for shard_index in (0, 1):
    item = inputs[shard_index]
    assigned = {row["id"] for index, row in enumerate(dataset) if index % 2 == shard_index}
    expected_keys = {(question_id, index) for question_id in assigned for index in range(n_samples)}
    if type(item["manifest"].get("total_samples")) is not int or item["manifest"]["total_samples"] != len(expected_keys) or len(item["rows"]) != len(expected_keys):
        raise SystemExit(f"Shard {shard_index} does not have its complete assigned sample count")
    keys = set()
    for row in item["rows"]:
        if (not isinstance(row, dict) or row.get("id") not in assigned or type(row.get("sample_index")) is not int
                or not 0 <= row["sample_index"] < n_samples):
            raise SystemExit(f"Shard {shard_index} contains an unexpected question assignment or sample index")
        key = (row["id"], row["sample_index"])
        if key in keys or key in merged_rows:
            raise SystemExit("Duplicate question/sample record")
        if any(row.get(field) != value for field, value in dataset_by_id[row["id"]].items()):
            raise SystemExit("Saved question/gold/group metadata differs from the frozen dataset")
        validated = runpy.run_path(str(validator_path), init_globals={"SAMPLE": row, "SETTINGS": settings})
        if validated.get("VALIDATED") is not True:
            raise SystemExit("Guarded answer validation did not complete")
        expected_seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{row['id']}:{row['sample_index']}".encode()).digest()[:4], "big")
        if type(row.get("sample_seed")) is not int or row["sample_seed"] != expected_seed:
            raise SystemExit("Saved sample seed differs from the unchanged deterministic generation protocol")
        for field, limit in (("output_tokens", settings["max_new_tokens"]), ("prompt_tokens", settings["max_prompt_length"])):
            if type(row.get(field)) is not int or not (1 if field == "prompt_tokens" else 0) <= row[field] <= limit:
                raise SystemExit("Invalid saved token count")
        if (type(row.get("seconds")) not in (int, float) or not math.isfinite(row["seconds"]) or row["seconds"] < 0
                or (row["truncated"] and row["output_tokens"] != row.get("generation_limit"))):
            raise SystemExit("Invalid saved duration or truncation flag")
        keys.add(key)
        merged_rows[key] = row
    if keys != expected_keys:
        raise SystemExit(f"Shard {shard_index} has missing assigned samples")
full_grid = {(row["id"], index) for row in dataset for index in range(n_samples)}
if set(merged_rows) != full_grid:
    raise SystemExit("Combined shards do not form the full frozen question-by-sample grid")
merger_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
identity = {"schema_version": 1, "merger_sha256": merger_hash, "verifier_sha256": verifier_hash, "guard_helpers_sha256": guard_hashes,
    "source_shards": [{"index": index, "manifest_sha256": inputs[index]["manifest_sha256"], "samples_sha256": inputs[index]["samples_sha256"]} for index in (0, 1)],
    "dataset_sha256": reference["dataset_sha256"], "dataset_manifest_sha256": reference["dataset_manifest_sha256"], "protocol_id": reference["protocol_id"]}

output.parent.mkdir(parents=True, exist_ok=True)
lock_path = output.parent / f".{output.name}.merge.lock"
if lock_path.is_symlink():
    raise SystemExit("Unsafe merge lock path")
lock_stream = lock_path.open("a")
fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
if output.exists():
    marker = output / "merge_ready.json"
    if marker.is_symlink() or not marker.is_file():
        raise SystemExit("Existing output is partial; preserve it and choose a new --output directory")
    ready = json.loads(marker.read_text())
    if ready.get("complete") is not True or ready.get("identity") != identity:
        raise SystemExit("Existing merged output belongs to different sources or code")
    artifacts = {}
    for path in output.rglob("*"):
        if path.is_symlink():
            raise SystemExit("Unsafe symlink in merged output")
        if path.is_file() and path != marker:
            artifacts[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if artifacts != ready.get("artifacts_sha256"):
        raise SystemExit("Completed merged output artifacts were changed")
    print(f"✅ [BOUNDARY MERGE CACHED] 100% | {len(merged_rows)} samples | {output}", flush=True)
    lock_stream.close()
    raise SystemExit(0)

print(f"[BOUNDARY MERGE 2/3] 67% | verified {len(dataset)} questions / {len(merged_rows)} answers | deterministic ordering", flush=True)
ordered = [merged_rows[(row["id"], index)] for row in dataset for index in range(n_samples)]
sample_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered).encode("utf-8")
merged_manifest = {key: value for key, value in reference.items() if key != "sharding" and not key.startswith(("invocation_", "peak_"))
                   and key not in {"gpu", "execution_cpu_threads", "recovered_complete_on_resume"}}
merged_manifest.update(complete=True, question_count=len(dataset), total_samples=len(ordered), samples_sha256=hashlib.sha256(sample_bytes).hexdigest())
merged_manifest["execution_provenance"] = {"kind": "validated_independent_worker_merge", "workers": 2, "assignment": "question_index_modulo",
    "generation_protocol_unchanged": True, "identity": identity,
    "shards": [{"index": index, "input_directory": str(inputs[index]["directory"].relative_to(root)) if inputs[index]["directory"].is_relative_to(root) else str(inputs[index]["directory"]),
        "gpu": inputs[index]["manifest"].get("gpu"), "execution_cpu_threads": inputs[index]["manifest"].get("execution_cpu_threads"),
        "invocation_seconds": inputs[index]["manifest"].get("invocation_seconds"),
        "peak_allocated_gib": inputs[index]["manifest"].get("peak_allocated_gib")} for index in (0, 1)]}
stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.merge-", dir=output.parent))
(stage / "samples.jsonl").write_bytes(sample_bytes)
(stage / "manifest.json").write_text(json.dumps(merged_manifest, ensure_ascii=False, indent=2) + "\n")
artifacts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(stage.iterdir())}
(stage / "merge_ready.json").write_text(json.dumps({"complete": True, "identity": identity, "artifacts_sha256": artifacts,
    "models_loaded": 0, "network_required": False}, indent=2) + "\n")
for path in guard_paths:
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != guard_hashes[path.name]:
        raise SystemExit("Guard/helper code changed while processing answers")
for path, expected in source_hashes.items():
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Source changed during merge; staged files retained: {stage}")
if hashlib.sha256(verifier_path.read_bytes()).hexdigest() != verifier_hash or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != merger_hash:
    raise SystemExit(f"Scoring or merge code changed; staged files retained: {stage}")
if output.exists():
    raise SystemExit(f"Output appeared during merge; staged files retained: {stage}")
os.rename(stage, output)
lock_stream.close()
print(f"✅ [BOUNDARY MERGE 3/3] 100% | COMPLETE | [SAVED] {output / 'manifest.json'}", flush=True)
