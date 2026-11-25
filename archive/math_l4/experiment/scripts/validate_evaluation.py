"""Validate one complete frozen evaluation without importing model packages."""
from pathlib import Path
import hashlib
import json
import runpy

root = Path(__file__).resolve().parents[1]
source = (root / Path(EVALUATION_PATH)).absolute()
if any(p.is_symlink() for p in (source, *source.parents)):
    raise ValueError("Evaluation path must not contain symlinks.")
source = source.resolve()
manifest_path = source / "manifest.json"
samples_path = source / "samples.jsonl"
if any(p.is_symlink() or not p.is_file() for p in (manifest_path, samples_path)):
    raise ValueError("Complete evaluation artifacts are missing.")
manifest_bytes = manifest_path.read_bytes()
samples_bytes = samples_path.read_bytes()
manifest = json.loads(manifest_bytes)
cfg = globals().get("VALIDATE_CONFIG")
if cfg is None:
    cfg = json.loads((root / "configs/math_l4.json").read_text())
if (manifest.get("schema_version") != 1 or manifest.get("experiment_type") != "math_l4_base_rl"
        or manifest.get("synthetic") is not False or manifest.get("complete") is not True
        or manifest.get("config") != cfg or manifest.get("path_base") != "code_root"
        or manifest.get("model_label") not in ("base", "rl") or manifest.get("split") not in ("engineering", "dev", "test")):
    raise ValueError("Evaluation identity/configuration is not a completed real MATH experiment.")
if manifest.get("sharding") is not None:
    raise ValueError("This experiment requires unsharded serial evaluations.")
if not samples_bytes.endswith(b"\n") or hashlib.sha256(samples_bytes).hexdigest() != manifest.get("samples_sha256"):
    raise ValueError("Completed sample hash is invalid.")
settings = manifest["settings"]
code_hashes = manifest["code_hashes"]
required_code = {"evaluate.py", "verify_answers.py", "load_model.py", "fingerprint_model.py", "repetition_guard.py",
                 "validate_guarded_sample.py", "doctor.py", "validate_evaluation.py", "validate_selection.py"}
if set(code_hashes) != required_code:
    raise ValueError("Incomplete evaluation code binding.")
for name, expected in code_hashes.items():
    path = root / "scripts" / name
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Evaluation code changed: {name}")
protocol = {k: manifest[k] for k in ("dataset_sha256", "dataset_manifest_sha256", "n_samples", "settings", "code_hashes", "experiment_type")}
if hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() != manifest.get("protocol_id"):
    raise ValueError("Evaluation protocol identity differs from its components.")
model_files = manifest.get("model_files")
if (not isinstance(model_files, dict) or not model_files
        or hashlib.sha256(json.dumps(model_files, sort_keys=True).encode()).hexdigest() != manifest.get("model_identity")
        or manifest["model_identity"] != cfg["model"]["expected_identity"]):
    raise ValueError("Wrong original Base model identity.")
expected_settings = {key: cfg["evaluation"][key] for key in ("temperature", "top_p", "top_k", "max_new_tokens", "max_prompt_length", "seed", "context_limit", "length_policy", "generation_guard")}
expected_settings.update(prompt_template=cfg["prompt"]["template"], dtype="bfloat16", backend="transformers-sdpa",
    batch_size=1, world_size=1, add_special_tokens=True, stopping_rule="tokenizer-eos-or-native-context-or-guard",
    repetition_penalty=1.0, eos_token_id=cfg["model"]["expected_eos_token_id"])
if any(settings.get(k) != v for k, v in expected_settings.items()) or not isinstance(settings.get("package_versions"), dict):
    raise ValueError("Saved generation settings differ from the frozen policy.")
for key, expected_relative, hash_key in (
        ("dataset_path", str(Path(cfg["data"]["output_dir"]) / (manifest["split"] + ".jsonl")), "dataset_sha256"),
        ("dataset_manifest_path", str(Path(cfg["data"]["output_dir"]) / "manifest.json"), "dataset_manifest_sha256")):
    if manifest.get(key) != expected_relative:
        raise ValueError("Split data path differs from the declared split.")
    path = root / expected_relative
    if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get(hash_key):
        raise ValueError("Frozen split data or manifest changed.")
data_manifest = json.loads((root / manifest["dataset_manifest_path"]).read_text())
dataset = [json.loads(line) for line in (root / manifest["dataset_path"]).read_text().splitlines() if line.strip()]
recorded = data_manifest["files"][Path(manifest["dataset_path"]).name]
if recorded["sha256"] != manifest["dataset_sha256"] or recorded["rows"] != len(dataset):
    raise ValueError("Prepared data manifest does not bind the evaluated split.")
by_id = {row["id"]: row for row in dataset}
if len(by_id) != len(dataset) or manifest.get("question_count") != len(dataset):
    raise ValueError("Invalid question identities/count.")
n = manifest["n_samples"]
if type(n) is not int or not 1 <= n <= 16:
    raise ValueError("Invalid sample count.")
rows = [json.loads(line) for line in samples_bytes.decode().splitlines() if line.strip()]
if len(rows) != len(dataset) * n or manifest.get("total_samples") != len(rows):
    raise ValueError("Incomplete evaluation grid.")
keys = set()
for row in rows:
    if row.get("id") not in by_id or type(row.get("sample_index")) is not int or not 0 <= row["sample_index"] < n:
        raise ValueError("Unexpected question/sample index.")
    key = (row["id"], row["sample_index"])
    if key in keys or any(row.get(k) != v for k,v in by_id[row["id"]].items()):
        raise ValueError("Duplicate sample or changed question/gold metadata.")
    keys.add(key)
    expected_seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{row['id']}:{row['sample_index']}".encode()).digest()[:4], "big")
    if row.get("sample_seed") != expected_seed:
        raise ValueError("Sample seed identity changed.")
    check = runpy.run_path(str(root / "scripts/validate_guarded_sample.py"), init_globals={"SAMPLE": row, "SETTINGS": settings})
    if check.get("VALIDATED") is not True:
        raise ValueError("Terminal sample validation failed.")
if manifest["model_label"] == "base":
    if any(manifest.get(k) is not None for k in ("adapter_path", "adapter_identity", "adapter_files", "selection_identity", "selection_path")):
        raise ValueError("Base evaluation contains an adapter/selection.")
else:
    if not isinstance(manifest.get("adapter_files"), dict) or set(manifest["adapter_files"]) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("Missing adapter content inventory.")
    adapter_path = (root / manifest["adapter_path"]).resolve()
    if not adapter_path.is_relative_to(root / "outputs"):
        raise ValueError("Adapter must belong to this experiment output tree.")
    if globals().get("VALIDATE_ADAPTER_FILES", True):
        digest = hashlib.sha256()
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            path = adapter_path / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("Missing or unsafe evaluated adapter.")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != manifest["adapter_files"][filename]:
                raise ValueError("Evaluated adapter bytes changed.")
            digest.update(filename.encode()); digest.update(data)
        if digest.hexdigest() != manifest["adapter_identity"]:
            raise ValueError("Adapter identity is invalid.")
    if manifest["split"] == "test":
        if not isinstance(manifest.get("selection_identity"), str) or len(manifest["selection_identity"]) != 64:
            raise ValueError("RL test has no frozen development selection.")
    elif manifest.get("selection_identity") is not None or manifest.get("selection_path") is not None:
        raise ValueError("Development/engineering evaluation cannot carry test selection metadata.")
EVALUATION_RESULT = {"manifest": manifest, "rows": rows, "dataset": dataset,
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "samples_sha256": hashlib.sha256(samples_bytes).hexdigest()}
