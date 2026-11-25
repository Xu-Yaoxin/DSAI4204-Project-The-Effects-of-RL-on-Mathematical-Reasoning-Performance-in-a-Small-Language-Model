"""Prepare immutable GSM8K splits from user-supplied official JSONL files."""

import argparse
import hashlib
import json
import random
import re
import runpy
import tempfile
import unicodedata
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/v1.json")
parser.add_argument("--allow-unverified-source", action="store_true", help="Explicitly allow local engineering fixtures without official provenance")
args = parser.parse_args()
code_root = Path(__file__).resolve().parents[1]
config_path = Path(args.config).expanduser()
if not config_path.is_absolute():
    config_path = code_root / config_path
config = json.loads(config_path.read_text(encoding="utf-8"))
data_config = config["data"]
seed = int(data_config.get("seed", config.get("seeds", {}).get("data", config.get("seed", 42))))
counts = {name: int(data_config.get(name + "_size", default)) for name, default in [("train", 512), ("pilot", 128), ("engineering", 32), ("dev", 200), ("test", 200)]}
if any(value <= 0 for value in counts.values()):
    raise SystemExit("All split sizes must be positive.")
if not counts["engineering"] <= counts["pilot"] <= counts["train"]:
    raise SystemExit("Require engineering_size <= pilot_size <= train_size.")
output_dir = Path(data_config["output_dir"]).expanduser()
if not output_dir.is_absolute():
    output_dir = code_root / output_dir
raw_pools = {}
source_manifest = {}
verifier_path = code_root / "scripts" / "verify_answers.py"
raw_paths = {}
for official_split in ["train", "test"]:
    raw_path = Path(data_config["raw_" + official_split]).expanduser()
    raw_paths[official_split] = raw_path if raw_path.is_absolute() else code_root / raw_path
if raw_paths["train"].resolve() == raw_paths["test"].resolve():
    raise SystemExit("raw_train and raw_test must be distinct official source files.")
provenance_path = raw_paths["train"].parent / "source_manifest.json"
raw_provenance = None
provenance_summary = {"status": "unverified_explicit_override"}
if provenance_path.is_file():
    provenance_bytes = provenance_path.read_bytes()
    raw_provenance = json.loads(provenance_bytes)
    source_revision = raw_provenance.get("revision", "")
    if raw_provenance.get("schema_version") != 2 or raw_provenance.get("repository") != "https://github.com/openai/grade-school-math" or not re.fullmatch(r"[0-9a-f]{40}", str(source_revision)) or raw_provenance.get("license") != "MIT" or raw_provenance.get("variant") != "main":
        raise SystemExit("Invalid official GSM8K source provenance. Restore the downloader manifest before preparing data.")
    if set(raw_provenance.get("files", {})) != {"train", "test", "license"}:
        raise SystemExit("Source provenance must include train, test, and license entries.")
    license_path = raw_paths["train"].parent / "LICENSE"
    if not license_path.is_file():
        raise SystemExit(f"Missing dataset license: {license_path}. Run download_data.py to restore it.")
    license_bytes = license_path.read_bytes()
    expected_license_url = f"https://raw.githubusercontent.com/openai/grade-school-math/{source_revision}/LICENSE"
    license_metadata = raw_provenance["files"]["license"]
    if hashlib.sha256(license_bytes).hexdigest() != license_metadata.get("sha256") or license_metadata.get("url") != expected_license_url or license_metadata.get("bytes") != len(license_bytes) or not license_bytes.startswith(b"MIT License"):
        raise SystemExit("Dataset license hash or source metadata mismatch; no processed files were written.")
    provenance_summary = {"status": "verified", "repository": raw_provenance["repository"], "revision": source_revision,
                          "source_manifest_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
                          "license": "MIT", "license_sha256": hashlib.sha256(license_bytes).hexdigest()}
elif not args.allow_unverified_source:
    raise SystemExit(f"Missing {provenance_path}. Run download_data.py; use --allow-unverified-source only for explicit local engineering fixtures.")

for official_split, path_key in [("train", "raw_train"), ("test", "raw_test")]:
    raw_path = raw_paths[official_split]
    if not raw_path.is_file():
        raise SystemExit(f"Missing {raw_path}. Place the official GSM8K main/{official_split}.jsonl here; this script never downloads data.")
    raw_bytes = raw_path.read_bytes()
    raw_rows = [json.loads(line) for line in raw_bytes.decode("utf-8").split("\n") if line.strip()]
    if raw_provenance:
        metadata = raw_provenance["files"][official_split]
        expected_url = f"https://raw.githubusercontent.com/openai/grade-school-math/{source_revision}/grade_school_math/data/{official_split}.jsonl"
        if metadata.get("sha256") != hashlib.sha256(raw_bytes).hexdigest() or metadata.get("rows") != len(raw_rows) or metadata.get("bytes") != len(raw_bytes) or metadata.get("url") != expected_url:
            raise SystemExit(f"Official {official_split} source hash/count/URL mismatch; no processed files were written.")
    score_input = []
    for row_index, row in enumerate(raw_rows):
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip() or not isinstance(row.get("answer"), str):
            raise SystemExit(f"Invalid question/answer row in {raw_path} at row {row_index}.")
        if "####" not in row["answer"]:
            raise SystemExit(f"Expected official GSM8K #### answer marker in {raw_path} at row {row_index}.")
        gold = row["answer"].rsplit("####", 1)[1].strip()
        score_input.append({"text": "#### " + gold, "answer": gold})
    scores = runpy.run_path(str(verifier_path), init_globals={"SCORE_INPUT": score_input})["SCORE_OUTPUT"]
    pool = []
    for row_index, (row, score) in enumerate(zip(raw_rows, scores)):
        if not score["correct"]:
            raise SystemExit(f"Invalid numeric gold answer in {raw_path} at row {row_index}: {score['status']}")
        record_id = f"gsm8k-{official_split}-{row_index:06d}"
        pool.append({"id": record_id, "question": row["question"].strip(), "answer": score["extracted"],
                     "family_id": record_id, "variant_type": "original",
                     "source": {"dataset": "GSM8K" if raw_provenance else "unverified-local-fixture", "official_split": official_split, "row_index": row_index}})
    raw_pools[official_split] = pool
    source_manifest[official_split] = {"path": str(data_config[path_key]), "sha256": hashlib.sha256(raw_bytes).hexdigest(), "rows": len(pool)}

if len(raw_pools["train"]) < counts["train"] + counts["dev"] or len(raw_pools["test"]) < counts["test"]:
    raise SystemExit("Raw pools are too small for disjoint train/dev and official-test samples.")
train_indices = list(range(len(raw_pools["train"])))
test_indices = list(range(len(raw_pools["test"])))
random.Random(seed).shuffle(train_indices)
random.Random(seed + 1).shuffle(test_indices)
selected = {
    "train": [raw_pools["train"][index] for index in train_indices[:counts["train"]]],
    "dev": [raw_pools["train"][index] for index in train_indices[counts["train"]:counts["train"] + counts["dev"]]],
    "test": [raw_pools["test"][index] for index in test_indices[:counts["test"]]],
}
seen_questions = {}
for split in ["train", "dev", "test"]:
    for record in selected[split]:
        normalized = re.sub(r"\W+", " ", unicodedata.normalize("NFKC", record["question"]).casefold()).strip()
        if not normalized:
            raise SystemExit(f"Question {record['id']} has no usable normalized content.")
        if normalized in seen_questions:
            previous_split, previous_id = seen_questions[normalized]
            raise SystemExit(f"Normalized question overlap: {previous_split}/{previous_id} and {split}/{record['id']}. No files were written.")
        seen_questions[normalized] = (split, record["id"])
selected["pilot"] = selected["train"][:counts["pilot"]]
selected["engineering"] = selected["train"][:counts["engineering"]]
output_bytes = {}
file_manifest = {}
for split, records in selected.items():
    records = [{**record, "split": split} for record in records]
    filename = split + ".jsonl"
    payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records).encode("utf-8")
    output_bytes[filename] = payload
    file_manifest[filename] = {"sha256": hashlib.sha256(payload).hexdigest(), "rows": len(records), "ids": [record["id"] for record in records]}
manifest = {
    "schema_version": 1, "kind": "gsm8k_splits", "seed": seed, "counts": counts,
    "sources": source_manifest, "source_provenance": provenance_summary, "files": file_manifest,
    "verifier_sha256": hashlib.sha256(verifier_path.read_bytes()).hexdigest(),
    "leakage_check": {"status": "passed", "normalization": "NFKC + casefold + non-word runs to spaces",
                      "unique_train_dev_test_questions": len(seen_questions), "nested_training_subsets": ["engineering", "pilot"]},
    "public_benchmark_note": "Disjoint post-training splits do not establish absence from model pretraining.",
}
output_bytes["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
for filename, payload in output_bytes.items():
    destination = output_dir / filename
    if destination.exists() and destination.read_bytes() != payload:
        raise SystemExit(f"Immutable data conflict: {destination}. Use a new output_dir for changed inputs/settings; existing files were preserved.")
output_dir.mkdir(parents=True, exist_ok=True)
written = 0
for filename, payload in output_bytes.items():
    destination = output_dir / filename
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".prepare-", delete=False) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
        written += 1
print(f"[DATA 100%] train={counts['train']} pilot={counts['pilot']} engineering={counts['engineering']} dev={counts['dev']} test={counts['test']}")
print(f"[SAVED] {output_dir / 'manifest.json'} | new_files={written} | overlap_check=passed")
