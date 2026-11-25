#!/usr/bin/env python3
"""Replay saved numerical answers offline; retain original training and selection."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys


root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, default=root.parent / "server_results/run_20261031/code")
parser.add_argument("--run-dir", type=Path, default=Path("outputs/v1_2"))
parser.add_argument("--output", type=Path, default=Path("outputs/rescore_v1_4"))
parser.add_argument("--bootstrap", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
archive = args.archive.absolute()
output = (root / args.output).absolute()
if args.bootstrap < 100:
    parser.error("--bootstrap must be at least 100")
for directory in (archive, output):
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise SystemExit(f"Unsafe symlink in directory path: {directory}")
archive = archive.resolve()
output = output.resolve()
if archive == root or archive in root.parents or root in archive.parents:
    raise SystemExit("Archive must be separate from the active code root")
if archive == output or archive in output.parents or output in archive.parents or output == root or output in root.parents:
    raise SystemExit("Output must not overlap archive or replace the code root")
if root in output.parents and root / "outputs" not in output.parents:
    raise SystemExit("Outputs inside the active code root must be under outputs/")
if args.run_dir.is_absolute() or ".." in args.run_dir.parts:
    parser.error("--run-dir must be an archive-relative directory without '..'")
run = args.run_dir
score_fields = ("correct", "reward", "extracted", "status", "format_compliant", "truncated")
source_bytes = {}
source_hashes = {}
print("=" * 72, flush=True)
print("[RESCORE 1/5] 0% | VALIDATE SAVED DATA, GENERATIONS AND FROZEN SELECTION", flush=True)

for relative in (Path("configs/v1.json"), Path("data/processed/manifest.json"), run / "selection/selection.json"):
    path = archive / relative
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise SystemExit(f"Missing or unsafe archive input: {relative}")
    source_bytes[str(relative)] = path.read_bytes()
    source_hashes[str(relative)] = hashlib.sha256(source_bytes[str(relative)]).hexdigest()
config = json.loads(source_bytes["configs/v1.json"])
prepared = json.loads(source_bytes["data/processed/manifest.json"])
selection = json.loads(source_bytes[str(run / "selection/selection.json")])
if selection.get("path_base") != "code_root" or selection.get("schema_version") != 1:
    raise SystemExit("Unsupported frozen selection path/schema")
candidates = selection.get("candidates", [])
if not candidates or any(type(item.get("step")) is not int or item["step"] < 1 for item in candidates):
    raise SystemExit("Frozen selection needs positive, distinct candidate steps")
if len({item["step"] for item in candidates}) != len(candidates):
    raise SystemExit("Duplicate candidate steps in frozen selection")
candidate_by_step = {item["step"]: item for item in candidates}
chosen = candidate_by_step.get(selection.get("step"))
if chosen is None or any(chosen.get(key) != selection.get(key) for key in ("adapter_path", "adapter_identity", "evaluation_path", "pass_at_1", "protocol_id")):
    raise SystemExit("Frozen selected checkpoint does not match its candidate")

plans = [("test_base", run / "test_base", "test", "base", None), ("test_rl", run / "test_rl", "test", "rl", chosen)]
plans += [(f"dev/checkpoint-{item['step']}", Path(item["evaluation_path"]), "dev", "rl", item) for item in sorted(candidates, key=lambda item: item["step"])]
required = {Path("scripts/verify_answers.py"), Path(selection["training_run"]) / "report.json", Path(selection["training_run"]) / "manifest.json"}
required.update(Path(info["path"]) for info in prepared["sources"].values())
required.update(Path("data/processed") / filename for filename in prepared["files"])
required.update(Path("scripts") / filename for filename in selection["code_hashes"])
for _, directory, _, _, candidate in plans:
    required.update((directory / "manifest.json", directory / "samples.jsonl"))
    if candidate:
        required.update(Path(candidate["adapter_path"]) / filename for filename in ("adapter_config.json", "adapter_model.safetensors"))
for relative in sorted(required):
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"Unsafe archive-relative input: {relative}")
    path = archive / relative
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise SystemExit(f"Missing or unsafe archive input: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    source_hashes[str(relative)] = digest.hexdigest()
    if path.suffix != ".safetensors":
        source_bytes[str(relative)] = path.read_bytes()
        if hashlib.sha256(source_bytes[str(relative)]).hexdigest() != digest.hexdigest():
            raise SystemExit(f"Source changed during validation: {relative}")

official = {}
for split, info in prepared["sources"].items():
    data = source_bytes[info["path"]]
    if hashlib.sha256(data).hexdigest() != info["sha256"]:
        raise SystemExit(f"Raw source hash mismatch: {split}")
    official[split] = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
    if len(official[split]) != info["rows"]:
        raise SystemExit(f"Raw source row count mismatch: {split}")
splits = {}
for filename, info in prepared["files"].items():
    data = source_bytes[str(Path("data/processed") / filename)]
    rows = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
    if hashlib.sha256(data).hexdigest() != info["sha256"] or len(rows) != info["rows"] or [row["id"] for row in rows] != info["ids"] or len(set(info["ids"])) != len(rows):
        raise SystemExit(f"Prepared data hash/count/id mismatch: {filename}")
    for row in rows:
        origin = row["source"]
        if type(origin["row_index"]) is not int or not 0 <= origin["row_index"] < len(official[origin["official_split"]]):
            raise SystemExit(f"Invalid official row index: {row['id']}")
        gold = official[origin["official_split"]][origin["row_index"]]
        if row["question"] != gold["question"] or Fraction(row["answer"]) != Fraction(gold["answer"].rsplit("####", 1)[-1].strip().replace(",", "")):
            raise SystemExit(f"Prepared question/gold differs from official source: {row['id']}")
    splits[Path(filename).stem] = {row["id"]: row for row in rows}
for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
    if set(splits[left]) & set(splits[right]) or {" ".join(row["question"].casefold().split()) for row in splits[left].values()} & {" ".join(row["question"].casefold().split()) for row in splits[right].values()}:
        raise SystemExit(f"Data split leakage: {left}/{right}")
if selection["dev_sha256"] != prepared["files"]["dev.jsonl"]["sha256"] or selection["test_sha256"] != prepared["files"]["test.jsonl"]["sha256"]:
    raise SystemExit("Frozen selection data hashes differ from archived data")
for filename, expected in selection["code_hashes"].items():
    if source_hashes[str(Path("scripts") / filename)] != expected:
        raise SystemExit(f"Original generation code hash mismatch: {filename}")
for filename, key in (("report.json", "training_report_sha256"), ("manifest.json", "training_manifest_sha256")):
    if source_hashes[str(Path(selection["training_run"]) / filename)] != selection[key]:
        raise SystemExit(f"Frozen selection training provenance mismatch: {filename}")
training_report = json.loads(source_bytes[str(Path(selection["training_run"]) / "report.json")])
if training_report.get("status") != "completed" or training_report.get("model_identity") != selection["model_identity"]:
    raise SystemExit("Original formal training was not completed for this model")
for candidate in candidates:
    digest = hashlib.sha256()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        digest.update(filename.encode())
        with (archive / candidate["adapter_path"] / filename).open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
    if digest.hexdigest() != candidate["adapter_identity"]:
        raise SystemExit(f"Adapter identity mismatch: checkpoint-{candidate['step']}")

evaluations = {}
old_scores = []
for name, directory, split, label, candidate in plans:
    manifest = json.loads(source_bytes[str(directory / "manifest.json")])
    data = source_bytes[str(directory / "samples.jsonl")]
    rows = [json.loads(line) for line in data.decode("utf-8").split("\n") if line.strip()]
    if manifest.get("complete") is not True or manifest.get("synthetic") is not False or manifest.get("schema_version") != 1:
        raise SystemExit(f"Incomplete, synthetic or unsupported evaluation: {name}")
    expected_n = config["evaluation"]["n_samples" if split == "test" else "dev_samples"]
    expected_identity = candidate["adapter_identity"] if candidate else None
    expected_path = candidate["adapter_path"] if candidate else None
    if manifest["model_label"] != label or manifest["split"] != split or manifest["n_samples"] != expected_n or manifest["adapter_identity"] != expected_identity or manifest.get("adapter_path") != expected_path:
        raise SystemExit(f"Evaluation model/split/sample/adapter mismatch: {name}")
    if manifest["settings"] != selection["settings"] or manifest["code_hashes"] != selection["code_hashes"] or manifest["config"] != config or manifest["model_identity"] != selection["model_identity"]:
        raise SystemExit(f"Evaluation settings/code/config/model mismatch: {name}")
    if hashlib.sha256(json.dumps(manifest["model_files"], sort_keys=True).encode()).hexdigest() != manifest["model_identity"]:
        raise SystemExit(f"Invalid recorded model inventory identity: {name}")
    if manifest["dataset_sha256"] != prepared["files"][f"{split}.jsonl"]["sha256"] or hashlib.sha256(data).hexdigest() != manifest["samples_sha256"]:
        raise SystemExit(f"Evaluation dataset/sample hash mismatch: {name}")
    protocol = {key: manifest[key] for key in ("dataset_sha256", "n_samples", "settings", "code_hashes")}
    if hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() != manifest["protocol_id"]:
        raise SystemExit(f"Evaluation protocol hash mismatch: {name}")
    if split == "test" and manifest["selection"] != selection:
        raise SystemExit(f"Test did not retain the frozen checkpoint selection: {name}")
    if split == "dev" and manifest["protocol_id"] != candidate["protocol_id"]:
        raise SystemExit(f"Candidate protocol mismatch: {name}")
    expected_grid = {(question_id, index) for question_id in splits[split] for index in range(expected_n)}
    if len(rows) != len(expected_grid) or {(row["id"], row["sample_index"]) for row in rows} != expected_grid or manifest["question_count"] != len(splits[split]) or manifest["total_samples"] != len(rows):
        raise SystemExit(f"Incomplete/duplicate sample grid: {name}")
    for row in rows:
        gold = splits[split][row["id"]]
        if any(row.get(key) != gold.get(key) for key in ("question", "answer", "family_id", "variant_type", "source", "split")):
            raise SystemExit(f"Sample metadata differs from archived gold: {name}/{row['id']}")
        if type(row["sample_index"]) is not int or type(row.get("sample_seed")) is not int or type(row.get("output_tokens")) is not int or row["output_tokens"] < 0 or not isinstance(row["text"], str):
            raise SystemExit(f"Invalid generation metadata: {name}/{row['id']}")
        if any(key not in row for key in score_fields) or any(type(row[key]) is not bool for key in ("correct", "format_compliant", "truncated")):
            raise SystemExit(f"Invalid original scoring fields: {name}/{row['id']}")
        if row["status"] in ("infrastructure_error", "invalid_gold"):
            raise SystemExit(f"Original evaluation contains an invalid result: {name}")
    replayed = runpy.run_path(str(archive / "scripts/verify_answers.py"), init_globals={"SCORE_INPUT": rows})["SCORE_OUTPUT"]
    if len(replayed) != len(rows) or any(any(row[key] != score.get(key) for key in score_fields) for row, score in zip(rows, replayed)):
        raise SystemExit(f"Original verifier replay differs from archived scores: {name}")
    if candidate and split == "dev":
        original_pass = sum(row["correct"] for row in rows) / len(rows)
        if original_pass != candidate["pass_at_1"]:
            raise SystemExit(f"Original dev score differs from frozen selection: {name}")
        old_scores.append((original_pass, -candidate["step"]))
    evaluations[name] = {"manifest": manifest, "rows": rows, "directory": directory, "candidate": candidate}
if -max(old_scores)[1] != selection["step"]:
    raise SystemExit("Frozen selection is inconsistent with its original dev criterion")
for field in ("protocol_id", "model_files", "settings", "code_hashes"):
    if evaluations["test_base"]["manifest"][field] != evaluations["test_rl"]["manifest"][field]:
        raise SystemExit(f"Main paired runs mismatch: {field}")
base_seeds = {(row["id"], row["sample_index"]): row["sample_seed"] for row in evaluations["test_base"]["rows"]}
rl_seeds = {(row["id"], row["sample_index"]): row["sample_seed"] for row in evaluations["test_rl"]["rows"]}
if base_seeds != rl_seeds:
    raise SystemExit("Main paired sample seeds differ")
dev_seed_grids = [{(row["id"], row["sample_index"]): row["sample_seed"] for row in evaluation["rows"]}
                  for name, evaluation in evaluations.items() if name.startswith("dev/")]
if any(grid != dev_seed_grids[0] for grid in dev_seed_grids[1:]):
    raise SystemExit("Development candidate sample seeds differ")

runtime_paths = [root / "scripts" / name for name in ("verify_answers.py", "analyze.py", "rescore_saved.py")]
if any(path.is_symlink() or not path.is_file() for path in runtime_paths):
    raise SystemExit("Missing or unsafe active scoring code")
runtime_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in runtime_paths}
identity = {"schema_version": 1, "kind": "posthoc_offline_rescore", "source_files_sha256": source_hashes,
            "scoring_code_sha256": runtime_hashes, "bootstrap": args.bootstrap, "seed": args.seed,
            "frozen_selected_step": selection["step"], "frozen_adapter_identity": selection["adapter_identity"]}
identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
if output.exists():
    marker = output / "ready.json"
    if not marker.is_file() or marker.is_symlink():
        raise SystemExit("Output is partial or unsafe; retain it for inspection and choose a new --output directory")
    ready = json.loads(marker.read_bytes())
    if ready.get("identity") != identity or ready.get("identity_sha256") != identity_hash or ready.get("complete") is not True:
        raise SystemExit("Existing output belongs to different source/scoring code/options; choose a new --output directory")
    actual_files = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"Unsafe output symlink: {path}")
        if path.is_file() and path != marker:
            actual_files.add(str(path.relative_to(output)))
    if actual_files != set(ready["artifacts_sha256"]):
        raise SystemExit("Existing output artifact inventory differs from its completed manifest")
    for relative, expected in ready["artifacts_sha256"].items():
        if hashlib.sha256((output / relative).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Existing output artifact hash mismatch: {relative}")
    print(f"✅ [RESCORE CACHED] 100% | verified immutable result | {output}", flush=True)
    raise SystemExit(0)

output.mkdir(parents=True, exist_ok=False)
print(f"[RESCORE 2/5] 20% | SCORE {sum(len(item['rows']) for item in evaluations.values())} SAVED ANSWERS | models=0 | network=0", flush=True)
diagnostics = {}
dev_scores = []
total = sum(len(item["rows"]) for item in evaluations.values())
processed = 0
changes = []
for name, evaluation in evaluations.items():
    rows = evaluation["rows"]
    scores = runpy.run_path(str(root / "scripts/verify_answers.py"), init_globals={"SCORE_INPUT": rows})["SCORE_OUTPUT"]
    if len(scores) != len(rows):
        raise SystemExit(f"Current verifier output length differs: {name}")
    rescored = []
    for row, score in zip(rows, scores):
        if set(score) != set(score_fields) or score["status"] in ("infrastructure_error", "invalid_gold"):
            raise SystemExit(f"Invalid new verifier result: {name}/{row['id']}")
        original = {key: row[key] for key in score_fields}
        revised = {**row, **score, "original_score": original}
        rescored.append(revised)
        if original != score:
            changes.append({"evaluation": name, "id": row["id"], "sample_index": row["sample_index"], "text": row["text"], "answer": row["answer"], "original_score": original, "revised_score": score})
    destination = output / name
    destination.mkdir(parents=True)
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rescored).encode("utf-8")
    (destination / "samples.jsonl").write_bytes(raw)
    manifest = dict(evaluation["manifest"])
    directory = evaluation["directory"]
    manifest["scoring_revision"] = {"kind": "posthoc_offline_rescore", "verifier_sha256": runtime_hashes["verify_answers.py"],
        "source_manifest_sha256": source_hashes[str(directory / "manifest.json")], "source_samples_sha256": source_hashes[str(directory / "samples.jsonl")],
        "original_training_reward_retained": True, "original_checkpoint_selection_retained": True}
    manifest["samples_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest["original_generation_manifest"] = evaluation["manifest"]
    manifest["generation_protocol_id"] = manifest["protocol_id"]
    manifest["scoring_protocol_id"] = hashlib.sha256(json.dumps({"generation_protocol_id": manifest["protocol_id"], "verifier_sha256": runtime_hashes["verify_answers.py"]}, sort_keys=True).encode()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnostics[name] = {"samples": len(rows), "original_correct": sum(row["correct"] for row in rows), "revised_correct": sum(row["correct"] for row in rescored),
        "original_status_counts": dict(Counter(row["status"] for row in rows)), "revised_status_counts": dict(Counter(row["status"] for row in rescored)),
        "changed_scores": sum(row["original_score"] != {key: row[key] for key in score_fields} for row in rescored),
        "incorrect_to_correct": sum(not row["original_score"]["correct"] and row["correct"] for row in rescored),
        "correct_to_incorrect": sum(row["original_score"]["correct"] and not row["correct"] for row in rescored)}
    if name.startswith("dev/"):
        dev_scores.append({"step": evaluation["candidate"]["step"], "original_pass_at_1": diagnostics[name]["original_correct"] / len(rows), "revised_pass_at_1": diagnostics[name]["revised_correct"] / len(rows)})
    processed += len(rows)
    print(f"[RESCORE PROGRESS] {processed}/{total} | {100 * processed / total:.1f}% | {name} | changed={diagnostics[name]['changed_scores']}", flush=True)

print("[RESCORE 3/5] 60% | PAIRED BOOTSTRAP | posthoc sensitivity; original training reward and selection retained", flush=True)
command = [sys.executable, "-S", str(root / "scripts/analyze.py"), "--base", str(output / "test_base"), "--rl", str(output / "test_rl"), "--output", str(output / "main_analysis"), "--bootstrap", str(args.bootstrap), "--seed", str(args.seed)]
analysis = subprocess.run(command, cwd=root, text=True, capture_output=True)
(output / "analysis.log").write_text(analysis.stdout + analysis.stderr, encoding="utf-8")
print(analysis.stdout, end="", flush=True)
if analysis.returncode:
    print(analysis.stderr, file=sys.stderr, end="", flush=True)
    raise SystemExit(f"Analysis failed; partial output retained at {output}; choose a new --output after repair")
new_best = sorted(dev_scores, key=lambda item: (-item["revised_pass_at_1"], item["step"]))[0]
sensitivity = {"kind": "posthoc_selection_sensitivity_only", "original_frozen_selection": selection, "candidate_scores": dev_scores,
    "would_select_step_under_revised_dev_scores": new_best["step"], "selection_would_change": new_best["step"] != selection["step"],
    "test_adapter_step_retained": selection["step"], "automatic_reselection_performed": False,
    "limitation": "These saved test answers remain from the original frozen adapter. Revised dev scores do not authorize reselection or validate another adapter on test."}
(output / "selection_sensitivity.json").write_text(json.dumps(sensitivity, indent=2) + "\n", encoding="utf-8")
(output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
(output / "changed_scores.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in changes), encoding="utf-8")
(output / "provenance.json").write_text(json.dumps({**identity, "source_archive": str(archive), "source_run": str(run), "generation_config": config,
    "original_selection": selection, "models_loaded": 0, "new_generations": 0, "network_required": False, "archive_modified": False,
    "limitations": ["Posthoc scoring sensitivity, not a new independent experiment.", "Original GRPO training rewards and original checkpoint selection are retained.", "Numerical answer equivalence does not establish reasoning correctness.", "Base model weights are not required or reverified; recorded model inventory identity and saved adapter bytes are verified."]}, indent=2) + "\n", encoding="utf-8")
notes = ["# Offline answer rescoring (posthoc sensitivity)", "", "This result reuses existing generated text. No model, GPU, training or network request is needed.",
    "", "Original archived files are unchanged. Original generation settings/code/protocol remain in each manifest; scoring_revision identifies the new verifier. Each row retains original_score.",
    "", "Original training rewards and the originally selected test adapter are retained. This is a posthoc sensitivity analysis, not a fresh experiment under the revised reward.",
    "", f"Frozen test checkpoint: {selection['step']}. Revised dev ranking would choose: {new_best['step']}. No automatic reselection was performed.",
    "", "| Evaluation | Samples | Original correct | Revised correct | Changed scores |", "|---|---:|---:|---:|---:|"]
for name, item in diagnostics.items():
    notes.append(f"| {name} | {item['samples']} | {item['original_correct']} | {item['revised_correct']} | {item['changed_scores']} |")
notes += ["", "See main_analysis/report.md for paired question-level bootstrap statistics, changed_scores.jsonl for changed decisions, selection_sensitivity.json for all development candidates, and ready.json for integrity hashes.", ""]
(output / "README.md").write_text("\n".join(notes), encoding="utf-8")

print("[RESCORE 4/5] 80% | RECHECK INPUT IMMUTABILITY AND SAVE PROVENANCE", flush=True)
for relative, expected in source_hashes.items():
    path = archive / relative
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise SystemExit(f"Archive input became a symlink: {relative}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit(f"Archive changed during rescoring: {relative}; output remains incomplete")
for path in runtime_paths:
    if hashlib.sha256(path.read_bytes()).hexdigest() != runtime_hashes[path.name]:
        raise SystemExit(f"Scoring code changed during rescoring: {path.name}; output remains incomplete")
artifacts = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.rglob("*")) if path.is_file()}
ready = {"complete": True, "created_utc": datetime.now(timezone.utc).isoformat(), "identity": identity, "identity_sha256": identity_hash,
    "artifacts_sha256": artifacts, "models_loaded": 0, "new_generations": 0, "gpu_execution": False, "source_archive_modified": False}
temporary_marker = output / ".ready.json.tmp"
temporary_marker.write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
temporary_marker.replace(output / "ready.json")
print(f"✅ [RESCORE 5/5] 100% | COMPLETE | {total} saved answers | [SAVED] {output / 'ready.json'}", flush=True)
