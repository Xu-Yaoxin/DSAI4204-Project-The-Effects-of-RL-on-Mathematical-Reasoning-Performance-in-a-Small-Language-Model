"""Evaluate all retained checkpoints on development data and freeze one selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/v1.json")
parser.add_argument("--run", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
if (int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1
        or int(os.environ.get("RANK", "0")) not in {-1, 0} or int(os.environ.get("LOCAL_RANK", "0")) not in {-1, 0}):
    raise ValueError("Checkpoint selection uses one process and one selected GPU; do not launch it with multi-process torchrun.")
root = Path(__file__).resolve().parents[1]
cfg = json.loads((root / args.config).read_text())
run = (root / args.run).resolve()
report = json.loads((run / "report.json").read_text())
if (report["status"] != "completed" or report["mode"] != "formal" or report["gates"].get("all_passed") is not True
        or report["gates"].get("loss_normalization") is not True
        or any(value is not True for value in report["gates"].values())):
    raise ValueError("Select from a completed formal run whose integrity gates all passed.")
training_manifest = json.loads((run / "manifest.json").read_text())
if report.get("model_identity") != training_manifest.get("model_identity"):
    raise ValueError("Training report and manifest refer to different base weights.")
if training_manifest["config"]["prompt"] != cfg["prompt"] or training_manifest["config"]["model"] != cfg["model"]:
    raise ValueError("Model/prompt changed since training.")
training_topology = report.get("topology")
training_world_size = 1
if training_topology is not None or training_manifest.get("topology") is not None:
    if not isinstance(training_topology, dict) or training_topology != training_manifest.get("topology"):
        raise ValueError("Training report and manifest have different GPU topologies.")
    topology_fields = ["world_size", "local_micro_batch", "gradient_accumulation_steps_per_rank",
        "global_effective_batch_size", "global_unique_prompts_per_update", "per_rank_generation_batch_size",
        "global_generation_batch_size", "per_rank_unique_prompts_per_update", "steps_per_generation"]
    if any(type(training_topology.get(field)) is not int or training_topology[field] < 1 for field in topology_fields):
        raise ValueError("Training topology has invalid integer batch dimensions.")
    training_world_size = training_topology["world_size"]
    if training_world_size not in {1, 2, 4} or training_topology.get("single_node") is not True:
        raise ValueError("Only completed single-node 1/2/4-GPU training topologies are supported.")
    training_settings = training_manifest["config"]["training"]
    micro_batch = training_topology["local_micro_batch"]
    accumulation = training_topology["gradient_accumulation_steps_per_rank"]
    generations = training_settings["num_generations"]
    global_batch = micro_batch * accumulation * training_world_size
    local_generation_batch = micro_batch * training_topology["steps_per_generation"]
    if (micro_batch != training_settings["per_device_train_batch_size"]
            or global_batch != micro_batch * training_settings["gradient_accumulation_steps"]
            or global_batch != training_topology["global_effective_batch_size"]
            or global_batch % generations or micro_batch * accumulation % generations
            or training_topology["global_unique_prompts_per_update"] != global_batch // generations
            or training_topology["per_rank_unique_prompts_per_update"] != micro_batch * accumulation // generations
            or local_generation_batch % generations
            or training_topology["per_rank_generation_batch_size"] != local_generation_batch
            or training_topology["global_generation_batch_size"] != local_generation_batch * training_world_size
            or accumulation % training_topology["steps_per_generation"]):
        raise ValueError("Training topology violates its recorded GRPO batch/group constraints.")
    rank_audits = report.get("rank_audits")
    if (report["gates"].get("rank_consistency") is not True or not isinstance(rank_audits, list)
            or len(rank_audits) != training_world_size
            or any(not isinstance(item, dict) or type(item.get("rank")) is not int for item in rank_audits)
            or {item["rank"] for item in rank_audits} != set(range(training_world_size))):
        raise ValueError("Training requires a complete, unique audit for every GPU rank.")
    if type(report.get("optimizer_steps")) is not int or report["optimizer_steps"] < 1:
        raise ValueError("Training optimizer-step count must be positive.")
    rank_flags = ["training_completed", "finite_metrics", "base_unchanged", "adapter_updated", "all_adapters_finite", "loss_normalization_verified"]
    for item in rank_audits:
        if (type(item.get("local_rank")) is not int or item["local_rank"] != item["rank"]
                or any(item.get(flag) is not True for flag in rank_flags)
                or type(item.get("optimizer_steps")) is not int or item["optimizer_steps"] != report["optimizer_steps"]
                or type(item.get("effective_update_steps")) is not int or not 1 <= item["effective_update_steps"] <= report["optimizer_steps"]
                or item.get("nonfinite_metrics") != []
                or type(item.get("reward_groups")) is not int
                or item["reward_groups"] != report["optimizer_steps"] * training_topology["per_rank_unique_prompts_per_update"]
                or any(not isinstance(item.get(field), str) or len(item[field]) != 64
                       or any(character not in "0123456789abcdef" for character in item[field])
                       for field in ["base_sha256_before", "base_sha256_after", "adapter_sha256_after", "reward_audit_sha256"])
                or item["base_sha256_before"] != item.get("base_sha256_after")
                ):
            raise ValueError(f"Training rank {item['rank']} failed its recorded integrity checks.")
        rank_bytes = (run / f"reward_audit.rank-{item['rank']}.jsonl").read_bytes()
        rank_groups = [json.loads(line) for line in rank_bytes.decode().split("\n") if line.strip()]
        if (hashlib.sha256(rank_bytes).hexdigest() != item.get("reward_audit_sha256")
                or len(rank_groups) != item["reward_groups"]
                or any(type(group.get("rank")) is not int or group["rank"] != item["rank"] for group in rank_groups)):
            raise ValueError(f"Training rank {item['rank']} reward evidence changed or is incomplete.")
    if (len({item["base_sha256_after"] for item in rank_audits}) != 1
            or len({item["adapter_sha256_after"] for item in rank_audits}) != 1):
        raise ValueError("Training ranks ended with inconsistent base or adapter parameters.")
expected_accumulation = (training_topology["gradient_accumulation_steps_per_rank"] if training_topology
    else training_manifest["config"]["training"]["gradient_accumulation_steps"])
normalization = report.get("loss_normalization")
if (not isinstance(normalization, dict) or normalization != training_manifest.get("loss_normalization")
        or any(type(normalization.get(field)) is not int for field in ["trainer_accumulation_steps",
            "accelerate_backward_divisor_before", "accelerate_backward_divisor"])
        or normalization.get("owner") != "trl_grpo"
        or normalization.get("trainer_accumulation_steps") != expected_accumulation
        or normalization.get("accelerate_backward_divisor_before") != expected_accumulation
        or normalization.get("accelerate_backward_divisor") != 1
        or normalization.get("compatibility_fix") != "transformers_5_3_0_accelerate_1_10_1_double_scaling"):
    raise ValueError("Training requires consistent verified loss-normalization provenance.")
data_dir = root / cfg["data"]["output_dir"]
for split in ["train", "dev", "test"]:
    if hashlib.sha256((data_dir / f"{split}.jsonl").read_bytes()).hexdigest() != report["split_sha256"][split]:
        raise ValueError(f"{split} data changed since training.")
output = (root / args.output).resolve()
if output.exists() and not args.resume:
    raise FileExistsError(f"Selection output exists: {output}; use --resume or a new path.")
output.mkdir(parents=True, exist_ok=True)
frozen = None
if (output / "selection.json").exists():
    if not args.resume:
        raise FileExistsError("Selection is already frozen. Use --resume to validate and reuse it without changes.")
    frozen = json.loads((output / "selection.json").read_text())
    if frozen.get("evaluation_config") != cfg["evaluation"] or frozen.get("prompt") != cfg["prompt"]:
        raise ValueError("The frozen selection configuration changed; refusing to reselect.")
candidates = sorted((item["step"], (root / item["path"]).resolve()) for item in report["checkpoints"])
if not any(step == report["optimizer_steps"] for step, _ in candidates):
    candidates.append((report["optimizer_steps"], (root / report["adapter_path"]).resolve()))
if not candidates or len({step for step, _ in candidates}) != len(candidates):
    raise ValueError("Checkpoint candidates must have unique optimizer steps.")
if len({adapter.name for _, adapter in candidates}) != len(candidates):
    raise ValueError("Checkpoint directory names must be unique for development output paths.")
dev_rows = {row["id"]: row for row in (json.loads(line) for line in (data_dir / "dev.jsonl").read_text().split("\n") if line.strip())}
scores = []
for step, adapter in candidates:
    destination = output / "dev" / adapter.name
    if frozen and (not (destination / "manifest.json").is_file()
            or json.loads((destination / "manifest.json").read_text()).get("complete") is not True):
        raise ValueError("A frozen selection requires every original completed development evaluation; refusing to rerun it.")
    command = [sys.executable, str(root / "scripts/evaluate.py"), "--config", str((root / args.config).resolve()),
               "--model", "rl", "--adapter", str(adapter), "--split", "dev", "--output", str(destination)]
    if destination.exists() and args.resume:
        command.append("--resume")
    subprocess.run(command, check=True)
    manifest = json.loads((destination / "manifest.json").read_text())
    samples = [json.loads(line) for line in (destination / "samples.jsonl").read_text().split("\n") if line.strip()]
    if not manifest["complete"] or manifest["split"] != "dev" or manifest["synthetic"]:
        raise ValueError("Checkpoint selection needs complete real development evaluations.")
    if manifest["model_identity"] != report["model_identity"]:
        raise ValueError("Checkpoint base identity differs from training.")
    if scores and manifest["protocol_id"] != scores[0]["protocol_id"]:
        raise ValueError("Development protocols differ across checkpoints.")
    keys = [(sample["id"], sample["sample_index"]) for sample in samples]
    expected = {(qid, index) for qid in dev_rows for index in range(manifest["n_samples"])}
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate development sample grid.")
    if hashlib.sha256((destination / "samples.jsonl").read_bytes()).hexdigest() != manifest["samples_sha256"]:
        raise ValueError("Development samples differ from their completed manifest hash.")
    for sample in samples:
        if (type(sample["correct"]) is not bool or sample["question"] != dev_rows[sample["id"]]["question"]
                or sample["answer"] != dev_rows[sample["id"]]["answer"] or sample["status"] in {"invalid_gold", "infrastructure_error"}):
            raise ValueError("Invalid development scoring metadata.")
    mean_pass1 = sum(sample["correct"] for sample in samples) / len(samples)
    scores.append({"step": step, "adapter_path": str(adapter.relative_to(root)) if adapter.is_relative_to(root) else str(adapter), "adapter_identity": manifest["adapter_identity"],
                   "pass_at_1": mean_pass1, "protocol_id": manifest["protocol_id"],
                   "evaluation_path": str(destination.relative_to(root)) if destination.is_relative_to(root) else str(destination)})
    print(f"[SELECT] step={step} | dev numerical pass@1={mean_pass1:.4f} | {adapter.name}", flush=True)
best = scores[0]
for item in scores[1:]:
    if item["pass_at_1"] > best["pass_at_1"]:
        best = item
selection = {"schema_version": 1, "path_base": "code_root", "criterion": "development numerical pass@1; ties choose earliest saved step",
             **best, "model_identity": report["model_identity"], "evaluation_config": cfg["evaluation"],
             "prompt": cfg["prompt"], "candidates": scores,
             "training_run": str(run.relative_to(root)) if run.is_relative_to(root) else str(run),
             "training_report_sha256": hashlib.sha256((run / "report.json").read_bytes()).hexdigest(),
             "training_manifest_sha256": hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
             "training_topology": training_topology, "training_world_size": training_world_size,
             "settings": manifest["settings"], "code_hashes": manifest["code_hashes"],
             "dev_sha256": report["split_sha256"]["dev"], "test_sha256": report["split_sha256"]["test"]}
selection["structural_sha256"] = (hashlib.sha256((data_dir / "structural.jsonl").read_bytes()).hexdigest()
    if (data_dir / "structural.jsonl").is_file() else None)
if frozen:
    if frozen != selection:
        raise ValueError("Frozen selection contents, source records or candidate scores changed; refusing to rewrite it.")
    print(f"[SELECTION CACHED] Validated the frozen selection without loading a GPU model; {output/'selection.json'}")
    raise SystemExit(0)
with (output / "selection.json").open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(selection, indent=2) + "\n")
print(f"✅ [SELECTION FROZEN] {output/'selection.json'}")
