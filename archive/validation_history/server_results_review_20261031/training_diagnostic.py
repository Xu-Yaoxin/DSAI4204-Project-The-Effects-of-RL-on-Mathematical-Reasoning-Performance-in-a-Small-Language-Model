"""Summarize archived training and rescore rewards for one post-hoc parser diagnostic."""
import hashlib
import json
from pathlib import Path
import statistics

output = Path(__file__).resolve().parent
archive = output.parents[3] / "server_results/run_20261031/code"
original_path = archive / "scripts/verify_answers.py"
patched_path = output / "nested_marker_diagnostic/verify_answers.nested_marker_only.py"
original = compile(original_path.read_text(), str(original_path), "exec")
patched = compile(patched_path.read_text(), str(patched_path), "exec")
records = {row["id"]: row for row in (json.loads(line) for line in
    (archive / "data/processed/train.jsonl").read_text().split("\n") if line.strip())}
results = {"scope": "Post-hoc diagnostic on recorded training completions; actual training and its rewards unchanged",
    "original_verifier_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
    "diagnostic_verifier_sha256": hashlib.sha256(patched_path.read_bytes()).hexdigest(), "stages": {}}
for stage in ("pilot", "formal"):
    directory = archive / f"outputs/v1_2/{stage}"
    report = json.loads((directory / "report.json").read_text())
    groups = [json.loads(line) for line in (directory / "reward_audit.jsonl").read_text().split("\n") if line.strip()]
    examples = []
    batch = []
    references = []
    for group in groups:
        for index, sample in enumerate(group["samples"]):
            batch.append({"text": sample["text"], "answer": records[group["question_id"]]["answer"],
                "truncated": sample["truncated"]})
            references.append((group["optimizer_step_before_update"], group["question_id"], index, sample["verifier"]))
    namespace = {"SCORE_INPUT": batch}
    exec(original, namespace)
    assert all(score == reference[3] for score, reference in zip(namespace["SCORE_OUTPUT"], references))
    primary_correct = sum(score["correct"] for score in namespace["SCORE_OUTPUT"])
    namespace = {"SCORE_INPUT": batch}
    exec(patched, namespace)
    diagnostic_correct = sum(score["correct"] for score in namespace["SCORE_OUTPUT"])
    changed_rewards = 0
    for row, reference, score in zip(batch, references, namespace["SCORE_OUTPUT"]):
        if score["reward"] != reference[3]["reward"]:
            changed_rewards += 1
            if len(examples) < 4:
                examples.append({"optimizer_step_before_update": reference[0], "question_id": reference[1],
                    "generation_index": reference[2], "text": row["text"], "answer": row["answer"],
                    "original_reward": reference[3]["reward"], "diagnostic_reward": score["reward"]})
    assert len(batch) == report["completion_count"] and primary_correct == report["correct_completion_count"]
    logs = [entry for entry in json.loads((directory / "trainer_logs.json").read_text()) if "reward" in entry]
    updates_per_epoch = report["optimizer_steps"] // 3
    epochs = []
    for epoch in (1, 2, 3):
        selected = [entry for entry in logs if (epoch - 1) * updates_per_epoch < entry["step"] <= epoch * updates_per_epoch]
        epochs.append({"epoch": epoch, "steps": len(selected), **{key: statistics.mean(entry[key] for entry in selected)
            for key in ("reward", "entropy", "frac_reward_zero_std", "completions/clipped_ratio", "step_time")}})
    results["stages"][stage] = {"optimizer_steps": report["optimizer_steps"], "effective_update_steps": report["effective_update_steps"],
        "training_hours": report["training_seconds"] / 3600,
        "peak_cuda_allocated_gib": report["peak_cuda_allocated_bytes"] / 1024**3,
        "peak_cuda_reserved_gib": report["peak_cuda_reserved_bytes"] / 1024**3,
        "recorded_completions": len(batch), "primary_correct": primary_correct,
        "diagnostic_correct": diagnostic_correct, "reward_changes": changed_rewards,
        "reward_change_examples": examples, "epochs": epochs}
(output / "training_diagnostic.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({stage: {key: value for key, value in data.items() if key not in ("epochs", "reward_change_examples")}
    for stage, data in results["stages"].items()}, indent=2))
