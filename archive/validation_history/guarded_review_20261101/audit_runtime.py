"""Audit downloaded token traces and runtime evidence without model execution."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import statistics

output = Path(__file__).resolve().parent
package = output.parents[2] / "boundary_guarded_v3"
source = Path("/Users/xu./Downloads/boundary_guarded_v3_outputs/guarded_v3")
rows = [json.loads(line) for line in (source / "base/samples.jsonl").read_text().splitlines()]
by_key = {(r["id"], r["sample_index"]): r for r in rows}
assert len(by_key) == len(rows) == 800
manifest = json.loads((source / "base/manifest.json").read_text())
source_code_matches = {}
for name, expected in manifest["code_hashes"].items():
    actual = hashlib.sha256((package / "scripts" / name).read_bytes()).hexdigest()
    source_code_matches[name] = actual == expected
assert all(source_code_matches.values()), source_code_matches
seen = set()
trace_chunks = 0
trace_tokens = 0
for index in (0, 1):
    shard_rows = [json.loads(line) for line in (source / f"worker_{index}/samples.jsonl").read_text().splitlines()]
    shard_keys = {(r["id"], r["sample_index"]) for r in shard_rows}
    for path in sorted((source / f"worker_{index}/live").glob("*.tokens.jsonl")):
        tokens = []
        key = None
        last_elapsed = -1
        for line in path.read_text().splitlines():
            event = json.loads(line)
            event_key = (event["id"], event["sample_index"])
            if key is None:
                key = event_key
            assert event_key == key and event["offset"] == len(tokens)
            assert event["elapsed_seconds"] >= last_elapsed
            last_elapsed = event["elapsed_seconds"]
            tokens.extend(event["token_ids"])
            trace_chunks += 1
        assert key in shard_keys and key not in seen, path
        assert tokens == by_key[key]["output_token_ids"], key
        assert last_elapsed <= by_key[key]["seconds"], key
        trace_tokens += len(tokens)
        seen.add(key)
assert seen == set(by_key)
invocations = [json.loads(p.read_text()) for p in source.glob("invocation-*.json")]
assert len(invocations) == 1
runtime = invocations[0]
assert runtime["status"] == "completed" and runtime["handled_attempts"] == 800
attempts = runtime["worker_attempts"]
assert len(attempts) == 2 and attempts[0]["finished_at_unix"] <= attempts[1]["started_at_unix"]
assert all(a["exit_code"] == 0 for a in attempts)
assert runtime["concurrent_workers"] == 1 and runtime["restarts_used"] == 0
all_seconds = sum(r["seconds"] for r in rows)
groups = {}
for group_id in sorted({r["group_id"] for r in rows}):
    group_rows = [r for r in rows if r["group_id"] == group_id]
    per_question = defaultdict(list)
    for row in group_rows:
        per_question[row["id"]].append(row)
    correct_histogram = Counter(sum(r["correct"] is True for r in qs) for qs in per_question.values())
    terms = {}
    for reason in sorted({r["termination_reason"] for r in group_rows}):
        selected = [r for r in group_rows if r["termination_reason"] == reason]
        terms[reason] = {"count": len(selected), "seconds_sum": sum(r["seconds"] for r in selected),
            "seconds_median": statistics.median(r["seconds"] for r in selected),
            "seconds_min": min(r["seconds"] for r in selected), "seconds_max": max(r["seconds"] for r in selected),
            "output_tokens_sum": sum(r["output_tokens"] for r in selected),
            "output_tokens_min": min(r["output_tokens"] for r in selected), "output_tokens_max": max(r["output_tokens"] for r in selected)}
    subjects = {}
    for subject in sorted({r["source"].get("subject", "not_applicable") for r in group_rows}):
        selected = [r for r in group_rows if r["source"].get("subject", "not_applicable") == subject]
        subjects[subject] = {"questions": len({r["id"] for r in selected}), "attempts": len(selected),
            "correct": sum(r["correct"] is True for r in selected),
            "success_rate": sum(r["correct"] is True for r in selected) / len(selected)}
    groups[group_id] = {"questions": len(per_question), "attempts": len(group_rows),
        "correct_per_question_histogram": {str(k): correct_histogram[k] for k in range(5)},
        "question_coverage": sum(any(r["correct"] is True for r in qs) for qs in per_question.values()) / len(per_question),
        "termination": terms, "subjects": subjects}
stop_seconds = sum(r["seconds"] for r in rows if r["termination_reason"] != "eos")
report = {"complete": True, "live_traces_verified": len(seen), "trace_chunks": trace_chunks,
    "trace_tokens": trace_tokens, "source_code_hash_matches": source_code_matches,
    "runtime": runtime, "strictly_sequential": True, "all_answer_seconds": all_seconds,
    "guarded_answer_seconds": stop_seconds, "guarded_time_share": stop_seconds / all_seconds,
    "timeout_time_share": sum(r["seconds"] for r in rows if r["termination_reason"] == "answer_timeout") / all_seconds,
    "groups": groups,
    "gpu_peak_allocated_gib": [json.loads((source / f"worker_{i}/manifest.json").read_text())["peak_allocated_gib"] for i in (0,1)],
    "weights_rehashed_locally": False}
initial = json.loads((output / "download_inventory.json").read_text())
final = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*") if p.is_file()}
assert initial == final, "Downloaded source files changed during the audit"
report["downloaded_files_unchanged"] = len(final)
(output / "runtime_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"trace_attempts": len(seen), "tokens": trace_tokens, "source_files_unchanged": len(final), "strictly_sequential": True, "guarded_time_share": report["guarded_time_share"]}, indent=2))
