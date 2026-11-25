"""Check the real reward bridge without loading a model or importing CUDA libraries."""

import ast
import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace

script = Path(__file__).resolve().parents[1] / "scripts" / "train_grpo.py"
tree = ast.parse(script.read_text(encoding="utf-8"))
assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
assert sum(isinstance(node, ast.Lambda) for node in ast.walk(tree)) == 1
test_count = 1

# Execute the production normalization adjustment without loading the training stack.
normalization_statements = next(value for node in ast.walk(tree) for _, value in ast.iter_fields(node)
    if isinstance(value, list) and any(isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "accelerator_accumulation_before"
                for target in item.targets) for item in value))
normalization_start = next(index for index, node in enumerate(normalization_statements)
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
        and target.id == "accelerator_accumulation_before" for target in node.targets))
normalization_end = next(index for index in range(normalization_start, len(normalization_statements))
    if isinstance(normalization_statements[index], ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "model"
        for target in normalization_statements[index].targets))
normalization_code = compile(ast.Module(body=normalization_statements[normalization_start:normalization_end],
    type_ignores=[]), str(script), "exec")
for world_size in (1, 2, 4):
    accumulation = 32 // world_size
    trainer = SimpleNamespace(accelerator=SimpleNamespace(gradient_accumulation_steps=accumulation),
        args=SimpleNamespace(gradient_accumulation_steps=accumulation), loss_is_scaled_for_ga=True,
        model_accepts_loss_kwargs=False, compute_loss_func="TRL disables HF scaling",
        is_deepspeed_enabled=False, is_fsdp_enabled=False)
    namespace = {"trainer": trainer, "accumulation_per_rank": accumulation,
                 "report": {}, "manifest": {}, "rank": 1}
    exec(normalization_code, namespace)
    assert trainer.accelerator.gradient_accumulation_steps == 1
    assert trainer.args.gradient_accumulation_steps == accumulation
    assert namespace["report"]["loss_normalization"] == namespace["manifest"]["loss_normalization"]
    assert namespace["report"]["loss_normalization"]["accelerate_backward_divisor_before"] == accumulation
    test_count += 1
for invalid_field, invalid_value in (("loss_is_scaled_for_ga", False), ("compute_loss_func", None),
        ("model_accepts_loss_kwargs", True), ("is_deepspeed_enabled", True), ("is_fsdp_enabled", True)):
    trainer = SimpleNamespace(accelerator=SimpleNamespace(gradient_accumulation_steps=32),
        args=SimpleNamespace(gradient_accumulation_steps=32), loss_is_scaled_for_ga=True,
        model_accepts_loss_kwargs=False, compute_loss_func="TRL disables HF scaling",
        is_deepspeed_enabled=False, is_fsdp_enabled=False)
    setattr(trainer, invalid_field, invalid_value)
    try:
        exec(normalization_code, {"trainer": trainer, "accumulation_per_rank": 32,
             "report": {}, "manifest": {}, "rank": 1})
    except RuntimeError as error:
        assert "unverified workaround" in str(error)
    else:
        raise AssertionError(f"Unsupported normalization path accepted: {invalid_field}")
    assert trainer.accelerator.gradient_accumulation_steps == 32
    test_count += 1
for accelerator_steps, trainer_steps in ((1, 32), (16, 32), (32, 16)):
    trainer = SimpleNamespace(accelerator=SimpleNamespace(gradient_accumulation_steps=accelerator_steps),
        args=SimpleNamespace(gradient_accumulation_steps=trainer_steps), loss_is_scaled_for_ga=True,
        model_accepts_loss_kwargs=False, compute_loss_func="TRL disables HF scaling",
        is_deepspeed_enabled=False, is_fsdp_enabled=False)
    try:
        exec(normalization_code, {"trainer": trainer, "accumulation_per_rank": 32,
             "report": {}, "manifest": {}, "rank": 1})
    except RuntimeError as error:
        assert "unverified workaround" in str(error)
    else:
        raise AssertionError("An unexpected initial accumulation state was accepted")
    assert trainer.accelerator.gradient_accumulation_steps == accelerator_steps
    test_count += 1

with tempfile.TemporaryDirectory(prefix="rlvr-training-test-") as directory:
    audit_path = Path(directory) / "reward.jsonl"
    request = {
        "completions": ["#### 2", "#### 3", "2", "#### 2\n#### 3"],
        "answer": ["2"] * 4,
        "prompts": ["What is 1 + 1?"] * 4,
        "question_id": ["question-1"] * 4,
        "completion_ids": [[1, 99], [2, 99], [3, 4], [5, 99]],
        "group_size": 4,
        "audit_path": str(audit_path),
        "step": 7,
        "max_completion_length": 2,
        "eos_token_ids": [99],
    }
    result = runpy.run_path(str(script), init_globals={"REWARD_REQUEST": request})
    assert result["REWARD_VALUES"] == [1.0, 0.0, 1.0, 0.0]
    groups = [json.loads(line) for line in audit_path.read_text().split("\n") if line]
    assert len(groups) == 1 and groups[0]["mixed_rewards"] is True
    assert groups[0]["optimizer_step_before_update"] == 7
    assert groups[0]["question_id"] == "question-1"
    assert groups[0]["samples"][2]["truncated"] is True
    assert groups[0]["samples"][2]["verifier"]["truncated"] is True
    assert groups[0]["samples"][0]["truncated"] is False
    assert groups[0]["samples"][3]["verifier"]["status"] == "ambiguous"
    assert groups[0]["samples"][2]["verifier"]["format_compliant"] is False
    test_count += 1

    bad_cases = [
        ("question_id", ["q1", "q2", "q1", "q1"], ValueError, "question IDs differ"),
        ("prompts", ["one", "two", "one", "one"], ValueError, "prompts or answers differ"),
        ("answer", ["2", "3", "2", "2"], ValueError, "prompts or answers differ"),
        ("answer", ["invalid"] * 4, ValueError, "Invalid gold"),
        ("completions", ["#### 2"] * 3, ValueError, "complete GRPO groups"),
        ("completions", [], ValueError, "empty"),
        ("completion_ids", [[1]] * 3, ValueError, "mismatched lengths"),
        ("completions", [[{"role": "assistant", "content": "2"}]] * 4, TypeError, "plain text"),
        ("group_size", 0, ValueError, "at least two"),
        ("group_size", True, ValueError, "at least two"),
        ("expected_batch_size", 8, ValueError, "configured generation batch"),
    ]
    for field, value, expected_type, fragment in bad_cases:
        invalid = copy.deepcopy(request)
        invalid[field] = value
        try:
            runpy.run_path(str(script), init_globals={"REWARD_REQUEST": invalid})
        except expected_type as error:
            assert fragment in str(error), str(error)
        else:
            raise AssertionError(f"Invalid reward request was accepted: {field}={value}")
        test_count += 1

    all_wrong = copy.deepcopy(request)
    all_wrong["completions"] = ["#### 3"] * 4
    result = runpy.run_path(str(script), init_globals={"REWARD_REQUEST": all_wrong})
    assert result["REWARD_VALUES"] == [0.0] * 4
    last_group = json.loads(audit_path.read_text().strip().split("\n")[-1])
    assert last_group["mixed_rewards"] is False
    test_count += 1

    batch_audit = Path(directory) / "eight_rollout_groups.jsonl"
    for question_index in range(4):
        grouped = copy.deepcopy(request)
        grouped.update({"group_size": 8, "expected_batch_size": 8, "audit_path": str(batch_audit),
                        "completions": ["Reasoning\u2028line\n#### 2"] * 4 + ["#### 3"] * 4,
                        "question_id": [f"q{question_index}"] * 8, "prompts": [f"Question {question_index}"] * 8,
                        "answer": ["2"] * 8, "completion_ids": [[1, 99]] * 8})
        result = runpy.run_path(str(script), init_globals={"REWARD_REQUEST": grouped})
        assert result["REWARD_VALUES"] == [1.0] * 4 + [0.0] * 4
    groups = [json.loads(line) for line in batch_audit.read_text().split("\n") if line]
    assert len(groups) == 4 and len({row["question_id"] for row in groups}) == 4
    assert all(row["optimizer_step_before_update"] == 7 for row in groups)
    assert sum(len(row["rewards"]) for row in groups) == 32
    test_count += 1

    duplicated = copy.deepcopy(grouped)
    duplicated["expected_batch_size"] = 16
    for key in ("completions", "question_id", "prompts", "answer", "completion_ids"):
        duplicated[key] *= 2
    before = batch_audit.read_bytes()
    try:
        runpy.run_path(str(script), init_globals={"REWARD_REQUEST": duplicated})
    except ValueError as error:
        assert "distinct question IDs" in str(error)
    else:
        raise AssertionError("Duplicate question groups accepted")
    assert batch_audit.read_bytes() == before
    test_count += 1

    worker_source = "import json,runpy,sys\nfrom pathlib import Path\nfor request in json.loads(Path(sys.argv[2]).read_text()):\n    result=runpy.run_path(sys.argv[1],init_globals={'REWARD_REQUEST':request})\n    assert len(result['REWARD_VALUES'])==8\n"
    for planned_world_size in (1, 2, 4):
        scenario_dir = Path(directory) / f"parallel-{planned_world_size}"
        scenario_dir.mkdir()
        processes = []
        for process_rank in range(planned_world_size):
            requests = []
            for optimizer_step in range(2):
                for local_generation in range(4 // planned_world_size):
                    question_index = optimizer_step * 4 + local_generation * planned_world_size + process_rank
                    local_request = copy.deepcopy(grouped)
                    local_request.update({"rank": process_rank, "step": optimizer_step,
                        "audit_path": str(scenario_dir / f"reward_audit.rank-{process_rank}.jsonl"),
                        "question_id": [f"q{question_index}"] * 8,
                        "prompts": [f"Question {question_index}"] * 8})
                    requests.append(local_request)
            request_path = scenario_dir / f"requests-{process_rank}.json"
            request_path.write_text(json.dumps(requests))
            processes.append(subprocess.Popen([sys.executable, "-c", worker_source, str(script), str(request_path)],
                                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr
        merged_groups = []
        for process_rank in range(planned_world_size):
            rank_rows = [json.loads(line) for line in (scenario_dir / f"reward_audit.rank-{process_rank}.jsonl").read_text().split("\n") if line]
            assert len(rank_rows) == 8 // planned_world_size
            assert {row["rank"] for row in rank_rows} == {process_rank}
            merged_groups.extend(rank_rows)
        for optimizer_step in range(2):
            step_rows = [row for row in merged_groups if row["optimizer_step_before_update"] == optimizer_step]
            assert len(step_rows) == 4 and len({row["question_id"] for row in step_rows}) == 4
            assert sum(len(row["rewards"]) for row in step_rows) == 32
        test_count += 1

print(f"[TEST 100%] {test_count} training reward/audit scenarios passed; GPU training was not executed")
