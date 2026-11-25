"""Test baseline-only calibration plans, cached integrity and mocked generation."""

import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


source_root = Path(__file__).resolve().parents[1]
source = source_root / "scripts/evaluate_boundary.py"
tree = ast.parse(source.read_text())
assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree))
checks = 1
with tempfile.TemporaryDirectory(prefix="rlvr-boundary-evaluation-") as temporary:
    root = Path(temporary) / "code"
    for directory in ("scripts", "configs", "data/boundary_v1", "model"):
        (root / directory).mkdir(parents=True)
    for name in ("evaluate_boundary.py", "analyze_boundary.py", "verify_answers.py", "load_model.py", "fingerprint_model.py"):
        shutil.copyfile(source_root / "scripts" / name, root / "scripts" / name)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (root / "model" / name).write_text("{}")
    header = json.dumps({"fixture.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    model_bytes = len(header).to_bytes(8, "little") + header + bytes(4)
    (root / "model/model.safetensors").write_bytes(model_bytes)
    identity = runpy.run_path(str(root / "scripts/fingerprint_model.py"), init_globals={"MODEL_PATH": root / "model"})
    cfg = {"version": "boundary-1.0", "model": {"path": "model", "dtype": "bfloat16", "repo_id": "Qwen/Qwen3.5-2B-Base", "expected_identity": identity["model_identity"]},
           "prompt": {"template": "Solve: {question}\nFinish with #### and the numerical answer.\n"},
           "data": {"path": "data/boundary_v1/questions.jsonl", "manifest": "data/boundary_v1/manifest.json"},
           "evaluation": {"seed": 271828, "n_samples": 2, "temperature": 0.6, "top_p": 0.95, "top_k": 0, "max_new_tokens": 262144, "context_limit": 262144, "length_policy": "native_context_minus_actual_prompt", "max_prompt_length": 24, "bootstrap_replicates": 2000}}
    config_path = root / "configs/confirmation_v2.json"
    config_path.write_text(json.dumps(cfg))
    rows = [{"id": f"q-{i}", "question": f"Compute the result of fixture question {i}.\u2028Use the quantities stated.", "answer": str(731 + i),
             "source_answer": str(731 + i), "group_id": f"g{i}", "group_label": f"Group {i}", "family_id": f"family-{i}",
             "difficulty_label": f"tier-{i}", "source": {"dataset": "fixture", "revision": "pinned", "split": "test", "row_index": i, "original_id": str(i)}} for i in range(2)]
    data_path = root / cfg["data"]["path"]
    data_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    data_path.write_bytes(data_bytes)
    data_manifest = {"schema_version": 1, "purpose": "exploratory_baseline_calibration",
                     "files": {"questions.jsonl": {"sha256": hashlib.sha256(data_bytes).hexdigest(), "rows": 2}},
                     "groups": [{"id": f"g{i}", "label": f"Group {i}", "count": 1} for i in range(2)], "sources": []}
    data_manifest_path = root / cfg["data"]["manifest"]
    data_manifest_path.write_text(json.dumps(data_manifest))
    script = root / "scripts/evaluate_boundary.py"
    # Match the parent interpreter's site-packages mode: software metadata is part
    # of the resume protocol even though no model packages are imported here.
    interpreter_command = [sys.executable] + (["-S"] if sys.flags.no_site else [])
    command = interpreter_command + [str(script)]
    result = subprocess.run(command + ["--plan"], cwd=Path(temporary), text=True, capture_output=True)
    assert result.returncode == 0 and "responses=4" in result.stdout and "max_output_tokens=1048576" in result.stdout, result.stderr
    assert "no model/GPU imported" in result.stdout and not (root / "outputs").exists()
    assert identity["model_identity"] in result.stdout
    checks += 1
    result = subprocess.run(command + ["--plan", "--samples", "1"], text=True, capture_output=True)
    assert result.returncode == 0 and "responses=2" in result.stdout, result.stderr
    checks += 1
    for flags in (["--cache-only"], ["--cache-only", "--resume", "--plan"]):
        result = subprocess.run(command + flags, text=True, capture_output=True)
        assert result.returncode == 2 and "--cache-only requires --resume" in result.stderr
        checks += 1
    with patch.dict("os.environ", {"RLVR_CPU_THREADS": "invalid-if-model-loading-started"}):
        result = subprocess.run(command + ["--resume", "--cache-only"], text=True, capture_output=True)
    assert result.returncode == 3 and "CACHE INCOMPLETE" in result.stdout, result.stderr
    assert not (root / "outputs").exists()
    checks += 1

    for mutation in ("data_hash", "duplicate_id", "duplicate_text", "group_count", "bad_gold", "gold_in_prompt", "adapter", "samples", "distributed", "bad_expected_identity"):
        current_cfg = copy.deepcopy(cfg)
        current_rows = copy.deepcopy(rows)
        current_manifest = copy.deepcopy(data_manifest)
        if mutation == "data_hash":
            current_rows[0]["question"] += " changed"
        elif mutation == "duplicate_id":
            current_rows[1]["id"] = current_rows[0]["id"]
        elif mutation == "duplicate_text":
            current_rows[1]["question"] = current_rows[0]["question"]
        elif mutation == "group_count":
            current_manifest["groups"][0]["count"] = 2
        elif mutation == "bad_gold":
            current_rows[0]["answer"] = "1+2"
        elif mutation == "gold_in_prompt":
            current_cfg["prompt"]["template"] = "Question: {question}, answer: {answer}"
        elif mutation == "adapter":
            current_cfg["model"]["adapter_path"] = "outputs/formal/checkpoint-384"
        elif mutation == "samples":
            current_cfg["evaluation"]["n_samples"] = True
        elif mutation == "bad_expected_identity":
            current_cfg["model"]["expected_identity"] = "not-a-digest"
        mutated_data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in current_rows).encode()
        if mutation != "data_hash":
            current_manifest["files"]["questions.jsonl"]["sha256"] = hashlib.sha256(mutated_data).hexdigest()
        config_path.write_text(json.dumps(current_cfg))
        data_manifest_path.write_text(json.dumps(current_manifest))
        data_path.write_bytes(mutated_data)
        with patch.dict("os.environ", {"WORLD_SIZE": "2"} if mutation == "distributed" else {}):
            result = subprocess.run(command + ["--plan"], text=True, capture_output=True)
        assert result.returncode != 0, mutation
        checks += 1
    config_path.write_text(json.dumps(cfg))
    data_manifest_path.write_text(json.dumps(data_manifest))
    data_path.write_bytes(data_bytes)

    package_versions = {}
    for package in ("torch", "transformers", "peft", "accelerate", "datasets", "trl", "safetensors", "flash-linear-attention", "fla-core", "causal-conv1d"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    tensor = MagicMock()
    tensor.shape = (1, 5)
    tensor.to.return_value = tensor
    tokenizer = MagicMock(return_value={"input_ids": tensor, "attention_mask": tensor})
    tokenizer.eos_token_id = 99
    tokenizer.pad_token_id = 99
    tokenizer.decode.side_effect = ["#### 731", "#### 0", "#### 732", "Answer: #### 732"]
    generated = MagicMock()
    generated.__getitem__.return_value.tolist.return_value = [10, 11, 99]
    model = SimpleNamespace(generate=MagicMock(return_value=generated), config=SimpleNamespace(max_position_embeddings=262144))
    torch_stub = SimpleNamespace(inference_mode=contextlib.nullcontext,
        set_num_threads=MagicMock(), set_num_interop_threads=MagicMock(),
        cuda=SimpleNamespace(get_device_name=lambda: "fixture GPU (not real)", reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0))
    set_seed = MagicMock()
    transformers_stub = SimpleNamespace(set_seed=set_seed)
    loaded = {"model": model, "tokenizer": tokenizer, "model_identity": identity["model_identity"], "model_files": identity["model_files"],
              "adapter_identity": None, "package_versions": package_versions}
    original_run_path = runpy.run_path
    loader_mock = MagicMock(side_effect=lambda path, **kwargs: loaded if Path(path).name == "load_model.py" else original_run_path(path, **kwargs))
    mismatched_cfg = copy.deepcopy(cfg)
    mismatched_cfg["model"]["expected_identity"] = "0" * 64
    config_path.write_text(json.dumps(mismatched_cfg))
    with patch("runpy.run_path", loader_mock), patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}), patch.object(sys, "argv", [str(script)]), contextlib.redirect_stdout(io.StringIO()):
        try:
            original_run_path(str(script), run_name="__main__")
        except ValueError as error:
            assert "Base identity differs" in str(error)
        else:
            raise AssertionError("Fresh baseline identity mismatch was accepted")
    assert model.generate.call_count == 0 and not (root / "outputs").exists()
    config_path.write_text(json.dumps(cfg))
    checks += 1
    with patch("runpy.run_path", loader_mock), patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}), patch.object(sys, "argv", [str(script)]), contextlib.redirect_stdout(io.StringIO()):
        original_run_path(str(script), run_name="__main__")
    assert model.generate.call_count == 4
    assert all(call.kwargs["max_new_tokens"] == 262139 for call in model.generate.call_args_list)
    checks += 1
    loader_call = next(call for call in loader_mock.call_args_list if Path(call.args[0]).name == "load_model.py")
    assert loader_call.kwargs["init_globals"]["ADAPTER_PATH"] is None and loader_call.kwargs["init_globals"]["TRAINING"] is False
    for call in tokenizer.call_args_list:
        assert "731" not in call.args[0] and "732" not in call.args[0]
        assert call.kwargs == {"return_tensors": "pt", "add_special_tokens": True}
    assert len(tokenizer.call_args_list) == 4
    destination = root / "outputs/confirmation_v2/base"
    manifest_path = destination / "manifest.json"
    samples_path = destination / "samples.jsonl"
    original_manifest = json.loads(manifest_path.read_text())
    original_sample_bytes = samples_path.read_bytes()
    generated_rows = [json.loads(line) for line in original_sample_bytes.decode().split("\n") if line.strip()]
    assert [row["correct"] for row in generated_rows] == [True, False, True, True]
    assert original_manifest["model_label"] == "base" and original_manifest["adapter_identity"] is None and original_manifest["complete"] is True
    assert [call.args[0] for call in set_seed.call_args_list] == [row["sample_seed"] for row in generated_rows]
    checks += 4

    analysis_command = interpreter_command + [str(root / "scripts/analyze_boundary.py"), "--bootstrap", "100"]
    result = subprocess.run(analysis_command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    produced_analysis = json.loads((root / "outputs/confirmation_v2/analysis/summary.json").read_text())
    assert produced_analysis["model_label"] == "base" and produced_analysis["adapter_identity"] is None
    assert produced_analysis["protocol_id"] == original_manifest["protocol_id"]
    assert produced_analysis["question_count"] == 2 and produced_analysis["total_samples"] == 4
    assert {group["group_id"]: group["pass_at_1"]["estimate"] for group in produced_analysis["groups"]} == {"g0": 0.5, "g1": 1.0}
    assert (root / "outputs/confirmation_v2/analysis/ready.json").is_file()
    checks += 1

    cached_command = command + ["--resume"]
    result = subprocess.run(cached_command, text=True, capture_output=True)
    assert result.returncode == 0 and "BOUNDARY CACHED" in result.stdout, result.stderr
    assert samples_path.read_bytes() == original_sample_bytes
    checks += 1
    manifest_before_probe = manifest_path.read_bytes()
    with patch.dict("os.environ", {"RLVR_CPU_THREADS": "invalid-if-model-loading-started"}):
        result = subprocess.run(cached_command + ["--cache-only"], text=True, capture_output=True)
    assert result.returncode == 0 and "BOUNDARY CACHED" in result.stdout, result.stderr
    assert samples_path.read_bytes() == original_sample_bytes and manifest_path.read_bytes() == manifest_before_probe
    checks += 1
    for mutation in ("score", "group", "seed", "duplicate", "missing", "reward_bool", "partial_tail", "complete_hash", "model_weights", "software_identity", "source_code", "expected_identity", "generation_limit", "context_overflow", "fake_truncation"):
        saved_manifest = copy.deepcopy(original_manifest)
        saved_rows = copy.deepcopy(generated_rows)
        if mutation == "score":
            saved_rows[0]["correct"] = False
        elif mutation == "group":
            saved_rows[0]["group_id"] = "g1"
        elif mutation == "seed":
            saved_rows[0]["sample_seed"] += 1
        elif mutation == "duplicate":
            saved_rows.append(copy.deepcopy(saved_rows[0]))
        elif mutation == "missing":
            saved_rows.pop()
        elif mutation == "reward_bool":
            saved_rows[0]["reward"] = True
        elif mutation == "model_weights":
            (root / "model/model.safetensors").write_bytes(model_bytes[:-1] + b"1")
        elif mutation == "generation_limit":
            saved_rows[0]["generation_limit"] += 1
        elif mutation == "context_overflow":
            saved_rows[0]["output_tokens"] = 262144
        elif mutation == "fake_truncation":
            saved_rows[0]["truncated"] = True
        elif mutation == "software_identity":
            saved_manifest["settings"]["package_versions"]["torch"] = "changed"
        elif mutation == "source_code":
            (root / "scripts/verify_answers.py").write_text((source_root / "scripts/verify_answers.py").read_text() + "\n# changed\n")
        elif mutation == "expected_identity":
            config_path.write_text(json.dumps(mismatched_cfg))
            saved_manifest["config"] = copy.deepcopy(mismatched_cfg)
        current_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in saved_rows).encode()
        if mutation == "partial_tail":
            saved_manifest["complete"] = False
            current_bytes += b'{"id":'
        if mutation != "complete_hash":
            saved_manifest["samples_sha256"] = hashlib.sha256(current_bytes).hexdigest()
        else:
            saved_manifest["samples_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(saved_manifest))
        samples_path.write_bytes(current_bytes)
        result = subprocess.run(cached_command, text=True, capture_output=True)
        assert result.returncode != 0, mutation
        assert "Expected the original unquantized" not in result.stderr, (mutation, result.stderr)
        if mutation == "expected_identity":
            assert "Base identity differs" in result.stderr, result.stderr
        cache_result = subprocess.run(cached_command + ["--cache-only"], text=True, capture_output=True)
        assert cache_result.returncode not in (0, 3), (mutation, cache_result.stdout, cache_result.stderr)
        checks += 1
        config_path.write_text(json.dumps(cfg))
        (root / "model/model.safetensors").write_bytes(model_bytes)
        shutil.copyfile(source_root / "scripts/verify_answers.py", root / "scripts/verify_answers.py")
    manifest_path.write_text(json.dumps(original_manifest))
    samples_path.write_bytes(original_sample_bytes)
    incomplete = copy.deepcopy(original_manifest)
    incomplete.update(complete=False)
    incomplete.pop("samples_sha256")
    manifest_path.write_text(json.dumps(incomplete))
    result = subprocess.run(cached_command, text=True, capture_output=True)
    assert result.returncode == 0 and "BOUNDARY CACHED" in result.stdout, result.stderr
    assert json.loads(manifest_path.read_text())["recovered_complete_on_resume"] is True
    checks += 1

    # All prompt lengths are checked before any new answer is generated or output created.
    tokenizer.reset_mock()
    tokenizer.side_effect = [{"input_ids": tensor, "attention_mask": tensor}, {"input_ids": SimpleNamespace(shape=(1, 100))}]
    model.generate.reset_mock()
    with patch("runpy.run_path", loader_mock), patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}), patch.object(sys, "argv", [str(script), "--output", "outputs/long"]), contextlib.redirect_stdout(io.StringIO()):
        try:
            original_run_path(str(script), run_name="__main__")
        except ValueError as error:
            assert "Prompt too long" in str(error)
        else:
            raise AssertionError("Long second question was accepted")
    assert model.generate.call_count == 0 and not (root / "outputs/long").exists()
    checks += 1

    # A failure retains completed JSONL lines; a restart generates only missing keys.
    tokenizer.side_effect = None
    tokenizer.decode.side_effect = ["#### 731"]
    model.generate.side_effect = [generated, RuntimeError("fixture interruption")]
    with patch("runpy.run_path", loader_mock), patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}), patch.object(sys, "argv", [str(script), "--output", "outputs/interrupted"]), contextlib.redirect_stdout(io.StringIO()):
        try:
            original_run_path(str(script), run_name="__main__")
        except RuntimeError as error:
            assert "fixture interruption" in str(error)
        else:
            raise AssertionError("Expected injected interruption")
    interrupted = root / "outputs/interrupted"
    prefix = (interrupted / "samples.jsonl").read_bytes()
    assert len([line for line in prefix.split(b"\n") if line]) == 1
    assert json.loads((interrupted / "manifest.json").read_text())["complete"] is False
    assert (interrupted / "infrastructure_errors.jsonl").is_file()
    interrupted_before_probe = {path.name: path.read_bytes() for path in interrupted.iterdir() if path.is_file()}
    with patch.dict("os.environ", {"RLVR_CPU_THREADS": "invalid-if-model-loading-started"}):
        result = subprocess.run(command + ["--output", "outputs/interrupted", "--resume", "--cache-only"], text=True, capture_output=True)
    assert result.returncode == 3 and "verified=1/4" in result.stdout, result.stderr
    assert {path.name: path.read_bytes() for path in interrupted.iterdir() if path.is_file()} == interrupted_before_probe
    checks += 1
    model.generate.reset_mock()
    model.generate.side_effect = None
    tokenizer.decode.side_effect = ["#### 731", "#### 732", "#### 732"]
    with patch("runpy.run_path", loader_mock), patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}), patch.object(sys, "argv", [str(script), "--output", "outputs/interrupted", "--resume"]), contextlib.redirect_stdout(io.StringIO()):
        original_run_path(str(script), run_name="__main__")
    assert model.generate.call_count == 3
    assert (interrupted / "samples.jsonl").read_bytes().startswith(prefix)
    assert json.loads((interrupted / "manifest.json").read_text())["complete"] is True
    checks += 2

    # Sharding partitions whole questions and leaves their original sample seeds
    # and protocol intact. Each independent process loads only the base weights.
    shard_rows = []
    shard_manifests = []
    for shard_index in (0, 1):
        model.generate.reset_mock()
        tokenizer.decode.side_effect = ["#### 731", "#### 0"] if shard_index == 0 else ["#### 732", "Answer: #### 732"]
        shard_flags = ["--num-shards", "2", "--shard-index", str(shard_index), "--output", f"outputs/shards/{shard_index}"]
        result = subprocess.run(command + shard_flags + ["--plan"], text=True, capture_output=True)
        assert result.returncode == 0 and "assigned_questions=1/2" in result.stdout and "responses=2" in result.stdout, result.stderr
        with patch("runpy.run_path", loader_mock), patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}), patch.object(sys, "argv", [str(script), *shard_flags]), contextlib.redirect_stdout(io.StringIO()):
            original_run_path(str(script), run_name="__main__")
        shard_destination = root / f"outputs/shards/{shard_index}"
        shard_manifest = json.loads((shard_destination / "manifest.json").read_text())
        current_shard_rows = [json.loads(line) for line in (shard_destination / "samples.jsonl").read_text().split("\n") if line.strip()]
        assert model.generate.call_count == 2
        assert {row["id"] for row in current_shard_rows} == {rows[shard_index]["id"]}
        assert shard_manifest["question_count"] == 2 and shard_manifest["total_samples"] == 2
        assert shard_manifest["groups"] == data_manifest["groups"]
        assert shard_manifest["sharding"] == {"count": 2, "index": shard_index, "assignment": "question_index_modulo"}
        assert shard_manifest["protocol_id"] == original_manifest["protocol_id"]
        shard_rows.extend(current_shard_rows)
        shard_manifests.append(shard_manifest)
        result = subprocess.run(command + shard_flags + ["--resume"], text=True, capture_output=True)
        assert result.returncode == 0 and "BOUNDARY CACHED" in result.stdout, result.stderr
        checks += 3
    assert {(row["id"], row["sample_index"]) for row in shard_rows} == {(row["id"], row["sample_index"]) for row in generated_rows}
    for row in shard_rows:
        original = next(item for item in generated_rows if (item["id"], item["sample_index"]) == (row["id"], row["sample_index"]))
        assert {key: value for key, value in row.items() if key != "seconds"} == {key: value for key, value in original.items() if key != "seconds"}
    assert shard_manifests[0]["model_identity"] == shard_manifests[1]["model_identity"]
    checks += 1
    for shard_flags in (["--num-shards", "2", "--shard-index", "1"], []):
        result = subprocess.run(command + ["--output", "outputs/shards/0", "--resume", *shard_flags], text=True, capture_output=True)
        assert result.returncode != 0 and "sharding changed" in result.stderr, result.stderr
        checks += 1
    for flags in (["--num-shards", "3"], ["--num-shards", "2", "--shard-index", "2"], ["--shard-index", "-1"]):
        result = subprocess.run(command + flags + ["--plan"], text=True, capture_output=True)
        assert result.returncode != 0
        checks += 1
    incorrect_partial = copy.deepcopy(shard_manifests[0])
    incorrect_partial["complete"] = False
    (root / "outputs/shards/0/manifest.json").write_text(json.dumps(incorrect_partial))
    (root / "outputs/shards/0/samples.jsonl").write_text(json.dumps(generated_rows[2], ensure_ascii=False) + "\n")
    result = subprocess.run(command + ["--output", "outputs/shards/0", "--num-shards", "2", "--shard-index", "0", "--resume"], text=True, capture_output=True)
    assert result.returncode != 0 and "Cached sample ID/index is invalid" in result.stderr, result.stderr
    checks += 1
    one_row_data = (json.dumps(rows[0], ensure_ascii=False) + "\n").encode()
    one_row_manifest = copy.deepcopy(data_manifest)
    one_row_manifest["groups"] = one_row_manifest["groups"][:1]
    one_row_manifest["files"]["questions.jsonl"].update(rows=1, sha256=hashlib.sha256(one_row_data).hexdigest())
    data_path.write_bytes(one_row_data)
    data_manifest_path.write_text(json.dumps(one_row_manifest))
    result = subprocess.run(command + ["--num-shards", "2", "--shard-index", "1", "--plan"], text=True, capture_output=True)
    assert result.returncode != 0 and "requested shard has no questions" in result.stderr, result.stderr
    data_path.write_bytes(data_bytes)
    data_manifest_path.write_text(json.dumps(data_manifest))
    checks += 1

    # Thread limits must be applied before model loading, independently of the
    # sampling protocol. The two worker processes receive the same explicit cap.
    trace = MagicMock()
    torch_stub.set_num_threads.reset_mock()
    torch_stub.set_num_interop_threads.reset_mock()
    thread_load = MagicMock(return_value=loaded)
    trace.attach_mock(torch_stub.set_num_threads, "intraop")
    trace.attach_mock(torch_stub.set_num_interop_threads, "interop")
    trace.attach_mock(thread_load, "load")
    tokenizer.decode.side_effect = ["#### 731", "#### 0"]
    thread_loader_mock = MagicMock(side_effect=lambda path, **kwargs: thread_load() if Path(path).name == "load_model.py" else original_run_path(path, **kwargs))
    with patch.dict("os.environ", {"RLVR_CPU_THREADS": "2"}), patch("runpy.run_path", thread_loader_mock), patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}), patch.object(sys, "argv", [str(script), "--num-shards", "2", "--output", "outputs/thread-limited"]), contextlib.redirect_stdout(io.StringIO()):
        original_run_path(str(script), run_name="__main__")
    assert [call[0] for call in trace.mock_calls[:3]] == ["intraop", "interop", "load"]
    torch_stub.set_num_threads.assert_called_once_with(2)
    torch_stub.set_num_interop_threads.assert_called_once_with(1)
    limited_manifest = json.loads((root / "outputs/thread-limited/manifest.json").read_text())
    assert limited_manifest["execution_cpu_threads"] == {"intraop": 2, "interop": 1}
    assert limited_manifest["protocol_id"] == original_manifest["protocol_id"]
    checks += 1
    for invalid_threads in ("0", "-1", "two"):
        with patch.dict("os.environ", {"RLVR_CPU_THREADS": invalid_threads}):
            result = subprocess.run(command + ["--output", "outputs/invalid-threads"], text=True, capture_output=True)
        assert result.returncode != 0 and "RLVR_CPU_THREADS must be a positive integer" in result.stderr, result.stderr
        checks += 1

    # A native-limit response is recognized as truncated and survives CPU resume.
    tokenizer.decode.side_effect = None
    tokenizer.decode.return_value = "#### 731"
    generated.__getitem__.return_value.tolist.return_value = [10] * 262139
    model.generate.reset_mock()
    with patch("runpy.run_path", loader_mock), patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}), patch.object(sys, "argv", [str(script), "--num-shards", "2", "--samples", "1", "--output", "outputs/native-limit"]), contextlib.redirect_stdout(io.StringIO()):
        original_run_path(str(script), run_name="__main__")
    native_row = json.loads((root / "outputs/native-limit/samples.jsonl").read_text())
    assert native_row["truncated"] is True
    assert native_row["output_tokens"] + native_row["prompt_tokens"] == 262144
    assert native_row["generation_limit"] == 262139
    result = subprocess.run(command + ["--num-shards", "2", "--samples", "1", "--output", "outputs/native-limit", "--resume", "--cache-only"], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    checks += 2
    # Never silently enlarge or shrink the pinned model context.
    model.config.max_position_embeddings = 32768
    with patch("runpy.run_path", loader_mock), patch.dict(sys.modules, {"torch": torch_stub, "transformers": transformers_stub}), patch.object(sys, "argv", [str(script), "--output", "outputs/wrong-context"]), contextlib.redirect_stdout(io.StringIO()):
        try:
            original_run_path(str(script), run_name="__main__")
        except ValueError as error:
            assert "Loaded model context differs" in str(error)
        else:
            raise AssertionError("Changed model native context accepted")
    assert not (root / "outputs/wrong-context").exists()
    checks += 1

print(f"✅ [BOUNDARY EVALUATION TEST 100%] {checks} plan, baseline-only, prompt, cache and interruption checks; model/GPU generation mocked")
