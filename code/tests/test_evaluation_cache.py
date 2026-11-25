"""Exercise cached-run integrity with temporary fixtures and a forbidden GPU loader."""
import copy
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

source_root = Path(__file__).resolve().parents[1]
configuration = json.loads((source_root / "configs/v1.json").read_text())
runtime_package_lists = []
for name in ["evaluate.py", "load_model.py"]:
    tree = ast.parse((source_root / "scripts" / name).read_text())
    package_loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == "package" and isinstance(node.iter, ast.List)]
    assert len(package_loops) == 1, f"Expected one runtime package list in {name}"
    runtime_package_lists.append({node.value for node in package_loops[0].iter.elts})
assert runtime_package_lists[0] == runtime_package_lists[1], "Evaluation and loader runtime fingerprints must match"
evaluation_tree = ast.parse((source_root / "scripts/evaluate.py").read_text())
code_hash_assignments = [node for node in ast.walk(evaluation_tree) if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "code_hashes" for target in node.targets)]
assert len(code_hash_assignments) == 1
protocol_sources = ast.literal_eval(code_hash_assignments[0].value.generators[0].iter)
assert {"verify_answers.py", "load_model.py", "fingerprint_model.py", "evaluate.py", "select_checkpoint.py"}.issubset(protocol_sources)
test_count = 0
with tempfile.TemporaryDirectory(prefix="rlvr-cache-tests-") as temporary:
    work = Path(temporary)
    fixture = work / "initial" / "code"
    (fixture / "scripts").mkdir(parents=True)
    for name in ["evaluate.py", "verify_answers.py", "fingerprint_model.py", "select_checkpoint.py"]:
        shutil.copyfile(source_root / "scripts" / name, fixture / "scripts" / name)
    (fixture / "scripts/load_model.py").write_text('raise AssertionError("GPU_LOADER_MUST_NOT_RUN_IN_CACHE_TEST")\n')
    config = copy.deepcopy(configuration)
    config["data"]["output_dir"] = "data"
    config["model"]["path"] = "model"
    config["evaluation"]["n_samples"] = 2
    config["evaluation"]["dev_samples"] = 2
    (fixture / "config.json").write_text(json.dumps(config))
    (fixture / "model").mkdir()
    (fixture / "model/config.json").write_text("{}")
    (fixture / "model/tokenizer.json").write_text("{}")
    (fixture / "model/tokenizer_config.json").write_text("{}")
    tensor_header = json.dumps({"fixture.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    model_fixture_bytes = len(tensor_header).to_bytes(8, "little") + tensor_header + bytes(4)
    (fixture / "model/model.safetensors").write_bytes(model_fixture_bytes)
    identity = runpy.run_path(str(fixture / "scripts/fingerprint_model.py"), init_globals={"MODEL_PATH": fixture / "model"})
    (fixture / "data").mkdir()
    split_rows = {}
    split_hashes = {}
    for split in ["engineering", "train", "dev", "test", "structural"]:
        row = {"id": split + "-0", "split": split, "question": "What is 1 + 1?", "answer": "2", "family_id": None, "variant_type": "original"}
        if split == "structural":
            row.update(family_id="family-0", variant_type="numerical", verified=False)
        split_rows[split] = row
        data_bytes = (json.dumps(row) + "\n").encode()
        (fixture / "data" / f"{split}.jsonl").write_bytes(data_bytes)
        split_hashes[split] = hashlib.sha256(data_bytes).hexdigest()
    review = {"approved": True, "reviewer": "temporary fixture reviewer", "dataset_sha256": split_hashes["structural"],
        "records": [{"id": split_rows["structural"]["id"], "verified": True, "reviewed_answer": "2",
            "question_sha256": hashlib.sha256(split_rows["structural"]["question"].encode()).hexdigest()}]}
    review_path = fixture / "data/structural_review.json"
    review_path.write_text(json.dumps(review))
    versions = {}
    for package in sorted(runtime_package_lists[1]):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    evaluation = config["evaluation"]
    settings = {"temperature": evaluation["temperature"], "top_p": evaluation["top_p"], "top_k": evaluation["top_k"],
        "max_new_tokens": evaluation["max_new_tokens"], "max_prompt_length": evaluation["max_prompt_length"],
        "seed": evaluation["seed"], "prompt_template": config["prompt"]["template"], "dtype": "bfloat16",
        "backend": "transformers-sdpa", "batch_size": 1, "world_size": 1, "add_special_tokens": True,
        "stopping_rule": "tokenizer-eos-or-max_new_tokens", "repetition_penalty": 1.0, "package_versions": versions}
    code_hashes = {name: hashlib.sha256((fixture / "scripts" / name).read_bytes()).hexdigest()
        for name in protocol_sources}
    cases = [("engineering", "base", "runs/base", None, 2),
             ("dev", "rl", "runs/selection/dev/checkpoint-1", "runs/training/checkpoint-1", 1),
             ("dev", "rl", "runs/selection/dev/checkpoint-2", "runs/training/checkpoint-2", 2)]
    baseline_manifest = None
    baseline_rows = None
    for split, label, output, adapter, correct_count in cases:
        destination = fixture / output
        destination.mkdir(parents=True)
        adapter_identity = None
        if adapter:
            adapter_path = fixture / adapter
            adapter_path.mkdir(parents=True)
            (adapter_path / "adapter_config.json").write_text("{}")
            (adapter_path / "adapter_model.safetensors").write_bytes(adapter.encode())
            digest = hashlib.sha256()
            for filename in ["adapter_config.json", "adapter_model.safetensors"]:
                digest.update(filename.encode())
                digest.update((adapter_path / filename).read_bytes())
            adapter_identity = digest.hexdigest()
        samples = []
        for sample_index in range(2):
            correct = sample_index < correct_count
            seed = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{split_rows[split]['id']}:{sample_index}".encode()).digest()[:4], "big")
            samples.append({**split_rows[split], "sample_index": sample_index, "correct": correct,
                "reward": float(correct), "extracted": "2" if correct else "0", "status": "correct" if correct else "incorrect",
                "text": "#### 2" if correct else "#### 0", "format_compliant": True, "truncated": False,
                "output_tokens": 2, "prompt_tokens": 4, "sample_seed": seed})
        sample_bytes = "".join(json.dumps(row) + "\n" for row in samples).encode()
        (destination / "samples.jsonl").write_bytes(sample_bytes)
        protocol = {"dataset_sha256": split_hashes[split], "n_samples": 2, "settings": settings, "code_hashes": code_hashes}
        manifest = {"schema_version": 1, "path_base": "code_root", "model_label": label, "n_samples": 2,
            "split": split, "dataset_sha256": split_hashes[split], "protocol_id": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
            "model_identity": identity["model_identity"], "adapter_identity": adapter_identity, "adapter_path": adapter,
            "synthetic": False, "complete": True, "settings": settings, "code_hashes": code_hashes,
            "question_count": 1, "total_samples": 2, "samples_sha256": hashlib.sha256(sample_bytes).hexdigest()}
        (destination / "manifest.json").write_text(json.dumps(manifest))
        if label == "base":
            baseline_manifest = copy.deepcopy(manifest)
            baseline_rows = copy.deepcopy(samples)
    normalization = {"owner": "trl_grpo", "trainer_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "accelerate_backward_divisor_before": config["training"]["gradient_accumulation_steps"],
        "accelerate_backward_divisor": 1, "compatibility_fix": "transformers_5_3_0_accelerate_1_10_1_double_scaling"}
    training_manifest = {"model_identity": identity["model_identity"], "config": config, "loss_normalization": normalization}
    (fixture / "runs/training/manifest.json").write_text(json.dumps(training_manifest))
    report = {"status": "completed", "mode": "formal", "gates": {"all_passed": True, "loss_normalization": True},
        "loss_normalization": normalization,
        "model_identity": identity["model_identity"], "split_sha256": split_hashes, "optimizer_steps": 2,
        "checkpoints": [{"step": 1, "path": "runs/training/checkpoint-1"}, {"step": 2, "path": "runs/training/checkpoint-2"}],
        "adapter_path": "runs/training/adapter-final"}
    (fixture / "runs/training/report.json").write_text(json.dumps(report))
    command = [sys.executable, str(fixture / "scripts/evaluate.py"), "--config", "config.json", "--model", "base",
        "--split", "engineering", "--output", "runs/base", "--resume"]
    manifest_path = fixture / "runs/base/manifest.json"
    sample_path = fixture / "runs/base/samples.jsonl"
    manifest_before = manifest_path.read_bytes()
    samples_before = sample_path.read_bytes()
    result = subprocess.run(command, cwd=work, capture_output=True, text=True)
    assert result.returncode == 0 and "[EVAL CACHED]" in result.stdout, result.stderr
    assert manifest_path.read_bytes() == manifest_before and sample_path.read_bytes() == samples_before
    test_count += 1

    mutations = ["checksum", "duplicate", "missing", "boolean", "score", "prompt_tokens", "truncation", "seed", "incomplete_bad_score", "model_weights"]
    for mutation in mutations:
        manifest = copy.deepcopy(baseline_manifest)
        samples = copy.deepcopy(baseline_rows)
        if mutation == "checksum":
            samples[0]["text"] = "#### 999"
        elif mutation == "duplicate":
            samples.append(copy.deepcopy(samples[0]))
        elif mutation == "missing":
            samples.pop()
        elif mutation == "boolean":
            samples[0]["correct"] = 1
        elif mutation == "score":
            samples[0]["correct"] = False
        elif mutation == "prompt_tokens":
            samples[0]["prompt_tokens"] = 0
        elif mutation == "truncation":
            samples[0]["truncated"] = True
        elif mutation == "seed":
            samples[0]["sample_seed"] = 0
        elif mutation == "incomplete_bad_score":
            manifest["complete"] = False
            samples[0]["reward"] = 0.0
        elif mutation == "model_weights":
            (fixture / "model/model.safetensors").write_bytes(model_fixture_bytes[:-1] + b"1")
        sample_bytes = "".join(json.dumps(row) + "\n" for row in samples).encode()
        if mutation != "checksum":
            manifest["samples_sha256"] = hashlib.sha256(sample_bytes).hexdigest()
        sample_path.write_bytes(sample_bytes)
        manifest_path.write_text(json.dumps(manifest))
        result = subprocess.run(command, cwd=work, capture_output=True, text=True)
        assert result.returncode != 0, f"Accepted cache mutation {mutation}"
        assert "GPU_LOADER_MUST_NOT_RUN_IN_CACHE_TEST" not in result.stderr, result.stderr
        test_count += 1
        (fixture / "model/model.safetensors").write_bytes(model_fixture_bytes)
    sample_path.write_bytes(samples_before)
    manifest_path.write_bytes(manifest_before)

    interrupted_manifest = copy.deepcopy(baseline_manifest)
    interrupted_manifest["complete"] = False
    interrupted_manifest.pop("samples_sha256")
    interrupted_manifest.pop("total_samples")
    manifest_path.write_text(json.dumps(interrupted_manifest))
    result = subprocess.run(command, cwd=work, capture_output=True, text=True)
    assert result.returncode == 0 and "[EVAL CACHED]" in result.stdout, result.stderr
    recovered_manifest = json.loads(manifest_path.read_text())
    assert recovered_manifest["complete"] is True and recovered_manifest["recovered_complete_on_resume"] is True
    assert recovered_manifest["total_samples"] == 2 and sample_path.read_bytes() == samples_before
    manifest_path.write_bytes(manifest_before)
    test_count += 1

    no_resume = command[:-1]
    result = subprocess.run(no_resume, cwd=work, capture_output=True, text=True)
    assert result.returncode != 0 and "Output exists" in result.stderr and "GPU_LOADER" not in result.stderr
    test_count += 1

    selection_command = [sys.executable, str(fixture / "scripts/select_checkpoint.py"), "--config", "config.json",
        "--run", "runs/training", "--output", "runs/selection", "--resume"]
    result = subprocess.run(selection_command, cwd=work, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    selection_path = fixture / "runs/selection/selection.json"
    selection = json.loads(selection_path.read_text())
    assert selection["step"] == 2 and selection["pass_at_1"] == 1.0
    assert selection["adapter_path"] == "runs/training/checkpoint-2"
    assert selection["training_run"] == "runs/training"
    assert selection["evaluation_path"] == "runs/selection/dev/checkpoint-2"
    selection_before = selection_path.read_bytes()
    result = subprocess.run(selection_command, cwd=work, capture_output=True, text=True)
    assert result.returncode == 0 and "[SELECTION CACHED]" in result.stdout, result.stderr
    assert selection_path.read_bytes() == selection_before
    test_count += 1

    for split in ["test", "structural"]:
        destination = fixture / "runs" / f"{split}-base"
        destination.mkdir()
        samples = []
        for sample_index in range(2):
            saved = {**baseline_rows[sample_index], **split_rows[split]}
            saved["sample_seed"] = int.from_bytes(hashlib.sha256(f"{settings['seed']}:{saved['id']}:{sample_index}".encode()).digest()[:4], "big")
            samples.append(saved)
        sample_bytes = "".join(json.dumps(row) + "\n" for row in samples).encode()
        (destination / "samples.jsonl").write_bytes(sample_bytes)
        protocol = {"dataset_sha256": split_hashes[split], "n_samples": 2, "settings": settings, "code_hashes": code_hashes}
        manifest = dict(baseline_manifest, split=split, dataset_sha256=split_hashes[split],
            protocol_id=hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(), selection=selection,
            samples_sha256=hashlib.sha256(sample_bytes).hexdigest(),
            structural_review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest() if split == "structural" else None)
        (destination / "manifest.json").write_text(json.dumps(manifest))
        formal_command = [sys.executable, str(fixture / "scripts/evaluate.py"), "--config", "config.json", "--model", "base",
            "--split", split, "--selection", "runs/selection/selection.json", "--output", f"runs/{split}-base", "--resume"]
        result = subprocess.run(formal_command, cwd=work, capture_output=True, text=True)
        assert result.returncode == 0 and "[EVAL CACHED]" in result.stdout, result.stderr
        test_count += 1
        if split == "test":
            for filename in ["report.json", "manifest.json"]:
                dependency_path = fixture / "runs/training" / filename
                original = dependency_path.read_bytes()
                dependency_path.write_bytes(original + b"\n")
                result = subprocess.run(formal_command, cwd=work, capture_output=True, text=True)
                assert result.returncode != 0 and f"Training dependency {filename} changed" in result.stderr, result.stderr
                assert "GPU_LOADER_MUST_NOT_RUN_IN_CACHE_TEST" not in result.stderr
                dependency_path.write_bytes(original)
                test_count += 1
            selector_path = fixture / "scripts/select_checkpoint.py"
            original = selector_path.read_bytes()
            selector_path.write_bytes(original + b"\n")
            result = subprocess.run(formal_command, cwd=work, capture_output=True, text=True)
            assert result.returncode != 0 and "protocol changed after development selection" in result.stderr, result.stderr
            assert "GPU_LOADER_MUST_NOT_RUN_IN_CACHE_TEST" not in result.stderr
            selector_path.write_bytes(original)
            test_count += 1
            changed_selection = dict(selection, pass_at_1=0.75)
            selection_path.write_text(json.dumps(changed_selection))
            result = subprocess.run(formal_command, cwd=work, capture_output=True, text=True)
            assert result.returncode != 0 and "Cannot resume: selection changed" in result.stderr, result.stderr
            selection_path.write_bytes(selection_before)
            test_count += 1
        else:
            original = review_path.read_bytes()
            review_path.write_text(json.dumps(dict(review, reviewer="different temporary fixture reviewer")))
            result = subprocess.run(formal_command, cwd=work, capture_output=True, text=True)
            assert result.returncode != 0 and "Cannot resume: structural_review_sha256 changed" in result.stderr, result.stderr
            assert "GPU_LOADER_MUST_NOT_RUN_IN_CACHE_TEST" not in result.stderr
            review_path.write_bytes(original)
            test_count += 1

    moved = work / "moved" / "code"
    moved.parent.mkdir()
    shutil.move(str(fixture), str(moved))
    selection_command[1] = str(moved / "scripts/select_checkpoint.py")
    result = subprocess.run(selection_command, cwd=work, capture_output=True, text=True)
    assert result.returncode == 0 and "[SELECTION CACHED]" in result.stdout, result.stderr
    assert (moved / "runs/selection/selection.json").read_bytes() == selection_before
    test_count += 1
    formal_command[1] = str(moved / "scripts/evaluate.py")
    result = subprocess.run(formal_command, cwd=work, capture_output=True, text=True)
    assert result.returncode == 0 and "[EVAL CACHED]" in result.stdout, result.stderr
    test_count += 1

    selection["pass_at_1"] = 0.25
    (moved / "runs/selection/selection.json").write_text(json.dumps(selection))
    changed_before = (moved / "runs/selection/selection.json").read_bytes()
    result = subprocess.run(selection_command, cwd=work, capture_output=True, text=True)
    assert result.returncode != 0 and "refusing to rewrite" in result.stderr, result.stderr
    assert (moved / "runs/selection/selection.json").read_bytes() == changed_before
    test_count += 1

    for world_size in (1, 2, 4):
        global_batch = config["training"]["per_device_train_batch_size"] * config["training"]["gradient_accumulation_steps"]
        generations = config["training"]["num_generations"]
        topology = {"world_size": world_size, "single_node": True, "local_micro_batch": 1,
            "gradient_accumulation_steps_per_rank": config["training"]["gradient_accumulation_steps"] // world_size,
            "global_effective_batch_size": global_batch, "global_unique_prompts_per_update": global_batch // generations,
            "per_rank_generation_batch_size": generations, "global_generation_batch_size": generations * world_size,
            "per_rank_unique_prompts_per_update": global_batch // generations // world_size, "steps_per_generation": generations}
        rank_normalization = dict(normalization, trainer_accumulation_steps=topology["gradient_accumulation_steps_per_rank"],
            accelerate_backward_divisor_before=topology["gradient_accumulation_steps_per_rank"])
        topology_manifest = dict(training_manifest, topology=topology, loss_normalization=rank_normalization)
        topology_report = copy.deepcopy(report)
        topology_report["topology"] = topology
        topology_report["loss_normalization"] = rank_normalization
        topology_report["gates"] = {"all_passed": True, "rank_consistency": True, "loss_normalization": True}
        topology_report["rank_audits"] = []
        for rank in range(world_size):
            group_count = report["optimizer_steps"] * topology["per_rank_unique_prompts_per_update"]
            rank_bytes = "".join(json.dumps({"rank": rank, "question_id": f"q-{rank}-{group}",
                "optimizer_step_before_update": group // topology["per_rank_unique_prompts_per_update"]}) + "\n" for group in range(group_count)).encode()
            (moved / "runs/training" / f"reward_audit.rank-{rank}.jsonl").write_bytes(rank_bytes)
            topology_report["rank_audits"].append({"rank": rank, "local_rank": rank,
                "optimizer_steps": report["optimizer_steps"], "training_completed": True, "finite_metrics": True,
                "nonfinite_metrics": [], "effective_update_steps": report["optimizer_steps"],
                "base_sha256_before": "0" * 64, "base_sha256_after": "0" * 64, "base_unchanged": True,
                "adapter_sha256_after": "2" * 64, "adapter_updated": True, "all_adapters_finite": True,
                "loss_normalization_verified": True,
                "reward_groups": group_count, "reward_audit_sha256": hashlib.sha256(rank_bytes).hexdigest()})
        (moved / "runs/training/manifest.json").write_text(json.dumps(topology_manifest))
        (moved / "runs/training/report.json").write_text(json.dumps(topology_report))
        destination = moved / "runs" / f"selection-world-{world_size}"
        shutil.copytree(moved / "runs/selection/dev", destination / "dev")
        for cached_manifest_path in (destination / "dev").glob("*/manifest.json"):
            cached_manifest = json.loads(cached_manifest_path.read_text())
            cached_manifest["training_topology"] = topology
            cached_manifest_path.write_text(json.dumps(cached_manifest))
        topology_command = list(selection_command)
        topology_command[topology_command.index("--output") + 1] = f"runs/selection-world-{world_size}"
        result = subprocess.run(topology_command, cwd=work, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        saved_selection = json.loads((destination / "selection.json").read_text())
        assert saved_selection["training_world_size"] == world_size
        assert saved_selection["training_topology"] == topology
        assert saved_selection["settings"]["world_size"] == 1
        test_count += 1

        if world_size == 2:
            mutation_messages = {"missing_rank": "complete, unique audit", "duplicate_rank": "complete, unique audit",
                "failed_rank": "failed its recorded integrity", "inconsistent_adapter": "inconsistent base or adapter",
                "reward_hash": "reward evidence changed", "topology_mismatch": "different GPU topologies",
                "wrong_batch": "GRPO batch/group constraints", "aggregate_gate": "integrity gates all passed",
                "normalization_gate_missing": "integrity gates all passed", "normalization_record_missing": "loss-normalization provenance",
                "normalization_record_mismatch": "loss-normalization provenance", "normalization_divisor": "loss-normalization provenance",
                "normalization_boolean": "loss-normalization provenance", "normalization_window": "loss-normalization provenance",
                "normalization_rank_missing": "failed its recorded integrity", "normalization_rank_false": "failed its recorded integrity"}
            for mutation, message in mutation_messages.items():
                invalid_report = copy.deepcopy(topology_report)
                invalid_manifest = copy.deepcopy(topology_manifest)
                if mutation == "missing_rank":
                    invalid_report["rank_audits"].pop()
                elif mutation == "duplicate_rank":
                    invalid_report["rank_audits"][1]["rank"] = 0
                elif mutation == "failed_rank":
                    invalid_report["rank_audits"][1]["adapter_updated"] = False
                elif mutation == "inconsistent_adapter":
                    invalid_report["rank_audits"][1]["adapter_sha256_after"] = "3" * 64
                elif mutation == "reward_hash":
                    invalid_report["rank_audits"][1]["reward_audit_sha256"] = "f" * 64
                elif mutation == "topology_mismatch":
                    invalid_manifest["topology"]["world_size"] = 4
                elif mutation == "wrong_batch":
                    invalid_report["topology"]["global_effective_batch_size"] += 1
                    invalid_manifest["topology"] = invalid_report["topology"]
                elif mutation == "aggregate_gate":
                    invalid_report["gates"]["base_unchanged"] = False
                elif mutation == "normalization_gate_missing":
                    invalid_report["gates"].pop("loss_normalization")
                elif mutation == "normalization_record_missing":
                    invalid_report.pop("loss_normalization")
                    invalid_manifest.pop("loss_normalization")
                elif mutation == "normalization_record_mismatch":
                    invalid_manifest["loss_normalization"]["owner"] = "changed"
                elif mutation in {"normalization_divisor", "normalization_boolean", "normalization_window"}:
                    field = "trainer_accumulation_steps" if mutation == "normalization_window" else "accelerate_backward_divisor"
                    invalid_report["loss_normalization"][field] = True if mutation == "normalization_boolean" else 2
                    invalid_manifest["loss_normalization"] = invalid_report["loss_normalization"]
                elif mutation == "normalization_rank_missing":
                    invalid_report["rank_audits"][1].pop("loss_normalization_verified")
                elif mutation == "normalization_rank_false":
                    invalid_report["rank_audits"][1]["loss_normalization_verified"] = False
                (moved / "runs/training/report.json").write_text(json.dumps(invalid_report))
                (moved / "runs/training/manifest.json").write_text(json.dumps(invalid_manifest))
                result = subprocess.run(topology_command, cwd=work, capture_output=True, text=True)
                assert result.returncode != 0 and message in result.stderr, result.stderr
                assert "GPU_LOADER_MUST_NOT_RUN_IN_CACHE_TEST" not in result.stderr
                test_count += 1

print(f"[TEST 100%] {test_count} cache integrity, GPU-skip and relocation scenarios passed; temporary fixtures only")
