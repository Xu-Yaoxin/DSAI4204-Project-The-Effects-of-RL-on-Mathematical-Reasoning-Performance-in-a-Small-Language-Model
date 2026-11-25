"""Train a fresh, local BF16 LoRA adapter; preserve audit evidence for every run."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import runpy
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parent.parent

# TRL requires a callable. The sole lambda below bridges to this linear reward branch.
if "REWARD_REQUEST" in globals():
    request = REWARD_REQUEST
    count = len(request["completions"])
    group_size = request["group_size"]
    if type(group_size) is not int or group_size < 2:
        raise ValueError("GRPO group_size must be an integer of at least two.")
    if count == 0 or count % group_size:
        raise ValueError("Reward batch is empty or does not contain complete GRPO groups.")
    if request.get("expected_batch_size", count) != count:
        raise ValueError("Reward batch size differs from the configured generation batch.")
    if not all(len(request[key]) == count for key in ("prompts", "answer", "question_id", "completion_ids")):
        raise ValueError("Reward inputs have mismatched lengths.")
    if not all(isinstance(text, str) for text in request["completions"]):
        raise TypeError("v1 expects plain text completions, not conversational messages.")
    group_question_ids = []
    for start in range(0, count, group_size):
        stop = start + group_size
        if len(set(request["question_id"][start:stop])) != 1:
            raise ValueError("TRL returned an invalid group: question IDs differ within the group.")
        if len(set(request["prompts"][start:stop])) != 1 or len(set(request["answer"][start:stop])) != 1:
            raise ValueError("TRL returned an invalid group: prompts or answers differ.")
        group_question_ids.append(request["question_id"][start])
    if len(set(group_question_ids)) != len(group_question_ids):
        raise ValueError("Different GRPO groups in one generation batch must have distinct question IDs.")
    scoring_inputs = []
    for index in range(count):
        token_ids = request["completion_ids"][index]
        ended_with_eos = bool(token_ids) and token_ids[-1] in request["eos_token_ids"]
        scoring_inputs.append({
            "text": request["completions"][index], "answer": request["answer"][index],
            "truncated": len(token_ids) >= request["max_completion_length"] and not ended_with_eos,
        })
    scoring = runpy.run_path(
        str(CODE_ROOT / "scripts" / "verify_answers.py"),
        init_globals={"SCORE_INPUT": scoring_inputs},
    )["SCORE_OUTPUT"]
    if len(scoring) != count or any(row["reward"] not in (0, 1, 0.0, 1.0) for row in scoring):
        raise ValueError("The verifier must return one binary correctness reward per completion.")
    if any(row["status"] in ("invalid_gold", "infrastructure_error") for row in scoring):
        raise ValueError("Invalid gold answers and infrastructure failures cannot be training rewards.")
    REWARD_VALUES = [float(row["reward"]) for row in scoring]
    with Path(request["audit_path"]).open("a", encoding="utf-8") as handle:
        for start in range(0, count, group_size):
            stop = start + group_size
            values = REWARD_VALUES[start:stop]
            samples = []
            for index in range(start, stop):
                token_ids = request["completion_ids"][index]
                samples.append({
                    "text": request["completions"][index],
                    "completion_tokens": len(token_ids),
                    "truncated": scoring_inputs[index]["truncated"],
                    "verifier": scoring[index],
                })
            handle.write(json.dumps({
                "rank": request.get("rank", 0),
                "optimizer_step_before_update": request["step"],
                "question_id": request["question_id"][start],
                "rewards": values,
                "mixed_rewards": min(values) < max(values),
                "samples": samples,
            }, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
else:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v1.json")
    parser.add_argument("--mode", choices=("engineering", "pilot", "formal"), required=True)
    parser.add_argument("--output", required=True, help="New run directory; existing directories are rejected.")
    parser.add_argument("--pilot-report", help="Successful, matching pilot report required for formal training.")
    parser.add_argument("--plan", action="store_true", help="Validate data and batching without importing GPU packages or writing files.")
    parser.add_argument("--world-size", type=int, choices=(1, 2, 4), help="Topology to inspect with --plan; real runs use torchrun WORLD_SIZE")
    args = parser.parse_args()
    if args.world_size is not None and not args.plan:
        parser.error("--world-size is for --plan only; launch real multi-GPU training with torchrun.")
    world_size = args.world_size if args.world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size not in (1, 2, 4):
        raise ValueError("Supported single-node GPU counts are 1, 2, and 4.")
    if not args.plan and (rank != local_rank or not 0 <= rank < world_size
            or int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size))) != world_size):
        raise ValueError("Expected one single-node torchrun process per selected GPU.")

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = CODE_ROOT / config_path
    config_path = config_path.resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    settings = config["training"]
    if config["model"]["dtype"] != "bfloat16":
        raise ValueError("v1 requires the same BF16 base precision for training and evaluation.")
    for key in ("num_generations", "per_device_train_batch_size", "gradient_accumulation_steps", "num_iterations",
                "epochs", "max_completion_length", "max_prompt_length", "lora_rank", "save_steps"):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f"training.{key} must be a positive integer.")
    if settings["num_generations"] < 2:
        raise ValueError("GRPO requires at least two generations per prompt.")
    if settings["num_iterations"] != 1:
        raise ValueError("v1 audits one optimizer pass per generated group; num_iterations must equal 1.")
    effective_batch = settings["per_device_train_batch_size"] * settings["gradient_accumulation_steps"]
    if effective_batch % settings["num_generations"]:
        raise ValueError("Global effective batch size must be divisible by num_generations.")
    if settings["gradient_accumulation_steps"] % world_size:
        raise ValueError("The configured accumulation budget must be divisible by world_size.")
    accumulation_per_rank = settings["gradient_accumulation_steps"] // world_size
    unique_prompts_per_update = effective_batch // settings["num_generations"]
    default_generation_steps = settings["num_generations"] // math.gcd(
        settings["num_generations"], settings["per_device_train_batch_size"])
    steps_per_generation = settings.get("steps_per_generation", default_generation_steps)
    if type(steps_per_generation) is not int or steps_per_generation <= 0:
        raise ValueError("steps_per_generation must be a positive integer.")
    generation_batch_size = settings["per_device_train_batch_size"] * steps_per_generation
    if generation_batch_size % settings["num_generations"]:
        raise ValueError("Generation batch must contain complete num_generations groups.")
    if accumulation_per_rank % steps_per_generation:
        raise ValueError("steps_per_generation must divide gradient_accumulation_steps so rollouts never cross policy updates.")
    if world_size > 1 and (settings["per_device_train_batch_size"] != 1 or settings["num_generations"] != 8
            or effective_batch != 32 or steps_per_generation != 8):
        raise ValueError("DDP requires microbatch=1, num_generations=8, global responses/update=32, steps_per_generation=8.")
    topology = {
        "world_size": world_size, "single_node": True,
        "local_micro_batch": settings["per_device_train_batch_size"],
        "gradient_accumulation_steps_per_rank": accumulation_per_rank,
        "global_effective_batch_size": effective_batch,
        "global_unique_prompts_per_update": unique_prompts_per_update,
        "per_rank_generation_batch_size": generation_batch_size,
        "global_generation_batch_size": generation_batch_size * world_size,
        "per_rank_unique_prompts_per_update": unique_prompts_per_update // world_size,
        "steps_per_generation": steps_per_generation,
    }
    for key in ("learning_rate", "temperature", "max_grad_norm", "lora_alpha"):
        if not isinstance(settings[key], (int, float)) or not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"training.{key} must be finite and positive.")
    if not 0 < settings["top_p"] <= 1 or not 0 <= settings["lora_dropout"] < 1:
        raise ValueError("top_p or lora_dropout is outside its permitted range.")
    if not math.isfinite(settings["beta"]) or settings["beta"] < 0:
        raise ValueError("beta must be finite and nonnegative.")
    if settings["loss_type"] != "grpo" or settings["scale_rewards"] != "group":
        raise ValueError("v1 explicitly implements GRPO with group-scaled binary rewards.")
    if settings["logging_steps"] != 1:
        raise ValueError("v1 requires logging_steps=1 to audit each optimizer update.")
    data_readiness = runpy.run_path(str(CODE_ROOT / "scripts" / "doctor.py"),
                                  init_globals={"DOCTOR_CONFIG": config})["report"]
    if not data_readiness["data_ready"]:
        failed_checks = [key for key, value in data_readiness["checks"].items() if not value]
        raise ValueError(f"Prepared data failed provenance/integrity checks: {failed_checks}. Run doctor.py --data-only.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    data_dir = Path(config["data"]["output_dir"]).expanduser()
    if not data_dir.is_absolute():
        data_dir = CODE_ROOT / data_dir
    data_dir = data_dir.resolve()
    split_name = {"engineering": "engineering", "pilot": "pilot", "formal": "train"}[args.mode]
    data_path = data_dir / (split_name + ".jsonl")
    data_bytes = data_path.read_bytes()
    records = [json.loads(line) for line in data_bytes.decode("utf-8").split("\n") if line.strip()]
    if not records or any(not {"id", "question", "answer"} <= row.keys() for row in records):
        raise ValueError("Training data must contain nonempty id/question/answer records.")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate question IDs in the training split.")
    size_key = {"engineering": "engineering_size", "pilot": "pilot_size", "formal": "train_size"}[args.mode]
    required_size = config["data"][size_key]
    if not 0 < config["data"]["engineering_size"] <= config["data"]["pilot_size"] <= config["data"]["train_size"]:
        raise ValueError("Data sizes must satisfy 0 < engineering <= pilot <= formal training.")
    if len(records) != required_size:
        raise ValueError(f"v1 {args.mode} requires {required_size} questions; found {len(records)}.")
    if len(records) % unique_prompts_per_update:
        raise ValueError("Training split size must be divisible by unique_prompts_per_update for complete audited data passes.")
    expected_optimizer_steps = len(records) // unique_prompts_per_update * settings["epochs"]
    if args.mode == "engineering":
        expected_optimizer_steps = min(2, expected_optimizer_steps)
    if any(not isinstance(row["question"], str) or not isinstance(row["answer"], str) for row in records):
        raise ValueError("Questions and canonical numeric answers must be strings.")
    prompt_template = config["prompt"]["template"]
    if prompt_template.count("{question}") != 1:
        raise ValueError("Prompt template must contain exactly one {question}.")
    training_protocol = {
        "model": config["model"], "prompt": config["prompt"], "training": settings,
        "verifier_sha256": hashlib.sha256((CODE_ROOT / "scripts" / "verify_answers.py").read_bytes()).hexdigest(),
        "training_script_sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256((CODE_ROOT / "scripts" / "load_model.py").read_bytes()).hexdigest(),
        "fingerprint_sha256": hashlib.sha256((CODE_ROOT / "scripts" / "fingerprint_model.py").read_bytes()).hexdigest(),
        "doctor_sha256": hashlib.sha256((CODE_ROOT / "scripts" / "doctor.py").read_bytes()).hexdigest(),
        "resolved_steps_per_generation": steps_per_generation,
        "topology": topology,
    }
    config_training_hash = hashlib.sha256(
        json.dumps(training_protocol, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    split_hashes = {}
    split_records = {}
    for split in ("engineering", "pilot", "train", "dev", "test"):
        split_file = data_dir / (split + ".jsonl")
        if not split_file.is_file():
            raise FileNotFoundError(f"Required prepared split missing: {split_file}")
        split_content = split_file.read_bytes()
        split_hashes[split] = hashlib.sha256(split_content).hexdigest()
        split_records[split] = [json.loads(line) for line in split_content.decode("utf-8").split("\n") if line.strip()]
    split_ids = {split: {row["id"] for row in values} for split, values in split_records.items()}
    if not split_ids["engineering"] <= split_ids["pilot"] <= split_ids["train"]:
        raise ValueError("Engineering and pilot IDs must be nested subsets of formal training IDs.")
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        left_questions = {" ".join(row["question"].casefold().split()) for row in split_records[left]}
        right_questions = {" ".join(row["question"].casefold().split()) for row in split_records[right]}
        if split_ids[left] & split_ids[right] or left_questions & right_questions:
            raise ValueError(f"Data leakage detected between {left} and {right}.")
    pilot_report = None
    if args.mode == "formal" and not args.plan:
        if not args.pilot_report:
            raise ValueError("Formal training requires --pilot-report from a completed pilot.")
        pilot_path = Path(args.pilot_report).expanduser()
        if not pilot_path.is_absolute():
            pilot_path = CODE_ROOT / pilot_path
        pilot_report = json.loads(pilot_path.read_text(encoding="utf-8"))
        required_gates = (
            "training_completed", "finite_metrics", "adapter_updated", "base_unchanged",
            "mixed_reward_groups", "effective_update_steps", "all_adapters_finite", "checkpoints_saved",
            "complete_prompt_groups", "prompt_coverage", "rank_consistency", "loss_normalization", "all_passed",
        )
        if pilot_report.get("mode") != "pilot" or pilot_report.get("status") != "completed":
            raise ValueError("The supplied report is not a completed pilot.")
        if not all(pilot_report.get("gates", {}).get(gate) is True for gate in required_gates):
            raise ValueError("Pilot acceptance gates did not all pass. Inspect its report before continuing.")
        if pilot_report.get("config_training_hash") != config_training_hash:
            raise ValueError("Pilot configuration or source code differs. Run a new pilot with this protocol.")
        if pilot_report.get("topology") != topology:
            raise ValueError("Formal training requires the same GPU topology as its accepted pilot.")
        if pilot_report.get("split_sha256") != split_hashes:
            raise ValueError("Prepared data differ from the successful pilot. Run a new pilot.")
        if pilot_report.get("prepared_manifest_sha256") != data_readiness["data"]["prepared_manifest_sha256"]:
            raise ValueError("Prepared data provenance differs from the successful pilot. Run a new pilot.")

    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = CODE_ROOT / output_dir
    output_dir = output_dir.resolve()
    saved_paths = {}
    for label, path in {"output_dir": output_dir, "config_path": config_path, "training_file": data_path,
                        "adapter_path": output_dir / "adapter"}.items():
        saved_paths[label] = str(path.relative_to(CODE_ROOT)) if path.is_relative_to(CODE_ROOT) else str(path)
    saved_pilot_path = None
    if pilot_report:
        pilot_path = pilot_path.resolve()
        saved_pilot_path = str(pilot_path.relative_to(CODE_ROOT)) if pilot_path.is_relative_to(CODE_ROOT) else str(pilot_path)
    batch_plan = {
        "effective_batch_size": effective_batch, "unique_prompts_per_update": unique_prompts_per_update,
        "generation_batch_size": generation_batch_size, "steps_per_generation": steps_per_generation,
        "unique_prompts_per_generation": generation_batch_size // settings["num_generations"],
        "generation_batches_per_update": accumulation_per_rank // steps_per_generation,
        "expected_optimizer_steps": expected_optimizer_steps,
        "expected_reward_groups": expected_optimizer_steps * unique_prompts_per_update,
        "topology": topology,
    }
    if args.plan:
        print(json.dumps({"mode": args.mode, "config_version": config.get("version"), "training_questions": len(records),
                          **batch_plan, **saved_paths, "path_base": "code_root", "gpu_checked": False,
                          "formal_requires_matching_pilot": args.mode == "formal"}, indent=2))
        raise SystemExit(0)
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import set_seed
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise RuntimeError("Install the documented GPU requirements before training; no fallback is used.") from error
    required_versions = {"trl": "1.16.0", "transformers": "5.3.0", "peft": "0.18.1", "accelerate": "1.10.1"}
    for package, expected in required_versions.items():
        actual = importlib.metadata.version(package)
        if actual.split("+")[0] != expected:
            raise RuntimeError(f"v1 expects {package}=={expected}; found {actual}.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        raise RuntimeError("Expose exactly the selected GPU count through CUDA_VISIBLE_DEVICES.")
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Real CUDA with BF16 support is required. CPU/MPS and simulated training are rejected.")
    distributed = world_size > 1
    if distributed:
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=300))
    creation_status = [None]
    if rank == 0:
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            creation_status[0] = f"{type(error).__name__}: {error}"
    if distributed:
        torch.distributed.broadcast_object_list(creation_status, src=0)
    if creation_status[0] is not None:
        raise FileExistsError(f"Cannot create a fresh run directory: {creation_status[0]}")
    started = time.perf_counter()
    report = {
        "version": config.get("version", "1.1.0"), "mode": args.mode, "status": "running", "path_base": "code_root",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "config_training_hash": config_training_hash,
        "data_sha256": split_hashes[split_name], "split_sha256": split_hashes,
        "training_questions": len(records), "gates": {"all_passed": False},
        "data_provenance": data_readiness["data"]["source_provenance"],
        "prepared_manifest_sha256": data_readiness["data"]["prepared_manifest_sha256"],
        "output_dir": saved_paths["output_dir"], "pilot_report": saved_pilot_path, **batch_plan,
    }
    if rank == 0:
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        print(f"\n{'=' * 72}\n[01/04] LOAD | rank={rank}/{world_size} | mode={args.mode} | questions={len(records)}", flush=True)
        set_seed(settings["seed"])
        torch.cuda.reset_peak_memory_stats()
        model_path = Path(config["model"]["path"]).expanduser()
        if not model_path.is_absolute():
            model_path = CODE_ROOT / model_path
        loaded = runpy.run_path(str(CODE_ROOT / "scripts" / "load_model.py"), init_globals={
            "MODEL_PATH": model_path.resolve(), "ADAPTER_PATH": None, "TRAINING": True,
        })
        model = loaded["model"]
        tokenizer = loaded["tokenizer"]
        model_identity = loaded["model_identity"]
        report["model_identity"] = model_identity
        report["package_versions"] = loaded["package_versions"]
        if pilot_report and pilot_report.get("model_identity") != model_identity:
            raise ValueError("Base checkpoint identity differs from the accepted pilot.")
        if pilot_report and pilot_report.get("package_versions") != loaded["package_versions"]:
            raise ValueError("Package versions differ from the accepted pilot. Run a new pilot.")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        allowed_leaves = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in allowed_leaves:
                if any(component in name.lower() for component in ("visual", "vision", "image", "projector")):
                    continue
                if ".layers." not in name:
                    continue
                target_modules.append(name)
        found_leaves = {name.rsplit(".", 1)[-1] for name in target_modules}
        if found_leaves != allowed_leaves:
            raise RuntimeError(f"Expected language attention and MLP projections; found {sorted(found_leaves)}.")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM", r=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
            lora_dropout=settings["lora_dropout"], bias="none", target_modules=target_modules,
        )
        model.config.use_cache = False
        dataset_rows = []
        prompt_lengths = []
        for row in records:
            prompt = prompt_template.format(question=row["question"])
            prompt_length = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            if prompt_length > settings["max_prompt_length"]:
                raise ValueError(f"Prompt {row['id']} has {prompt_length} tokens; v1 refuses silent truncation.")
            prompt_lengths.append(prompt_length)
            dataset_rows.append({"question_id": row["id"], "prompt": prompt, "answer": row["answer"]})
        train_dataset = Dataset.from_list(dataset_rows)
        eos_token_ids = model.generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = tokenizer.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        if not eos_token_ids:
            raise ValueError("No EOS token IDs available for generation and truncation auditing.")
        reward_audit_path = output_dir / f"reward_audit.rank-{rank}.jsonl"
        reward_callback = lambda prompts, completions, answer, question_id, completion_ids, trainer_state, **kwargs: runpy.run_path(
            str(SCRIPT_PATH), init_globals={"REWARD_REQUEST": {
                "prompts": prompts, "completions": completions, "answer": answer, "question_id": question_id,
                "completion_ids": completion_ids, "step": trainer_state.global_step,
                "group_size": settings["num_generations"], "audit_path": str(reward_audit_path),
                "expected_batch_size": generation_batch_size,
                "rank": rank,
                "max_completion_length": settings["max_completion_length"], "eos_token_ids": eos_token_ids,
            }},
        )["REWARD_VALUES"]
        reward_callback.__name__ = "numerical_correctness"
        trainer_kwargs = {
            "output_dir": str(output_dir), "seed": settings["seed"], "data_seed": settings["seed"],
            "num_train_epochs": settings["epochs"], "max_steps": expected_optimizer_steps if args.mode == "engineering" else -1,
            "learning_rate": settings["learning_rate"], "lr_scheduler_type": "constant",
            "warmup_steps": 0, "weight_decay": 0.0, "optim": "adamw_torch",
            "per_device_train_batch_size": settings["per_device_train_batch_size"],
            "gradient_accumulation_steps": accumulation_per_rank,
            "steps_per_generation": steps_per_generation,
            "num_generations": settings["num_generations"], "temperature": settings["temperature"],
            "top_p": settings["top_p"], "top_k": 0, "repetition_penalty": 1.0,
            "max_completion_length": settings["max_completion_length"],
            "generation_kwargs": {"use_cache": True, "do_sample": True},
            "loss_type": settings["loss_type"], "beta": settings["beta"],
            "epsilon": settings["epsilon"], "scale_rewards": settings["scale_rewards"],
            "num_iterations": settings["num_iterations"], "max_grad_norm": settings["max_grad_norm"],
            "bf16": True, "fp16": False, "tf32": False,
            "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "remove_unused_columns": False, "use_vllm": False, "use_liger_kernel": False,
            "mask_truncated_completions": False, "report_to": [], "push_to_hub": False,
            "save_strategy": "steps" if args.mode == "engineering" else settings.get("save_strategy", "epoch"),
            "save_steps": 1 if args.mode == "engineering" else settings["save_steps"],
            "save_only_model": False, "save_total_limit": None,
            "logging_strategy": "steps", "logging_steps": settings["logging_steps"],
            "logging_nan_inf_filter": False, "eval_strategy": "no", "load_best_model_at_end": False,
            "dataloader_num_workers": 0, "disable_tqdm": False,
            "ddp_find_unused_parameters": False, "ddp_timeout": 300,
            "accelerator_config": {"split_batches": False, "dispatch_batches": False},
        }
        unknown_fields = set(trainer_kwargs) - set(GRPOConfig.__dataclass_fields__)
        if unknown_fields:
            raise RuntimeError(f"Installed TRL does not support the pinned configuration: {sorted(unknown_fields)}")
        training_args = GRPOConfig(**trainer_kwargs)
        manifest = {
            **report, "config": config, "config_path": saved_paths["config_path"],
            "training_file": saved_paths["training_file"], "training_sha256": report["data_sha256"],
            "train_ids": [row["id"] for row in records], "model_identity": model_identity,
            "package_versions": loaded["package_versions"], "training_protocol": training_protocol,
            "trainer_arguments": training_args.to_dict(), "max_observed_prompt_tokens": max(prompt_lengths),
            "gpu_name": torch.cuda.get_device_name(local_rank), "cuda_version": torch.version.cuda,
            "fresh_adapter_from_base": True, "checkpoint_selection": "external development-set numerical pass@1",
        }
        manifest["trainer_arguments"]["output_dir"] = saved_paths["output_dir"]
        if rank == 0:
            (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        # Let TRL install the fresh adapter so its reference policy disables LoRA.
        trainer = GRPOTrainer(
            model=model, args=training_args, train_dataset=train_dataset,
            reward_funcs=reward_callback, processing_class=tokenizer, peft_config=lora_config,
        )
        # TRL already divides GRPO loss by Trainer's accumulation window. Transformers
        # 5.3 also gives Accelerate 1.10.1 that divisor, causing a second division in
        # Accelerator.backward. Trainer owns microbatch counting and sync boundaries;
        # disable only Accelerate's extra backward divisor for this pinned stack.
        accelerator_accumulation_before = trainer.accelerator.gradient_accumulation_steps
        if (accelerator_accumulation_before != accumulation_per_rank
                or trainer.args.gradient_accumulation_steps != accumulation_per_rank
                or trainer.loss_is_scaled_for_ga is not True
                or trainer.model_accepts_loss_kwargs is not False
                or trainer.compute_loss_func is None
                or trainer.is_deepspeed_enabled or trainer.is_fsdp_enabled):
            raise RuntimeError("Unexpected pinned Trainer/Accelerate loss normalization; refusing an unverified workaround.")
        trainer.accelerator.gradient_accumulation_steps = 1
        loss_normalization = {
            "owner": "trl_grpo", "trainer_accumulation_steps": accumulation_per_rank,
            "accelerate_backward_divisor_before": accelerator_accumulation_before,
            "accelerate_backward_divisor": trainer.accelerator.gradient_accumulation_steps,
            "compatibility_fix": "transformers_5_3_0_accelerate_1_10_1_double_scaling",
        }
        if (trainer.args.gradient_accumulation_steps != accumulation_per_rank
                or trainer.accelerator.gradient_accumulation_steps != 1):
            raise RuntimeError("Loss normalization adjustment changed the Trainer accumulation window.")
        report["loss_normalization"] = loss_normalization
        manifest["loss_normalization"] = loss_normalization
        if rank == 0:
            (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        model = trainer.model
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable_names or any(".lora_" not in name for name in trainable_names):
            raise RuntimeError("Only LoRA adapter parameters may be trainable.")
        initial_adapters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        initial_adapter_hash = hashlib.sha256()
        for name, value in initial_adapters.items():
            initial_adapter_hash.update(name.encode("utf-8"))
            initial_adapter_hash.update(memoryview(value.contiguous().reshape(-1).view(torch.uint8).numpy()))
        base_hash_before = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                base_hash_before.update(name.encode("utf-8"))
                base_hash_before.update(memoryview(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()))
        module_audit = {
            "target_modules": target_modules, "trainable_parameters": trainable_names,
            "trainable_numel": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "total_numel": sum(parameter.numel() for parameter in model.parameters()),
            "base_sha256_before": base_hash_before.hexdigest(),
        }
        initial_rank_states = [None] * world_size
        initial_rank_state = {"base_sha256": base_hash_before.hexdigest(), "model_identity": model_identity,
                              "package_versions": loaded["package_versions"], "trainable_parameters": trainable_names,
                              "adapter_sha256": initial_adapter_hash.hexdigest(), "loss_normalization": loss_normalization}
        if distributed:
            torch.distributed.all_gather_object(initial_rank_states, initial_rank_state)
        else:
            initial_rank_states[0] = initial_rank_state
        if any(item != initial_rank_states[0] for item in initial_rank_states):
            raise RuntimeError("Ranks loaded different base weights, adapters, or software.")
        if rank == 0:
            (output_dir / "module_audit.json").write_text(json.dumps(module_audit, indent=2) + "\n", encoding="utf-8")
        print(f"[02/04] TRAIN | 25% | trainable={module_audit['trainable_numel']:,} | targets={len(target_modules)} | "
              f"responses/update={effective_batch} | questions/update={unique_prompts_per_update} | generation_batch={generation_batch_size}", flush=True)
        training_started = time.perf_counter()
        train_result = trainer.train()
        torch.cuda.synchronize()
        report["training_seconds"] = time.perf_counter() - training_started
        print("[03/04] AUDIT | 75% | checking parameter updates, rewards and checkpoints", flush=True)
        trainer.save_model(str(output_dir / "adapter"))
        if rank == 0:
            tokenizer.save_pretrained(str(output_dir / "adapter"))
            trainer.save_state()
        changed_names = []
        adapter_max_abs_delta = 0.0
        all_adapters_finite = True
        base_hash_after = hashlib.sha256()
        adapter_hash_after = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                current = parameter.detach().cpu()
                adapter_hash_after.update(name.encode("utf-8"))
                adapter_hash_after.update(memoryview(current.contiguous().reshape(-1).view(torch.uint8).numpy()))
                all_adapters_finite = all_adapters_finite and bool(torch.isfinite(current).all())
                if not torch.equal(current, initial_adapters[name]):
                    changed_names.append(name)
                delta = (current.float() - initial_adapters[name].float()).abs().max().item()
                if math.isfinite(delta):
                    adapter_max_abs_delta = max(adapter_max_abs_delta, delta)
                else:
                    all_adapters_finite = False
            else:
                base_hash_after.update(name.encode("utf-8"))
                base_hash_after.update(memoryview(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()))
        module_audit.update({
            "changed_adapter_parameters": changed_names, "adapter_max_abs_delta": adapter_max_abs_delta,
            "base_sha256_after": base_hash_after.hexdigest(), "all_adapters_finite": all_adapters_finite,
        })
        finite_metrics = True
        numeric_metrics = 0
        nonfinite_metrics = []
        for index, log in enumerate(trainer.state.log_history):
            for key, value in log.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_metrics += 1
                    if not math.isfinite(value):
                        finite_metrics = False
                        nonfinite_metrics.append({"log_index": index, "key": key, "value": str(value)})
        loss_logs = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        effective_update_steps = sum(
            isinstance(log.get("grad_norm"), (int, float))
            and math.isfinite(log["grad_norm"]) and log["grad_norm"] > 0
            for log in trainer.state.log_history
        )
        finite_metrics = finite_metrics and numeric_metrics > 0 and bool(loss_logs)
        local_reward_bytes = reward_audit_path.read_bytes()
        local_reward_groups = [json.loads(line) for line in local_reward_bytes.decode("utf-8").split("\n") if line.strip()]
        local_rank_audit = {
            "rank": rank, "local_rank": local_rank, "gpu_name": torch.cuda.get_device_name(local_rank),
            "optimizer_steps": trainer.state.global_step,
            "training_completed": trainer.state.global_step == trainer.state.max_steps == expected_optimizer_steps,
            "finite_metrics": finite_metrics, "nonfinite_metrics": nonfinite_metrics,
            "effective_update_steps": effective_update_steps,
            "loss_normalization_verified": (trainer.args.gradient_accumulation_steps == accumulation_per_rank
                and trainer.accelerator.gradient_accumulation_steps == 1),
            "base_sha256_before": base_hash_before.hexdigest(), "base_sha256_after": base_hash_after.hexdigest(),
            "base_unchanged": base_hash_before.hexdigest() == base_hash_after.hexdigest(),
            "adapter_sha256_after": adapter_hash_after.hexdigest(),
            "adapter_updated": bool(changed_names) and adapter_max_abs_delta > 0,
            "all_adapters_finite": all_adapters_finite, "adapter_max_abs_delta": adapter_max_abs_delta,
            "reward_groups": len(local_reward_groups), "reward_audit_sha256": hashlib.sha256(local_reward_bytes).hexdigest(),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "training_seconds": report["training_seconds"],
        }
        rank_audits = [None] * world_size
        if distributed:
            torch.distributed.all_gather_object(rank_audits, local_rank_audit)
        else:
            rank_audits[0] = local_rank_audit
        rank_consistency = (
            {item["rank"] for item in rank_audits} == set(range(world_size))
            and len({item["base_sha256_after"] for item in rank_audits}) == 1
            and len({item["adapter_sha256_after"] for item in rank_audits}) == 1
            and len({item["optimizer_steps"] for item in rank_audits}) == 1
            and all(item["reward_groups"] == expected_optimizer_steps * unique_prompts_per_update // world_size
                    for item in rank_audits)
        )
        completion_packet = [None]
        if rank == 0:
            module_audit["ranks"] = rank_audits
            (output_dir / "module_audit.json").write_text(json.dumps(module_audit, indent=2) + "\n", encoding="utf-8")
            (output_dir / "trainer_logs.json").write_text(json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8")
            ordered_reward_groups = []
            for other_rank in range(world_size):
                rank_file = output_dir / f"reward_audit.rank-{other_rank}.jsonl"
                rank_bytes = rank_file.read_bytes()
                if hashlib.sha256(rank_bytes).hexdigest() != rank_audits[other_rank]["reward_audit_sha256"]:
                    raise RuntimeError("A rank reward audit changed after its integrity check.")
                for ordinal, line in enumerate(rank_bytes.decode("utf-8").split("\n")):
                    if line.strip():
                        group = json.loads(line)
                        if group.get("rank") != other_rank:
                            raise RuntimeError("Rank-local reward audit contains records from another rank.")
                        ordered_reward_groups.append((group["optimizer_step_before_update"], other_rank, ordinal, group))
            reward_groups = [item[3] for item in sorted(ordered_reward_groups)]
            with (output_dir / "reward_audit.jsonl").open("x", encoding="utf-8") as merged:
                for group in reward_groups:
                    merged.write(json.dumps(group, ensure_ascii=False, allow_nan=False) + "\n")
            mixed_groups = sum(group["mixed_rewards"] for group in reward_groups)
            observed_question_ids = {group["question_id"] for group in reward_groups}
            prompt_ids_by_step = {}
            for group in reward_groups:
                prompt_ids_by_step.setdefault(group["optimizer_step_before_update"], []).append(group["question_id"])
            complete_prompt_groups = (
                set(prompt_ids_by_step) == set(range(trainer.state.global_step))
                and all(len(ids) == unique_prompts_per_update and len(set(ids)) == unique_prompts_per_update
                        for ids in prompt_ids_by_step.values())
                and len(reward_groups) == expected_optimizer_steps * unique_prompts_per_update
            )
            samples = [sample for group in reward_groups for sample in group["samples"]]
            checkpoints = []
            incomplete_checkpoints = []
            required_checkpoint_files = ["adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
                                         "optimizer.pt", "scheduler.pt"]
            required_checkpoint_files += ([f"rng_state_{other_rank}.pth" for other_rank in range(world_size)]
                                          if distributed else ["rng_state.pth"])
            for path in sorted(output_dir.glob("checkpoint-*")):
                saved_checkpoint_path = str(path.relative_to(CODE_ROOT)) if path.is_relative_to(CODE_ROOT) else str(path)
                missing_files = [filename for filename in required_checkpoint_files if not (path / filename).is_file()]
                if not missing_files:
                    checkpoints.append({"path": saved_checkpoint_path, "step": int(path.name.split("-")[-1])})
                else:
                    incomplete_checkpoints.append({"path": saved_checkpoint_path, "missing_files": missing_files})
            gates = {
                "training_completed": all(item["training_completed"] for item in rank_audits),
                "finite_metrics": all(item["finite_metrics"] for item in rank_audits),
                "adapter_updated": all(item["adapter_updated"] for item in rank_audits),
                "base_unchanged": all(item["base_unchanged"] for item in rank_audits),
                "mixed_reward_groups": mixed_groups > 0,
                "effective_update_steps": all(item["effective_update_steps"] > 0 for item in rank_audits),
                "all_adapters_finite": all(item["all_adapters_finite"] for item in rank_audits),
                "checkpoints_saved": bool(checkpoints) and not incomplete_checkpoints,
                "complete_prompt_groups": complete_prompt_groups, "rank_consistency": rank_consistency,
                "loss_normalization": all(item["loss_normalization_verified"] for item in rank_audits),
                "prompt_coverage": observed_question_ids == {row["id"] for row in records} if args.mode != "engineering"
                                   else len(observed_question_ids) == min(len(records), expected_optimizer_steps * unique_prompts_per_update),
            }
            gates["all_passed"] = all(gates.values())
            report.update({
                "status": "completed", "gates": gates, "optimizer_steps": trainer.state.global_step,
                "epochs_completed": trainer.state.epoch, "adapter_path": saved_paths["adapter_path"],
                "observed_unique_prompt_count": len(observed_question_ids),
                "observed_prompts_per_update": {str(step): len(ids) for step, ids in prompt_ids_by_step.items()},
                "effective_update_steps": effective_update_steps, "rank_audits": rank_audits,
                "checkpoints": checkpoints, "metrics": train_result.metrics, "incomplete_checkpoints": incomplete_checkpoints,
                "reward_groups": len(reward_groups), "mixed_reward_groups": mixed_groups,
                "mixed_reward_fraction": mixed_groups / len(reward_groups) if reward_groups else 0.0,
                "completion_count": len(samples), "truncation_count": sum(sample["truncated"] for sample in samples),
                "correct_completion_count": sum(bool(sample["verifier"]["correct"]) for sample in samples),
                "format_compliant_count": sum(bool(sample["verifier"].get("format_compliant", False)) for sample in samples),
                "changed_adapter_parameters": len(changed_names), "adapter_max_abs_delta": adapter_max_abs_delta,
                "nonfinite_metrics": [dict(item, rank=entry["rank"]) for entry in rank_audits for item in entry["nonfinite_metrics"]],
                "peak_cuda_allocated_bytes": max(item["peak_cuda_allocated_bytes"] for item in rank_audits),
                "peak_cuda_reserved_bytes": max(item["peak_cuda_reserved_bytes"] for item in rank_audits),
                "training_seconds": max(item["training_seconds"] for item in rank_audits),
                "memory_reporting": "maximum per-device peak; DDP does not pool GPU memory",
                "gate_scope": "Engineering integrity only; no mathematical improvement or resource feasibility claim.",
            })
            completion_packet[0] = {"status": report["status"], "gates": report["gates"]}
        if distributed:
            torch.distributed.broadcast_object_list(completion_packet, src=0)
        if rank != 0:
            report.update(completion_packet[0])
    except BaseException as error:
        report.update({
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "gates": {"all_passed": False},
        })
        (output_dir / f"failure.rank-{rank}.json").write_text(json.dumps({
            "rank": rank, "local_rank": local_rank, "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc()}, indent=2) + "\n", encoding="utf-8")
        # Never enter collectives on an exception path; torchrun terminates failed peers.
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if rank == 0:
            (output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
            print(f"[04/04] SAVE | 100% | status={report['status']} | {output_dir / 'report.json'}", flush=True)
    if distributed:
        torch.distributed.destroy_process_group()
    if not report["gates"]["all_passed"]:
        sys.exit("Training ended, but acceptance gates failed. Inspect report.json before any formal run.")
