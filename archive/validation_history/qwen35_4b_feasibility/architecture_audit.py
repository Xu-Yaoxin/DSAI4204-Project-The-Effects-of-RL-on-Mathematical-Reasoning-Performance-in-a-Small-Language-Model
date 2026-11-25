"""Inspect pinned Qwen3.5 text architectures and LoRA on the meta device only."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--metadata-dir", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
args = parser.parse_args()
root = Path(__file__).resolve().parents[3]
versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate")}
assert versions == {"torch": "2.10.0", "transformers": "5.3.0", "peft": "0.18.1", "accelerate": "1.10.1"}, versions
training = json.loads((root / "configs/v1.json").read_text())["training"]
allowed_leaves = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
results = []
for label in ("2B", "4B"):
    path = args.metadata_dir / f"{label}_config.json"
    raw = path.read_bytes()
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config.text_config, dtype=torch.bfloat16, attn_implementation="sdpa")
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    tied = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    assert tied == config.text_config.tie_word_embeddings
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    base_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    headers = {}
    header_sources = []
    for header_path in sorted(args.metadata_dir.glob(f"{label}_*.header.json")):
        header_raw = header_path.read_bytes()
        shard = json.loads(header_raw)
        shard.pop("__metadata__", None)
        assert not set(headers).intersection(shard), header_path
        headers.update(shard)
        header_sources.append({"file": header_path.name, "sha256": hashlib.sha256(header_raw).hexdigest()})
    assert headers
    text_headers = {name.replace("model.language_model", "model", 1): value for name, value in headers.items()
                    if name.startswith("model.language_model.") or name.startswith("lm_head.")}
    parameter_shapes = {name: list(parameter.shape) for name, parameter in model.named_parameters()}
    assert set(text_headers) == set(parameter_shapes), {"missing": sorted(set(parameter_shapes) - set(text_headers)),
                                                       "unexpected": sorted(set(text_headers) - set(parameter_shapes))}
    assert all(value["shape"] == parameter_shapes[name] for name, value in text_headers.items())
    excluded_names = sorted(set(headers) - {name for name in headers if name.startswith(("model.language_model.", "lm_head."))})
    assert all(name.startswith(("model.visual.", "mtp.", "model.mtp.")) for name in excluded_names)
    assert all(value["dtype"] in ("BF16", "F32") for value in text_headers.values())
    target_modules = []
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in allowed_leaves:
            if any(component in name.lower() for component in ("visual", "vision", "image", "projector")) or ".layers." not in name:
                continue
            target_modules.append(name)
            targets.append({"name": name, "in_features": module.in_features, "out_features": module.out_features,
                            "lora_parameters": training["lora_rank"] * (module.in_features + module.out_features)})
    assert {name.rsplit(".", 1)[-1] for name in target_modules} == allowed_leaves
    model.requires_grad_(False)
    with torch.device("meta"):
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=training["lora_rank"],
            lora_alpha=training["lora_alpha"], lora_dropout=training["lora_dropout"], bias="none", target_modules=target_modules))
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    trainable_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters() if parameter.requires_grad)
    trainable_dtype_counts = Counter(str(parameter.dtype) for parameter in model.parameters() if parameter.requires_grad)
    assert trainable_parameters == sum(item["lora_parameters"] for item in targets)
    assert all("lora_" in name for name, parameter in model.named_parameters() if parameter.requires_grad)
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    if label == "2B":
        assert len(target_modules) == 96
        assert trainable_parameters == 10_911_744
    results.append({"model": f"Qwen/Qwen3.5-{label}-Base", "config_sha256": hashlib.sha256(raw).hexdigest(),
        "text_model_class": type(model.base_model.model).__name__, "text_hidden_size": config.text_config.hidden_size,
        "text_num_layers": config.text_config.num_hidden_layers, "text_parameters": base_parameters,
        "bf16_text_weight_bytes": base_bytes, "bf16_text_weight_gib": base_bytes / 1024**3,
        "tied_input_output_embeddings": tied, "target_module_count": len(target_modules),
        "target_leaves": dict(Counter(name.rsplit(".", 1)[-1] for name in target_modules)),
        "lora_trainable_parameters": trainable_parameters, "lora_parameter_bytes": trainable_bytes,
        "lora_parameter_dtypes": dict(trainable_dtype_counts), "lora_rank": training["lora_rank"],
        "target_modules": targets, "all_parameters_meta": True,
        "checkpoint_header_sources": header_sources, "checkpoint_unique_text_tensor_count": len(text_headers),
        "checkpoint_text_keys_and_shapes_match": True,
        "checkpoint_text_dtype_counts": dict(Counter(value["dtype"] for value in text_headers.values())),
        "excluded_nontext_tensor_count": len(excluded_names),
        "conversion": "Pinned Transformers qwen3_5_text renames ^model.language_model to model; tied lm_head aliases embedding"})
    print(f"[META PASSED] {label} | text_parameters={base_parameters:,} | BF16_weights={base_bytes / 1024**3:.4f} GiB | LoRA={trainable_parameters:,}", flush=True)
    del model

sources = {}
for module_name in ("transformers.models.qwen3_5.modeling_qwen3_5", "transformers.models.qwen3_5.configuration_qwen3_5",
                    "peft.tuners.lora.layer", "peft.peft_model"):
    module = sys.modules[module_name]
    path = Path(module.__file__)
    sources[module_name] = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
report = {"status": "passed", "scope": "Full-sized model construction and LoRA injection on meta tensors; no weights or numerical execution",
    "package_versions": versions, "platform": platform.platform(), "python": platform.python_version(),
    "torch_cuda_build": torch.version.cuda, "gpu_executed": False, "weights_loaded": False,
    "forward_backward_executed": False, "optimizer_executed": False, "training_memory_measured": False,
    "sources": sources, "models": results,
    "ratios": {"text_parameters_4b_over_2b": results[1]["text_parameters"] / results[0]["text_parameters"],
               "lora_parameters_4b_over_2b": results[1]["lora_trainable_parameters"] / results[0]["lora_trainable_parameters"]},
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "limitations": ["Mac ARM64 torch build, not Linux CUDA runtime", "Meta tensors have no storage and do not measure VRAM",
        "No checkpoint loading, generation, GRPO optimization, real math benchmark or training peak measured",
        "Current production loader rejects 4B until separately adapted and tested"]}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(f"[SAVED] {args.output}", flush=True)
