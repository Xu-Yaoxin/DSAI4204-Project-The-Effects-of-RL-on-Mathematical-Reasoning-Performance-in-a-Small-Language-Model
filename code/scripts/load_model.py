"""Shared local text-model loading steps. No downloads or model substitution."""
import importlib.metadata
import json
import os
from pathlib import Path
import runpy

if "MODEL_PATH" not in globals():
    raise SystemExit("Internal script: use train_grpo.py or evaluate.py.")
code_root = Path(__file__).resolve().parents[1]
model_path = Path(MODEL_PATH).expanduser()
if not model_path.is_absolute():
    model_path = code_root / model_path
model_path = model_path.resolve()
if not (model_path / "config.json").is_file():
    raise FileNotFoundError(f"Missing local model config in {model_path}; see READM.md.")
local_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
text_config = local_config.get("text_config", {})
if (local_config.get("model_type") != "qwen3_5" or text_config.get("model_type") != "qwen3_5_text"
        or text_config.get("hidden_size") != 2048 or text_config.get("num_hidden_layers") != 24
        or "quantization_config" in local_config or "quantization_config" in text_config):
    raise ValueError("Expected the original unquantized Qwen3.5-2B-Base config. See READM.md.")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
package_versions = {}
for package in ["torch", "transformers", "peft", "accelerate", "datasets", "trl", "safetensors",
                "flash-linear-attention", "fla-core", "causal-conv1d"]:
    try:
        package_versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        package_versions[package] = None
for package, expected in {"torch": "2.10.0", "transformers": "5.3.0", "peft": "0.18.1", "accelerate": "1.10.1"}.items():
    if not package_versions[package] or package_versions[package].split("+")[0] != expected:
        raise RuntimeError(f"Expected {package}=={expected}, found {package_versions[package]}. Install requirements-gpu.txt in a fresh environment.")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

world_size = int(os.environ.get("WORLD_SIZE", "1"))
rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
training_load = bool(globals().get("TRAINING", False))
if world_size not in (1, 2, 4) or (world_size > 1 and not training_load):
    raise RuntimeError("Training supports one-node DDP on 1/2/4 GPUs; evaluation requires one process.")
if not torch.cuda.is_available() or torch.cuda.device_count() != world_size or not 0 <= local_rank < world_size:
    raise RuntimeError("CUDA_VISIBLE_DEVICES must expose the selected GPUs, with one training process per GPU.")
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("Real v1 runs require an NVIDIA CUDA device with BF16 support. CPU tests use tests/run_all.py.")
distributed_load = training_load and world_size > 1
if distributed_load and not torch.distributed.is_initialized():
    raise RuntimeError("Initialize the training process group before distributed model loading.")
identity_packet = [None]
if rank == 0:
    try:
        fingerprint = runpy.run_path(str(Path(__file__).with_name("fingerprint_model.py")), init_globals={"MODEL_PATH": model_path})
        identity_packet[0] = {key: fingerprint[key] for key in ("model_identity", "model_files", "model_layout")}
    except (OSError, ValueError, KeyError, TypeError) as error:
        identity_packet[0] = {"error": f"{type(error).__name__}: {error}"}
if distributed_load:
    torch.distributed.broadcast_object_list(identity_packet, src=0)
identity = identity_packet[0]
if identity is None or "error" in identity:
    raise RuntimeError(f"Local checkpoint fingerprint failed: {identity}")
model_identity, model_files = identity["model_identity"], identity["model_files"]
model_layout = identity["model_layout"]
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
if tokenizer.eos_token_id is None:
    raise ValueError("Tokenizer must define eos_token_id.")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
# Transformers 5.3.0 maps the official multimodal checkpoint to its text backbone.
# Its qwen3_5_text conversion maps model.language_model.* to model.*.
model, loading_info = AutoModelForCausalLM.from_pretrained(
    model_path, config=config.text_config, dtype=torch.bfloat16,
    attn_implementation="sdpa", local_files_only=True, trust_remote_code=False,
    use_safetensors=True, output_loading_info=True,
)
unexpected = [key for key in loading_info.get("unexpected_keys", [])
              if not key.startswith(("model.visual.", "mtp.", "model.mtp."))]
if loading_info.get("missing_keys") or loading_info.get("mismatched_keys") or unexpected or loading_info.get("error_msgs"):
    raise RuntimeError(f"Checkpoint did not fully load the text weights: {loading_info}")
model.requires_grad_(False)
model.to(device)
model.generation_config = GenerationConfig(
    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
)
adapter_identity = None
if globals().get("ADAPTER_PATH"):
    from peft import PeftModel
    import hashlib
    adapter_path = Path(ADAPTER_PATH).expanduser()
    if not adapter_path.is_absolute():
        adapter_path = code_root / adapter_path
    adapter_path = adapter_path.resolve()
    adapter_digest = hashlib.sha256()
    for filename in ["adapter_config.json", "adapter_model.safetensors"]:
        adapter_digest.update(filename.encode())
        with (adapter_path / filename).open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                adapter_digest.update(chunk)
    adapter_identity = adapter_digest.hexdigest()
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False, local_files_only=True)
    model.requires_grad_(False)
if globals().get("TRAINING", False):
    model.config.use_cache = False
else:
    model.eval()
    model.config.use_cache = True
