"""Content identity for a local Transformers checkpoint; invoked through runpy."""
import hashlib
import json
import math
from pathlib import Path

if "MODEL_PATH" not in globals():
    raise SystemExit("Internal script: call doctor.py or evaluate.py instead.")
model_path = Path(MODEL_PATH).expanduser()
if not model_path.is_absolute():
    model_path = Path(__file__).resolve().parents[1] / model_path
model_path = model_path.resolve()
if not (model_path / "config.json").is_file():
    raise FileNotFoundError(f"Missing local config.json in {model_path}; see READM.md.")
weight_paths = sorted(model_path.glob("*.safetensors"))
if not weight_paths:
    raise FileNotFoundError(f"No safetensors weights in {model_path}.")
index_path = model_path / "model.safetensors.index.json"
weight_map = None
if index_path.is_file():
    weight_index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = weight_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Checkpoint index must contain a nonempty weight_map.")
    declared_files = set(weight_map.values())
    if any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors") for name in declared_files):
        raise ValueError("Checkpoint index contains invalid or nonlocal shard names.")
    if declared_files != {path.name for path in weight_paths}:
        raise ValueError("Checkpoint index and local shards differ; finish or repair the original download.")
elif len(weight_paths) != 1 or weight_paths[0].name != "model.safetensors":
    raise ValueError("Multiple checkpoint shards require model.safetensors.index.json; a single file must be model.safetensors.")
for filename in ("tokenizer_config.json", "tokenizer.json"):
    if not (model_path / filename).is_file():
        raise FileNotFoundError(f"Missing local tokenizer file: {filename}")

# Inspect only safetensors metadata before any expensive hash or GPU allocation.
tensor_locations = {}
dtype_bytes = {"BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4,
               "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}
for weight_path in weight_paths:
    file_size = weight_path.stat().st_size
    with weight_path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors file: {weight_path.name}")
        header_length = int.from_bytes(prefix, "little")
        if not 2 <= header_length <= min(100_000_000, file_size - 8):
            raise ValueError(f"Invalid safetensors header length: {weight_path.name}")
        header = json.loads(stream.read(header_length))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {weight_path.name}")
    data_bytes = file_size - header_length - 8
    intervals = []
    for tensor_name, tensor in header.items():
        if tensor_name == "__metadata__":
            continue
        if tensor_name in tensor_locations or not isinstance(tensor, dict):
            raise ValueError("Duplicate tensor name or invalid tensor metadata in checkpoint.")
        shape, offsets, dtype = tensor.get("shape"), tensor.get("data_offsets"), tensor.get("dtype")
        if (not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= data_bytes or dtype not in dtype_bytes):
            raise ValueError(f"Invalid or unsupported tensor metadata: {tensor_name}")
        if offsets[1] - offsets[0] != math.prod(shape) * dtype_bytes[dtype]:
            raise ValueError(f"Truncated or inconsistent tensor bytes: {tensor_name}")
        tensor_locations[tensor_name] = weight_path.name
        if offsets[1] > offsets[0]:
            intervals.append(tuple(offsets))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError(f"Overlapping or missing tensor bytes in {weight_path.name}")
        cursor = end
    if cursor != data_bytes:
        raise ValueError(f"Unexpected trailing tensor bytes in {weight_path.name}")
if not tensor_locations or (weight_map is not None and tensor_locations != weight_map):
    raise ValueError("Checkpoint tensor metadata does not match its weight index.")
model_layout = {"tensor_count": len(tensor_locations), "weight_files": [path.name for path in weight_paths],
                "weight_bytes": sum(path.stat().st_size for path in weight_paths), "indexed": weight_map is not None}
model_files = {}
identity_paths = sorted(set(weight_paths + [p for p in model_path.iterdir()
    if p.name in {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json",
                  "model.safetensors.index.json"}]))
for identity_path in identity_paths:
    model_files[identity_path.name] = {"bytes": identity_path.stat().st_size}
    if globals().get("HASH_MODEL", True):
        digest = hashlib.sha256()
        with identity_path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        model_files[identity_path.name]["sha256"] = digest.hexdigest()
model_identity = (hashlib.sha256(json.dumps(model_files, sort_keys=True).encode()).hexdigest()
                  if globals().get("HASH_MODEL", True) else None)
