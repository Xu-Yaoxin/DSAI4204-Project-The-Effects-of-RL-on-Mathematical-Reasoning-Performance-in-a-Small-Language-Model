"""Read pinned official metadata and safetensors headers, never full weights."""
import datetime
import hashlib
import json
from pathlib import Path
import ssl
import urllib.request
import certifi

root = Path(__file__).resolve().parent
root.mkdir(parents=True, exist_ok=True)
reports = []
tls = ssl.create_default_context(cafile=certifi.where())
for name, fixed_revision in [("2B", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"), ("4B", None)]:
    repo = f"Qwen/Qwen3.5-{name}-Base"
    endpoint = f"https://huggingface.co/api/models/{repo}"
    if fixed_revision:
        endpoint += f"/revision/{fixed_revision}"
    with urllib.request.urlopen(endpoint + "?blobs=true", timeout=40, context=tls) as response:
        raw = response.read(4_000_001)
    assert len(raw) <= 4_000_000
    metadata = json.loads(raw)
    revision = metadata["sha"]
    assert fixed_revision is None or revision == fixed_revision
    (root / f"{name}_official_api.json").write_bytes(raw)
    files = {item["rfilename"]: item for item in metadata["siblings"]}
    fetched = {}
    for filename in ["config.json", "model.safetensors.index.json", "README.md"]:
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        with urllib.request.urlopen(url, timeout=40, context=tls) as response:
            body = response.read(4_000_001)
        assert len(body) <= 4_000_000 and len(body) == files[filename]["size"]
        blob_hash = hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest()
        assert blob_hash == files[filename]["blobId"]
        target = root / f"{name}_{filename}"
        target.write_bytes(body)
        fetched[filename] = {"bytes": len(body), "git_blob_sha1": blob_hash, "url": url}
    index = json.loads((root / f"{name}_model.safetensors.index.json").read_text())
    tensor_headers = {}
    shard_receipts = []
    for filename in sorted(set(index["weight_map"].values())):
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        request = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
        with urllib.request.urlopen(request, timeout=60, context=tls) as response:
            assert response.status == 206
            content_range = response.headers["Content-Range"]
            assert content_range == f"bytes 0-7/{files[filename]['size']}"
            prefix = response.read(9)
        assert len(prefix) == 8
        header_length = int.from_bytes(prefix, "little")
        assert 2 <= header_length <= 4_000_000
        request = urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + header_length}"})
        with urllib.request.urlopen(request, timeout=60, context=tls) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == f"bytes 8-{7 + header_length}/{files[filename]['size']}"
            header_bytes = response.read(header_length + 1)
        assert len(header_bytes) == header_length
        header = json.loads(header_bytes)
        (root / f"{name}_{filename}.header.json").write_bytes(header_bytes)
        for key, value in header.items():
            if key == "__metadata__":
                continue
            assert key not in tensor_headers and index["weight_map"][key] == filename
            tensor_headers[key] = value
        shard_receipts.append({"filename": filename, "bytes": files[filename]["size"],
            "expected_full_file_sha256": files[filename]["lfs"]["sha256"],
            "header_bytes": header_length, "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
            "full_weight_hash_verified": False})
        print(f"[HEADER VERIFIED] {name} | {filename} | {header_length} bytes", flush=True)
    assert set(tensor_headers) == set(index["weight_map"])
    groups = {}
    for key, tensor in tensor_headers.items():
        group = ("text" if key.startswith(("model.language_model.", "lm_head."))
            else "vision" if key.startswith("model.visual.") else "other")
        entry = groups.setdefault(group, {"parameters": 0, "stored_bytes": 0, "tensors": 0, "dtypes": {}})
        count = 1
        for dimension in tensor["shape"]:
            count *= dimension
        entry["parameters"] += count
        entry["stored_bytes"] += tensor["data_offsets"][1] - tensor["data_offsets"][0]
        entry["tensors"] += 1
        entry["dtypes"][tensor["dtype"]] = entry["dtypes"].get(tensor["dtype"], 0) + count
    reports.append({"name": name, "repo": repo, "revision": revision,
        "api_sha256": hashlib.sha256(raw).hexdigest(), "metadata_files": fetched,
        "weights": shard_receipts, "tensor_groups": groups,
        "total_download_bytes_if_all_repository_files": sum(item["size"] for item in files.values()),
        "full_weights_downloaded": False, "gpu_execution": False})
    (root / "metadata_report.json").write_text(json.dumps({"retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "models": reports}, ensure_ascii=False, indent=2) + "\n")
    print(f"[METADATA SAVED] {repo} | revision={revision} | groups={groups}", flush=True)
