"""Download or verify the pinned official model with resumable transfers and byte progress."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", default="models/qwen3.5-2b-base-manifest.json")
parser.add_argument("--output-dir", default="models/Qwen3.5-2B-Base")
parser.add_argument("--offline", "--verify", dest="offline", action="store_true",
                    help="Verify all local bytes without contacting Hugging Face")
parser.add_argument("--timeout", type=int, default=60)
parser.add_argument("--retries", type=int, default=3, help="Maximum transfer attempts per missing file")
args = parser.parse_args()
if args.timeout < 1 or not 1 <= args.retries <= 5:
    raise SystemExit("--timeout must be positive and --retries must be between 1 and 5.")
root = Path(__file__).resolve().parents[1]
manifest_path = Path(args.manifest).expanduser()
if not manifest_path.is_absolute():
    manifest_path = root / manifest_path
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
if (manifest.get("schema_version") != 1 or manifest.get("repo_id") != "Qwen/Qwen3.5-2B-Base"
        or manifest.get("revision") != "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
        or not isinstance(manifest.get("files"), list) or not manifest["files"]):
    raise SystemExit("Expected the pinned official Qwen3.5-2B-Base file manifest.")
entries = {}
for entry in manifest["files"]:
    if not isinstance(entry, dict):
        raise SystemExit("Invalid model file metadata.")
    name = entry.get("path")
    if (not isinstance(name, str) or not name or PurePosixPath(name).is_absolute()
            or PurePosixPath(name).as_posix() != name or "\\" in name
            or any(part in {".", "..", ".cache"} for part in PurePosixPath(name).parts)
            or name in {"source_manifest.json", ".model-download.lock"} or name in entries):
        raise SystemExit("Unsafe, reserved, or duplicate model file path in manifest.")
    if type(entry.get("size")) is not int or entry["size"] < 0:
        raise SystemExit("Invalid model file size in manifest.")
    hash_fields = {key for key in ("sha256", "git_blob_sha1") if key in entry}
    if len(hash_fields) != 1:
        raise SystemExit("Each model file must have exactly one authoritative hash.")
    hash_name = next(iter(hash_fields))
    if not re.fullmatch(r"[0-9a-f]{64}" if hash_name == "sha256" else r"[0-9a-f]{40}", str(entry[hash_name])):
        raise SystemExit("Invalid authoritative model hash.")
    entries[name] = entry
total_bytes = sum(entry["size"] for entry in entries.values())
if manifest.get("total_bytes") != total_bytes:
    raise SystemExit("Model manifest total size does not match its files.")
output_dir = Path(args.output_dir).expanduser()
if not output_dir.is_absolute():
    output_dir = root / output_dir
if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
    raise SystemExit("Model destination must be a regular directory; existing content was preserved.")
output_dir.mkdir(parents=True, exist_ok=True)
output_dir = output_dir.resolve()
receipt_path = output_dir / "source_manifest.json"
lock_path = output_dir / ".model-download.lock"
for path in (receipt_path, lock_path):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SystemExit(f"Refusing non-regular model metadata destination: {path}")
lock_stream = lock_path.open("a+b")
try:
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError as error:
    lock_stream.close()
    raise SystemExit("Another model download/verification is running in this directory.") from error

receipt = {"schema_version": 1, "repo_id": manifest["repo_id"], "revision": manifest["revision"],
           "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "status": "checking",
           "all_files_verified": False, "checked_utc": datetime.now(timezone.utc).isoformat(),
           "files": {}, "downloaded_files": 0, "reused_files": 0, "total_bytes": total_bytes,
           "transferred_bytes": 0, "resumed_bytes": 0,
           "offline": args.offline, "model_loaded": False, "training_executed": False}
previous_receipt = None
failure = None
try:
    if receipt_path.exists():
        previous_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    with tempfile.NamedTemporaryFile(mode="w", dir=output_dir, prefix=".model-receipt-", delete=False) as stream:
        temporary_path = Path(stream.name)
        stream.write(json.dumps(receipt, indent=2) + "\n")
    temporary_path.replace(receipt_path)
    if previous_receipt is not None and (previous_receipt.get("repo_id") != manifest["repo_id"]
            or previous_receipt.get("revision") != manifest["revision"]
            or previous_receipt.get("manifest_sha256") != receipt["manifest_sha256"]):
        raise ValueError("Existing model provenance conflicts with the pinned manifest; files were preserved.")
    cache_dir = output_dir / ".cache/rlvr-download" / manifest["revision"]
    for cache_path in (output_dir / ".cache", output_dir / ".cache/rlvr-download", cache_dir):
        if cache_path.is_symlink() or (cache_path.exists() and not cache_path.is_dir()):
            raise ValueError(f"Refusing non-directory or symlink model download cache: {cache_path}")
    existing = []
    missing = []
    for name in entries:
        path = output_dir / name
        for component in [path, *path.parents]:
            if component == output_dir:
                break
            if component.is_symlink():
                raise ValueError(f"Refusing symlink inside model destination: {component}")
        if path.exists() and not path.is_file():
            raise ValueError(f"Refusing non-regular model file: {path}")
        (existing if path.is_file() else missing).append(name)
    model_affecting_names = {"generation_config.json", "added_tokens.json", "special_tokens_map.json",
                            "adapter_config.json", "adapter_model.bin", "pytorch_model.bin", "pytorch_model.bin.index.json"}
    extra_files = {path.relative_to(output_dir).as_posix() for path in output_dir.glob("*.safetensors")}
    extra_files |= {name for name in model_affecting_names if (output_dir / name).exists()}
    extra_files -= set(entries)
    if extra_files:
        raise ValueError(f"Unlisted model-affecting files were preserved: {sorted(extra_files)}")
    from tqdm import tqdm

    print(f"[MODEL] {manifest['repo_id']} | revision={manifest['revision']} | files={len(entries)} | bytes={total_bytes}", flush=True)
    with tqdm(total=total_bytes, desc="Model integrity", unit="B", unit_scale=True,
              dynamic_ncols=True, disable=False) as progress:
        for phase, names in (("local", existing), ("download", missing)):
            if phase == "download" and names:
                if args.offline:
                    raise FileNotFoundError(f"Offline model verification requires all files; missing: {', '.join(names)}")
                cache_dir.mkdir(parents=True, exist_ok=True)
                default_ca_paths = ssl.get_default_verify_paths()
                ca_file = ("/etc/ssl/cert.pem" if not default_ca_paths.cafile and not default_ca_paths.capath
                           and Path("/etc/ssl/cert.pem").is_file() else None)
                ssl_context = ssl.create_default_context(cafile=ca_file)
            for name in names:
                entry = entries[name]
                path = output_dir / name
                verification_path = path
                if phase == "download":
                    partial_path = cache_dir / (hashlib.sha256(name.encode()).hexdigest() + ".part")
                    if partial_path.is_symlink() or (partial_path.exists() and not partial_path.is_file()):
                        raise ValueError(f"Refusing non-regular partial model file: {partial_path}")
                    offset = partial_path.stat().st_size if partial_path.exists() else 0
                    if offset > entry["size"]:
                        raise ValueError(f"Partial model file is oversized and was preserved: {partial_path}")
                    receipt["resumed_bytes"] += offset
                    print(f"[MODEL DOWNLOAD] {name} | bytes={entry['size']} | resume_offset={offset}", flush=True)
                    url = (f"https://huggingface.co/{manifest['repo_id']}/resolve/{manifest['revision']}/"
                           + urllib.parse.quote(name, safe="/") + "?download=true")
                    with tqdm(total=entry["size"], initial=offset, desc=f"Download {name}", unit="B",
                              unit_scale=True, dynamic_ncols=True, disable=False) as transfer_progress:
                        for attempt in range(1, args.retries + 1):
                            offset = partial_path.stat().st_size if partial_path.exists() else 0
                            if offset == entry["size"] and partial_path.is_file():
                                break
                            request = urllib.request.Request(url, headers={"User-Agent": "DSAI4207-RLVR-model-v1.2.2",
                                "Accept-Encoding": "identity", "Range": f"bytes={offset}-"})
                            try:
                                with urllib.request.urlopen(request, timeout=args.timeout, context=ssl_context) as response:
                                    if not response.geturl().startswith("https://"):
                                        raise ValueError("Refusing a model transfer redirected away from HTTPS.")
                                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                                        raise ValueError("Unexpected compressed model transfer; partial bytes were preserved.")
                                    if response.status == 206:
                                        expected_range = f"bytes {offset}-{entry['size'] - 1}/{entry['size']}"
                                        if response.headers.get("Content-Range") != expected_range:
                                            raise ValueError(f"Unexpected model Content-Range; expected {expected_range}. Partial bytes were preserved.")
                                    elif response.status != 200 or offset != 0:
                                        raise ValueError("The server ignored a resume Range or returned an unexpected status; partial bytes were preserved.")
                                    content_length = response.headers.get("Content-Length")
                                    if content_length is not None and int(content_length) != entry["size"] - offset:
                                        raise ValueError("Unexpected model Content-Length; partial bytes were preserved.")
                                    with partial_path.open("ab") as stream:
                                        received = offset
                                        while chunk := response.read(1024 * 1024):
                                            if received + len(chunk) > entry["size"]:
                                                raise ValueError("Model response exceeds its pinned size; extra bytes were not appended.")
                                            stream.write(chunk)
                                            received += len(chunk)
                                            receipt["transferred_bytes"] += len(chunk)
                                            transfer_progress.update(len(chunk))
                                        stream.flush()
                                        os.fsync(stream.fileno())
                                    if received != entry["size"]:
                                        raise OSError("Model transfer ended early; partial bytes were retained for resume.")
                                break
                            except (urllib.error.URLError, TimeoutError, OSError, http.client.IncompleteRead) as error:
                                retryable = not isinstance(error, urllib.error.HTTPError) or error.code in {408, 429, 500, 502, 503, 504}
                                if not retryable or attempt == args.retries:
                                    raise RuntimeError(f"Model transfer failed: {name}. Partial bytes are retained; rerun setup to resume. {error}") from error
                                print(f"[MODEL RETRY {attempt}/{args.retries}] {name} | retaining downloaded bytes", flush=True)
                                time.sleep(min(2 ** (attempt - 1), 4))
                    verification_path = partial_path
                if verification_path.is_symlink() or not verification_path.is_file() or verification_path.stat().st_size != entry["size"]:
                    raise ValueError(f"Model file size/type mismatch: {name}; existing content was preserved.")
                sha256 = hashlib.sha256()
                git_blob = hashlib.sha1(f"blob {entry['size']}\0".encode()) if "git_blob_sha1" in entry else None
                with verification_path.open("rb") as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        sha256.update(chunk)
                        if git_blob is not None:
                            git_blob.update(chunk)
                        progress.update(len(chunk))
                if ("sha256" in entry and sha256.hexdigest() != entry["sha256"]
                        or git_blob is not None and git_blob.hexdigest() != entry["git_blob_sha1"]):
                    raise ValueError(f"Model hash mismatch: {name}; existing content was preserved. Restore or move this conflicting file explicitly: {verification_path}")
                if phase == "download":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    os.link(partial_path, path)
                    partial_path.unlink()
                receipt["files"][name] = {"bytes": entry["size"], "sha256": sha256.hexdigest()}
                if git_blob is not None:
                    receipt["files"][name]["git_blob_sha1"] = git_blob.hexdigest()
                receipt["reused_files" if phase == "local" else "downloaded_files"] += 1
    receipt.update(status="verified", all_files_verified=True)
except BaseException as error:
    receipt.update(status="failed", all_files_verified=False, error=f"{type(error).__name__}: {error}")
    failure = error
finally:
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output_dir, prefix=".model-receipt-", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(json.dumps(receipt, indent=2) + "\n")
        temporary_path.replace(receipt_path)
    finally:
        fcntl.flock(lock_stream, fcntl.LOCK_UN)
        lock_stream.close()
if failure is not None:
    raise failure
print(f"✅ [MODEL VERIFIED 100%] files={len(entries)} | downloaded={receipt['downloaded_files']} | reused={receipt['reused_files']}", flush=True)
print(f"[SAVED] {receipt_path}", flush=True)
