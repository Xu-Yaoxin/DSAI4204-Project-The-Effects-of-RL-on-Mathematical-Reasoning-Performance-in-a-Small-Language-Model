"""Download and verify official GSM8K at a pinned revision without overwrites."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import tempfile
import urllib.error
import urllib.request


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/v1.json")
parser.add_argument("--revision", default="3101c7d5072418e28b9008a6636bde82a006892c", help="Full immutable Git commit SHA")
parser.add_argument("--ca-file", help="Optional trusted CA bundle; TLS verification is always enabled")
parser.add_argument("--offline", action="store_true", help="Verify a complete existing download without network access")
parser.add_argument("--timeout", type=int, default=45)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
config_path = Path(args.config).expanduser()
if not config_path.is_absolute():
    config_path = root / config_path
cfg = json.loads(config_path.read_text(encoding="utf-8"))
revision = args.revision.lower()
if not re.fullmatch(r"[0-9a-f]{40}", revision):
    raise SystemExit("--revision must be a full 40-character Git SHA, not a moving branch or tag.")
if args.timeout < 1:
    raise SystemExit("--timeout must be positive.")
repository = "https://github.com/openai/grade-school-math"
destinations = {}
for split in ["train", "test"]:
    path = Path(cfg["data"]["raw_" + split]).expanduser()
    destinations[split] = path if path.is_absolute() else root / path
destinations["license"] = destinations["train"].parent / "LICENSE"
provenance_path = destinations["train"].parent / "source_manifest.json"
if len({str(path.absolute()) for path in list(destinations.values()) + [provenance_path]}) != 4:
    raise SystemExit("Dataset, license, and provenance destinations must be distinct.")
for path in list(destinations.values()) + [provenance_path]:
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SystemExit(f"Refusing non-regular output destination: {path}")
remote_paths = {"train": "grade_school_math/data/train.jsonl", "test": "grade_school_math/data/test.jsonl", "license": "LICENSE"}
urls = {name: f"https://raw.githubusercontent.com/openai/grade-school-math/{revision}/{remote_path}" for name, remote_path in remote_paths.items()}
existing_provenance = None
if provenance_path.exists():
    existing_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if existing_provenance.get("schema_version") != 2 or existing_provenance.get("repository") != repository or existing_provenance.get("revision") != revision or existing_provenance.get("license") != "MIT":
        raise SystemExit(f"Provenance conflict: {provenance_path}. Use a new raw-data directory for a different revision; no files were changed.")
    if set(existing_provenance.get("files", {})) != set(destinations):
        raise SystemExit(f"Incomplete provenance metadata: {provenance_path}. No files were changed.")
    for name, path in destinations.items():
        metadata = existing_provenance["files"][name]
        if metadata.get("url") != urls[name] or not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("sha256", ""))):
            raise SystemExit(f"Invalid pinned source metadata for {name}; no files were changed.")
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise SystemExit(f"Raw-data hash mismatch: {path}. Existing content was preserved; restore or move the conflicting file explicitly.")

ca_file = args.ca_file
default_ca_paths = ssl.get_default_verify_paths()
if ca_file is None and not default_ca_paths.cafile and not default_ca_paths.capath and Path("/etc/ssl/cert.pem").is_file():
    ca_file = "/etc/ssl/cert.pem"
ssl_context = ssl.create_default_context(cafile=ca_file)
downloaded = {}
provenance = {"schema_version": 2, "dataset": "GSM8K", "repository": repository, "revision": revision,
              "license": "MIT", "variant": "main", "files": {}}
network_files = 0
for name, path in destinations.items():
    if existing_provenance and path.exists():
        content = path.read_bytes()
    else:
        if args.offline:
            raise SystemExit(f"Offline verification requires complete pinned data and provenance: {path}")
        request = urllib.request.Request(urls[name], headers={"User-Agent": "DSAI4207-RLVR-data-v1.1"})
        try:
            with urllib.request.urlopen(request, timeout=args.timeout, context=ssl_context) as response:
                content = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise SystemExit(f"Download failed for {name}: {error}. No dataset files were installed.") from error
        network_files += 1
    try:
        local_path = str(path.relative_to(root))
    except ValueError:
        local_path = str(path)
    metadata = {"url": urls[name], "local_path": local_path, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    if name in ["train", "test"]:
        try:
            records = [json.loads(line) for line in content.decode("utf-8").split("\n") if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"Invalid official {name} JSONL response; no dataset files were installed.") from error
        if not records or any(not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip() or not isinstance(row.get("answer"), str) or "####" not in row["answer"] for row in records):
            raise SystemExit(f"Invalid official {name} question/answer rows; no dataset files were installed.")
        if revision == "3101c7d5072418e28b9008a6636bde82a006892c" and len(records) != {"train": 7473, "test": 1319}[name]:
            raise SystemExit(f"Unexpected row count for pinned {name}: {len(records)}. No dataset files were installed.")
        metadata["rows"] = len(records)
    elif not content.startswith(b"MIT License") or b"Copyright (c) 2021 OpenAI" not in content:
        raise SystemExit("Unexpected GSM8K license response; no dataset files were installed.")
    if existing_provenance and metadata != existing_provenance["files"][name]:
        raise SystemExit(f"Provenance metadata mismatch for {name}; existing files were preserved.")
    if path.exists() and path.read_bytes() != content:
        raise SystemExit(f"Refusing to overwrite conflicting raw data: {path}. No files were changed.")
    downloaded[name] = content
    provenance["files"][name] = metadata

provenance_bytes = (json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
if existing_provenance and existing_provenance != provenance:
    raise SystemExit(f"Provenance conflict: {provenance_path}. Existing files were preserved.")
pending = {destinations[name]: content for name, content in downloaded.items() if not destinations[name].exists()}
if not provenance_path.exists():
    pending[provenance_path] = provenance_bytes
staged_paths = []
installed_paths = []
try:
    for path, content in pending.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".gsm8k-stage-", delete=False) as stream:
            staged_path = Path(stream.name)
            staged_paths.append((staged_path, path))
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    for staged_path, path in staged_paths:
        try:
            os.link(staged_path, path)
            installed_paths.append((path, path.stat().st_ino))
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != pending[path]:
                raise RuntimeError(f"Concurrent output conflict: {path}. Existing files were preserved.")
except BaseException:
    for path, inode in reversed(installed_paths):
        if path.is_file() and not path.is_symlink() and path.stat().st_ino == inode:
            path.unlink()
    raise
finally:
    for staged_path, path in staged_paths:
        staged_path.unlink(missing_ok=True)
for name in destinations:
    count = f"rows={provenance['files'][name]['rows']}" if name in ["train", "test"] else "license=MIT"
    print(f"[DATA VERIFIED] {name} | {count} | sha256={provenance['files'][name]['sha256']}")
print(f"[DATA 100%] revision={revision} | network_files={network_files} | installed_files={len(installed_paths)}")
print(f"[SAVED] {provenance_path}")
