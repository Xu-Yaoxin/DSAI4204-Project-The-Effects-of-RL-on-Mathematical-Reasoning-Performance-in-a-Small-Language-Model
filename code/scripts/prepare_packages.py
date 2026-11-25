"""Resolve, download, or verify the Linux CPython 3.11 CUDA 12.8 wheel bundle."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from unittest.mock import patch


parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--resolve", action="store_true", help="Create a new immutable target lock using pip metadata resolution")
mode.add_argument("--verify", action="store_true", help="Verify the complete local wheelhouse without network access")
parser.add_argument("--bundle", default="packages/linux-cp311-cu128")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--ca-file")
parser.add_argument("--fetch-part", help=argparse.SUPPRESS)
parser.add_argument("--range-start", type=int, help=argparse.SUPPRESS)
parser.add_argument("--range-end", type=int, help=argparse.SUPPRESS)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
bundle = Path(args.bundle).expanduser()
if not bundle.is_absolute():
    bundle = root / bundle
manifest_path = bundle / "manifest.json"
lock_path = bundle / "requirements-lock.txt"
wheelhouse = bundle / "wheels"
target_environment = {
    "implementation_name": "cpython", "implementation_version": "3.11.0", "os_name": "posix",
    "platform_machine": "x86_64", "platform_release": "", "platform_system": "Linux", "platform_version": "",
    "python_full_version": "3.11.0", "platform_python_implementation": "CPython", "python_version": "3.11", "sys_platform": "linux",
}
platform_tags = [f"manylinux_2_{minor}_x86_64" for minor in range(28, 16, -1)] + ["manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
ca_file = args.ca_file
verify_paths = ssl.get_default_verify_paths()
if ca_file is None and not verify_paths.cafile and not verify_paths.capath and Path("/etc/ssl/cert.pem").is_file():
    ca_file = "/etc/ssl/cert.pem"
context = ssl.create_default_context(cafile=ca_file)
if not 1 <= args.workers <= 8:
    raise SystemExit("--workers must be between 1 and 8.")

if args.resolve:
    if manifest_path.exists() or lock_path.exists():
        raise SystemExit("A package lock already exists. Resolve into a new --bundle directory instead of overwriting provenance.")
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Lock resolution must run with Python 3.11; downloading and verification are standard-library only.")
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.tags import compatible_tags, cpython_tags, parse_tag
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._internal.operations.prepare import RequirementPreparer
    bundle.mkdir(parents=True, exist_ok=True)
    source_requirements = (root / "requirements-gpu.txt").read_text(encoding="utf-8")
    torch_url = "https://download-r2.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=1d01ffaebf64715c0f507a39463149cb19e596ff702bd4bcf862601f2881dabc"
    resolve_requirements = source_requirements
    resolve_requirements += "\npip>=25,<27\nsetuptools>=77,<83\n"
    report_path = bundle / "resolution-report.json"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as stream:
        stream.write(resolve_requirements)
        temporary_requirements = Path(stream.name)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".html", delete=False) as stream:
        stream.write(f'<a href="{torch_url}" data-core-metadata="sha256=7f56e1b49211bfcfe45a8c7df42d96a054f69a96fbf6bc4f1b6b6fe8d3f097a7" data-dist-info-metadata="sha256=7f56e1b49211bfcfe45a8c7df42d96a054f69a96fbf6bc4f1b6b6fe8d3f097a7">torch CUDA wheel</a>')
        temporary_links = Path(stream.name)
    pip_arguments = ["pip", "install", "--dry-run", "--ignore-installed", "--only-binary=:all:", "--no-cache-dir",
                     "--disable-pip-version-check", "--index-url", "https://pypi.org/simple", "--report", str(report_path),
                     "--python-version", "3.11", "--implementation", "cp", "--abi", "cp311", "--find-links", str(temporary_links), "-r", str(temporary_requirements)]
    for platform_tag in platform_tags:
        pip_arguments.extend(["--platform", platform_tag])
    if ca_file:
        pip_arguments.extend(["--cert", ca_file])
    if not report_path.exists():
        try:
            metadata_cache = {}
            original_metadata_fetch = RequirementPreparer._fetch_metadata_only
            metadata_fetch = lambda preparer, requirement: metadata_cache[requirement.link.url] if requirement.link.url in metadata_cache else metadata_cache.setdefault(requirement.link.url, original_metadata_fetch(preparer, requirement) or (_ for _ in ()).throw(RuntimeError("Missing metadata-only wheel distribution; refusing full prefetch")))
            original_get_dist = __import__("pip._internal.req.req_install", fromlist=["InstallRequirement"]).InstallRequirement.get_dist
            metadata_get_dist = lambda requirement: metadata_cache[requirement.link.url] if requirement.link and requirement.link.url in metadata_cache else original_get_dist(requirement)
            metadata_report = lambda requirement: {"download_info": {"url": requirement.link.url_without_fragment, "archive_info": {"hashes": {requirement.link.hash_name: requirement.link.hash}}}, "metadata": metadata_cache[requirement.link.url].metadata_dict}
            with patch.object(sys, "argv", pip_arguments), patch("pip._vendor.packaging.markers.default_environment", side_effect=target_environment.copy), patch("pip._internal.operations.prepare.RequirementPreparer._fetch_metadata_only", new=metadata_fetch), patch("pip._internal.operations.prepare.RequirementPreparer.prepare_linked_requirements_more", return_value=None), patch("pip._internal.models.installation_report.InstallationReport._install_req_to_dict", side_effect=metadata_report), patch("pip._internal.req.req_install.InstallRequirement.get_dist", new=metadata_get_dist):
                try:
                    runpy.run_module("pip", run_name="__main__")
                except SystemExit as error:
                    if error.code not in [None, 0]:
                        raise
        finally:
            temporary_requirements.unlink(missing_ok=True)
            temporary_links.unlink(missing_ok=True)
    else:
        temporary_requirements.unlink(missing_ok=True)
        temporary_links.unlink(missing_ok=True)
        print("[RESOLVE] Reusing cached metadata report; direct requirements and Linux closure are checked below.", flush=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    packages = []
    for entry in report["install"]:
        url = entry["download_info"]["url"]
        metadata = entry["metadata"]
        sha256 = entry["download_info"]["archive_info"]["hashes"]["sha256"]
        filename = urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1])
        if not filename.endswith(".whl") or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise SystemExit(f"Expected a hashed binary wheel: {url}")
        wheel_parts = filename[:-4].rsplit("-", 3)
        wheel_tags = parse_tag("-".join(wheel_parts[-3:]))
        supported_tags = set(cpython_tags(python_version=(3, 11), abis=["cp311"], platforms=platform_tags)) | set(compatible_tags(python_version=(3, 11), interpreter="cp311", platforms=platform_tags))
        if not supported_tags.intersection(wheel_tags):
            raise SystemExit(f"Incompatible wheel selected for Linux CPython 3.11: {filename}")
        packages.append({"name": canonicalize_name(metadata["name"]), "version": metadata["version"], "filename": filename,
                         "url": url, "sha256": sha256, "requires_dist": metadata.get("requires_dist", []),
                         "requires_python": metadata.get("requires_python", "")})
    versions = {entry["name"]: entry["version"] for entry in packages}
    for requirement_text in resolve_requirements.split("\n"):
        if not requirement_text.strip() or requirement_text.lstrip().startswith("#"):
            continue
        direct_requirement = Requirement(requirement_text)
        direct_name = canonicalize_name(direct_requirement.name)
        if direct_name not in versions or not direct_requirement.specifier.contains(versions[direct_name], prereleases=True):
            raise SystemExit(f"Cached resolver report does not satisfy direct requirement: {requirement_text}. Use a new bundle directory.")
    if report.get("environment", {}).get("sys_platform") != "linux" or report.get("environment", {}).get("python_version") != "3.11":
        raise SystemExit("Resolver report environment must be Linux CPython 3.11.")
    extras = {entry["name"]: set() for entry in packages}
    closure_edges = []
    changed = True
    while changed:
        changed = False
        closure_edges = []
        for entry in packages:
            for dependency_text in entry["requires_dist"]:
                dependency = Requirement(dependency_text)
                if dependency.marker and not any(dependency.marker.evaluate({**target_environment, "extra": extra}) for extra in extras[entry["name"]] | {""}):
                    continue
                name = canonicalize_name(dependency.name)
                if name not in versions or not dependency.specifier.contains(versions[name], prereleases=True):
                    raise SystemExit(f"Linux dependency closure failure: {entry['name']} requires {dependency_text}; selected {versions.get(name)}")
                new_extras = set(dependency.extras) - extras[name]
                if new_extras:
                    extras[name].update(new_extras)
                    changed = True
                closure_edges.append({"from": entry["name"], "requirement": dependency_text, "to": name, "version": versions[name]})
    print(f"[RESOLVE] Linux dependency closure passed: {len(packages)} wheels; retrieving exact download sizes.", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        size_requests = [(entry, pool.submit(urllib.request.urlopen, urllib.request.Request(entry["url"], method="HEAD"), context=context, timeout=60)) for entry in packages]
        for entry, future in size_requests:
            with future.result() as response:
                entry["bytes"] = int(response.headers["Content-Length"])
                entry["etag"] = response.headers.get("ETag")
                entry["accept_ranges"] = response.headers.get("Accept-Ranges", "").lower() == "bytes"
    packages.sort(key=lambda entry: entry["name"])
    lock_text = "# Linux x86_64, CPython 3.11, CUDA 12.8. Generated from manifest.json.\n" + "\n".join(f"{entry['name']}=={entry['version']} --hash=sha256:{entry['sha256']}" for entry in packages) + "\n"
    manifest = {"schema_version": 1, "target": {"id": "linux-cp311-cu128", "python": "3.11", "os": "Linux", "machine": "x86_64",
                "glibc_min": "2.28", "cuda": "12.8", "driver_min": "570.26", "gpu": "RTX 4090", "compute_capability": "8.9"},
                "marker_environment": target_environment, "platform_tags": platform_tags, "packages": packages,
                "total_bytes": sum(entry["bytes"] for entry in packages), "package_count": len(packages),
                "requirements_sha256": hashlib.sha256(source_requirements.encode("utf-8")).hexdigest(),
                "lock_sha256": hashlib.sha256(lock_text.encode("utf-8")).hexdigest(),
                "dependency_closure": {"verified": True, "edges": closure_edges},
                "validation": {"metadata_resolved": True, "gpu_runtime_tested": False, "model_training_tested": False},
                "optional_kernels": {"installed": False, "reason": "Pinned Transformers provides PyTorch fallbacks; latest FLA requires a newer Transformers and causal-conv1d has no PyPI wheel."},
                "sources": ["https://pytorch.org/get-started/previous-versions/", "https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html", "https://pypi.org/"]}
    lock_path.write_text(lock_text, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(f"[LOCK SAVED] {manifest_path} | wheels={len(packages)} | download_gib={manifest['total_bytes'] / 2**30:.2f}", flush=True)
    raise SystemExit(0)

if not manifest_path.is_file() or not lock_path.is_file():
    raise SystemExit("Missing package manifest/lock. Obtain the prepared bundle metadata or run --resolve explicitly with Python 3.11.")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("target", {}).get("id") != "linux-cp311-cu128" or manifest.get("schema_version") != 1:
    raise SystemExit("Unsupported package manifest target/schema.")
if hashlib.sha256(lock_path.read_bytes()).hexdigest() != manifest["lock_sha256"]:
    raise SystemExit("Package lock hash mismatch; refusing to download or install changed dependencies.")
if hashlib.sha256((root / "requirements-gpu.txt").read_bytes()).hexdigest() != manifest["requirements_sha256"]:
    raise SystemExit("requirements-gpu.txt changed after package locking. Resolve a new bundle explicitly.")
packages = manifest["packages"]
if len(packages) != manifest["package_count"] or len({entry["filename"] for entry in packages}) != len(packages) or len({entry["name"] for entry in packages}) != len(packages):
    raise SystemExit("Invalid package count or duplicate wheel filenames in manifest.")
for entry in packages:
    if not isinstance(entry.get("bytes"), int) or entry["bytes"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
        raise SystemExit("Invalid locked package size or SHA-256.")
expected_lock_lines = sorted(f"{entry['name']}=={entry['version']} --hash=sha256:{entry['sha256']}" for entry in packages)
actual_lock_lines = sorted(line.strip() for line in lock_path.read_text(encoding="utf-8").split("\n") if line.strip() and not line.lstrip().startswith("#"))
if actual_lock_lines != expected_lock_lines or manifest["total_bytes"] != sum(entry["bytes"] for entry in packages):
    raise SystemExit("Package manifest and exact hash lock disagree.")
wheelhouse.mkdir(parents=True, exist_ok=True)
if args.fetch_part:
    matches = [entry for entry in packages if entry["name"] == args.fetch_part]
    if len(matches) != 1:
        raise SystemExit("Unknown package part request.")
    entry = matches[0]
    start, end = args.range_start, args.range_end
    if start is None or end is None or not 0 <= start <= end < entry["bytes"]:
        raise SystemExit("Invalid locked package range.")
    if Path(entry["filename"]).name != entry["filename"] or urllib.parse.urlsplit(entry["url"]).scheme != "https" or urllib.parse.urlsplit(entry["url"]).hostname not in {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"}:
        raise SystemExit("Unsafe locked package URL or filename.")
    part_path = wheelhouse / (entry["filename"] + f".part-{start:012d}-{end:012d}")
    if part_path.is_symlink():
        raise SystemExit(f"Refusing a symlink partial download: {part_path}")
    expected_size = end - start + 1
    full_wheel = start == 0 and expected_size == entry["bytes"]
    last_error = None
    for attempt in range(1, 4):
        received = part_path.stat().st_size if part_path.exists() else 0
        if received > expected_size:
            raise SystemExit(f"Oversized partial download preserved: {part_path}")
        if received == expected_size:
            last_error = None
            break
        headers = {"User-Agent": "DSAI4207-wheelhouse-v1.2"}
        if not full_wheel or received:
            headers["Range"] = f"bytes={start + received}-{end}"
        if entry.get("etag") and not entry["etag"].startswith("W/"):
            headers["If-Match"] = entry["etag"]
        request = urllib.request.Request(entry["url"], headers=headers)
        try:
            with urllib.request.urlopen(request, context=context, timeout=60) as response:
                if "Range" in headers and response.status == 206:
                    expected_range = f"bytes {start + received}-{end}/{entry['bytes']}"
                    if response.headers.get("Content-Range") != expected_range:
                        raise ValueError(f"Unexpected range response: {response.headers.get('Content-Range')}")
                elif "Range" in headers and full_wheel and response.status == 200:
                    received = 0
                elif "Range" in headers:
                    raise ValueError(f"Range request not honored: HTTP {response.status}")
                with part_path.open("ab" if received else "wb") as stream:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        if stream.tell() > expected_size:
                            raise ValueError("Package part exceeds locked length")
            if part_path.stat().st_size != expected_size:
                raise OSError("Package part ended early")
            last_error = None
            break
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(attempt)
    if last_error:
        raise SystemExit(f"Part retained for resume: {part_path.name}: {last_error}")
    print(f"[PART SAVED] {entry['name']} bytes={start}-{end}", flush=True)
    raise SystemExit(0)

pending = []
verified_bytes = 0
for index, entry in enumerate(packages, 1):
    if Path(entry["filename"]).name != entry["filename"] or not entry["filename"].endswith(".whl"):
        raise SystemExit("Unsafe wheel filename in manifest.")
    path = wheelhouse / entry["filename"]
    if path.is_symlink():
        raise SystemExit(f"Refusing symlink wheel: {path}")
    if path.exists():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                block = stream.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise SystemExit(f"Wheel hash mismatch; existing file preserved: {path}")
        verified_bytes += entry["bytes"]
    else:
        pending.append(entry)
if args.verify:
    if pending:
        raise SystemExit(f"Wheelhouse incomplete: {len(pending)} missing wheels, {sum(entry['bytes'] for entry in pending) / 2**30:.2f} GiB. Run download_packages.sh online.")
    print(f"[VERIFIED 100%] wheels={len(packages)} bytes={verified_bytes} | network=unused | target=Linux-cp311-cu128")
    raise SystemExit(0)
needed_bytes = sum(entry["bytes"] for entry in pending)
free_bytes = shutil.disk_usage(wheelhouse).free
if free_bytes < needed_bytes + 2 * 2**30:
    raise SystemExit(f"Insufficient disk: need {needed_bytes / 2**30:.2f} GiB downloads plus 2 GiB reserve, have {free_bytes / 2**30:.2f} GiB.")
print(f"[DOWNLOAD] target=Linux-cp311-cu128 | verified={len(packages)-len(pending)}/{len(packages)} | remaining_gib={needed_bytes / 2**30:.2f}", flush=True)
if args.workers > 1 and pending:
    import fcntl
    with (wheelhouse / ".download.lock").open("a+") as bundle_lock:
        try:
            fcntl.flock(bundle_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another wheel download is already using this bundle; wait for it before resuming.")
        jobs = []
        remaining_parts = {}
        ranges = {}
        entries = {entry["name"]: entry for entry in pending}
        for entry in pending:
            chunked = entry["bytes"] > 64 * 1024**2 and entry.get("accept_ranges") and entry.get("etag") and not entry["etag"].startswith("W/")
            chunk_size = 16 * 1024**2 if chunked else entry["bytes"]
            ranges[entry["name"]] = [(start, min(start + chunk_size, entry["bytes"]) - 1) for start in range(0, entry["bytes"], chunk_size)]
            remaining_parts[entry["name"]] = len(ranges[entry["name"]])
            for start, end in ranges[entry["name"]]:
                command = [sys.executable, str(Path(__file__).resolve()), "--bundle", str(bundle), "--fetch-part", entry["name"], "--range-start", str(start), "--range-end", str(end)]
                if args.ca_file:
                    command.extend(["--ca-file", args.ca_file])
                jobs.append((entry["name"], command))
        jobs.sort(key=lambda job: int(job[1][job[1].index("--range-start") + 1]))
        completed_jobs = 0
        completed_packages = len(packages) - len(pending)
        received_bytes = verified_bytes
        progress_started = time.monotonic()
        progress_updated = progress_started - 1
        print(f"[PARALLEL] workers={args.workers} range_jobs={len(jobs)} | large_wheels_use_verified_HTTP_ranges", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(subprocess.run, command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600):
                       (name, int(command[command.index('--range-end') + 1]) - int(command[command.index('--range-start') + 1]) + 1)
                       for name, command in jobs}
            for future in as_completed(futures):
                name, part_bytes = futures[future]
                result = future.result()
                if result.returncode:
                    for other in futures:
                        other.cancel()
                    raise SystemExit(f"Package download paused safely for {name}; resume by rerunning.\n{result.stdout}")
                completed_jobs += 1
                received_bytes += part_bytes
                progress_now = time.monotonic()
                if progress_now - progress_updated >= 1 or completed_jobs == len(jobs):
                    fraction = received_bytes / manifest['total_bytes']
                    filled = min(28, int(28 * fraction))
                    bar = '#' * filled + '-' * (28 - filled)
                    print(f"[WHEEL DOWNLOAD] [{bar}] {fraction:6.1%} | "
                          f"{received_bytes / 2**30:.2f}/{manifest['total_bytes'] / 2**30:.2f} GiB | "
                          f"parts={completed_jobs}/{len(jobs)} | elapsed={progress_now - progress_started:.0f}s", flush=True)
                    progress_updated = progress_now
                remaining_parts[name] -= 1
                if remaining_parts[name]:
                    continue
                entry = entries[name]
                path = wheelhouse / entry["filename"]
                assembly = path.with_suffix(path.suffix + ".assemble")
                if assembly.is_symlink():
                    raise SystemExit(f"Refusing symlink wheel assembly: {assembly}")
                digest = hashlib.sha256()
                with assembly.open("wb") as output:
                    for start, end in ranges[name]:
                        part_path = wheelhouse / (entry["filename"] + f".part-{start:012d}-{end:012d}")
                        if part_path.is_symlink() or part_path.stat().st_size != end - start + 1:
                            raise SystemExit(f"Part size changed during assembly: {part_path}")
                        with part_path.open("rb") as stream:
                            while True:
                                block = stream.read(8 * 1024 * 1024)
                                if not block:
                                    break
                                digest.update(block)
                                output.write(block)
                if assembly.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                    raise SystemExit(f"Assembled wheel hash mismatch; chunks retained for inspection: {assembly}")
                os.link(assembly, path)
                assembly.unlink()
                for start, end in ranges[name]:
                    (wheelhouse / (entry["filename"] + f".part-{start:012d}-{end:012d}")).unlink()
                completed_packages += 1
                verified_bytes += entry["bytes"]
                print(f"[SAVED {completed_packages}/{len(packages)}] {entry['filename']} | verified_bytes={verified_bytes}/{manifest['total_bytes']} ({verified_bytes/manifest['total_bytes']:.1%})", flush=True)
    print(f"[PACKAGES COMPLETE] wheels={len(packages)} | bytes={verified_bytes} | {wheelhouse}", flush=True)
    raise SystemExit(0)
if args.workers == 1 and pending:
    import fcntl
    single_download_lock = (wheelhouse / ".download.lock").open("a+")
    try:
        fcntl.flock(single_download_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another wheel download is already using this bundle; wait for it before resuming.")
for index, entry in enumerate(pending, 1):
    path = wheelhouse / entry["filename"]
    partial_path = path.with_suffix(path.suffix + ".part")
    if partial_path.is_symlink():
        raise SystemExit(f"Refusing symlink partial download: {partial_path}")
    if urllib.parse.urlsplit(entry["url"]).scheme != "https" or urllib.parse.urlsplit(entry["url"]).hostname not in {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"}:
        raise SystemExit(f"Unexpected wheel source: {entry['url']}")
    last_error = None
    for attempt in range(1, 4):
        offset = partial_path.stat().st_size if partial_path.exists() else 0
        if offset > entry["bytes"]:
            raise SystemExit(f"Oversized partial download preserved for inspection: {partial_path}")
        if offset == entry["bytes"]:
            break
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "DSAI4207-wheelhouse-v1.2", **({"Range": f"bytes={offset}-"} if offset else {})})
        try:
            with urllib.request.urlopen(request, context=context, timeout=60) as response:
                status = response.status
                if offset and status == 206 and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected resume range")
                if offset and status == 200:
                    offset = 0
                elif offset and status != 206:
                    raise ValueError(f"Cannot safely resume download: HTTP {status}")
                with partial_path.open("ab" if offset else "wb") as stream:
                    while True:
                        block = response.read(8 * 1024 * 1024)
                        if not block:
                            break
                        stream.write(block)
                        if stream.tell() > entry["bytes"]:
                            raise ValueError("Download exceeds the locked wheel size")
            if partial_path.stat().st_size != entry["bytes"]:
                raise OSError("Response ended before the locked wheel size")
            last_error = None
            break
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            last_error = error
            print(f"[RETRY {attempt}/3] {entry['name']} | saved_partial_bytes={partial_path.stat().st_size if partial_path.exists() else 0} | {error}", flush=True)
            if attempt < 3:
                time.sleep(attempt)
    if last_error:
        raise SystemExit(f"Download paused safely; rerun to resume {entry['filename']}: {last_error}")
    digest = hashlib.sha256()
    with partial_path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    if digest.hexdigest() != entry["sha256"]:
        raise SystemExit(f"Downloaded wheel hash mismatch. Partial retained, not installed: {partial_path}")
    os.link(partial_path, path)
    partial_path.unlink()
    verified_bytes += entry["bytes"]
    print(f"[SAVED {index}/{len(pending)}] {entry['filename']} | total={verified_bytes / manifest['total_bytes']:.1%}", flush=True)
print(f"[PACKAGES COMPLETE] wheels={len(packages)} | bytes={verified_bytes} | {wheelhouse}", flush=True)
