"""Preserve legacy training and repair only an interrupted baseline JSONL tail."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-dir", default="outputs/v1_2")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
relative_run = Path(args.run_dir)
if (relative_run.is_absolute() or ".." in relative_run.parts or len(relative_run.parts) < 2
        or relative_run.parts[0] != "outputs"):
    parser.error("Use a project-local run directory below outputs/.")
raw_run = root / relative_run
if raw_run.is_symlink():
    raise ValueError("Run directory must not be a symlink.")
run = raw_run.resolve()
if not run.is_relative_to(root / "outputs"):
    raise ValueError("Run directory escapes project outputs.")
if not run.exists():
    print("[UPGRADE] No previous experiment outputs; ready for a new resumable run.", flush=True)
    raise SystemExit(0)
if not run.is_dir():
    raise ValueError("Run path is not a directory.")
legacy = []
for stage, mode in (("engineering_train", "engineering"), ("pilot", "pilot"), ("formal", "formal")):
    folder = run / stage
    if not folder.exists() and not folder.is_symlink():
        continue
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(f"Unsafe training directory: {folder}")
    report_path = folder / "report.json"
    if report_path.is_symlink() or not report_path.is_file():
        raise ValueError(f"Cannot classify a training directory without a regular report: {folder}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not report or report.get("mode") != mode:
        raise ValueError(f"Invalid training report; preserved without changes: {report_path}")
    if type(report.get("resume_schema_version")) is not int or report["resume_schema_version"] != 1:
        legacy.append(folder)
archives = list(legacy)
if legacy:
    for name in ("selection", "test_base", "test_rl", "main_analysis"):
        path = run / name
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"Unsafe legacy downstream output: {path}")
            archives.append(path)

baseline = run / "engineering_base"
repair = None
if baseline.exists() or baseline.is_symlink():
    if baseline.is_symlink() or not baseline.is_dir():
        raise ValueError("Baseline output must be a local directory.")
    manifest_path = baseline / "manifest.json"
    samples_path = baseline / "samples.jsonl"
    if (manifest_path.is_symlink() or not manifest_path.is_file() or samples_path.is_symlink()
            or (samples_path.exists() and not samples_path.is_file())):
        raise ValueError("Baseline manifest or samples are missing or unsafe; preserved without changes.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or type(manifest.get("complete")) is not bool:
        raise ValueError("Baseline manifest must declare a boolean complete state.")
    if samples_path.exists():
        original = samples_path.read_bytes()
        lines = original.splitlines(keepends=True)
        offset = 0
        for index, line in enumerate(lines):
            if line.strip():
                try:
                    json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    if index != len(lines) - 1 or line.endswith(b"\n") or manifest["complete"]:
                        raise ValueError("Invalid baseline JSONL record is not a recoverable incomplete last line.") from error
                    repair = {"action": "remove_interrupted_last_line", "bytes": original[:offset]}
            offset += len(line)
        if (repair is None and original and not original.endswith(b"\n") and not manifest["complete"]):
            repair = {"action": "terminate_valid_last_line", "bytes": original + b"\n"}
        if repair:
            backup_root = baseline / "recovery_backups"
            pending = baseline / "samples.jsonl.upgrade-pending"
            if (backup_root.is_symlink() or (backup_root.exists() and not backup_root.is_dir())
                    or pending.is_symlink()):
                raise ValueError("Unsafe baseline recovery path; no files were changed.")

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
if archives:
    archive = run / ("legacy-before-resume-" + timestamp)
    archive.mkdir(exist_ok=False)
    for path in archives:
        path.rename(archive / path.name)
    (archive / "upgrade.json").write_text(json.dumps({
        "reason": "Training predates the resumable protocol; all artifacts preserved before starting new training.",
        "archived": [path.name for path in archives], "created_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[UPGRADE ARCHIVED] {archive}", flush=True)
if repair:
    backup_root.mkdir(exist_ok=True)
    backup = backup_root / ("samples.before-upgrade-" + timestamp + ".jsonl")
    shutil.copy2(samples_path, backup)
    with pending.open("wb") as stream:
        stream.write(repair["bytes"])
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(samples_path)
    (backup.with_suffix(".json")).write_text(json.dumps({
        "action": repair["action"], "original_sha256": hashlib.sha256(original).hexdigest(),
        "repaired_sha256": hashlib.sha256(repair["bytes"]).hexdigest(),
        "original_bytes": len(original), "repaired_bytes": len(repair["bytes"]),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[BASELINE TAIL RECOVERED] {repair['action']} | backup={backup}", flush=True)
print("[UPGRADE READY] Baseline manifest and completed samples preserved; resumable runs retained.", flush=True)
