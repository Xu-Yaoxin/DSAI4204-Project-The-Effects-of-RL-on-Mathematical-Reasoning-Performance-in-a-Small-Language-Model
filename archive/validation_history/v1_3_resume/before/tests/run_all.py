"""Run model-free correctness checks with the current Python interpreter."""
import ast
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="outputs/validation/v1_2_2/test_report.json")
args = parser.parse_args()
started = time.perf_counter()
scripts = sorted((root / "scripts").glob("*.py"))
for path in scripts:
    tree = ast.parse(path.read_text(), filename=str(path))
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree)):
        raise AssertionError(f"Expected linear scripts without custom function/class encapsulation: {path}")
print(f"✅ [SYNTAX] {len(scripts)} linear Python scripts", flush=True)
tests = sorted((root / "tests").glob("test_*.py"))
results = []
for index, path in enumerate(tests, 1):
    result = subprocess.run([sys.executable, str(path)], cwd=root, text=True, capture_output=True)
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    results.append({"suite": path.name, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    print(f"{'✅' if result.returncode == 0 else '❌'} [TEST {index}/{len(tests)}] {path.name}", flush=True)
passed = all(result["exit_code"] == 0 for result in results)
print(f"{'✅' if passed else '❌'} [CPU CHECKS COMPLETE] {len(tests)} suites | {time.perf_counter()-started:.2f}s | GPU training not tested")
report_path = root / args.output
report_path.parent.mkdir(parents=True, exist_ok=True)
tracked_sources = scripts + tests + [Path(__file__).resolve(), root / "configs/v1.json"]
tracked_sources += sorted(root.glob("*.sh")) + sorted(root.glob("requirements*.txt"))
tracked_sources += [root / "packages/linux-cp311-cu128/manifest.json", root / "packages/linux-cp311-cu128/requirements-lock.txt"]
tracked_sources += [root / "models/qwen3.5-2b-base-manifest.json"]
report_path.write_text(json.dumps({"checked_utc": datetime.now(timezone.utc).isoformat(), "suite_count": len(tests), "passed": passed, "gpu_training_tested": False,
    "python": sys.version, "results": results, "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tracked_sources}}, indent=2) + "\n")
print(f"[SAVED] {report_path}", flush=True)
raise SystemExit(0 if passed else 1)
