"""Run isolated CPU validation; never invoke model inference or network access."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
started = time.perf_counter()
for path in sorted((root / "scripts").glob("*.py")):
    tree = ast.parse(path.read_text(), filename=str(path))
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree)):
        raise AssertionError(f"Production scripts must remain linear: {path}")
print("[CHECK 0/4] Python syntax and linear production structure passed", flush=True)
results = []
for index, name in enumerate(("test_verifier_v2.py", "test_boundary_evaluation.py", "test_boundary_merge.py", "test_boundary_parallel.py"), 1):
    result = subprocess.run([sys.executable, "-S", str(root / "tests" / name)], cwd=root)
    results.append({"suite": name, "returncode": result.returncode})
    print(f"[CHECK {index}/4] {100*index/4:.0f}% | {name} | {'PASS' if result.returncode == 0 else 'FAIL'}", flush=True)
passed = all(result["returncode"] == 0 for result in results)
print(json.dumps({"passed": passed, "suites": results, "elapsed_seconds": time.perf_counter()-started,
    "gpu_inference_tested": False, "network_used": False}, indent=2), flush=True)
raise SystemExit(0 if passed else 1)
