"""Run CPU-only validation suites without model loading or network access."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
started = time.perf_counter()
for path in sorted((root/'scripts').glob('*.py')):
    tree = ast.parse(path.read_text(), filename=str(path))
    if any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for node in ast.walk(tree)):
        raise AssertionError(f'Production scripts must remain linear: {path}')
suites = sorted((root/'tests').glob('test_*.py'))
print(f'[CPU CHECKS] {len(suites)} suites | no model/GPU/network',flush=True)
results = []
environment = dict(os.environ)
environment.setdefault('RLVR_CPU_THREADS','4')
print(f'[CPU CHECKS ENV] RLVR_CPU_THREADS={environment["RLVR_CPU_THREADS"]} | launcher thread-setting branch enabled',flush=True)
for index,path in enumerate(suites,1):
    result = subprocess.run([sys.executable,'-S',str(path)],cwd=root,env=environment)
    results.append({'suite':path.name,'returncode':result.returncode})
    print(f'[{index:02d}/{len(suites):02d}] {100*index/len(suites):.0f}% | {path.name} | {"PASS" if result.returncode==0 else "FAIL"}',flush=True)
passed=all(item['returncode']==0 for item in results)
print(json.dumps({'passed':passed,'suites':results,'elapsed_seconds':time.perf_counter()-started,
    'gpu_training_tested':False,'network_used':False},indent=2),flush=True)
raise SystemExit(0 if passed else 1)
