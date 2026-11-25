"""Exercise real process supervision with small CPU fixtures, never a GPU model."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

source_root = Path(__file__).resolve().parents[1]
source = source_root / "scripts/run_boundary_parallel.py"
assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(ast.parse(source.read_text())))
checks = 1
worker_source = '''import argparse, json, os, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output'); p.add_argument('--config'); p.add_argument('--num-shards'); p.add_argument('--shard-index', type=int)
p.add_argument('--samples', type=int, default=2); p.add_argument('--resume', action='store_true'); p.add_argument('--plan', action='store_true'); p.add_argument('--cache-only',action='store_true')
a=p.parse_args()
if a.plan:
    print('[FIXTURE PLAN]')
    raise SystemExit(0)
out=Path(a.output)
if a.cache_only:
    raise SystemExit(0 if (out/'manifest.json').is_file() else 3)
out.mkdir(parents=True, exist_ok=True)
(out.parent/f'started-{a.shard_index}').write_text(str(os.getpid()))
if os.environ.get('TEST_PAIR')=='1':
    deadline=time.monotonic()+5
    while not all((out.parent/f'started-{i}').exists() for i in range(2)):
        if time.monotonic()>deadline: raise SystemExit(7)
        time.sleep(.02)
if os.environ.get('TEST_FAIL')=='1':
    if a.shard_index==0: raise SystemExit(9)
    time.sleep(60)
rows=[{'id':f'q{a.shard_index}', 'sample_index':i,'output_tokens':3} for i in range(a.samples)]
(out/'samples.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in rows))
(out/'manifest.json').write_text(json.dumps({'complete':True}))
(out.parent/f'finished-{a.shard_index}').write_text(str(time.monotonic()))
'''
merge_source = '''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--inputs',nargs=2);p.add_argument('--output');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
(out/'samples.jsonl').write_text(''.join((Path(x)/'samples.jsonl').read_text() for x in a.inputs))
'''
analysis_source = '''import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'report.md').write_text('Fixture only')
'''
original_read_text = Path.read_text
original_is_file = Path.is_file
original_term_handler = signal.getsignal(signal.SIGTERM)
with tempfile.TemporaryDirectory(prefix="boundary parallel fixture ") as temporary:
    root = Path(temporary) / "code"
    for folder in ("scripts", "configs", "data"):
        (root / folder).mkdir(parents=True)
    shutil.copyfile(source, root / "scripts/run_boundary_parallel.py")
    (root / "scripts/evaluate_boundary.py").write_text(worker_source)
    (root / "scripts/merge_boundary.py").write_text(merge_source)
    (root / "scripts/analyze_boundary.py").write_text(analysis_source)
    (root / "configs/confirmation_v2.json").write_text(json.dumps({"evaluation":{"n_samples":2},"data":{"path":"data/questions.jsonl"}}))
    (root / "data/questions.jsonl").write_text('{"id":"q0"}\n{"id":"q1"}\n')
    script = root / "scripts/run_boundary_parallel.py"
    for case in ("plan", "parallel", "cached", "sequential", "low_gpu", "low_ram", "failure", "outside", "cpu_quota"):
        gpu_free = (10 if case == "low_gpu" else 22)*1024**3
        ram_kib = (16 if case == "low_ram" else 48)*1024**2
        cuda = SimpleNamespace(is_available=MagicMock(return_value=True), device_count=MagicMock(return_value=1),
            is_bf16_supported=MagicMock(return_value=True), mem_get_info=MagicMock(return_value=(gpu_free,24*1024**3)),
            get_device_name=MagicMock(return_value="CPU fixture, no real GPU"))
        torch = SimpleNamespace(cuda=cuda, set_num_threads=MagicMock())
        args = [str(script), "--output", f"outputs/{'parallel' if case=='cached' else case}"]
        if case == "plan": args += ["--plan"]
        if case == "sequential": args += ["--workers", "1"]
        if case == "outside": args = [str(script), "--output", "../outside"]
        environment = {"TEST_PAIR":"1" if case in ("parallel", "failure") else "0", "TEST_FAIL":"1" if case=="failure" else "0",
                       "RLVR_CPU_THREADS":"4" if case=="cpu_quota" else "2", "WORLD_SIZE":"1","LOCAL_WORLD_SIZE":"1","RANK":"0","LOCAL_RANK":"0"}
        text_reader = lambda path, *a, **kw: f"MemAvailable: {ram_kib} kB\n" if str(path)=="/proc/meminfo" else "100000 100000" if str(path)=="/sys/fs/cgroup/cpu.max" else original_read_text(path,*a,**kw)
        file_checker = lambda path: (case=="cpu_quota" and str(path)=="/sys/fs/cgroup/cpu.max") if str(path).startswith("/sys/") else original_is_file(path)
        error = None
        with patch.dict(os.environ, environment), patch.dict(sys.modules,{"torch":torch}), patch.object(sys,"platform","linux"), \
                patch.object(sys,"argv",args), patch.object(Path,"read_text",text_reader), patch.object(Path,"is_file",file_checker), \
                patch.object(os,"cpu_count",return_value=8), contextlib.redirect_stdout(io.StringIO()):
            try:
                runpy.run_path(str(script),run_name="__main__")
            except (SystemExit, RuntimeError) as caught:
                error = caught
            finally:
                signal.signal(signal.SIGTERM, original_term_handler)
        destination = root / "outputs" / case
        if case == "plan":
            assert isinstance(error,SystemExit) and error.code==0
            assert not destination.exists() and not cuda.is_available.called
        elif case == "cached":
            assert isinstance(error,SystemExit) and error.code==0
            assert not cuda.is_available.called and (root/"outputs/parallel/analysis/report.md").is_file()
        elif case in ("parallel","sequential"):
            assert error is None, error
            invocation = json.loads(next(destination.glob("invocation-*.json")).read_text())
            assert invocation["status"]=="completed" and invocation["worker_exit_codes"]==[0,0]
            assert invocation["new_answers"]==4 and invocation["new_output_tokens"]==12
            assert invocation["concurrent_workers"]==(1 if case=="sequential" else 2)
            assert (destination/"analysis/report.md").is_file()
            for index in range(2):
                assert (destination/f"finished-{index}").is_file()
        elif case=="failure":
            assert isinstance(error,RuntimeError), error
            assert not (destination/"base").exists() and not (destination/"analysis").exists()
            invocation = json.loads(next(destination.glob("invocation-*.json")).read_text())
            assert invocation["status"]=="failed" and 9 in invocation["worker_exit_codes"]
            for index in range(2):
                pid = int((destination/f"started-{index}").read_text())
                try: os.kill(pid,0)
                except ProcessLookupError: pass
                else: raise AssertionError("Failed sibling process was left alive")
        else:
            assert isinstance(error,SystemExit) and error.code!=0, (case,error)
            assert not destination.exists()
        checks += 1
print(f"✅ [BOUNDARY PARALLEL] {checks} supervision checks | two real CPU fixture subprocesses, resource gates and sibling cleanup | no GPU/model executed")
