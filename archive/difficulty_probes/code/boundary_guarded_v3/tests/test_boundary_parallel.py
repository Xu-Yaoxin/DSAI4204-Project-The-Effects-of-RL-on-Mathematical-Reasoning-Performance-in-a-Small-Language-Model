"""Exercise guarded serial supervision using real CPU subprocess fixtures."""
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
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

source_root = Path(__file__).resolve().parents[1]
source = source_root / 'scripts/run_boundary_parallel.py'
assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(ast.parse(source.read_text())))
checks = 1
worker_source = '''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--config');p.add_argument('--num-shards');p.add_argument('--shard-index',type=int)
p.add_argument('--samples',type=int,default=2);p.add_argument('--resume',action='store_true');p.add_argument('--plan',action='store_true');p.add_argument('--cache-only',action='store_true');a=p.parse_args()
if a.plan:raise SystemExit(0)
out=Path(a.output)
if a.cache_only:
 if os.environ.get('TEST_CASE')=='completed_last_guard' and (out/'samples.jsonl').is_file() and len((out/'samples.jsonl').read_text().splitlines())==a.samples:
  (out/'manifest.json').write_text(json.dumps({'complete':True}))
  (out.parent/f'finished-{a.shard_index}').write_text(str(time.monotonic()))
 raise SystemExit(0 if (out/'manifest.json').is_file() else 3)
out.mkdir(parents=True,exist_ok=True)
with (out.parent/'process_events.jsonl').open('a') as f:f.write(json.dumps({'event':'start','worker':a.shard_index,'time':time.monotonic(),'pid':os.getpid()})+'\\n')
(out.parent/f'started-{a.shard_index}').write_text(str(os.getpid()))
if a.shard_index==1 and not (out.parent/'finished-0').exists():raise SystemExit(98)
case=os.environ.get('TEST_CASE')
if case=='failure':raise SystemExit(9)
if case=='oom':raise SystemExit(137)
if case=='interrupt':time.sleep(60)
(out/'live').mkdir(exist_ok=True)
now=time.time()
(out/'live/current.json').write_text(json.dumps({'id':f'q{a.shard_index}','sample_index':0,'generated_tokens':2,'started_at_unix':now-10,'updated_at_unix':now,'last_token_unix':now,'worker_pid':os.getpid(),'elapsed_seconds':10,'tokens_per_second':.2,'status':'generating','partial_text':'Fixture partial output; no gold'}))
time.sleep(.08)
sample_path=out/'samples.jsonl'
rows=[json.loads(x) for x in sample_path.read_text().splitlines()] if sample_path.exists() else []
if a.shard_index==0 and case=='unsafe75':raise SystemExit(75)
if a.shard_index==0 and case in ('restart','no_token_restart','repeat_restart','restart_limit','completed_last_guard') and (not rows or case in ('restart_limit','completed_last_guard')):
 r={'id':'q0','sample_index':len(rows),'output_tokens':3,'correct':False,'reward':0,'termination_reason':'no_token_timeout' if case=='no_token_restart' else 'repetition_guard' if case=='repeat_restart' else 'answer_timeout'}
 if case in ('no_token_restart','repeat_restart'):r.update(correct=None,reward=None,status='unscored')
 with sample_path.open('a') as f:f.write(json.dumps(r)+'\\n');f.flush();os.fsync(f.fileno())
 raise SystemExit(75)
with sample_path.open('a') as f:
 for i in range(len(rows),a.samples):f.write(json.dumps({'id':f'q{a.shard_index}','sample_index':i,'output_tokens':3,'correct':True,'reward':1,'termination_reason':'eos'})+'\\n')
 f.flush();os.fsync(f.fileno())
(out/'manifest.json').write_text(json.dumps({'complete':True}))
(out.parent/f'finished-{a.shard_index}').write_text(str(time.monotonic()))
with (out.parent/'process_events.jsonl').open('a') as f:f.write(json.dumps({'event':'finish','worker':a.shard_index,'time':time.monotonic(),'pid':os.getpid()})+'\\n')
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
with tempfile.TemporaryDirectory(prefix='guarded serial fixture ') as temporary:
    root = Path(temporary) / 'code'
    for folder in ('scripts','configs','data'):(root/folder).mkdir(parents=True)
    shutil.copyfile(source,root/'scripts/run_boundary_parallel.py')
    (root/'scripts/evaluate_boundary.py').write_text(worker_source)
    (root/'scripts/merge_boundary.py').write_text(merge_source)
    (root/'scripts/analyze_boundary.py').write_text(analysis_source)
    (root/'configs/guarded_v3.json').write_text(json.dumps({'evaluation':{'n_samples':2,'generation_guard':{'max_worker_restarts':2,'answer_timeout_seconds':300}},'data':{'path':'data/questions.jsonl'}}))
    (root/'data/questions.jsonl').write_text('{"id":"q0"}\n{"id":"q1"}\n')
    script = root/'scripts/run_boundary_parallel.py'
    for case in ('plan','serial','cached','reject_parallel','low_gpu','low_ram','failure','oom','outside','cpu_quota','restart','no_token_restart','repeat_restart','restart_limit','completed_last_guard','unsafe75','interrupt'):
        gpu_free=(10 if case=='low_gpu' else 22)*1024**3
        ram_kib=(15 if case=='low_ram' else 48)*1024**2
        cuda=SimpleNamespace(is_available=MagicMock(return_value=True),device_count=MagicMock(return_value=1),is_bf16_supported=MagicMock(return_value=True),mem_get_info=MagicMock(return_value=(gpu_free,24*1024**3)),get_device_name=MagicMock(return_value='CPU fixture'))
        torch=SimpleNamespace(cuda=cuda,set_num_threads=MagicMock())
        args=[str(script),'--output',f"outputs/{'serial' if case=='cached' else case}"]
        if case=='plan':args+=['--plan']
        if case=='reject_parallel':args+=['--workers','2']
        if case=='outside':args=[str(script),'--output','../outside']
        if case=='restart_limit':args+=['--samples','4']
        if case=='completed_last_guard':args+=['--samples','3']
        environment={'TEST_CASE':case,'RLVR_CPU_THREADS':'4' if case=='cpu_quota' else '2','WORLD_SIZE':'1','LOCAL_WORLD_SIZE':'1','RANK':'0','LOCAL_RANK':'0'}
        text_reader=lambda path,*a,**kw: f'MemAvailable: {ram_kib} kB\n' if str(path)=='/proc/meminfo' else '100000 100000' if str(path)=='/sys/fs/cgroup/cpu.max' else original_read_text(path,*a,**kw)
        file_checker=lambda path: (case=='cpu_quota' and str(path)=='/sys/fs/cgroup/cpu.max') if str(path).startswith('/sys/') else original_is_file(path)
        if case=='cached':
            stale=root/'outputs/serial/live/current.json'
            prior=json.loads(stale.read_text());prior['status']='failed';stale.write_text(json.dumps(prior))
        error=None
        timer=None
        with patch.dict(os.environ,environment),patch.dict(sys.modules,{'torch':torch}),patch.object(sys,'platform','linux'),patch.object(sys,'argv',args),patch.object(Path,'read_text',text_reader),patch.object(Path,'is_file',file_checker),patch.object(os,'cpu_count',return_value=8),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            try:
                if case=='interrupt':
                    timer=threading.Timer(1.2,os.kill,args=(os.getpid(),signal.SIGTERM));timer.start()
                runpy.run_path(str(script),run_name='__main__')
            except (SystemExit,RuntimeError,KeyboardInterrupt) as caught:error=caught
            finally:
                if timer:timer.cancel();timer.join()
                signal.signal(signal.SIGTERM,original_term_handler)
        destination=root/'outputs'/case
        if case=='plan':
            assert isinstance(error,SystemExit) and error.code==0
            assert not destination.exists() and not cuda.is_available.called
        elif case=='cached':
            assert isinstance(error,SystemExit) and error.code==0 and not cuda.is_available.called
            assert json.loads((root/'outputs/serial/live/current.json').read_text())['status']=='completed'
        elif case in ('serial','restart','no_token_restart','repeat_restart'):
            assert error is None,(case,error)
            inv=json.loads(next(destination.glob('invocation-*.json')).read_text())
            assert inv['status']=='completed' and inv['concurrent_workers']==1
            assert inv['new_answers']==4 and inv['new_output_tokens']==12
            expected=[75,0,0] if case!='serial' else [0,0]
            assert inv['worker_exit_codes']==expected
            assert inv['restarts_used']==(case!='serial') and (destination/'analysis/report.md').is_file()
            assert [x['worker_index'] for x in inv['worker_attempts']]==([0,0,1] if case!='serial' else [0,1])
            summary=json.loads((destination/'live/current.json').read_text())
            assert summary['status']=='completed' and summary['handled_attempts']==4
            if case!='serial':assert summary['termination_counts']['answer_timeout' if case=='restart' else 'repetition_guard' if case=='repeat_restart' else 'no_token_timeout']==1
            events=[json.loads(x) for x in (destination/'process_events.jsonl').read_text().splitlines()]
            first_end=next(x['time'] for x in events if x['event']=='finish' and x['worker']==0)
            second_start=next(x['time'] for x in events if x['event']=='start' and x['worker']==1)
            assert second_start>=first_end
            if case!='serial':assert 'restart=1' in (destination/'worker_0.log').read_text()
        elif case=='completed_last_guard':
            assert error is None,error
            inv=json.loads(next(destination.glob('invocation-*.json')).read_text())
            assert inv['status']=='completed' and inv['restarts_used']==2
            assert inv['worker_exit_codes']==[75,75,75,0]
            assert [x['worker_index'] for x in inv['worker_attempts']]==[0,0,0,1]
            assert inv['worker_attempts'][2]['post_exit_cache_code']==0
            assert inv['new_answers']==6 and inv['new_output_tokens']==18
            assert (destination/'analysis/report.md').is_file()
            summary=json.loads((destination/'live/current.json').read_text())
            assert summary['handled_attempts']==6 and summary['termination_counts']['answer_timeout']==3
        elif case in ('failure','oom','restart_limit','unsafe75','interrupt'):
            assert isinstance(error,KeyboardInterrupt if case=='interrupt' else RuntimeError),(case,error)
            assert not (destination/'base').exists() and not (destination/'analysis').exists()
            inv=json.loads(next(destination.glob('invocation-*.json')).read_text())
            assert inv['status']==('interrupted' if case=='interrupt' else 'failed')
            assert not (destination/'started-1').exists()
            if case=='restart_limit':
                assert inv['worker_exit_codes']==[75,75,75] and inv['restarts_used']==2 and inv['handled_attempts']==3
                assert json.loads((destination/'live/current.json').read_text())['handled_attempts']==3
            elif case=='unsafe75':assert inv['worker_exit_codes']==[75] and inv['restarts_used']==0
            elif case=='oom':assert inv['worker_exit_codes']==[137] and inv['restarts_used']==0
            for attempt in inv['worker_attempts']:
                try:os.kill(attempt['pid'],0)
                except ProcessLookupError:pass
                else:raise AssertionError('Worker process left alive')
        else:
            assert isinstance(error,SystemExit) and error.code!=0,(case,error)
            assert not destination.exists()
        checks+=1
print(f'✅ [GUARDED SERIAL] {checks} checks | real CPU subprocesses: timeout resume, restart limit, no parallelism, fail-fast and SIGTERM cleanup | no GPU/model')
