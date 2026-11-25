"""Exercise the serial experiment coordinator with actual CPU subprocess fixtures."""
import ast
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import time

source_root = Path(__file__).resolve().parents[1]
source = source_root / 'scripts/run_experiment.py'
assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(ast.parse(source.read_text())))
checks = 1
fixture_source = r'''import json,os,sys,time
from pathlib import Path
name=Path(__file__).stem
args=sys.argv[1:]
root=Path.cwd()
case=os.environ.get('FIXTURE_CASE','normal')
log=root/'events.jsonl'
values={}
for i,arg in enumerate(args):
 if arg.startswith('--') and i+1<len(args) and not args[i+1].startswith('--'):values[arg]=args[i+1]
plan='--plan' in args
cache='--cache-only' in args
label=name
if name=='preflight':label+=':cpu' if '--cpu-only' in args else ':gpu'
if name=='train_grpo':label+=':'+values['--mode']
if name=='evaluate':label+=':'+values['--split']+':'+values['--model']
if plan:label+=':plan'
if cache:label+=':cache'
with log.open('a') as f:f.write(json.dumps({'event':'start','label':label,'pid':os.getpid(),'time':time.monotonic(),'args':args})+'\n')
lock=root/'active_job'
fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
os.write(fd,str(os.getpid()).encode());os.close(fd)
exit_code=0
try:
 if plan:pass
 elif name=='run_all' and case=='fail_checks':exit_code=9
 elif name=='train_grpo':
  if values['--mode']=='pilot' and case=='fail_pilot':exit_code=9
  else:
   out=root/values['--output'];out.mkdir(parents=True,exist_ok=True)
   if values['--mode']=='formal':
    cfg=json.loads((root/'configs/math_l4.json').read_text())
    per_epoch=cfg['data']['train_size']*cfg['training']['num_generations']//(cfg['training']['per_device_train_batch_size']*cfg['training']['gradient_accumulation_steps'])
    report={'status':'completed','mode':'formal','optimizer_steps':per_epoch*cfg['training']['epochs'],'gates':{'all_passed':case!='bad_formal'},'checkpoints':[{'step':per_epoch*i} for i in range(1,cfg['training']['epochs']+1)]}
    (out/'report.json').write_text(json.dumps(report))
 elif name=='evaluate':
  out=root/values['--output'];out.mkdir(parents=True,exist_ok=True)
  sample=out/'samples.jsonl'
  rows=[json.loads(line) for line in sample.read_text().splitlines()] if sample.exists() else []
  engineering=values['--split']=='engineering'
  if cache:
   exit_code=4 if case=='bad_cache' else 0 if (out/'complete').exists() else 3
  elif engineering and case=='interrupt':
   (root/'active_eval_pid').write_text(str(os.getpid()))
   time.sleep(60)
  elif engineering and case=='unsafe75':exit_code=75
  elif engineering and case=='eval_failure':exit_code=137
  elif engineering and case in ('restart','bad_cache','restart_limit','complete_after_limit','complete_no_restart') and (not rows or case in ('restart_limit','complete_after_limit')):
   row={'id':'fixture','sample_index':len(rows),'correct':False,'reward':0.0,'termination_reason':'answer_timeout'}
   with sample.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
   if case=='complete_no_restart' or (case=='complete_after_limit' and len(rows)==2):(out/'complete').write_text('done')
   exit_code=75
  else:
   if not (out/'complete').exists():
    with sample.open('a') as f:f.write(json.dumps({'id':'fixture','sample_index':len(rows),'correct':True,'reward':1.0,'termination_reason':'eos'})+'\n')
   (out/'complete').write_text('done')
 elif name=='select_checkpoint':
  out=root/values['--output'];out.parent.mkdir(parents=True,exist_ok=True);out.write_text('{}')
 elif name=='analyze':
  out=root/values['--output'];out.mkdir(parents=True,exist_ok=True);(out/'report.md').write_text('fixture')
 elif name=='preflight' and '--output' in values:
  out=root/values['--output'];out.parent.mkdir(parents=True,exist_ok=True);out.write_text('{}')
finally:
 lock.unlink(missing_ok=True)
 with log.open('a') as f:f.write(json.dumps({'event':'finish','label':label,'pid':os.getpid(),'time':time.monotonic(),'exit':exit_code})+'\n')
raise SystemExit(exit_code)
'''
bootstrap = "import runpy,sys; sys.platform='linux'; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
with tempfile.TemporaryDirectory(prefix='math pipeline fixture ') as temporary:
    base = Path(temporary)
    for case in ('plan','normal','prepare','train','analyze','fail_checks','fail_pilot','bad_formal','unsafe75','eval_failure','bad_cache','restart','restart_limit','complete_after_limit','complete_no_restart','outside','outside_config','state_symlink','tmp_symlink','interrupt'):
        root=base/case
        for directory in ('scripts','tests','configs'):(root/directory).mkdir(parents=True)
        shutil.copyfile(source,root/'scripts/run_experiment.py')
        for name in ('preflight','train_grpo','evaluate','select_checkpoint','analyze'):(root/'scripts'/f'{name}.py').write_text(fixture_source)
        (root/'tests/run_all.py').write_text(fixture_source)
        cfg=json.loads((source_root/'configs/math_l4.json').read_text())
        cfg['data'].update(train_size=8,dev_size=2,test_size=2)
        if case=='complete_no_restart':cfg['execution']['max_eval_worker_restarts']=0
        (root/'configs/math_l4.json').write_text(json.dumps(cfg))
        (root/'bundle_manifest.json').write_text('{}')
        stage=case if case in ('plan','prepare','train','analyze') else 'all'
        args=[str(root/'scripts/run_experiment.py'),stage]
        output=root/'outputs/math_l4_v1'
        if case=='outside':args+=['--run-dir','../outside']
        if case=='outside_config':
            external=base/'external.json';external.write_text(json.dumps(cfg));args+=['--config',str(external)]
        if case in ('state_symlink','tmp_symlink'):
            output.mkdir(parents=True)
            external=base/f'{case}-untouched.txt';external.write_text('sentinel')
            (output/('run_state.json' if case=='state_symlink' else 'run_state.json.tmp')).symlink_to(external)
        env=dict(os.environ,FIXTURE_CASE=case,PYTHONUNBUFFERED='1')
        command=[sys.executable,'-S','-c',bootstrap,*args]
        if case=='interrupt':
            handle=(root/'captured.txt').open('w')
            process=subprocess.Popen(command,cwd=root,env=env,stdout=handle,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+20
            while not (root/'active_eval_pid').exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.05)
            assert (root/'active_eval_pid').exists(),(root/'captured.txt').read_text()
            child_pid=int((root/'active_eval_pid').read_text())
            second=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=10)
            assert second.returncode!=0 and 'temporarily unavailable' in second.stderr.lower(),second.stderr
            checks+=1
            other=subprocess.run(command+['--run-dir','outputs/other'],cwd=root,env=env,capture_output=True,text=True,timeout=10)
            assert other.returncode!=0 and 'temporarily unavailable' in other.stderr.lower(),other.stderr
            assert not (root/'outputs/other/run_state.json').exists()
            checks+=1
            process.send_signal(signal.SIGTERM);process.wait(timeout=20);handle.close()
            try:os.kill(child_pid,0)
            except ProcessLookupError:pass
            else:raise AssertionError('Worker survived coordinator SIGTERM')
            state=json.loads((output/'run_state.json').read_text())
            assert state['status']=='interrupted' and state['child_pid'] is None
            checks+=1
            continue
        result=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=40)
        events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()] if (root/'events.jsonl').exists() else []
        starts=[event['label'] for event in events if event['event']=='start']
        successful=case in ('plan','normal','prepare','train','analyze','restart','complete_after_limit','complete_no_restart')
        assert (result.returncode==0)==successful,(case,result.stdout,result.stderr)
        active=set()
        for event in events:
            if event['event']=='start':
                assert not active,(case,event,active)
                active.add(event['pid'])
            else:
                assert event['pid'] in active
                active.remove(event['pid'])
        assert not active
        if case=='plan':
            assert not output.exists()
            assert starts==['preflight:cpu','train_grpo:engineering:plan','train_grpo:pilot:plan','train_grpo:formal:plan','evaluate:engineering:base:plan','evaluate:dev:base:plan','evaluate:test:base:plan']
        elif case in ('outside','outside_config','state_symlink','tmp_symlink'):
            assert not events
            if case in ('state_symlink','tmp_symlink'):assert external.read_text()=='sentinel'
        else:
            state=json.loads((output/'run_state.json').read_text())
            assert state['status']==('completed' if successful else 'failed')
            if case=='normal':
                assert starts==['preflight:cpu','run_all','preflight:gpu','evaluate:engineering:base','train_grpo:engineering','train_grpo:pilot','train_grpo:formal','evaluate:dev:rl','evaluate:dev:rl','evaluate:dev:rl','select_checkpoint','evaluate:test:base','evaluate:test:rl','analyze'],starts
                for event in events:
                    if event['event']=='start' and event['label'].startswith(('train_grpo:','evaluate:')):assert '--resume' in event['args']
                    if event['event']=='start' and event['label']=='analyze':assert '--config' in event['args'] and event['args'][event['args'].index('--config')+1]=='configs/math_l4.json'
                earlier=(output/'run_state.json').read_bytes()
                cfg['training']['learning_rate']*=2
                (root/'configs/math_l4.json').write_text(json.dumps(cfg))
                rerun=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True,timeout=10)
                assert rerun.returncode!=0 and 'different frozen protocol' in rerun.stderr
                assert (output/'run_state.json').read_bytes()==earlier
                checks+=1
            elif case=='prepare':assert starts==['preflight:cpu','run_all','preflight:gpu']
            elif case=='train':assert 'select_checkpoint' not in starts and not any(':test:' in x for x in starts)
            elif case=='analyze':assert starts==['preflight:cpu','analyze']
            elif case=='fail_checks':assert starts==['preflight:cpu','run_all']
            elif case=='fail_pilot':assert starts[-1]=='train_grpo:pilot' and 'train_grpo:formal' not in starts
            elif case=='bad_formal':assert starts[-1]=='train_grpo:formal' and 'select_checkpoint' not in starts
            elif case in ('unsafe75','eval_failure'):assert starts[-1]=='evaluate:engineering:base' and not any(x.endswith(':cache') for x in starts)
            elif case=='bad_cache':assert starts[-1]=='evaluate:engineering:base:cache' and starts.count('evaluate:engineering:base')==1
            elif case=='restart':
                assert starts.count('evaluate:engineering:base')==2 and starts.count('evaluate:engineering:base:cache')==1
                records=[json.loads(line) for line in (output/'engineering_base/samples.jsonl').read_text().splitlines()]
                assert [r['sample_index'] for r in records]==[0,1]
            elif case=='restart_limit':assert starts.count('evaluate:engineering:base')==3 and 'train_grpo:engineering' not in starts
            elif case=='complete_after_limit':assert starts.count('evaluate:engineering:base')==3 and starts.count('evaluate:engineering:base:cache')==3
            elif case=='complete_no_restart':assert starts.count('evaluate:engineering:base')==1 and starts.count('evaluate:engineering:base:cache')==1
        checks+=1
    reward_path=base/'reward_audit.jsonl'
    request={'completions':['#### 17']*8,'group_size':8,'expected_batch_size':8,'prompts':['p']*8,'answer':['17']*8,'question_id':['q']*8,
        'completion_ids':[[8,248044] if i%2==0 else [8] for i in range(8)],'eos_token_ids':[248044],'max_completion_length':2048,
        'audit_path':str(reward_path),'rank':0,'step':0}
    rewards=runpy.run_path(str(source_root/'scripts/train_grpo.py'),init_globals={'REWARD_REQUEST':request})['REWARD_VALUES']
    assert rewards==[1.,0.,1.,0.,1.,0.,1.,0.]
    audit=json.loads(reward_path.read_text())
    assert all(s['verifier']['status']=='unfinished' and s['verifier']['extracted'] is None for s in audit['samples'][1::2])
    checks+=1
print(f'✅ [MATH PIPELINE] {checks} CPU checks | real subprocess ordering, single-active-job lock, staged execution, durable75/cache recovery, restart limits, fail-fast, SIGTERM cleanup, path/protocol guards and EOS-only training rewards | no GPU/model')
