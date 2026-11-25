"""Run one serial MATH experiment with sealed training resume and guarded evaluations."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage', nargs='?', default='all', choices=('all','plan','prepare','train','evaluate','analyze'))
parser.add_argument('--config',default='configs/math_l4.json')
parser.add_argument('--run-dir',default='outputs/math_l4_v1')
args = parser.parse_args()
config_path = (root/args.config).resolve()
if not config_path.is_relative_to(root):raise SystemExit('Configuration must stay within package')
cfg = json.loads(config_path.read_text())
run = (root/args.run_dir).absolute()
if any(p.is_symlink() for p in (run,*run.parents)):raise SystemExit('Run path must not contain symlinks')
run = run.resolve()
if not run.is_relative_to(root/'outputs') or run==root/'outputs':raise SystemExit('Use a dedicated directory under outputs/')
config_arg = str(config_path.relative_to(root))
run_arg = str(run.relative_to(root))
steps_per_epoch = cfg['data']['train_size']*cfg['training']['num_generations']//(cfg['training']['per_device_train_batch_size']*cfg['training']['gradient_accumulation_steps'])
checkpoint_steps = [steps_per_epoch*epoch for epoch in range(1,cfg['training']['epochs']+1)]
budget = {'train_questions':cfg['data']['train_size'],'dev_questions':cfg['data']['dev_size'],'test_questions':cfg['data']['test_size'],
    'training_updates':steps_per_epoch*cfg['training']['epochs'],'training_completion_cap':cfg['training']['max_completion_length'],
    'dev_samples_per_question':cfg['evaluation']['dev_samples'],'test_samples_per_question':cfg['evaluation']['n_samples'],
    'dev_responses_all_candidates':cfg['data']['dev_size']*cfg['evaluation']['dev_samples']*len(checkpoint_steps),
    'test_responses_base_and_rl':2*cfg['data']['test_size']*cfg['evaluation']['n_samples'],
    'evaluation_deadline_seconds':cfg['evaluation']['generation_guard']['answer_timeout_seconds'],
    'concurrent_model_workers':1,'epoch_checkpoints':checkpoint_steps}
print('[EXPERIMENT PLAN] '+json.dumps(budget),flush=True)
if args.stage=='plan':
    subprocess.run([sys.executable,'-S',str(root/'scripts/preflight.py'),'--config',config_arg,'--cpu-only'],cwd=root,check=True)
    for mode in ('engineering','pilot','formal'):
        subprocess.run([sys.executable,'-S',str(root/'scripts/train_grpo.py'),'--config',config_arg,'--mode',mode,'--output',f'{run_arg}/{mode}','--plan'],cwd=root,check=True)
    for split,samples in (('engineering',cfg['evaluation']['engineering_samples']),('dev',cfg['evaluation']['dev_samples']),('test',cfg['evaluation']['n_samples'])):
        subprocess.run([sys.executable,'-S',str(root/'scripts/evaluate.py'),'--config',config_arg,'--model','base','--split',split,'--samples',str(samples),'--output',f'{run_arg}/plan_{split}','--plan'],cwd=root,check=True)
    print('✅ [PLAN COMPLETE] No model inference/training and no experiment output created',flush=True)
    raise SystemExit(0)
if sys.platform!='linux':raise SystemExit('Run GPU stages on the Linux server; use plan locally')
(root/'outputs').mkdir(exist_ok=True)
global_lock_path=root/'outputs/.single_gpu_experiment.lock'
if global_lock_path.is_symlink():raise SystemExit('Unsafe global experiment lock')
global_lock_stream=global_lock_path.open('a')
fcntl.flock(global_lock_stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
run.mkdir(parents=True,exist_ok=True)
lock=run/'.experiment.lock'
if lock.is_symlink():raise SystemExit('Unsafe experiment lock')
lock_stream=lock.open('a')
fcntl.flock(lock_stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
identity={'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
    'bundle_sha256':hashlib.sha256((root/'bundle_manifest.json').read_bytes()).hexdigest()}
state_path=run/'run_state.json'
if state_path.is_symlink() or state_path.with_suffix('.json.tmp').is_symlink():
    raise SystemExit('Unsafe state or temporary state path')
if state_path.is_file():
    prior=json.loads(state_path.read_text())
    if prior['identity']!=identity:raise SystemExit('Existing run has a different frozen protocol; choose a new run directory')
state={'identity':identity,'status':'running','pid':os.getpid(),'stage_requested':args.stage,'started_utc':datetime.now(timezone.utc).isoformat(),'budget':budget,'completed_commands':[]}
job_list=[("BUNDLE AND DATA","cpu",[sys.executable,"-S","scripts/preflight.py","--config",config_arg,"--cpu-only"],None)]
if args.stage in ('all','prepare','train','evaluate'):
    job_list += [('CPU CHECKS','cpu',[sys.executable,'-S','tests/run_all.py'],None),
        ('ENVIRONMENT','cpu',[sys.executable,'scripts/preflight.py','--config',config_arg,'--output',f'{run_arg}/environment/{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}.json'],None)]
if args.stage in ('all','train'):
    job_list.append(('ENGINEERING BASE','eval',[sys.executable,'scripts/evaluate.py','--config',config_arg,'--model','base','--split','engineering','--samples',str(cfg['evaluation']['engineering_samples']),'--output',f'{run_arg}/engineering_base','--resume'],run/'engineering_base'))
    for mode in ('engineering','pilot','formal'):
        command=[sys.executable,'scripts/train_grpo.py','--config',config_arg,'--mode',mode,'--output',f'{run_arg}/{mode}','--resume']
        if mode=='formal':command+=['--pilot-report',f'{run_arg}/pilot/report.json']
        job_list.append((mode.upper()+' TRAIN','train',command,run/mode))
if args.stage in ('all','evaluate'):
    job_list.append(('VALIDATE FORMAL','formal_guard',[],run/'formal'))
    for step in checkpoint_steps:
        job_list.append((f'DEV CHECKPOINT {step}','eval',[sys.executable,'scripts/evaluate.py','--config',config_arg,'--model','rl','--split','dev','--samples',str(cfg['evaluation']['dev_samples']),'--adapter',f'{run_arg}/formal/checkpoint-{step}','--output',f'{run_arg}/dev/checkpoint-{step}','--resume'],run/'dev'/f'checkpoint-{step}'))
    job_list.append(('FREEZE SELECTION','cpu',[sys.executable,'-S','scripts/select_checkpoint.py','--config',config_arg,'--training-report',f'{run_arg}/formal/report.json','--dev-root',f'{run_arg}/dev','--output',f'{run_arg}/selection.json'],None))
    job_list.append(('TEST BASE','eval',[sys.executable,'scripts/evaluate.py','--config',config_arg,'--model','base','--split','test','--samples',str(cfg['evaluation']['n_samples']),'--output',f'{run_arg}/test_base','--resume'],run/'test_base'))
    job_list.append(('TEST RL','eval',[sys.executable,'scripts/evaluate.py','--config',config_arg,'--model','rl','--split','test','--samples',str(cfg['evaluation']['n_samples']),'--selection',f'{run_arg}/selection.json','--output',f'{run_arg}/test_rl','--resume'],run/'test_rl'))
if args.stage in ('all','evaluate','analyze'):
    job_list.append(('PAIRED ANALYSIS','cpu',[sys.executable,'-S','scripts/analyze.py','--config',config_arg,'--base',f'{run_arg}/test_base','--rl',f'{run_arg}/test_rl','--output',f'{run_arg}/comparison'],None))
child=None
signal.signal(signal.SIGTERM,signal.default_int_handler)
started=time.monotonic()
try:
    for index,(name,kind,command,stage_output) in enumerate(job_list,1):
        state.update(phase=name,phase_index=index,phase_count=len(job_list),phase_kind=kind,phase_output=str(stage_output.relative_to(root)) if stage_output else None)
        print('\n'+'='*72,flush=True)
        print(f'[{index:02d}/{len(job_list):02d}] {name} | {100*(index-1)/len(job_list):.0f}% | single GPU, serial',flush=True)
        if kind=='formal_guard':
            report=json.loads((stage_output/'report.json').read_text())
            if report.get('status')!='completed' or report.get('mode')!='formal' or not all(report.get('gates',{}).values()) or report.get('gates',{}).get('all_passed') is not True:
                raise RuntimeError('A completed accepted formal training report is required before checkpoint selection')
            if report.get('optimizer_steps')!=steps_per_epoch*cfg['training']['epochs']:
                raise RuntimeError('Formal optimizer steps differ from the frozen plan')
            steps=sorted(entry['step'] for entry in report.get('checkpoints',[]))
            if steps!=checkpoint_steps:raise RuntimeError('Formal report epoch checkpoints differ from the plan')
            continue
        restarts=0
        while True:
            before=0
            sample_path=stage_output/'samples.jsonl' if kind=='eval' else None
            if sample_path and sample_path.is_file():
                before=sum(bool(line.strip()) for line in sample_path.read_text().splitlines())
            print('$ '+' '.join(command),flush=True)
            child=subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,start_new_session=True)
            state.update(child_pid=child.pid,updated_at_unix=time.time(),eval_restarts=restarts)
            phase_started=time.monotonic()
            last_heartbeat=0
            last_console=0
            while child.poll() is None:
                now=time.monotonic()
                if now-last_heartbeat>=5:
                    state.update(updated_at_unix=time.time(),elapsed_seconds=now-started,phase_elapsed_seconds=now-phase_started)
                    state.pop('answer',None)
                    if kind=='eval' and (stage_output/'live/current.json').is_file():
                        live=json.loads((stage_output/'live/current.json').read_text())
                        if live.get('worker_pid')==child.pid:state['answer']=live
                    temporary=state_path.with_suffix('.json.tmp')
                    temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(state_path)
                    if now-last_console>=15 and kind=='train':
                        print(f'[TRAIN HEARTBEAT] {name} | phase_elapsed={now-phase_started:.0f}s | pid={child.pid}',flush=True)
                        last_console=now
                    last_heartbeat=now
                time.sleep(.5)
            exit_code=child.returncode
            child=None
            if exit_code==0:break
            if kind!='eval' or exit_code!=75:
                raise RuntimeError(f'{name} exited {exit_code}. Saved work is preserved; rerun the same command after resolving the error')
            records=[json.loads(line) for line in sample_path.read_text().splitlines() if line.strip()] if sample_path.is_file() else []
            if len(records)<=before or records[-1].get('termination_reason') not in ('answer_timeout','repetition_guard','no_token_timeout'):
                raise RuntimeError('Refusing guarded restart without a new durable stop record')
            probe=subprocess.run([*command,'--cache-only'],cwd=root)
            if probe.returncode==0:break
            if probe.returncode!=3:raise RuntimeError('Guarded evaluation cache failed validation')
            if restarts>=cfg['execution']['max_eval_worker_restarts']:
                raise RuntimeError('Evaluation restart limit reached; inspect the logs before resuming')
            restarts+=1
            print(f'[GUARDED RESUME] {name} | restart={restarts} | preserved={len(records)}',flush=True)
        state['completed_commands'].append({'phase':name,'command':command,'finished_utc':datetime.now(timezone.utc).isoformat()})
    state.pop('answer',None)
    state.update(status='completed',phase='COMPLETE',child_pid=None,updated_at_unix=time.time(),elapsed_seconds=time.monotonic()-started)
    result_location=run/'comparison' if args.stage in ('all','evaluate','analyze') else state_path
    print(f'✅ [EXPERIMENT COMPLETE] stage={args.stage} | [SAVED] {result_location}',flush=True)
except BaseException as error:
    state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error),updated_at_unix=time.time())
    raise
finally:
    if child is not None and child.poll() is None:
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
    state['child_pid']=None
    temporary=state_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(state_path)
    lock_stream.close()
    global_lock_stream.close()
