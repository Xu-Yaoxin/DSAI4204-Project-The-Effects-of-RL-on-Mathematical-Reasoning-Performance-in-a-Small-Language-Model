"""Real threaded CPU producers exercise streamed evaluation, cache and recovery."""
import ast
import contextlib
import copy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

source_root = Path(__file__).resolve().parents[1]
assert not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in ast.walk(ast.parse((source_root/'scripts/evaluate.py').read_text())))
checks = 1
real_perf = time.perf_counter
real_sleep = time.sleep
producer_state = {}

# These fixture methods simulate the documented Transformers streamer protocol;
# production scripts retain their linear, no-custom-callback implementation.
class Tensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values[0])) if values and isinstance(values[0], list) else (len(values),)
    def to(self, device):
        return self
    def tolist(self):
        return self.values
    def __getitem__(self, key):
        return Tensor(self.values[key[0]][key[1]])

class Tokenizer:
    eos_token_id = 248044
    pad_token_id = 248044
    def __call__(self, prompt, **kwargs):
        assert 'GOLD_CANARY' not in prompt and '731' not in prompt and '732' not in prompt
        producer_state.setdefault('prompts', []).append(prompt)
        length = producer_state.get('prompt_length', 5)
        return {'input_ids': Tensor([[100]*length]), 'attention_mask': Tensor([[1]*length])}
    def decode(self, tokens, **kwargs):
        if not tokens:
            return ''
        return ('<SPECIAL> ' if not kwargs['skip_special_tokens'] else '') + '#### ' + str(tokens[0])

class Criterion:
    def __init__(self, max_time):
        self.max_time = max_time

# The actual producer runs in a ThreadPoolExecutor thread and observes stop state.
def produce(**kwargs):
    assert threading.current_thread() is not threading.main_thread()
    producer_state['calls'] = producer_state.get('calls', 0)+1
    if producer_state.get('fail_call') == producer_state['calls']:
        raise RuntimeError('fixture interruption')
    plan = producer_state['plans'].pop(0)
    streamer = kwargs['streamer']
    criterion = kwargs['stopping_criteria'][0]
    producer_state.setdefault('criteria', []).append(criterion)
    producer_state.setdefault('generation_settings', []).append({k:kwargs[k] for k in ('temperature','top_p','top_k','repetition_penalty','max_new_tokens')})
    prompt = kwargs['input_ids'].tolist()[0]
    streamer.put(Tensor([prompt]))
    tokens = []
    if plan.get('stall'):
        while criterion.max_time != 0:
            real_sleep(.005)
        producer_state['stop_observed'] = True
    for token in plan.get('tokens', []):
        if plan.get('delay'):
            real_sleep(plan['delay'])
        if criterion.max_time == 0 and not plan.get('ignore_stop'):
            producer_state['stop_observed'] = True
            break
        tokens.append(token)
        streamer.put(Tensor([token]))
    streamer.end()
    return Tensor([prompt+tokens])

with tempfile.TemporaryDirectory(prefix='guarded evaluator fixture ') as temporary:
    root = Path(temporary)/'code'
    for folder in ('scripts','configs','data','model'):
        (root/folder).mkdir(parents=True)
    for name in ('evaluate.py','verify_answers.py','fingerprint_model.py','load_model.py','repetition_guard.py','validate_guarded_sample.py','validate_evaluation.py','validate_selection.py'):
        shutil.copyfile(source_root/'scripts'/name, root/'scripts'/name)
    (root/'scripts/doctor.py').write_text("report={'data_ready':True}\n")
    for name in ('config.json','tokenizer.json','tokenizer_config.json'):
        (root/'model'/name).write_text('{}')
    header = json.dumps({'fixture.weight':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
    model_bytes = len(header).to_bytes(8,'little')+header+bytes(4)
    (root/'model/model.safetensors').write_bytes(model_bytes)
    original_run = runpy.run_path
    identity = original_run(str(root/'scripts/fingerprint_model.py'),init_globals={'MODEL_PATH':root/'model'})
    policy = json.loads((source_root/'configs/math_l4.json').read_text())['evaluation']['generation_guard']
    cfg = {'version':'guarded-fixture','model':{'path':'model','dtype':'bfloat16','repo_id':'Qwen/Qwen3.5-2B-Base','expected_identity':identity['model_identity'],'expected_eos_token_id':248044},
           'prompt':{'template':'Solve: {question}\nFinal numerical answer:'},
           'data':{'output_dir':'data'},
           'evaluation':{'seed':271828,'n_samples':2,'dev_samples':2,'engineering_samples':1,'temperature':.6,'top_p':.95,'top_k':0,'max_new_tokens':262144,'context_limit':262144,'length_policy':'native_context_minus_actual_prompt','max_prompt_length':24,'bootstrap_replicates':100,'generation_guard':policy}}
    config_path = root/'configs/math_l4.json'
    config_path.write_text(json.dumps(cfg))
    rows=[{'id':f'q-{i}','question':f'Compute fixture question {i}.','answer':str(731+i),'source_answer':'GOLD_CANARY','group_id':f'g{i}','group_label':f'Group {i}','family_id':f'f{i}','difficulty_label':'fixture','source':{'dataset':'fixture','revision':'pinned','split':'test','subject':'algebra','row_index':i,'original_id':str(i)},'split':'test'} for i in range(1)]
    data_bytes=''.join(json.dumps(r)+'\n' for r in rows).encode()
    data_path=root/'data/test.jsonl'; data_path.write_bytes(data_bytes)
    dataset_manifest={'schema_version':1,'purpose':'formal_base_rl_comparison','kind':'math_level4_splits','files':{'test.jsonl':{'sha256':hashlib.sha256(data_bytes).hexdigest(),'rows':1}},'groups':[{'id':f'g{i}','label':f'Group {i}','count':1} for i in range(2)],'sources':[]}
    (root/'data/manifest.json').write_text(json.dumps(dataset_manifest))
    script=root/'scripts/evaluate.py'
    interpreter=[sys.executable]+(['-S'] if sys.flags.no_site else [])
    command=interpreter+[str(script),'--model','base','--split','test']
    environment={k:'1' if k in ('WORLD_SIZE','LOCAL_WORLD_SIZE') else '0' for k in ('WORLD_SIZE','LOCAL_WORLD_SIZE','RANK','LOCAL_RANK')}
    # Match the launcher even when this test is invoked directly without it.
    environment['RLVR_CPU_THREADS']='4'
    versions={}
    for package in ('torch','transformers','peft','accelerate','datasets','trl','safetensors','flash-linear-attention','fla-core','causal-conv1d'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    tokenizer=Tokenizer()
    model=SimpleNamespace(generate=MagicMock(side_effect=produce),config=SimpleNamespace(max_position_embeddings=262144))
    torch=SimpleNamespace(set_num_threads=MagicMock(),set_num_interop_threads=MagicMock(),cuda=SimpleNamespace(get_device_name=lambda:'CPU fixture',reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:0))
    transformers=SimpleNamespace(set_seed=MagicMock(),MaxTimeCriteria=Criterion)
    loaded={'model':model,'tokenizer':tokenizer,'model_identity':identity['model_identity'],'model_files':identity['model_files'],'adapter_identity':None,'package_versions':versions}
    loader=MagicMock(side_effect=lambda path,**kwargs:loaded if Path(path).name=='load_model.py' else original_run(path,**kwargs))

    for flags in (['--plan'],['--plan','--samples','1']):
        result=subprocess.run(command+flags,text=True,capture_output=True,env={**os.environ,**environment})
        assert result.returncode==0 and 'PLAN COMPLETE' in result.stdout,result.stderr
        assert not (root/'outputs').exists()
        checks+=1
    result=subprocess.run(command+['--resume','--cache-only'],text=True,capture_output=True)
    assert result.returncode==3 and not (root/'outputs').exists();checks+=1
    for mutation in ('hash','prompt_gold','adapter','identity','context','eos','guard'):
        mutated=copy.deepcopy(cfg)
        if mutation=='hash': data_path.write_bytes(data_bytes+b' ')
        elif mutation=='prompt_gold': mutated['prompt']['template']='{question} {answer}'
        elif mutation=='adapter': mutated['model']['adapter_path']='trained'
        elif mutation=='identity': mutated['model']['expected_identity']='invalid'
        elif mutation=='context': mutated['evaluation']['context_limit']=65536
        elif mutation=='eos': mutated['model']['expected_eos_token_id']='248044'
        else: mutated['evaluation']['generation_guard']['check_every_tokens']=0
        config_path.write_text(json.dumps(mutated))
        result=subprocess.run(command+['--plan'],text=True,capture_output=True)
        assert result.returncode!=0,mutation
        config_path.write_text(json.dumps(cfg));data_path.write_bytes(data_bytes);checks+=1

    # Fresh generation, exact seeds and both one-question shards use actual queues.
    outcomes={}
    for name,flags,plans in [('fresh',[],[731,0])]:
        producer_state.clear();producer_state['plans']=[{'tokens':[n,248044]} for n in plans]
        with patch.dict(os.environ,environment),patch.dict(sys.modules,{'torch':torch,'transformers':transformers}),patch('runpy.run_path',loader),patch.object(sys,'argv',[str(script),'--model','base','--split','test','--output',f'outputs/{name}',*flags]),contextlib.redirect_stdout(io.StringIO()):
            original_run(str(script),run_name='__main__')
        dest=root/'outputs'/name
        torch.set_num_threads.assert_called_once_with(4)
        torch.set_num_interop_threads.assert_called_once_with(1)
        assert json.loads((dest/'manifest.json').read_text())['execution_cpu_threads']=={'intraop':4,'interop':1}
        checks+=1
        records=[json.loads(l) for l in (dest/'samples.jsonl').read_text().splitlines()]
        assert [r['correct'] for r in records]==[True,False]
        assert all(r['termination_reason']=='eos' and r['scoring_evaluable'] and r['output_token_ids'][-1]==248044 for r in records)
        assert all(s['temperature']==.6 and s['top_p']==.95 and s['top_k']==0 and s['repetition_penalty']==1 for s in producer_state['generation_settings'])
        assert all(r['sample_seed']==int.from_bytes(hashlib.sha256(f"271828:{r['id']}:{r['sample_index']}".encode()).digest()[:4],'big') for r in records)
        live=json.loads((dest/'live/current.json').read_text());assert live['status']=='saved' and live['generated_tokens']==2
        assert '<SPECIAL>' in (dest/'live/current.txt').read_text() and all('<SPECIAL>' in r['raw_text'] and '<SPECIAL>' not in r['text'] for r in records)
        for chunk_path in (dest/'live').glob('*.tokens.jsonl'):
            chunks=[json.loads(l) for l in chunk_path.read_text().splitlines()]
            assert all(len(chunk['token_ids'])==1 and 'answer' not in chunk and 'question' not in chunk for chunk in chunks)
        outcomes[name]=records
        result=subprocess.run(command+['--output',f'outputs/{name}','--resume','--cache-only',*flags],text=True,capture_output=True)
        assert result.returncode==0 and 'EVAL CACHED' in result.stdout,result.stderr
        checks+=6
    assert all(call.kwargs['init_globals']['ADAPTER_PATH'] is None and call.kwargs['init_globals']['TRAINING'] is False for call in loader.call_args_list if Path(call.args[0]).name=='load_model.py');checks+=1

    # Cached input/score/token/stop tampering is rejected before any GPU loading.
    dest=root/'outputs/fresh';saved_bytes=(dest/'samples.jsonl').read_bytes();saved_manifest=json.loads((dest/'manifest.json').read_text())
    for mutation in ('score','metadata','seed','duplicate','missing','tail','hash','tokens','stop','budget','model','code'):
        samples=copy.deepcopy(outcomes['fresh']); manifest=copy.deepcopy(saved_manifest)
        if mutation=='score': samples[0]['correct']=False
        elif mutation=='metadata': samples[0]['question']='changed'
        elif mutation=='seed': samples[0]['sample_seed']+=1
        elif mutation=='duplicate': samples.append(samples[0])
        elif mutation=='missing': samples.pop()
        elif mutation=='tokens': samples[0]['output_token_ids']=[1,248044,248044]
        elif mutation=='stop': samples[0]['termination_reason']='repetition_guard'
        elif mutation=='budget': samples[0]['generation_limit']-=1
        elif mutation=='model': (root/'model/model.safetensors').write_bytes(model_bytes[:-1]+b'1')
        elif mutation=='code': (root/'scripts/repetition_guard.py').write_text((source_root/'scripts/repetition_guard.py').read_text()+'\n# changed\n')
        current=''.join(json.dumps(r)+'\n' for r in samples).encode()
        if mutation=='tail':current+=b'{"id":';manifest['complete']=False
        manifest['samples_sha256']=hashlib.sha256(current).hexdigest() if mutation!='hash' else '0'*64
        (dest/'samples.jsonl').write_bytes(current);(dest/'manifest.json').write_text(json.dumps(manifest))
        result=subprocess.run(command+['--output','outputs/fresh','--resume','--cache-only'],capture_output=True,text=True)
        assert result.returncode not in (0,3),(mutation,result.stdout,result.stderr)
        (root/'model/model.safetensors').write_bytes(model_bytes)
        shutil.copyfile(source_root/'scripts/repetition_guard.py',root/'scripts/repetition_guard.py')
        checks+=1
    (dest/'samples.jsonl').write_bytes(saved_bytes);(dest/'manifest.json').write_text(json.dumps(saved_manifest))

    # Repetition can finish faster than the consumer. The earliest 1024-token
    # periodic prefix still wins over the queued later EOS, and remains unscored.
    for name,plan,reason,scale in [('repeat',{'tokens':[7]*1056+[248044],'ignore_stop':True},'repetition_guard',1),
                                  ('inactivity',{'stall':True},'no_token_timeout',1800),
                                  ('deadline',{'tokens':list(range(500,1500)),'delay':.015},'answer_timeout',1800)]:
        producer_state.clear();producer_state['plans']=[plan]
        clock_start=real_perf()
        with patch.dict(os.environ,environment),patch.dict(sys.modules,{'torch':torch,'transformers':transformers}),patch('runpy.run_path',loader),patch.object(time,'perf_counter',side_effect=lambda:clock_start+(real_perf()-clock_start)*scale),patch.object(sys,'argv',[str(script),'--model','base','--split','test','--output',f'outputs/{name}','--samples','1']),contextlib.redirect_stdout(io.StringIO()) as logs:
            original_run(str(script),run_name='__main__')
        record=json.loads((root/f'outputs/{name}/samples.jsonl').read_text())
        assert record['termination_reason']==reason and not record['scoring_evaluable'],record
        assert '[TOKEN PROGRESS]' in logs.getvalue()
        if reason=='answer_timeout': assert record['correct'] is False and record['reward']==0 and record['status']=='timeout'
        else: assert record['correct'] is None and record['reward'] is None and record['status']=='unscored'
        if reason=='repetition_guard': assert record['guard_evidence']['detected_at_tokens']==1024
        else: assert producer_state['stop_observed'] and record['timeout_evidence']['elapsed_seconds']>=policy['no_token_timeout_seconds' if reason=='no_token_timeout' else 'answer_timeout_seconds']
        result=subprocess.run(command+['--output',f'outputs/{name}','--samples','1','--resume','--cache-only'],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        checks+=4

    # A failed second attempt leaves one complete first attempt; resume produces
    # only the missing attempt without changing the saved byte prefix.
    producer_state.clear();producer_state.update(plans=[{'tokens':[731,248044]},{'tokens':[0,248044]}],fail_call=2)
    with patch.dict(os.environ,environment),patch.dict(sys.modules,{'torch':torch,'transformers':transformers}),patch('runpy.run_path',loader),patch.object(sys,'argv',[str(script),'--model','base','--split','test','--output','outputs/interrupted']),contextlib.redirect_stdout(io.StringIO()):
        try: original_run(str(script),run_name='__main__')
        except RuntimeError as e: assert 'fixture interruption' in str(e)
        else: raise AssertionError('Failure was not propagated')
    prefix=(root/'outputs/interrupted/samples.jsonl').read_bytes()
    producer_state.clear();producer_state['plans']=[{'tokens':[0,248044]}]
    with patch.dict(os.environ,environment),patch.dict(sys.modules,{'torch':torch,'transformers':transformers}),patch('runpy.run_path',loader),patch.object(sys,'argv',[str(script),'--model','base','--split','test','--output','outputs/interrupted','--resume']),contextlib.redirect_stdout(io.StringIO()):
        original_run(str(script),run_name='__main__')
    assert producer_state['calls']==1 and (root/'outputs/interrupted/samples.jsonl').read_bytes().startswith(prefix);checks+=1

    # A long prompt leaves four native-context tokens. Even a correct-looking
    # partial answer remains unknown when no EOS is generated.
    native_cfg=copy.deepcopy(cfg);native_cfg['evaluation']['max_prompt_length']=262143
    config_path.write_text(json.dumps(native_cfg))
    producer_state.clear();producer_state.update(prompt_length=262140,plans=[{'tokens':[731,1,2,3]}])
    with patch.dict(os.environ,environment),patch.dict(sys.modules,{'torch':torch,'transformers':transformers}),patch('runpy.run_path',loader),patch.object(sys,'argv',[str(script),'--model','base','--split','test','--output','outputs/context','--samples','1']),contextlib.redirect_stdout(io.StringIO()):
        original_run(str(script),run_name='__main__')
    context_record=json.loads((root/'outputs/context/samples.jsonl').read_text())
    assert context_record['termination_reason']=='context_limit' and context_record['truncated'] and context_record['correct'] is None
    assert context_record['generation_limit']==4 and producer_state['generation_settings'][0]['max_new_tokens']==4
    config_path.write_text(json.dumps(cfg));checks+=1
    producer_state.clear();producer_state.update(prompt_length=25,plans=[])
    with patch.dict(os.environ,environment),patch.dict(sys.modules,{'torch':torch,'transformers':transformers}),patch('runpy.run_path',loader),patch.object(sys,'argv',[str(script),'--model','base','--split','test','--output','outputs/prompt-overflow']),contextlib.redirect_stdout(io.StringIO()):
        try: original_run(str(script),run_name='__main__')
        except ValueError as e: assert 'Prompt too long' in str(e)
        else: raise AssertionError('Prompt was silently truncated')
    assert not (root/'outputs/prompt-overflow').exists() and producer_state.get('calls',0)==0
    producer_state.clear();checks+=1

    # This subprocess has a genuinely unresponsive generation thread. It must
    # fsync the timed-out attempt and exit75, avoiding executor shutdown hangs.
    source_text=Path(__file__).read_text()
    classes='\n\n'.join(ast.get_source_segment(source_text,n) for n in ast.parse(source_text).body if isinstance(n,ast.ClassDef))
    driver=root/'hang_fixture.py'
    driver.write_text('''import importlib.metadata,json,runpy,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
producer_state={}
''' + classes + '''
root=Path(__file__).parent
original_run=runpy.run_path
identity=original_run(str(root/'scripts/fingerprint_model.py'),init_globals={'MODEL_PATH':root/'model'})
versions={}
for package in ('torch','transformers','peft','accelerate','datasets','trl','safetensors','flash-linear-attention','fla-core','causal-conv1d'):
    try: versions[package]=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: versions[package]=None
real_perf=time.perf_counter
real_sleep=time.sleep
thread_calls=[]
def hang(**kwargs):
    assert thread_calls==[['intraop',4],['interop',1]],thread_calls
    (root/'thread_calls.json').write_text(json.dumps(thread_calls))
    kwargs['streamer'].put(kwargs['input_ids'])
    while True: real_sleep(.01)
model=SimpleNamespace(generate=hang,config=SimpleNamespace(max_position_embeddings=262144))
loaded={'model':model,'tokenizer':Tokenizer(),'model_identity':identity['model_identity'],'model_files':identity['model_files'],'adapter_identity':None,'package_versions':versions}
torch=SimpleNamespace(set_num_threads=lambda count:thread_calls.append(['intraop',count]),set_num_interop_threads=lambda count:thread_calls.append(['interop',count]),cuda=SimpleNamespace(get_device_name=lambda:'CPU hung-thread fixture',reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:0))
transformers=SimpleNamespace(set_seed=lambda seed:None,MaxTimeCriteria=Criterion)
clock_start=real_perf()
sys.argv=[str(root/'scripts/evaluate.py'),'--model','base','--split','test','--output','outputs/hard-exit']
with patch.dict(sys.modules,{'torch':torch,'transformers':transformers}),patch('runpy.run_path',side_effect=lambda path,**kwargs:loaded if Path(path).name=='load_model.py' else original_run(path,**kwargs)),patch.object(time,'perf_counter',side_effect=lambda:clock_start+(real_perf()-clock_start)*4000):
    original_run(sys.argv[0],run_name='__main__')
''')
    result=subprocess.run(interpreter+[str(driver)],capture_output=True,text=True,timeout=10,env={**os.environ,**environment})
    assert result.returncode==75,(result.returncode,result.stdout,result.stderr)
    assert json.loads((root/'thread_calls.json').read_text())==[['intraop',4],['interop',1]]
    checks+=1
    hard_path=root/'outputs/hard-exit/samples.jsonl';prefix=hard_path.read_bytes()
    hard_record=json.loads(prefix)
    assert hard_record['termination_reason']=='answer_timeout' and hard_record['correct'] is False and hard_record['output_token_ids']==[]
    producer_state.clear();producer_state['plans']=[{'tokens':[731,248044]}]
    with patch.dict(os.environ,environment),patch.dict(sys.modules,{'torch':torch,'transformers':transformers}),patch('runpy.run_path',loader),patch.object(sys,'argv',[str(script),'--model','base','--split','test','--output','outputs/hard-exit','--resume']),contextlib.redirect_stdout(io.StringIO()):
        original_run(str(script),run_name='__main__')
    assert producer_state['calls']==1 and hard_path.read_bytes().startswith(prefix)
    assert len(hard_path.read_text().splitlines())==2
    checks+=3

    # A live-decoding exception while the producer hangs is an infrastructure
    # failure, not a scored timeout. It must exit1 after bounded thread cleanup.
    infrastructure_cfg=copy.deepcopy(cfg)
    infrastructure_cfg['evaluation']['generation_guard'].update(heartbeat_seconds=.01,stop_grace_seconds=.05)
    (root/'configs/infra.json').write_text(json.dumps(infrastructure_cfg))
    infrastructure_driver=driver.read_text().replace("kwargs['streamer'].put(kwargs['input_ids'])", "kwargs['streamer'].put(kwargs['input_ids'])\n    kwargs['streamer'].put(Tensor([731]))")
    infrastructure_driver=infrastructure_driver.replace("return ('<SPECIAL> ' if not kwargs['skip_special_tokens'] else '') + '#### ' + str(tokens[0])", "raise RuntimeError('fixture live decoding failure')")
    infrastructure_driver=infrastructure_driver.replace("'outputs/hard-exit'", "'outputs/infra-hang','--config','configs/infra.json'").replace('(real_perf()-clock_start)*4000','(real_perf()-clock_start)')
    (root/'infra_fixture.py').write_text(infrastructure_driver)
    (root/'thread_calls.json').unlink()
    result=subprocess.run(interpreter+[str(root/'infra_fixture.py')],capture_output=True,text=True,timeout=10,env={**os.environ,**environment})
    assert result.returncode==1 and '[INFRASTRUCTURE FAILURE]' in result.stdout,(result.returncode,result.stdout,result.stderr)
    assert json.loads((root/'thread_calls.json').read_text())==[['intraop',4],['interop',1]]
    checks+=1
    assert 'fixture live decoding failure' in (root/'outputs/infra-hang/infrastructure_errors.jsonl').read_text()
    assert (root/'outputs/infra-hang/samples.jsonl').read_bytes()==b''
    checks+=2

    # A real streamed RL development run passes the adapter to the loader,
    # records its bytes and rejects a changed checkpoint before GPU reuse.
    dev_rows=[dict(r,split='dev') for r in rows]
    dev_bytes=''.join(json.dumps(r)+'\n' for r in dev_rows).encode()
    (root/'data/dev.jsonl').write_bytes(dev_bytes)
    dataset_manifest['files']['dev.jsonl']={'sha256':hashlib.sha256(dev_bytes).hexdigest(),'rows':len(dev_rows)}
    (root/'data/manifest.json').write_text(json.dumps(dataset_manifest))
    adapter=root/'outputs/adapter';adapter.mkdir()
    (adapter/'adapter_config.json').write_text('{}')
    (adapter/'adapter_model.safetensors').write_bytes(b'fixture adapter')
    adapter_digest=hashlib.sha256()
    for filename in ('adapter_config.json','adapter_model.safetensors'):
        adapter_digest.update(filename.encode());adapter_digest.update((adapter/filename).read_bytes())
    loaded_rl=dict(loaded,adapter_identity=adapter_digest.hexdigest())
    rl_loader=MagicMock(side_effect=lambda path,**kwargs:loaded_rl if Path(path).name=='load_model.py' else original_run(path,**kwargs))
    producer_state.clear();producer_state['plans']=[{'tokens':[731,248044]},{'tokens':[0,248044]}]
    with patch.dict(os.environ,environment),patch.dict(sys.modules,{'torch':torch,'transformers':transformers}),patch('runpy.run_path',rl_loader),patch.object(sys,'argv',[str(script),'--model','rl','--split','dev','--adapter','outputs/adapter','--output','outputs/dev-rl']),contextlib.redirect_stdout(io.StringIO()):
        original_run(str(script),run_name='__main__')
    assert all(call.kwargs['init_globals']['ADAPTER_PATH']==adapter.resolve() for call in rl_loader.call_args_list if Path(call.args[0]).name=='load_model.py')
    rl_manifest=json.loads((root/'outputs/dev-rl/manifest.json').read_text())
    assert rl_manifest['model_label']=='rl' and rl_manifest['adapter_identity']==adapter_digest.hexdigest()
    result=subprocess.run(command+['--model','rl','--split','dev','--adapter','outputs/adapter','--output','outputs/dev-rl','--resume','--cache-only'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    (adapter/'adapter_model.safetensors').write_bytes(b'changed adapter')
    result=subprocess.run(command+['--model','rl','--split','dev','--adapter','outputs/adapter','--output','outputs/dev-rl','--resume','--cache-only'],capture_output=True,text=True)
    assert result.returncode!=0
    checks+=4

    # Explicit Base/RL routing and serial-only guards fail before model loading.
    for flags in (['--num-shards','2'], ['--model','base','--adapter','outputs/adapter'], ['--model','rl'], ['--model','rl','--adapter','outputs/adapter']):
        result=subprocess.run(command+flags+['--plan'],capture_output=True,text=True)
        assert result.returncode!=0,flags
        checks+=1
    result=subprocess.run(command+['--model','rl','--selection','outputs/selection.json','--plan'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    checks+=1

print(f'✅ [MATH EVALUATION] {checks} streaming, timeout, serial routing and resume checks; no GPU/model/network execution')
