"""Temporary CPU fixtures test dev-only selection and paired guarded inference statistics."""
import copy
import hashlib
import json
import math
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

source=Path(__file__).resolve().parents[1]
checks=0
with tempfile.TemporaryDirectory(prefix='math-paired-fixture-') as tmp:
    root=Path(tmp)/'code'
    for name in ('scripts','configs','data/processed','outputs/formal','outputs/dev'):(root/name).mkdir(parents=True)
    for path in (source/'scripts').glob('*.py'):shutil.copy2(path,root/'scripts'/path.name)
    cfg=json.loads((source/'configs/math_l4.json').read_text())
    model_files={'fixture.config':'a'*64,'fixture.weights':'b'*64}
    cfg['model']['expected_identity']=hashlib.sha256(json.dumps(model_files,sort_keys=True).encode()).hexdigest()
    cfg['training']['epochs']=2
    cfg['data'].update(train_size=8,pilot_size=4,engineering_size=4,dev_size=2,test_size=2)
    cfg['evaluation'].update(dev_samples=4,n_samples=16,bootstrap_replicates=100)
    (root/'configs/math_l4.json').write_text(json.dumps(cfg))
    data={}
    for split,size in (('train',8),('dev',2),('test',2)):
        data[split]=[{'id':f'{split}-{i}','question':f'Compute {split} fixture {i}.','answer':str(731+i),
            'source_answer':str(731+i),'split':split,'family_id':f'{split}-family-{i}',
            'group_id':'math_l4','group_label':'MATH Level4','difficulty_label':'Level 4',
            'source':{'subject':'algebra','split':'test' if split=='test' else 'train','dataset':'fixture','revision':'pinned'}} for i in range(size)]
    for split in ('pilot','engineering'):data[split]=[dict(r,split=split) for r in data['train'][:4]]
    dm={'schema_version':1,'kind':'math_level4_splits','purpose':'formal_base_rl_comparison','files':{}}
    for split,rows in data.items():
        content=''.join(json.dumps(r)+'\n' for r in rows).encode();(root/f'data/processed/{split}.jsonl').write_bytes(content)
        dm['files'][split+'.jsonl']={'sha256':hashlib.sha256(content).hexdigest(),'rows':len(rows)}
    (root/'data/processed/manifest.json').write_text(json.dumps(dm))
    dmh=hashlib.sha256((root/'data/processed/manifest.json').read_bytes()).hexdigest()
    settings={k:cfg['evaluation'][k] for k in ('temperature','top_p','top_k','max_new_tokens','max_prompt_length','seed','context_limit','length_policy','generation_guard')}
    settings.update(prompt_template=cfg['prompt']['template'],dtype='bfloat16',backend='transformers-sdpa',batch_size=1,world_size=1,add_special_tokens=True,
        stopping_rule='tokenizer-eos-or-native-context-or-guard',repetition_penalty=1.0,eos_token_id=248044,package_versions={'torch':'fixture'})
    code_hashes={name:hashlib.sha256((root/'scripts'/name).read_bytes()).hexdigest() for name in ('evaluate.py','verify_answers.py','load_model.py','fingerprint_model.py','repetition_guard.py','validate_guarded_sample.py','doctor.py','validate_evaluation.py','validate_selection.py')}
    proto={'training':cfg['training']}
    for field,name in (('verifier_sha256','verify_answers.py'),('training_script_sha256','train_grpo.py'),('loader_sha256','load_model.py'),('fingerprint_sha256','fingerprint_model.py'),('doctor_sha256','doctor.py'),('resume_helper_sha256','training_resume.py'),('checkpoint_event_sha256','training_checkpoint_event.py')):
        proto[field]=hashlib.sha256((root/'scripts'/name).read_bytes()).hexdigest()
    protocol_hash=hashlib.sha256(json.dumps(proto,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    tm={'config':cfg,'mode':'formal','fresh_adapter_from_base':True,'model_identity':cfg['model']['expected_identity'],
        'config_training_hash':protocol_hash,'training_protocol':proto,'resume_binding':{'fixture':'sealed'}}
    (root/'outputs/formal/manifest.json').write_text(json.dumps(tm))
    report={'mode':'formal','status':'completed','gates':{'all_passed':True,'adapter_updated':True},'model_identity':cfg['model']['expected_identity'],
        'config_training_hash':protocol_hash,'prepared_manifest_sha256':dmh,'package_versions':settings['package_versions'],
        'split_sha256':{s:dm['files'][s+'.jsonl']['sha256'] for s in data},'checkpoints':[{'step':s,'path':f'outputs/formal/checkpoint-{s}'} for s in (2,4)],
        'optimizer_steps':4,'completion_files':{'manifest.json':hashlib.sha256((root/'outputs/formal/manifest.json').read_bytes()).hexdigest()}}
    (root/'outputs/formal/report.json').write_text(json.dumps(report))
    adapters={}
    for step in (2,4):
        path=root/f'outputs/formal/checkpoint-{step}';path.mkdir()
        (path/'adapter_config.json').write_text('{}');(path/'adapter_model.safetensors').write_bytes(f'fixture adapter {step}'.encode())
        files={};digest=hashlib.sha256()
        for name in ('adapter_config.json','adapter_model.safetensors'):
            content=(path/name).read_bytes();files[name]=hashlib.sha256(content).hexdigest();digest.update(name.encode());digest.update(content)
        adapters[step]={'path':str(path.relative_to(root)),'identity':digest.hexdigest(),'files':files}
        receipt={'step':step,'binding':tm['resume_binding'],'checkpoint_name':path.name,'files':{name:{'sha256':h} for name,h in files.items()}}
        (path/'resume_state.json').write_text(json.dumps(receipt))
    # Test fixture generator only; production modules remain linear.
    def write_eval(relative,label,split,n,correct_counts,unknown_counts=None,timeout_counts=None,step=2,selection=None):
        destination=root/relative;destination.mkdir(parents=True,exist_ok=True)
        rows=[]
        for qi,q in enumerate(data[split]):
            unknown=(unknown_counts or [0,0])[qi];timeout=(timeout_counts or [0,0])[qi]
            for i in range(n):
                eos=i>=unknown+timeout
                if i<unknown:
                    ids=[7]*1024;reason='repetition_guard';text='loop';score={'correct':None,'reward':None,'extracted':None,'status':'unscored','format_compliant':None,'truncated':False}
                    evidence=runpy.run_path(str(root/'scripts/repetition_guard.py'),init_globals={'TOKEN_IDS':ids,'GUARD_POLICY':settings['generation_guard']})['GUARD_RESULT']['evidence'];timing=None;seconds=40
                elif i<unknown+timeout:
                    ids=[];reason='answer_timeout';text='';score={'correct':False,'reward':0.0,'extracted':None,'status':'timeout','format_compliant':None,'truncated':False};evidence=None;seconds=300.2
                    timing={'timeout_seconds':300,'elapsed_seconds':300.1,'last_token_age_seconds':300.1,'observed_tokens_at_timeout':0}
                else:
                    value=int(q['answer']) if i-unknown-timeout<correct_counts[qi] else 0
                    ids=[value,248044];reason='eos';text=f'#### {value}';seconds=1;evidence=None;timing=None
                    score=runpy.run_path(str(root/'scripts/verify_answers.py'),init_globals={'SCORE_INPUT':[{'text':text,'answer':q['answer'],'truncated':False}]})['SCORE_OUTPUT'][0]
                rows.append({**q,**score,'sample_index':i,'sample_seed':int.from_bytes(hashlib.sha256(f"{settings['seed']}:{q['id']}:{i}".encode()).digest()[:4],'big'),
                    'text':text,'raw_text':text,'output_token_ids':ids,'output_tokens':len(ids),'prompt_tokens':5,'generation_limit':262139,'seconds':seconds,
                    'termination_reason':reason,'scoring_evaluable':eos,'guard_evidence':evidence,'timeout_evidence':timing})
        content=''.join(json.dumps(row)+'\n' for row in rows).encode();(destination/'samples.jsonl').write_bytes(content)
        m={'schema_version':1,'experiment_type':'math_l4_base_rl','synthetic':False,'model_label':label,'split':split,'path_base':'code_root','complete':True,
            'adapter_path':adapters[step]['path'] if label=='rl' else None,'adapter_identity':adapters[step]['identity'] if label=='rl' else None,
            'adapter_files':adapters[step]['files'] if label=='rl' else None,'selection_identity':selection['selection_id'] if selection else None,
            'selection_path':'outputs/selection.json' if selection else None,'dataset_path':f'data/processed/{split}.jsonl','dataset_sha256':dm['files'][split+'.jsonl']['sha256'],
            'dataset_manifest_path':'data/processed/manifest.json','dataset_manifest_sha256':dmh,'n_samples':n,'question_count':2,'total_samples':len(rows),
            'config':cfg,'settings':settings,'code_hashes':code_hashes,'model_identity':cfg['model']['expected_identity'],'model_files':model_files,
            'samples_sha256':hashlib.sha256(content).hexdigest()}
        protocol={k:m[k] for k in ('dataset_sha256','dataset_manifest_sha256','n_samples','settings','code_hashes','experiment_type')}
        m['protocol_id']=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
        (destination/'manifest.json').write_text(json.dumps(m))
        return rows,m
    write_eval('outputs/dev/checkpoint-2','rl','dev',4,[2,2],step=2)
    write_eval('outputs/dev/checkpoint-4','rl','dev',4,[2,2],step=4)
    interpreter=[sys.executable,'-S']
    select=interpreter+[str(root/'scripts/select_checkpoint.py'),'--training-report','outputs/formal/report.json','--dev-root','outputs/dev','--output','outputs/selection.json']
    result=subprocess.run(select,capture_output=True,text=True);assert result.returncode==0,result.stderr
    selection=json.loads((root/'outputs/selection.json').read_text());assert selection['selected_step']==2 and selection['test_outcomes_used'] is False;checks+=2
    result=subprocess.run(select,capture_output=True,text=True);assert result.returncode==0 and 'CACHED' in result.stdout;checks+=1
    write_eval('outputs/test_base','base','test',16,[4,0],unknown_counts=[1,0],timeout_counts=[1,2])
    write_eval('outputs/test_rl','rl','test',16,[8,1],unknown_counts=[0,2],timeout_counts=[0,1],selection=selection)
    command=interpreter+[str(root/'scripts/analyze.py'),'--base','outputs/test_base','--rl','outputs/test_rl','--output','outputs/comparison','--bootstrap','100']
    result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==0,result.stderr
    summary=json.loads((root/'outputs/comparison/summary.json').read_text())
    assert summary['metrics']['base_pass_at_1']['estimate']==4/32
    assert summary['metrics']['rl_pass_at_1']['estimate']==9/32
    assert summary['metrics']['delta_pass_at_1']['estimate']==5/32
    assert summary['metrics']['base_pass_at_16']['estimate']==.5 and summary['metrics']['rl_pass_at_16']['estimate']==1
    assert summary['metrics']['delta_pass_at_1_minus_delta_pass_at_16']['estimate']==5/32-.5
    assert summary['metrics']['base_pass_at_1']['unknown_bounds']==[4/32,5/32]
    assert summary['diagnostics']['base']['termination_counts']['answer_timeout']==3
    assert summary['observed_coverage_counts']=={'both':1,'rl_only':1}
    assert abs(summary['metrics']['base_pass_at_4']['estimate']-(1-math.comb(12,4)/math.comb(16,4))/2)<1e-12
    checks+=9
    result=subprocess.run(command,capture_output=True,text=True);assert result.returncode==0 and 'CACHED' in result.stdout;checks+=1
    original=(root/'outputs/test_rl/samples.jsonl').read_bytes();original_manifest=(root/'outputs/test_rl/manifest.json').read_bytes()
    for mutation in ('seed','score','unknown_zero','wrong_gold','duplicate','missing','stop','config','protocol','base_identity','selection','adapter_path'):
        rows=[json.loads(line) for line in original.decode().splitlines()];m=json.loads(original_manifest)
        if mutation=='seed':rows[0]['sample_seed']+=1
        elif mutation=='score':rows[0]['correct']=False
        elif mutation=='unknown_zero':rows[16]['correct']=False;rows[16]['reward']=0
        elif mutation=='wrong_gold':rows[0]['answer']='999'
        elif mutation=='duplicate':rows[-1]=rows[0]
        elif mutation=='missing':rows.pop()
        elif mutation=='stop':rows[0]['termination_reason']='answer_timeout'
        elif mutation=='config':m['config']['evaluation']['temperature']=.7
        elif mutation=='protocol':m['protocol_id']='0'*64
        elif mutation=='base_identity':m['model_identity']='0'*64
        elif mutation=='selection':m['selection_identity']='0'*64
        else:m['adapter_path']='outputs/formal/checkpoint-4'
        content=''.join(json.dumps(row)+'\n' for row in rows).encode();m['samples_sha256']=hashlib.sha256(content).hexdigest()
        (root/'outputs/test_rl/samples.jsonl').write_bytes(content);(root/'outputs/test_rl/manifest.json').write_text(json.dumps(m))
        rejected_command=command.copy();rejected_command[rejected_command.index('--output')+1]='outputs/rejected-'+mutation
        result=subprocess.run(rejected_command,capture_output=True,text=True);assert result.returncode!=0,(mutation,result.stdout,result.stderr)
        checks+=1
    (root/'outputs/test_rl/samples.jsonl').write_bytes(original);(root/'outputs/test_rl/manifest.json').write_bytes(original_manifest)
    # An altered but hash-consistent candidate must not silently reselect using test results.
    selected_bytes=(root/'outputs/selection.json').read_bytes()
    for mutation in ('choose_later','missing_candidate','test_as_dev','changed_adapter','changed_training','changed_dev_score','changed_scoring'):
        restores={}
        if mutation=='choose_later':
            s=json.loads(selected_bytes);s['selected_step']=4;(root/'outputs/selection.json').write_text(json.dumps(s))
        elif mutation=='missing_candidate':
            p=root/'outputs/formal/report.json';restores[p]=p.read_bytes();r=json.loads(restores[p]);r['checkpoints'].pop();p.write_text(json.dumps(r))
        elif mutation=='test_as_dev':
            p=root/'outputs/dev/checkpoint-2/manifest.json';restores[p]=p.read_bytes();m=json.loads(restores[p]);m['split']='test';p.write_text(json.dumps(m))
        elif mutation=='changed_adapter':
            p=root/'outputs/formal/checkpoint-2/adapter_model.safetensors';restores[p]=p.read_bytes();p.write_bytes(b'changed')
        elif mutation=='changed_training':
            p=root/'outputs/formal/manifest.json';restores[p]=p.read_bytes();p.write_bytes(restores[p]+b' ')
        elif mutation=='changed_dev_score':
            p=root/'outputs/dev/checkpoint-2/samples.jsonl';restores[p]=p.read_bytes();p.write_bytes(restores[p]+b' ')
        else:
            p=root/'scripts/verify_answers.py';restores[p]=p.read_bytes();p.write_bytes(restores[p]+b'\n# changed\n')
        rejected_command=command.copy();rejected_command[rejected_command.index('--output')+1]='outputs/rejected-'+mutation
        result=subprocess.run(rejected_command,capture_output=True,text=True);assert result.returncode!=0,(mutation,result.stdout,result.stderr)
        for p,b in restores.items():p.write_bytes(b)
        (root/'outputs/selection.json').write_bytes(selected_bytes)
        checks+=1
    # Monitor reads only pipeline status and exits on completion.
    state={'status':'completed','phase':'COMPARE','phase_index':9,'phase_count':9,'elapsed_seconds':123,'updated_at_unix':0,'answer':None}
    (root/'outputs/run_state.json').write_text(json.dumps(state))
    result=subprocess.run(interpreter+[str(root/'scripts/monitor.py'),'--run-dir','outputs','--follow'],capture_output=True,text=True,timeout=3)
    assert result.returncode==0 and 'completed' in result.stdout;checks+=1
print(f'✅ [MATH COMPARISON] {checks} dev-only freeze, paired statistics, unknown-stop and integrity checks; temporary CPU fixtures only')
