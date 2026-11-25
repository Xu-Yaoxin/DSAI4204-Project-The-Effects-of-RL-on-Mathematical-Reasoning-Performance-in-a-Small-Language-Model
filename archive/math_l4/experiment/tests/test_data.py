"""Exercise real frozen split integrity, provenance, leakage and deterministic selection."""
import ast
import builtins
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
config = {'data': {'output_dir':'data/processed','seed':20261102,'engineering_size':32,
    'pilot_size':128,'train_size':512,'dev_size':128,'test_size':200}}
checks = 0
for script in ('prepare_data.py','doctor.py'):
    tree=ast.parse((root/'scripts'/script).read_text())
    assert not any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for node in ast.walk(tree))
    checks+=1
original_import=builtins.__import__
def isolated_import(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','pyarrow','datasets','peft','trl','accelerate'}:
        raise AssertionError('Data readiness must not import model or parquet packages: '+name)
    return original_import(name,*args,**kwargs)
builtins.__import__=isolated_import
try:
    result=runpy.run_path(str(root/'scripts/doctor.py'),init_globals={'DOCTOR_CONFIG':config})['report']
finally:
    builtins.__import__=original_import
assert result['data_ready'] and all(result['checks'].values()),result
checks+=1
m=json.loads((root/'data/processed/manifest.json').read_text())
assert m['eligible_by_official_split']=={'test':662,'train':1168}
assert m['model_scores_used_for_selection'] is False
checks+=1
records={split:[json.loads(line) for line in (root/f'data/processed/{split}.jsonl').read_text().split('\n') if line.strip()] for split in ('engineering','pilot','train','dev','test')}
ids={name:{r['id'] for r in rows} for name,rows in records.items()}
assert ids['engineering']<=ids['pilot']<=ids['train']
assert not ids['train']&ids['dev'] and not ids['train']&ids['test'] and not ids['dev']&ids['test']
checks+=1
assert all(r['source']['split']=='train' for s in ('engineering','pilot','train','dev') for r in records[s])
assert all(r['source']['split']=='test' for r in records['test'])
checks+=1
history=json.loads((root/'provenance/history_exclusions.json').read_text())
old=set(history['excluded_normalized_question_sha256'])
assert not old&{r['normalized_question_sha256'] for s in ('train','dev','test') for r in records[s]}
checks+=1
pool=[json.loads(line) for line in (root/'provenance/eligible_pool.jsonl').read_text().split('\n') if line.strip()]
pool_norm={r['normalized_question_sha256'] for r in pool}
assert len(pool_norm)==len(pool)==1830
checks+=1
with tempfile.TemporaryDirectory(prefix='math-l4-data-') as temp:
    fixture=Path(temp)/'package';fixture.mkdir()
    for folder in ('data','provenance'):
        shutil.copytree(root/folder,fixture/folder)
    (fixture/'scripts').mkdir()
    for name in ('prepare_data.py','doctor.py','verify_answers.py'):
        shutil.copy2(root/'scripts'/name,fixture/'scripts'/name)
    snapshot={p.relative_to(fixture):p.read_bytes() for p in fixture.rglob('*') if p.is_file()}
    def restore():
        for relative,payload in snapshot.items():
            path=fixture/relative
            if path.is_symlink():path.unlink()
            if not path.exists() or path.read_bytes()!=payload:path.write_bytes(payload)
    def rebind_split(split, rows):
        path=fixture/f'data/processed/{split}.jsonl'
        payload=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows).encode()
        path.write_bytes(payload)
        manifest=json.loads((fixture/'data/processed/manifest.json').read_text())
        manifest['files'][split+'.jsonl'].update(sha256=hashlib.sha256(payload).hexdigest(),rows=len(rows),ids=[r['id'] for r in rows])
        (fixture/'data/processed/manifest.json').write_text(json.dumps(manifest))
    def assert_rejected(label):
        result=runpy.run_path(str(fixture/'scripts/doctor.py'),init_globals={'DOCTOR_CONFIG':config})['report']
        assert not result['data_ready'],label
        restore()
    # Content tampering fails even before a manifest can authenticate it.
    path=fixture/'data/processed/test.jsonl';path.write_bytes(path.read_bytes()+b' ')
    assert_rejected('raw split bytes changed');checks+=1
    altered=deepcopy(records['train']);altered[0]['answer']='999999';rebind_split('train',altered)
    assert_rejected('edited answer with rebound hash');checks+=1
    altered=deepcopy(records['train']);altered[0]=deepcopy(records['test'][0]);altered[0]['split']='train';rebind_split('train',altered)
    assert_rejected('official test inserted into training');checks+=1
    altered=deepcopy(records['dev']);altered[0]=deepcopy(records['train'][0]);altered[0]['split']='dev';rebind_split('dev',altered)
    assert_rejected('train/dev overlap');checks+=1
    altered=deepcopy(records['engineering']);outside=next(r for r in records['train'] if r['id'] not in ids['pilot']);altered[0]={**outside,'split':'engineering'};rebind_split('engineering',altered)
    assert_rejected('engineering outside nested pilot');checks+=1
    altered=deepcopy(records['test']);altered.reverse();rebind_split('test',altered)
    assert_rejected('changed deterministic order with rebound hashes');checks+=1
    altered=deepcopy(records['train']);available=next(r for r in pool if r['source']['split']=='train' and r['source']['subject']==altered[0]['source']['subject'] and r['id'] not in ids['train']|ids['dev']);altered[0]={**available,'split':'train'};rebind_split('train',altered)
    assert_rejected('valid but score-selected replacement violates frozen sample');checks+=1
    altered=deepcopy(records['test']);altered[0]['source']['row_index']=999999;rebind_split('test',altered)
    assert_rejected('source row beyond pinned file bounds');checks+=1
    altered=deepcopy(records['test']);altered[0]['source']['level']=5;rebind_split('test',altered)
    assert_rejected('wrong difficulty level');checks+=1
    altered=deepcopy(records['test']);altered[0]['question']+=' altered';rebind_split('test',altered)
    assert_rejected('question no longer matches source pool');checks+=1
    path=fixture/'data/raw/math_full/algebra/train-00000-of-00001.parquet';path.write_bytes(path.read_bytes()+b'bad')
    assert_rejected('pinned raw source changed');checks+=1
    h=deepcopy(history);h['excluded_normalized_question_sha256'].append(records['test'][0]['normalized_question_sha256']);h['known_prior_question_count']+=1
    path=fixture/'provenance/history_exclusions.json';path.write_text(json.dumps(h));manifest=json.loads((fixture/'data/processed/manifest.json').read_text());manifest['history_exclusions_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();(fixture/'data/processed/manifest.json').write_text(json.dumps(manifest))
    assert_rejected('prior calibration question included');checks+=1
    path=fixture/'provenance/eligible_pool.jsonl';payload=path.read_bytes()+path.read_bytes().split(b'\n')[0]+b'\n';path.write_bytes(payload)
    manifest=json.loads((fixture/'data/processed/manifest.json').read_text());manifest['eligible_pool_sha256']=hashlib.sha256(payload).hexdigest();manifest['eligible_pool_rows']+=1;(fixture/'data/processed/manifest.json').write_text(json.dumps(manifest))
    assert_rejected('duplicate normalized family in eligible pool');checks+=1
    path=fixture/'data/processed/test.jsonl';path.unlink();path.symlink_to(root/'data/processed/test.jsonl')
    assert_rejected('symlink prepared split');checks+=1
    manifest=json.loads((fixture/'data/processed/manifest.json').read_text());manifest['eligible_pool_path']='../../outside.jsonl';(fixture/'data/processed/manifest.json').write_text(json.dumps(manifest))
    assert_rejected('unsafe provenance path');checks+=1
    changed=deepcopy(config);changed['data']['seed']+=1
    result=runpy.run_path(str(fixture/'scripts/doctor.py'),init_globals={'DOCTOR_CONFIG':changed})['report']
    assert not result['data_ready'];checks+=1
    changed=deepcopy(config);changed['data']['train_size']=508
    result=runpy.run_path(str(fixture/'scripts/doctor.py'),init_globals={'DOCTOR_CONFIG':changed})['report']
    assert not result['data_ready'];checks+=1
    (fixture/'data/processed/manifest.json').write_text('{broken')
    assert_rejected('invalid manifest');checks+=1
    result=subprocess.run([sys.executable,'-S',str(fixture/'scripts/prepare_data.py'),'--verify'],text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr
    checks+=1
print(f'✅ [MATH DATA] {checks} actual-data/provenance/official-split/leakage/stratification/tamper checks; no model/GPU/network')
