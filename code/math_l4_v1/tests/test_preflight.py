"""Validate frozen bundle/data and environment gates with isolated CPU fixtures."""
import ast
import builtins
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import runpy
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

source_root=Path(__file__).resolve().parents[1]
source=source_root/'scripts/preflight.py'
assert not any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for node in ast.walk(ast.parse(source.read_text())))
checks=1
original_read_text=Path.read_text
original_is_file=Path.is_file
original_import=builtins.__import__
versions={'torch':'2.10.0+cu128','transformers':'5.3.0','trl':'1.16.0','peft':'0.18.1','accelerate':'1.10.1'}
with tempfile.TemporaryDirectory(prefix='math preflight fixture ') as temporary:
    base=Path(temporary)
    for case in ('cpu','gpu_fixture','missing_bundle','changed_bundle','bad_data','outside_config','file_symlink','parent_symlink','unsafe_bundle_entry','outside_output','output_symlink','python_version','package_version','ddp','gpu_count','bf16','low_gpu','low_ram','cgroup_low_ram','wrong_model','wrong_eos','long_prompt'):
        root=base/case
        for directory in ('scripts','configs','data/processed'):(root/directory).mkdir(parents=True)
        shutil.copyfile(source,root/'scripts/preflight.py')
        cfg=json.loads((source_root/'configs/math_l4.json').read_text())
        cfg['data']['output_dir']='data/processed'
        (root/'configs/math_l4.json').write_text(json.dumps(cfg))
        (root/'scripts/doctor.py').write_text('report='+repr({'data_ready':case!='bad_data','checks':{'fixture':case!='bad_data'}})+'\n')
        (root/'scripts/fingerprint_model.py').write_text('model_identity='+repr('bad' if case=='wrong_model' else cfg['model']['expected_identity'])+'\n')
        for split in ('train','dev','test'):(root/f'data/processed/{split}.jsonl').write_text('{"id":"q","question":"What is 1+1?","answer":"2"}\n')
        bundle={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
        (root/'bundle_manifest.json').write_text(json.dumps({'files_sha256':bundle}))
        cpu_only=case in ('cpu','missing_bundle','changed_bundle','bad_data','outside_config','file_symlink','parent_symlink','unsafe_bundle_entry','outside_output','output_symlink')
        args=[str(root/'scripts/preflight.py')]
        if cpu_only:args+=['--cpu-only']
        if case=='missing_bundle':(root/'bundle_manifest.json').unlink()
        if case=='changed_bundle':(root/'scripts/doctor.py').write_text('report={"data_ready":True}\n# mutation\n')
        if case=='outside_config':
            external=base/'outside.json';external.write_text(json.dumps(cfg));args+=['--config',str(external)]
        if case=='file_symlink':
            target=root/'scripts/doctor.py';outside=base/'external_doctor.py';outside.write_bytes(target.read_bytes());target.unlink();target.symlink_to(outside)
        if case=='parent_symlink':
            outside=base/'external_data';(root/'data').rename(outside);(root/'data').symlink_to(outside,target_is_directory=True)
        if case=='unsafe_bundle_entry':
            (root/'bundle_manifest.json').write_text(json.dumps({'files_sha256':{'../outside.json':'0'*64}}))
        if case=='outside_output':args+=['--output','../outside_result.json']
        if case=='output_symlink':
            (root/'outputs').mkdir();external=base/'untouched.txt';external.write_text('sentinel');(root/'outputs/link.json').symlink_to(external);args+=['--output','outputs/link.json']
        if case in ('cpu','gpu_fixture'):args+=['--output','outputs/preflight.json']
        tensor=MagicMock();product=MagicMock();equal=MagicMock()
        tensor.__matmul__.return_value=product;product.__eq__.return_value=equal;equal.all.return_value=True
        cuda=SimpleNamespace(is_available=MagicMock(return_value=True),device_count=MagicMock(return_value=2 if case=='gpu_count' else 1),
            is_bf16_supported=MagicMock(return_value=case!='bf16'),synchronize=MagicMock(),empty_cache=MagicMock(),
            mem_get_info=MagicMock(return_value=((10 if case=='low_gpu' else 22)*1024**3,24*1024**3)),get_device_name=MagicMock(return_value='CPU fixture'))
        torch=SimpleNamespace(cuda=cuda,bfloat16='fixture-bf16',ones=MagicMock(return_value=tensor))
        tokenizer=MagicMock(return_value={'input_ids':[1]*(3000 if case=='long_prompt' else 10)})
        tokenizer.eos_token_id=0 if case=='wrong_eos' else cfg['model']['expected_eos_token_id']
        load_tokenizer=MagicMock(return_value=tokenizer)
        transformer=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer))
        def read_text(path,*args,**kwargs):
            if str(path)=='/proc/meminfo':return f'MemAvailable: {(8 if case=="low_ram" else 48)*1024**2} kB\n'
            if case=='cgroup_low_ram' and str(path)=='/sys/fs/cgroup/memory.max':return str(20*1024**3)
            if case=='cgroup_low_ram' and str(path)=='/sys/fs/cgroup/memory.current':return str(8*1024**3)
            return original_read_text(path,*args,**kwargs)
        def is_file(path):
            if str(path).startswith('/sys/'):return case=='cgroup_low_ram' and str(path) in ('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current')
            return original_is_file(path)
        def get_version(name):
            return '0.0.0' if case=='package_version' and name=='trl' else versions[name]
        def guarded_import(name,*args,**kwargs):
            if cpu_only and name in ('torch','transformers','trl','peft','accelerate'):
                raise AssertionError('CPU-only preflight imported model packages')
            return original_import(name,*args,**kwargs)
        error=None
        output=io.StringIO()
        environment={'WORLD_SIZE':'2' if case=='ddp' else '1','LOCAL_WORLD_SIZE':'1','RANK':'0','LOCAL_RANK':'0'}
        with patch.object(sys,'argv',args),patch.object(sys,'version_info',(3,10,0) if case=='python_version' else (3,11,0)),patch.object(platform,'system',return_value='Linux'),patch.dict(sys.modules,{'torch':torch,'transformers':transformer}),patch.object(importlib.metadata,'version',get_version),patch.object(Path,'read_text',read_text),patch.object(Path,'is_file',is_file),patch.dict(os.environ,environment),patch.object(builtins,'__import__',guarded_import),contextlib.redirect_stdout(output),contextlib.redirect_stderr(io.StringIO()):
            try:runpy.run_path(str(root/'scripts/preflight.py'),run_name='__main__')
            except (SystemExit,ValueError,AssertionError) as caught:error=caught
        if case in ('cpu','gpu_fixture'):
            assert error is None,(case,error,output.getvalue())
            saved=json.loads((root/'outputs/preflight.json').read_text())
            assert saved['ready'] and saved['gpu_checked']==(case=='gpu_fixture') and not saved['model_loaded']
            assert cuda.is_available.called==(case=='gpu_fixture')
            if case=='gpu_fixture':assert saved['prompt_preflight']['test']['max_prompt_tokens']==10 and load_tokenizer.call_args.kwargs['local_files_only']
        else:
            assert isinstance(error,SystemExit),(case,error,output.getvalue())
            if case in ('package_version','python_version','ddp'):assert not cuda.is_available.called
            if case in ('low_gpu','low_ram','cgroup_low_ram','gpu_count','bf16'):assert not load_tokenizer.called
            if case=='output_symlink':assert external.read_text()=='sentinel'
        checks+=1
print(f'✅ [MATH PREFLIGHT] {checks} isolated CPU checks | immutable bundle/data, unsafe paths, no CPU-only model imports, pinned packages, topology, BF16/resource gates, model/EOS identity and prompt budget | no real GPU/model')
