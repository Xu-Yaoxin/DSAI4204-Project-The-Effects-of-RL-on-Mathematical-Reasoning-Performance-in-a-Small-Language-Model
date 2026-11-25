"""Verify the file-only live viewer, including refresh and terminal-state exit."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

root=Path(__file__).resolve().parents[1]
source=root/'scripts/monitor_guarded.py'
assert not any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for node in ast.walk(ast.parse(source.read_text())))
checks=1
with tempfile.TemporaryDirectory(prefix='guarded monitor ') as temporary:
    fixture=Path(temporary)/'code'
    (fixture/'scripts').mkdir(parents=True)
    shutil.copyfile(source,fixture/'scripts/monitor_guarded.py')
    command=[sys.executable,'-S',str(fixture/'scripts/monitor_guarded.py')]
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==0 and 'Waiting for live status' in result.stdout
    checks+=1
    live=fixture/'outputs/guarded_v3/live';live.mkdir(parents=True)
    state={'schema_version':1,'status':'running','updated_at_unix':time.time(),'total_attempts':800,'handled_attempts':11,'restarts_used':1,'termination_counts':{'answer_timeout':1,'eos':10},'workers':[{'worker_index':0,'state':'running','active_question_id':'q17','sample_index':2,'generated_tokens':513,'tokens_per_second':42.5,'last_token_age_seconds':.5,'answer_remaining_seconds':287,'partial_text':'fixture prefix '+('A'*500)+'\\boxed{24}'},{'worker_index':1,'state':'queued'}]}
    (live/'current.json').write_text(json.dumps(state))
    result=subprocess.run(command+['--tail-chars','30'],text=True,capture_output=True)
    assert result.returncode==0 and 'handled=11/800' in result.stdout and 'question=q17 sample=2' in result.stdout
    assert 'tokens=513 rate=42.5/s' in result.stdout and 'answer_remaining=287s' in result.stdout
    assert '\\boxed{24}' in result.stdout and 'fixture prefix' not in result.stdout
    checks+=3
    child=subprocess.Popen(command+['--follow','--interval','.2'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    time.sleep(.25)
    state.update(status='completed',handled_attempts=800,updated_at_unix=time.time())
    (live/'current.json.tmp').write_text(json.dumps(state));(live/'current.json.tmp').replace(live/'current.json')
    stdout,stderr=child.communicate(timeout=5)
    assert child.returncode==0 and 'running' in stdout and 'completed' in stdout and 'handled=800/800' in stdout
    checks+=1
    for arguments in (['--interval','0'],['--interval','nan'],['--tail-chars','-1'],['--output','../outside']):
        result=subprocess.run(command+arguments,text=True,capture_output=True)
        assert result.returncode!=0
        checks+=1
    (live/'current.json').write_text('{')
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==0 and 'Waiting' in result.stdout
    checks+=1
print(f'✅ [LIVE MONITOR] {checks} file-only status, partial-output, follow and validation checks; no GPU/model/network')
