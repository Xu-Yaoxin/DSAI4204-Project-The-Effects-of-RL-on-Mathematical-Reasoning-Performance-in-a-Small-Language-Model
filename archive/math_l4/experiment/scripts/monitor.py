"""Read saved pipeline state and live generation text without loading model packages."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir',default='outputs/math_l4_v1')
parser.add_argument('--follow',action='store_true')
parser.add_argument('--interval',type=float,default=5)
args=parser.parse_args()
run=(root/args.run_dir).resolve()
if not run.is_relative_to(root/'outputs') or not math.isfinite(args.interval) or args.interval<.2:
    raise ValueError('Use a run under outputs/ and interval>=0.2 seconds.')
try:
    while True:
        path=run/'run_state.json'
        state=None
        if path.is_file() and not path.is_symlink():
            try:state=json.loads(path.read_text())
            except (OSError,ValueError):pass
        stamp=datetime.now().strftime('%H:%M:%S')
        if not isinstance(state,dict):
            print(f'[{stamp}] Waiting for pipeline status: {path}',flush=True)
        else:
            age=max(0,time.time()-state.get('updated_at_unix',time.time()))
            print(f"[{stamp}] {state.get('status')} | phase={state.get('phase')} ({state.get('phase_index','?')}/{state.get('phase_count','?')}) | elapsed={state.get('elapsed_seconds',0):.0f}s | snapshot_age={age:.0f}s",flush=True)
            answer=state.get('answer')
            if isinstance(answer,dict):
                print(f"question={answer.get('id')} sample={answer.get('sample_index')} | tokens={answer.get('generated_tokens',0)} | rate={answer.get('tokens_per_second',0):.1f}/s | remaining={answer.get('remaining_seconds',0):.1f}s | reason={answer.get('termination_reason')}",flush=True)
                print('Termination counts: '+json.dumps(answer.get('termination_counts',{}),sort_keys=True),flush=True)
                partial=''.join(c for c in str(answer.get('partial_text','')) if c in '\n\t' or ord(c)>=32).replace('\x7f','')
                if partial:print('Partial output:\n'+partial,flush=True)
            elif state.get('phase_output'):
                print('Stage output: '+str(state['phase_output']),flush=True)
            if state.get('status') in ('completed','failed','interrupted'):
                break
        if not args.follow:break
        time.sleep(args.interval)
except KeyboardInterrupt:
    print('\nViewer stopped; the background pipeline continues.',flush=True)
