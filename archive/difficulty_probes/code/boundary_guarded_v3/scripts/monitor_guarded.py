"""Display short live progress from saved files without importing model packages."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('outputs/guarded_v3'))
parser.add_argument('--follow', action='store_true', help='Refresh every5 seconds until the run finishes; Ctrl+C only stops this viewer')
parser.add_argument('--interval', type=float, default=5.0)
parser.add_argument('--tail-chars', type=int, default=400)
args = parser.parse_args()
if not math.isfinite(args.interval) or args.interval < 0.2 or not 0 <= args.tail_chars <= 4000:
    parser.error('Use interval>=0.2 seconds and tail-chars between0 and4000')
output = (root / args.output).resolve()
if not output.is_relative_to(root / 'outputs') or output == root / 'outputs':
    parser.error('Use a dedicated run directory under outputs/')
try:
    while True:
        path = output / 'live/current.json'
        state = None
        if path.is_file() and not path.is_symlink():
            try:
                state = json.loads(path.read_text())
            except (ValueError, OSError):
                pass
        stamp = datetime.now().strftime('%H:%M:%S')
        if not isinstance(state, dict):
            print(f'[{stamp}] Waiting for live status: {path}', flush=True)
        else:
            status = state.get('status', 'unknown')
            updated = state.get('updated_at_unix')
            age = f'{max(0,time.time()-updated):.0f}s' if type(updated) in (int, float) else 'unknown'
            print(f"[{stamp}] {status} | handled={state.get('handled_attempts',0)}/{state.get('total_attempts','?')} | snapshot_age={age} | restarts={state.get('restarts_used',0)}", flush=True)
            print('Termination counts: ' + json.dumps(state.get('termination_counts', {}), ensure_ascii=False, sort_keys=True), flush=True)
            for worker in state.get('workers', []):
                if not isinstance(worker, dict):
                    continue
                rate = worker.get('tokens_per_second')
                token_age = worker.get('last_token_age_seconds')
                remaining = worker.get('answer_remaining_seconds')
                rate_text = f'{rate:.1f}' if type(rate) in (int, float) else '?'
                age_text = f'{token_age:.1f}s' if type(token_age) in (int, float) else '?'
                remaining_text = f'{remaining:.0f}s' if type(remaining) in (int, float) else '?'
                print(f"worker={worker.get('worker_index')} {worker.get('state')} | question={worker.get('active_question_id')} sample={worker.get('sample_index')} | tokens={worker.get('generated_tokens',0)} rate={rate_text}/s | since_token={age_text} | answer_remaining={remaining_text}", flush=True)
                partial = worker.get('partial_text', '')
                if partial and args.tail_chars:
                    partial = ''.join(c for c in str(partial) if c in '\n\t' or ord(c) >= 32).replace('\x7f', '')
                    print('Partial output:\n' + partial[-args.tail_chars:], flush=True)
            if status in ('completed', 'failed', 'interrupted'):
                break
        if not args.follow:
            break
        time.sleep(args.interval)
except KeyboardInterrupt:
    print('\nViewer stopped; the remote evaluation process continues.', flush=True)
