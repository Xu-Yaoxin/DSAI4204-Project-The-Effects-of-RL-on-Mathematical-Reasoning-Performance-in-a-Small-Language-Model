"""Read-only shard-0 validation and exploratory comparison on its paired Base questions."""
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import random
import runpy
import statistics
import sys
import time

output = Path(__file__).resolve().parent
code = output.parents[2]
bundle = code / 'math_l4_rl_split_v1'
experiment = Path('/Users/xu./Downloads/math_l4_selected_20261104/math_l4_v1')
run = experiment / 'outputs/math_l4_v1_fix1'
cfg = json.loads((experiment / 'configs/math_l4.json').read_text())
started = time.monotonic()
frozen = json.loads((experiment / 'bundle_manifest.json').read_text())
for name, expected in frozen['files_sha256'].items():
    assert hashlib.sha256((experiment / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((code / 'math_l4_v1' / name).read_bytes()).hexdigest() == expected
print('[CHECK 1/4] Frozen code/data match; replaying selected-checkpoint evidence and shard 0.', flush=True)

merge_path = bundle / 'scripts/merge_shards.py'
merge_source = merge_path.read_text()
boundary = "left, right = [item['manifest'] for item in validated]"
assert merge_source.count(boundary) == 1
prefix = merge_source.split(boundary, 1)[0]
loop = "for index in (0, 1):\n    source = run / 'rl_shards' / f'shard_{index}'"
assert prefix.count(loop) == 1
prefix = prefix.replace(loop, "for index in (0,):\n    source = run / 'rl_shards' / f'shard_{index}'", 1)
previous_argv = sys.argv
sys.argv = [str(merge_path), '--experiment-root', str(experiment)]
namespace = {'__name__': '__main__', '__file__': str(merge_path)}
exec(compile(prefix, str(merge_path), 'exec'), namespace)
sys.argv = previous_argv
rl_checked = namespace['validated'][0]
selection = namespace['selection']
rl = rl_checked['manifest']
del namespace
print('[CHECK 2/4] 1536 development answers and 1600 RL answers replayed; validating full Base.', flush=True)
base_checked = runpy.run_path(str(experiment / 'scripts/validate_evaluation.py'), init_globals={
    'EVALUATION_PATH': run / 'test_base', 'VALIDATE_CONFIG': cfg})['EVALUATION_RESULT']
base = base_checked['manifest']
receipt = json.loads((run / 'base_stop_receipt.json').read_text())
assert base['samples_sha256'] == receipt['base_samples_sha256']
assert receipt['complete'] is True and receipt['base_total_samples'] == 3200
for name in ['schema_version', 'experiment_type', 'synthetic', 'split', 'dataset_path', 'dataset_sha256',
             'dataset_manifest_path', 'dataset_manifest_sha256', 'n_samples', 'question_count', 'config',
             'settings', 'code_hashes', 'protocol_id', 'path_base', 'model_identity', 'model_files',
             'gpu', 'groups', 'execution_cpu_threads']:
    assert base[name] == rl[name], name
state = json.loads((run / 'rl_shards/worker_0_state.json').read_text())
hardware = json.loads((run / 'rl_shards/worker_0_hardware.json').read_text())
assert state['status'] == 'completed' and state['child_pid'] is None
assert state['completed_answers'] == state['target_answers'] == 1600 and state['shard_index'] == 0
assert hardware['ready'] is True and hardware['shard_index'] == 0
assert hardware['gpu'] == rl['gpu'] and hardware['source_gpu'] == base['gpu']
assert hardware['expected_package_versions'] == rl['settings']['package_versions']
assert hardware['config_sha256'] == hashlib.sha256((experiment / 'configs/math_l4.json').read_bytes()).hexdigest()
assigned = base_checked['dataset'][0::2]
assigned_ids = {q['id'] for q in assigned}
assert len(assigned_ids) == 100
paired_base = [r for r in base_checked['rows'] if r['id'] in assigned_ids]
validated = {'base': {'rows': paired_base, 'dataset': assigned}, 'rl': rl_checked}
assert len(paired_base) == len(rl_checked['rows']) == 1600
print('[CHECK 3/4] Same 100 questions, seeds, scoring, generation policy and checkpoint verified.', flush=True)

analysis_path = experiment / 'scripts/analyze.py'
analysis_source = analysis_path.read_text()
start_marker = "records = {label: {(r['id'],r['sample_index']): r for r in validated[label]['rows']} for label in ('base','rl')}"
end_marker = 'limitations = ['
assert analysis_source.count(start_marker) == analysis_source.count(end_marker) == 1
statistics_code = start_marker + analysis_source.split(start_marker, 1)[1].split(end_marker, 1)[0]
args = SimpleNamespace(seed=42)
repeats = cfg['evaluation']['bootstrap_replicates']
stats = {'validated': validated, 'base': base, 'args': args, 'repeats': repeats,
         'Counter': Counter, 'defaultdict': defaultdict, 'math': math, 'random': random}
exec(compile(statistics_code, str(analysis_path), 'exec'), stats)
per_question = stats['per_question']
metrics = stats['metrics']
diagnostics = stats['diagnostics']
for label in ['base', 'rl']:
    values = validated[label]['rows']
    diagnostics[label].update(
        mean_answer_seconds=statistics.mean(r['seconds'] for r in values),
        median_answer_seconds=statistics.median(r['seconds'] for r in values),
        median_output_tokens=statistics.median(r['output_tokens'] for r in values),
        observed_tokens_per_answer_second=sum(r['output_tokens'] for r in values)/sum(r['seconds'] for r in values),
        termination_seconds={reason: sum(r['seconds'] for r in values if r['termination_reason'] == reason)
                            for reason in sorted({r['termination_reason'] for r in values})})
subjects = {}
for subject in sorted({q['subject'] for q in per_question}):
    selected = [q for q in per_question if q['subject'] == subject]
    subjects[subject] = {'questions': len(selected)}
    for k in [1, 4, 16]:
        for label in ['base', 'rl']:
            subjects[subject][f'{label}_pass_at_{k}'] = statistics.mean(q[label][f'pass_at_{k}']['estimate'] for q in selected)
        subjects[subject][f'delta_pass_at_{k}'] = statistics.mean(q[f'delta_pass_at_{k}'] for q in selected)
changes = Counter('rl_higher' if q['rl']['correct_count'] > q['base']['correct_count'] else
                  'base_higher' if q['rl']['correct_count'] < q['base']['correct_count'] else 'equal'
                  for q in per_question)
summary = {
    'scope': 'Exploratory paired comparison of completed RL shard 0 and its same 100 Base questions; not the final 200-question result.',
    'complete_within_shard': True, 'full_experiment_complete': False,
    'model_identity': base['model_identity'], 'adapter_identity': rl['adapter_identity'],
    'selection_identity': selection['selection_id'], 'selected_step': selection['selected_step'],
    'base_full_samples_sha256': base['samples_sha256'], 'rl_shard_samples_sha256': rl['samples_sha256'],
    'protocol_id': base['protocol_id'], 'execution_extension': rl['execution_extension'],
    'question_count': len(per_question), 'family_count': len(stats['families']), 'n_samples': 16,
    'bootstrap_replicates': repeats, 'bootstrap_seed': args.seed,
    'interval_method': 'paired family-cluster percentile bootstrap, 95%, linear quantiles; unchanged production statistics block',
    'metrics': metrics, 'diagnostics': diagnostics, 'subjects': subjects,
    'question_success_count_comparison': dict(changes),
    'observed_coverage_counts': dict(stats['coverage']),
    'integrity': {'frozen_files_verified': len(frozen['files_sha256']), 'base_rows_replayed': 3200,
                  'rl_rows_replayed': 1600, 'dev_rows_replayed': 1536, 'worker_restarts': state['restarts'],
                  'merge_script_sha256': hashlib.sha256(merge_path.read_bytes()).hexdigest(),
                  'analysis_script_sha256': hashlib.sha256(analysis_path.read_bytes()).hexdigest()},
    'limitations': [
        'Checkpoint and scoring were not changed in response to test outcomes.',
        'This is an interim 100-question subset and a single training run, not a final full-test conclusion.',
        'Timeouts are zero by the frozen policy. Repetition stops remain unknown and contribute no confirmed success.',
        'Unknown sensitivity bounds are not confidence intervals. Bootstrap intervals are marginal, not multiple-testing adjusted.',
        'Matching GPU model and software do not equalize measured throughput; RL runs near 20 tokens/s and Base near 40. A 300-second deadline therefore permits different token budgets.',
        'Frozen scorer replay establishes consistency, not mathematical correctness of every answer. Small semantic spot checks found two RL false negatives from Markdown headings.',
        'No observed success in 16 attempts does not establish zero capability. pass@16 measures limited-sample coverage.',
        'No GPU, model generation, remote operations or downloaded-file changes were used.'],
    'review_seconds': time.monotonic() - started}
(output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
(output / 'per_question.json').write_text(json.dumps(per_question, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({key: summary[key] for key in ['metrics', 'diagnostics', 'question_success_count_comparison', 'observed_coverage_counts', 'review_seconds']}, indent=2))
print('[CHECK 4/4] COMPLETE | CPU-only paired interim analysis | saved=' + str(output / 'summary.json'), flush=True)
