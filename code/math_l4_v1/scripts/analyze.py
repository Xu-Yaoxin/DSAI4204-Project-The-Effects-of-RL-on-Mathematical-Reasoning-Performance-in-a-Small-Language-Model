"""Paired Base/RL analysis under identical guarded generation and frozen scoring."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import runpy
import tempfile
import shutil

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', default='configs/math_l4.json')
parser.add_argument('--base', required=True)
parser.add_argument('--rl', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--bootstrap', type=int)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
cfg = json.loads((root / args.config).read_text())
repeats = cfg['evaluation']['bootstrap_replicates'] if args.bootstrap is None else args.bootstrap
if type(repeats) is not int or repeats < 100 or args.seed < 0:
    raise ValueError('Use at least100 bootstrap replicates and a nonnegative seed.')
output = (root / args.output).absolute()
if any(p.is_symlink() for p in (output, *output.parents)):
    raise ValueError('Unsafe analysis output.')
output = output.resolve()
if not output.is_relative_to(root / 'outputs'):
    raise ValueError('Analysis output must be under outputs/.')
validated = {}
for label, relative in (('base', args.base), ('rl', args.rl)):
    path = (root / relative).resolve()
    if output == path or path in output.parents or output in path.parents:
        raise ValueError('Analysis and evaluation inputs must be separate.')
    validated[label] = runpy.run_path(str(root / 'scripts/validate_evaluation.py'), init_globals={
        'EVALUATION_PATH': path, 'VALIDATE_CONFIG': cfg})['EVALUATION_RESULT']
    m = validated[label]['manifest']
    if m['model_label'] != label or m['split'] != 'test' or m['n_samples'] != cfg['evaluation']['n_samples']:
        raise ValueError('Primary analysis requires complete Base/RL test runs at the predeclared sample count.')
base = validated['base']['manifest']; rl = validated['rl']['manifest']
for field in ('protocol_id','settings','dataset_sha256','dataset_manifest_sha256','n_samples','question_count','model_identity','model_files','code_hashes'):
    if base[field] != rl[field]:
        raise ValueError(f'Unmatched Base/RL comparison: {field}')
selection = runpy.run_path(str(root / 'scripts/validate_selection.py'), init_globals={
    'SELECTION_PATH': root / rl['selection_path'], 'SELECTION_CONFIG': cfg})['SELECTION_RESULT']
if selection['selection_id'] != rl['selection_identity'] or selection['adapter_identity'] != rl['adapter_identity']:
    raise ValueError('RL test adapter differs from the frozen development selection.')
identity = {label: {k: validated[label][k] for k in ('manifest_sha256','samples_sha256')} for label in ('base','rl')}
identity.update(selection_identity=selection['selection_id'], bootstrap=repeats, seed=args.seed,
    analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
if output.exists():
    ready_path = output / 'ready.json'
    if not ready_path.is_file() or ready_path.is_symlink():
        raise ValueError('Analysis output is partial; preserve it and choose a new directory.')
    ready = json.loads(ready_path.read_text())
    if ready.get('identity') != identity or ready.get('complete') is not True:
        raise ValueError('Existing analysis belongs to a different comparison.')
    inventory = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file() and p != ready_path}
    if inventory != ready['artifacts_sha256']:
        raise ValueError('Completed analysis artifacts changed.')
    print(f'[ANALYSIS CACHED] {output}', flush=True)
    raise SystemExit(0)
print('[COMPARE 1/3] 33% | exact identities, paired samples and development selection verified', flush=True)
records = {label: {(r['id'],r['sample_index']): r for r in validated[label]['rows']} for label in ('base','rl')}
if set(records['base']) != set(records['rl']):
    raise ValueError('Base/RL grids are not paired.')
for key in records['base']:
    if records['base'][key]['sample_seed'] != records['rl'][key]['sample_seed']:
        raise ValueError('Paired sampling seeds differ.')
n = base['n_samples']
ks = [k for k in (1,4,16) if k <= n]
questions = validated['base']['dataset']
per_question = []
family_positions = defaultdict(list)
for question in sorted(questions,key=lambda q:q['id']):
    q = {'id':question['id'],'family_id':question['family_id'],'subject':question['source']['subject'],'n_samples':n}
    for label in ('base','rl'):
        values = [records[label][(question['id'],i)] for i in range(n)]
        correct = sum(r['correct'] is True for r in values)
        unknown = sum(r['correct'] is None for r in values)
        result = {'correct_count':correct,'unknown_count':unknown,
            'completed_count':sum(r['termination_reason']=='eos' for r in values),
            'timeout_count':sum(r['termination_reason']=='answer_timeout' for r in values),
            'termination_counts':dict(Counter(r['termination_reason'] for r in values))}
        for k in ks:
            low = 1 - math.comb(n-correct,k)/math.comb(n,k)
            high = 1 - math.comb(n-correct-unknown,k)/math.comb(n,k)
            result[f'pass_at_{k}'] = {'estimate':low,'unknown_bounds':[low,high]}
        q[label]=result
    for k in ks:
        q[f'delta_pass_at_{k}'] = q['rl'][f'pass_at_{k}']['estimate']-q['base'][f'pass_at_{k}']['estimate']
        q[f'delta_pass_at_{k}_unknown_bounds'] = [q['rl'][f'pass_at_{k}']['unknown_bounds'][0]-q['base'][f'pass_at_{k}']['unknown_bounds'][1],
            q['rl'][f'pass_at_{k}']['unknown_bounds'][1]-q['base'][f'pass_at_{k}']['unknown_bounds'][0]]
    family_positions[q['family_id']].append(len(per_question))
    per_question.append(q)
count = len(per_question)
series = {}
for k in ks:
    for label in ('base','rl'):
        series[f'{label}_pass_at_{k}'] = [q[label][f'pass_at_{k}']['estimate'] for q in per_question]
    series[f'delta_pass_at_{k}'] = [q[f'delta_pass_at_{k}'] for q in per_question]
if 16 in ks:
    series['delta_pass_at_1_minus_delta_pass_at_16'] = [q['delta_pass_at_1']-q['delta_pass_at_16'] for q in per_question]
boot = {name:[] for name in series}
rng = random.Random(args.seed)
families = list(family_positions)
for iteration in range(repeats):
    selected_families = rng.choices(families,k=len(families))
    indices = [i for family in selected_families for i in family_positions[family]]
    for name,values in series.items():
        boot[name].append(sum(values[i] for i in indices)/len(indices))
metrics = {}
for name,values in series.items():
    samples = sorted(boot[name])
    interval = []
    for probability in (.025,.975):
        position = probability*(len(samples)-1)
        lo = math.floor(position); hi = math.ceil(position)
        interval.append(samples[lo]+(samples[hi]-samples[lo])*(position-lo))
    metrics[name] = {'estimate':sum(values)/count,'ci95':interval}
for k in ks:
    for label in ('base','rl'):
        metrics[f'{label}_pass_at_{k}']['unknown_bounds'] = [sum(q[label][f'pass_at_{k}']['unknown_bounds'][j] for q in per_question)/count for j in (0,1)]
    metrics[f'delta_pass_at_{k}']['unknown_bounds'] = [sum(q[f'delta_pass_at_{k}_unknown_bounds'][j] for q in per_question)/count for j in (0,1)]
coverage = Counter()
for q in per_question:
    left,right=q['base']['correct_count']>0,q['rl']['correct_count']>0
    coverage['both' if left and right else 'base_only' if left else 'rl_only' if right else 'neither']+=1
diagnostics = {}
for label in ('base','rl'):
    values = validated[label]['rows']
    completed = sum(r['termination_reason']=='eos' for r in values)
    correct = sum(r['correct'] is True for r in values)
    diagnostics[label] = {'requested_count':len(values),'correct_count':correct,'completed_count':completed,
        'completed_only_accuracy': correct/completed if completed else None,
        'termination_counts':dict(Counter(r['termination_reason'] for r in values)),
        'status_counts':dict(Counter(r['status'] for r in values)),
        'unknown_count':sum(r['correct'] is None for r in values),
        'mean_output_tokens':sum(r['output_tokens'] for r in values)/len(values),
        'summed_answer_seconds':sum(r['seconds'] for r in values)}
limitations = [
    'Primary estimates are operational success under the fixed native-context/300-second/repetition policy. Unknown stopped attempts contribute no confirmed success; this does not label them mathematical errors.',
    'Answer timeouts are zero by the prespecified policy. Exact repetition, no-token stalls and context stops retain unknown correctness; unknown bounds are not confidence intervals.',
    'Base and RL use the same questions, deterministic sample seeds, scorer, prompt, dtype, backend and generation settings; the only intended model difference is the selected LoRA adapter.',
    'Checkpoint selection used all predeclared epoch candidates on development only; test outcomes never choose a checkpoint.',
    'Confidence intervals use paired whole-question-family bootstrap; repeated answers are not independent questions. Intervals are marginal, not multiplicity adjusted.',
    'Zero observed success in16 attempts does not establish zero capability; pass@16 is limited-sample coverage, not a complete reasoning boundary.',
    'This is one training seed and one source-filtered Level4 subset. It does not establish universal RL effects or validate every reasoning step.'
]
summary = {'schema_version':1,'experiment_type':'math_l4_base_rl','synthetic':False,
    'model_identity':base['model_identity'],'adapter_identity':rl['adapter_identity'],
    'selection_identity':selection['selection_id'],'selected_step':selection['selected_step'],
    'protocol_id':base['protocol_id'],'settings':base['settings'],'question_count':count,'family_count':len(families),
    'n_samples':n,'bootstrap':repeats,'bootstrap_seed':args.seed,
    'interval_method':'paired family-cluster percentile bootstrap,95%,linear quantiles',
    'metrics':metrics,'observed_coverage_counts':dict(coverage),'diagnostics':diagnostics,
    'identity':identity,'limitations':limitations}
lines=['# MATH Level4 Base/RL comparison','',
    f'Questions: {count}; samples/model/question: {n}; selected optimizer step: {selection["selected_step"]}.',
    'Primary scores use the fixed300-second and repetition policy; all requested attempts remain in the denominator.','',
    '| Metric | Base | RL | Paired difference | 95% interval for difference |',
    '|---|---:|---:|---:|---:|']
for k in ks:
    left=metrics[f'base_pass_at_{k}'];right=metrics[f'rl_pass_at_{k}'];delta=metrics[f'delta_pass_at_{k}']
    lines.append(f'| Operational pass@{k} | {100*left["estimate"]:.2f}% | {100*right["estimate"]:.2f}% | {100*delta["estimate"]:+.2f}pp | [{100*delta["ci95"][0]:+.2f}, {100*delta["ci95"][1]:+.2f}]pp |')
if 'delta_pass_at_1_minus_delta_pass_at_16' in metrics:
    delta=metrics['delta_pass_at_1_minus_delta_pass_at_16']
    lines += ['',f'Delta pass@1 minus delta pass@16: {100*delta["estimate"]:+.2f}pp; 95% interval [{100*delta["ci95"][0]:+.2f}, {100*delta["ci95"][1]:+.2f}]pp.']
lines += ['', 'Observed coverage counts: '+json.dumps(dict(coverage),sort_keys=True), '', '## Completion diagnostics','']
for label in ('base','rl'):
    lines.append(f'- {label}: '+json.dumps(diagnostics[label],sort_keys=True))
lines += ['', 'Unknown-outcome bounds (not confidence intervals):','']
for k in ks:
    lines.append(f'- pass@{k}: Base {metrics[f"base_pass_at_{k}"]["unknown_bounds"]}; RL {metrics[f"rl_pass_at_{k}"]["unknown_bounds"]}; difference {metrics[f"delta_pass_at_{k}"]["unknown_bounds"]}.')
lines += ['', '## Interpretation limits','']+['- '+item for item in limitations]+['']
print('[COMPARE 2/3] 67% | paired bootstrap and unknown bounds complete',flush=True)
output.parent.mkdir(parents=True,exist_ok=True)
stage=Path(tempfile.mkdtemp(prefix='.comparison-',dir=output.parent))
try:
    (stage/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (stage/'per_question.jsonl').write_text(''.join(json.dumps(q,sort_keys=True)+'\n' for q in per_question))
    (stage/'report.md').write_text('\n'.join(lines))
    ready={'complete':True,'identity':identity,'artifacts_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir() if p.is_file()}}
    (stage/'ready.json').write_text(json.dumps(ready,indent=2)+'\n')
    stage.rename(output)
finally:
    if stage.exists():shutil.rmtree(stage)
print(f'✅ [COMPARE 3/3] 100% | [SAVED] {output/"report.md"}',flush=True)
