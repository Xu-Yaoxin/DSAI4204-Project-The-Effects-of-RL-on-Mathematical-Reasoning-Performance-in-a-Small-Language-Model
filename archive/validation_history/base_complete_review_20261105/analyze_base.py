"""Audit the downloaded complete Base grid and summarize frozen recorded scores."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import statistics

snapshot = Path('/Users/xu./Downloads/math_l4_selected_20261104/math_l4_v1')
trusted = Path(__file__).resolve().parents[3] / 'math_l4_v1'
run = snapshot / 'outputs/math_l4_v1_fix1'
source = run / 'test_base'
output = Path(__file__).resolve().parent
manifest = json.loads((source / 'manifest.json').read_text())
receipt = json.loads((run / 'base_stop_receipt.json').read_text())
cfg = json.loads((trusted / 'configs/math_l4.json').read_text())
assert manifest['complete'] is True and manifest['model_label'] == 'base'
assert manifest['split'] == 'test' and manifest['synthetic'] is False
assert manifest['config'] == cfg
assert all(manifest.get(k) is None for k in ['adapter_path', 'adapter_identity', 'selection_identity'])
assert receipt['complete'] is True and receipt['status'] == 'stopped_after_base'
for name, expected in manifest['code_hashes'].items():
    assert hashlib.sha256((trusted / 'scripts' / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256((snapshot / 'scripts' / name).read_bytes()).hexdigest() == expected
for name, key in [('dataset_path', 'dataset_sha256'), ('dataset_manifest_path', 'dataset_manifest_sha256')]:
    assert hashlib.sha256((snapshot / manifest[name]).read_bytes()).hexdigest() == manifest[key]
    assert hashlib.sha256((trusted / manifest[name]).read_bytes()).hexdigest() == manifest[key]
protocol = {k: manifest[k] for k in ['dataset_sha256', 'dataset_manifest_sha256', 'n_samples', 'settings', 'code_hashes', 'experiment_type']}
assert hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == manifest['protocol_id']
dataset = [json.loads(line) for line in (snapshot / manifest['dataset_path']).read_text().splitlines() if line.strip()]
by_id = {r['id']: r for r in dataset}
n = manifest['n_samples']
assert n == 16 and len(by_id) == len(dataset) == manifest['question_count'] == 200
expected_grid = {(r['id'], i) for r in dataset for i in range(n)}
seen = set()
rows = []
digest = hashlib.sha256()
with (source / 'samples.jsonl').open('rb') as stream:
    for line in stream:
        digest.update(line)
        assert line.endswith(b'\n') and line.strip()
        row = json.loads(line)
        key = (row['id'], row['sample_index'])
        assert key in expected_grid and key not in seen
        assert all(row.get(k) == v for k, v in by_id[row['id']].items())
        expected_seed = int.from_bytes(hashlib.sha256(f"{manifest['settings']['seed']}:{key[0]}:{key[1]}".encode()).digest()[:4], 'big')
        assert row['sample_seed'] == expected_seed
        assert row['output_tokens'] == len(row['output_token_ids'])
        assert row['correct'] is True or row['correct'] is False or row['correct'] is None
        assert row['seconds'] >= 0 and math.isfinite(row['seconds'])
        seen.add(key)
        rows.append({k: row[k] for k in ['id', 'sample_index', 'correct', 'status', 'termination_reason', 'seconds', 'output_tokens', 'format_compliant']})
assert seen == expected_grid
assert len(rows) == manifest['total_samples'] == receipt['base_total_samples'] == 3200
assert digest.hexdigest() == manifest['samples_sha256'] == receipt['base_samples_sha256']
groups = defaultdict(list)
for row in rows:
    groups[row['id']].append(row)
per_question = []
for item in dataset:
    answers = groups[item['id']]
    c = sum(r['correct'] is True for r in answers)
    u = sum(r['correct'] is None for r in answers)
    result = {'id': item['id'], 'subject': item['source']['subject'], 'correct': c, 'unknown': u}
    for k in [1, 4, 16]:
        result[f'pass_at_{k}'] = 1 - math.comb(n - c, k) / math.comb(n, k)
        result[f'pass_at_{k}_unknown_upper'] = 1 - math.comb(n - c - u, k) / math.comb(n, k)
    per_question.append(result)
summary = {
    'samples_sha256': digest.hexdigest(), 'questions': len(dataset), 'answers': len(rows),
    'integrity': 'sample hashes match manifest and stop receipt; full unique question/sample grid, metadata, seeds, protocol and frozen code hashes verified',
    'scoring_scope': 'Recorded frozen scores; this review does not rejudge mathematical answers or replay every guard event. No model, GPU or network used.',
    'correct': sum(r['correct'] is True for r in rows),
    'incorrect_or_policy_failure': sum(r['correct'] is False for r in rows),
    'unknown': sum(r['correct'] is None for r in rows),
    'statuses': dict(Counter(r['status'] for r in rows)),
    'termination_counts': dict(Counter(r['termination_reason'] for r in rows)),
    'question_correct_histogram': dict(sorted(Counter(r['correct'] for r in per_question).items())),
    'no_confirmed_success_questions': sum(r['correct'] == 0 for r in per_question),
    'all_16_correct_questions': sum(r['correct'] == n for r in per_question),
    'mean_output_tokens': statistics.mean(r['output_tokens'] for r in rows),
    'median_output_tokens': statistics.median(r['output_tokens'] for r in rows),
    'mean_answer_seconds': statistics.mean(r['seconds'] for r in rows),
    'median_answer_seconds': statistics.median(r['seconds'] for r in rows),
    'summed_answer_seconds': sum(r['seconds'] for r in rows),
    'invocation_seconds': manifest['invocation_seconds'],
    'termination_seconds': {reason: sum(r['seconds'] for r in rows if r['termination_reason'] == reason) for reason in sorted({r['termination_reason'] for r in rows})},
    'format_compliant_count': sum(r['format_compliant'] is True for r in rows),
    'metrics': {}, 'subjects': {},
    'limitations': ['Unknown-bound intervals are sensitivity bounds, not confidence intervals.',
                    'Repeated attempts are not independent questions.',
                    'No success in 16 attempts does not prove a problem is impossible for Base.',
                    'Results apply to the frozen scalar text-only MATH Level 4 subset and its 300-second/repetition policy.',
                    'Base-only results do not establish an RL improvement.']}
for k in [1, 4, 16]:
    summary['metrics'][f'pass_at_{k}'] = {
        'confirmed_success_estimate': statistics.mean(q[f'pass_at_{k}'] for q in per_question),
        'unknown_upper_bound': statistics.mean(q[f'pass_at_{k}_unknown_upper'] for q in per_question)}
for subject in sorted({q['subject'] for q in per_question}):
    selected = [q for q in per_question if q['subject'] == subject]
    summary['subjects'][subject] = {'questions': len(selected), 'answers': len(selected) * n,
        'correct': sum(q['correct'] for q in selected),
        **{f'pass_at_{k}': statistics.mean(q[f'pass_at_{k}'] for q in selected) for k in [1, 4, 16]}}
(output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
(output / 'per_question.json').write_text(json.dumps(per_question, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(summary, indent=2, ensure_ascii=False))
print('[BASE REVIEW COMPLETE] 3200 answers; CPU-only integrity and descriptive statistics; saved=' + str(output / 'summary.json'))
