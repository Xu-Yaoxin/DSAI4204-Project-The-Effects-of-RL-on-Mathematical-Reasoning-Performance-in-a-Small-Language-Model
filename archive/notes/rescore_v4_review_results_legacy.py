"""Independently verify the completed real-data rescore and publish its paths."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

package = Path(__file__).resolve().parent
output = package.parent / 'outputs/rescore_20261106'
ready = json.loads((output / 'ready.json').read_text())
summary = json.loads((output / 'summary.json').read_text())
assert ready['complete'] is True and ready['posthoc'] is True
assert ready['question_count'] == 200 and ready['answers_per_model'] == 3200
assert summary['question_count'] == summary['family_count'] == 200 and summary['n_samples'] == 16
assert summary['generation_provenance']['selected_step'] == 384
assert summary['generation_provenance']['checkpoint_selection_unchanged'] is True
assert summary['generation_provenance']['new_generation_performed'] is False
assert ready['identity']['scorer_sha256'] == hashlib.sha256((package / 'scorer.py').read_bytes()).hexdigest()
assert ready['identity']['runner_sha256'] == hashlib.sha256((package / 'rescore.py').read_bytes()).hexdigest()
observed_inventory = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in output.rglob('*') if path.is_file() and path.name != 'ready.json'}
assert observed_inventory == ready['artifacts_sha256']
source = Path(ready['identity']['experiment_root'])
run = source / 'outputs/math_l4_v1_fix1'
score_fields = ('correct', 'reward', 'extracted', 'status', 'format_compliant', 'truncated')
changes = [json.loads(line) for line in (output / 'changes.jsonl').read_text().splitlines()]
actual_changes = {}
independent_metrics = {}
checks = {}
for label, directory in [('base', 'test_base'), ('rl', 'test_rl_split_merged')]:
    original_bytes = (run / directory / 'samples.jsonl').read_bytes()
    original = [json.loads(line) for line in original_bytes.splitlines()]
    rescored = [json.loads(line) for line in (output / label / 'samples.jsonl').read_bytes().splitlines()]
    manifest = json.loads((output / label / 'manifest.json').read_text())
    assert hashlib.sha256(original_bytes).hexdigest() == manifest['generation_provenance']['source_samples_sha256']
    assert len(original) == len(rescored) == 3200
    old_by_key = {(row['id'], row['sample_index']): row for row in original}
    new_by_key = {(row['id'], row['sample_index']): row for row in rescored}
    assert len(old_by_key) == len(new_by_key) == 3200 and old_by_key.keys() == new_by_key.keys()
    counts = Counter()
    per_question = defaultdict(list)
    for key, row in new_by_key.items():
        old = old_by_key[key]
        assert row['old_score'] == {field: old[field] for field in score_fields}
        assert {field: value for field, value in row.items() if field not in (*score_fields, 'old_score', 'rescore')} == {
            field: value for field, value in old.items() if field not in score_fields}
        assert row['rescore']['scorer_version'] == '4.0.0'
        assert row['rescore']['scorer_sha256'] == ready['identity']['scorer_sha256']
        assert row['rescore']['scoring_applied'] == (old['termination_reason'] == 'eos')
        if old['termination_reason'] != 'eos':
            assert all(row[field] == old[field] for field in score_fields)
        changed = [field for field in score_fields if row[field] != old[field]]
        if changed:
            actual_changes[(label, *key)] = {'old_score': row['old_score'],
                'new_score': {field: row[field] for field in score_fields}, 'changed_fields': changed}
        counts['false_to_true'] += old['correct'] is False and row['correct'] is True
        counts['true_to_false'] += old['correct'] is True and row['correct'] is False
        counts['correct'] += row['correct'] is True
        counts['changed_rows'] += bool(changed)
        counts['stopped_preserved'] += old['termination_reason'] != 'eos'
        per_question[row['id']].append(row['correct'] is True)
    assert len(per_question) == 200 and all(len(values) == 16 for values in per_question.values())
    for k in (1, 4, 16):
        exact = sum((1 - Fraction(math.comb(16 - sum(values), k), math.comb(16, k))
            for values in per_question.values()), Fraction(0)) / 200
        assert abs(float(exact) - summary['metrics'][f'{label}_pass_at_{k}']['estimate']) < 1e-12
        independent_metrics[f'{label}_pass_at_{k}'] = {'exact_fraction': str(exact), 'estimate': float(exact)}
    checks[label] = dict(counts)
assert len(changes) == len(actual_changes) == summary['changed_samples']
assert len({(row['model_label'], row['id'], row['sample_index']) for row in changes}) == len(changes)
for row in changes:
    assert {field: row[field] for field in ('old_score', 'new_score', 'changed_fields')} == actual_changes[(row['model_label'], row['id'], row['sample_index'])]

download_run = Path('/Users/xu./Downloads/math_l4_selected_20261104/math_l4_v1/outputs/math_l4_v1_fix1')
original_paths = {'baseline': download_run / 'test_base/samples.jsonl',
    'rl_shard_0': download_run / 'rl_shards/shard_0/samples.jsonl',
    'rl_shard_1': download_run / 'rl_shards/shard_1/samples.jsonl',
    'rl_merged': run / 'test_rl_split_merged/samples.jsonl'}
for path in original_paths.values():
    assert path.is_file()
    original_manifest = json.loads((path.parent / 'manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_manifest['samples_sha256']
    if path.is_relative_to(download_run):
        counterpart = run / path.relative_to(download_run)
        assert hashlib.sha256(counterpart.read_bytes()).hexdigest() == original_manifest['samples_sha256']
verification = {'complete': True, 'scorer_version': '4.0.0', 'original_downloads_unchanged': True,
    'all_generation_fields_preserved': True, 'all_non_eos_scores_preserved': True,
    'changed_rows_audit_matches': True, 'counts': checks, 'exact_metric_verification': independent_metrics,
    'raw_answer_paths': {label: str(path) for label, path in original_paths.items()},
    'rescored_answer_paths': {label: str(output / label / 'samples.jsonl') for label in ('base', 'rl')},
    'ready_sha256': hashlib.sha256((output / 'ready.json').read_bytes()).hexdigest()}
(package / 'validation/completed_rescore_review.json').write_text(json.dumps(verification, indent=2) + '\n')

lines = ['**判分器 v4 本地重评分结果（2026-11-06）**', '',
    '全部 6,400 条回答已统一处理，模型、问题、标准答案和生成内容未改变。只有自然结束的回答应用 v4；超时和重复停止保留原评分。新数据中 old_score 保存旧判分，rescore 保存来源、版本和哈希。', '',
    '| 模型 | 原正确数 / 3,200 | 新正确数 / 3,200 | 错误转正确 | 正确转错误 | 全部评分变化记录 |',
    '|---|---:|---:|---:|---:|---:|']
for label in ('base', 'rl'):
    counts = checks[label]
    lines.append(f'| {label} | {summary["original_statistics"]["diagnostics"][label]["correct_count"]} | {counts["correct"]} | {counts["false_to_true"]} | {counts["true_to_false"]} | {counts["changed_rows"]} |')
lines += ['', '| 新评分指标 | Base | RL | RL−Base（百分点） | 配对 95% 区间 |', '|---|---:|---:|---:|---:|']
for k in (1, 4, 16):
    b, r, d = (summary['metrics'][f'{label}_pass_at_{k}'] for label in ('base', 'rl', 'delta'))
    lines.append(f'| pass@{k} | {b["estimate"]*100:.2f}% | {r["estimate"]*100:.2f}% | {d["estimate"]*100:+.2f} | [{d["ci95"][0]*100:+.2f}, {d["ci95"][1]*100:+.2f}] |')
lines += ['', '这是根据已观察到的解析问题进行的事后重评分。checkpoint-384 的选择沿用原开发集结果；没有根据测试结果重选模型。bootstrap 仍按题目配对，2,000 次、种子 42。一次训练种子、有限采样、固定 300 秒规则和不同机器的吞吐差异，仍限制结论的推广。', '',
    '修复限于有明确定义的标题和最终答案句式，不把所有提取失败都改判正确。空 #### 后换行给裸数字、冲突答案、未知或否定表述等仍按既有保守规则处理。该判分器核对最终标量答案，并不证明推理过程全部正确。', '',
    '**原始回答地址**', '']
for label, path in original_paths.items():
    lines.append(f'- {label}: [{path.name}](<{path}>)')
lines += ['', '**重评分结果地址**', '']
for label in ('base', 'rl'):
    path = output / label / 'samples.jsonl'
    lines.append(f'- {label}: [{path.name}](<{path}>)')
for filename, description in [('summary.json', '完整 Base/RL 对比'), ('changes.jsonl', '全部评分变化'), ('ready.json', '完成状态和哈希清单')]:
    path = output / filename
    lines.append(f'- {description}: [{filename}](<{path}>)')
(package / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(verification['counts'], indent=2))
print('[VERIFIED] 6400 records, unchanged raw payloads and stop scores, complete change audit, exact pass@k arithmetic.')
print('[SAVED] ' + str(package / 'RESULTS.md'))
