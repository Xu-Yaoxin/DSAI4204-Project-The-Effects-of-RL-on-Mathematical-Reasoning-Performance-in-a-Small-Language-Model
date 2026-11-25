"""Build a readable report from the validated full comparison and independent checks."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import statistics

review = Path(__file__).resolve().parent
experiment = review / 'experiment'
run = experiment / 'outputs/math_l4_v1_fix1'
analysis = run / 'main_analysis'
summary = json.loads((analysis / 'summary.json').read_text())
ready = json.loads((analysis / 'ready.json').read_text())
assert ready['complete'] is True
model_check = json.loads((review / 'model_checks.json').read_text())
assert model_check['verified'] is True and model_check['model_identity'] == summary['model_identity']
data_check = json.loads((review / 'data_checks.json').read_text())
assert data_check['data_ready'] is True and all(data_check['checks'].values())
for filename, expected in ready['artifacts_sha256'].items():
    assert hashlib.sha256((analysis / filename).read_bytes()).hexdigest() == expected, filename
independent = json.loads((review / 'independent_stats.json').read_text())
for name, values in summary['metrics'].items():
    for key, value in values.items():
        other = independent['scopes']['full']['metrics'][name][key]
        if isinstance(value, list):
            assert len(value) == len(other) and all(abs(a - b) < 1e-12 for a, b in zip(value, other)), (name, key)
        else:
            assert abs(value - other) < 1e-12, (name, key)
assert summary['question_count'] == summary['family_count'] == 200 and summary['n_samples'] == 16
assert summary['observed_coverage_counts'] == independent['scopes']['full']['coverage']

inventory = json.loads((review / 'source_inventory.json').read_text())
for relative, expected in inventory['files'].items():
    for parent in (Path(inventory['source']), experiment):
        payload = (parent / relative).read_bytes()
        assert len(payload) == expected['bytes'] and hashlib.sha256(payload).hexdigest() == expected['sha256'], str(parent / relative)
merged = [json.loads(line) for line in (run / 'test_rl_split_merged/samples.jsonl').read_text().splitlines()]
originals = {}
worker_checks = {}
for index in (0, 1):
    shard = run / 'rl_shards' / f'shard_{index}'
    rows = [json.loads(line) for line in (shard / 'samples.jsonl').read_text().splitlines()]
    manifest = json.loads((shard / 'manifest.json').read_text())
    state = json.loads((run / 'rl_shards' / f'worker_{index}_state.json').read_text())
    hardware = json.loads((run / 'rl_shards' / f'worker_{index}_hardware.json').read_text())
    assert state['status'] == 'completed' and state['completed_answers'] == state['target_answers'] == 1600
    assert state['shard_index'] == index and state['child_pid'] is None
    assert hardware['ready'] is True and hardware['shard_index'] == index
    assert hardware['gpu'] == manifest['gpu'] == 'NVIDIA GeForce RTX 4090'
    assert hardware['expected_package_versions'] == manifest['settings']['package_versions']
    assert hardware['config_sha256'] == hashlib.sha256((experiment / 'configs/math_l4.json').read_bytes()).hexdigest()
    assert manifest['execution_cpu_threads'] == {'intraop': 4, 'interop': 1}
    for row in rows:
        key = (row['id'], row['sample_index'])
        assert key not in originals
        originals[key] = row
    seconds = sum(r['seconds'] for r in rows)
    tokens = sum(r['output_tokens'] for r in rows)
    worker_checks[str(index)] = {'completed_answers': len(rows), 'restarts': state['restarts'],
        'gpu': hardware['gpu'], 'tokens_per_answer_second': tokens / seconds,
        'summed_answer_hours': seconds / 3600, 'mean_answer_seconds': seconds / len(rows),
        'mean_output_tokens': tokens / len(rows),
        'timeout_median_output_tokens': statistics.median(r['output_tokens'] for r in rows if r['termination_reason'] == 'answer_timeout')}
assert len(merged) == len(originals) == 3200
assert {(r['id'], r['sample_index']): r for r in merged} == originals
base_rows = [json.loads(line) for line in (run / 'test_base/samples.jsonl').read_text().splitlines()]
base_manifest = json.loads((run / 'test_base/manifest.json').read_text())
assert base_manifest['gpu'] == 'NVIDIA GeForce RTX 4090'
assert base_manifest['execution_cpu_threads'] == {'intraop': 4, 'interop': 1}
base_seconds = sum(r['seconds'] for r in base_rows)
base_tokens = sum(r['output_tokens'] for r in base_rows)
per_question = [json.loads(line) for line in (analysis / 'per_question.jsonl').read_text().splitlines()]
changes = Counter('rl_higher' if q['rl']['correct_count'] > q['base']['correct_count'] else
    'base_higher' if q['rl']['correct_count'] < q['base']['correct_count'] else 'equal' for q in per_question)
assert dict(changes) == independent['scopes']['full']['correct_count_comparison']
training = json.loads((run / 'formal/report.json').read_text())
diagnostics = summary['diagnostics']
metrics = summary['metrics']
subjects = {name.split(':', 1)[1]: value for name, value in independent['scopes'].items() if name.startswith('subject:')}
audit = {'complete': True, 'question_count': 200, 'base_answers': 3200, 'rl_answers': 3200,
    'original_downloads_unchanged': True, 'staged_source_files_verified': len(inventory['files']),
    'local_base_model_files_verified': len(model_check['files']),
    'dataset_checks_passed': len(data_check['checks']),
    'scoring_and_guard_replay': 'Production merge and analysis validators completed successfully for both test grids and development selection.',
    'merged_records_equal_original_records': True, 'independent_statistics_match': True,
    'workers': worker_checks, 'base_tokens_per_answer_second': base_tokens / base_seconds,
    'question_success_count_comparison': dict(changes),
    'training_evidence': {k: training[k] for k in ('optimizer_steps', 'effective_update_steps', 'changed_adapter_parameters', 'adapter_max_abs_delta', 'mixed_reward_groups', 'reward_groups')},
    'source_inventory_sha256': hashlib.sha256((review / 'source_inventory.json').read_bytes()).hexdigest(),
    'analysis_ready_sha256': hashlib.sha256((analysis / 'ready.json').read_bytes()).hexdigest(),
    'merge_provenance_sha256': hashlib.sha256((run / 'test_rl_split_merged/merge_provenance.json').read_bytes()).hexdigest()}
(review / 'integrity_audit.json').write_text(json.dumps(audit, indent=2) + '\n')

lines = [
    '**MATH Level 4：完整 200 题 Base/RL 对比（2026-11-06）**', '',
    '这次 LoRA-GRPO 训练在冻结的测试协议下有效。2B 模型的单次成功率从 43.53% 提高到 60.94%；16 次采样内至少出现一次正确答案的题目，从 171 道增加到 187 道。单次采样的提升更大，同时有限采样下的覆盖率也有提升。', '',
    '评测对象为 Qwen3.5-2B-Base 与 checkpoint-384 的 LoRA 版本。两批 RL 各含 100 题、1,600 次回答，合并后与 Base 的同一套 200 题逐题配对；每题每个模型均有 16 次回答，总计 6,400 次。所有尝试均保留在分母中。', '',
    '| 指标 | Base | RL | RL−Base（百分点） | 差值的配对 95% 区间 |',
    '|---|---:|---:|---:|---:|'
]
for k in (1, 4, 16):
    b, r, d = (metrics[f'{label}_pass_at_{k}'] for label in ('base', 'rl', 'delta'))
    lines.append(f'| pass@{k} | {b["estimate"]*100:.2f}% | {r["estimate"]*100:.2f}% | {d["estimate"]*100:+.2f} | [{d["ci95"][0]*100:+.2f}, {d["ci95"][1]*100:+.2f}] |')
lines += ['',
    'pass@1 是从每题 16 次回答估计的单次成功率，并非只看第一次回答；pass@4 使用预先固定的组合估计量；pass@16 是这 16 次中至少成功一次的题目比例。置信区间按题目配对 bootstrap，重复 2,000 次，种子 42；200 个题目族均唯一。没有把同一道题的 16 次回答当作 16 个独立题目。区间为逐指标区间，未进行多重比较校正。', '',
    '正确回答总数为 1,393/3,200 → 1,950/3,200，增加 557 次。相对于 Base，单次成功率约提高 40.0%。146 道题的正确次数增加，29 道持平，25 道减少；并非每道题都变好。', '',
    '| 16 次采样中的观察结果 | 题目数 |', '|---|---:|',
    '| 两个模型都至少成功一次 | 167 |', '| 只有 RL 至少成功一次 | 20 |',
    '| 只有 Base 至少成功一次 | 4 |', '| 两个模型均未观察到成功 | 9 |', '',
    '完整结果支持“RL 提高采样成功率，并改善本协议下的有限采样覆盖率”。相较此前仅分析第一批 100 题，完整数据的 pass@16 提升区间已高于零。', '',
    'pass@1 的增幅比 pass@16 大 9.41 个百分点，其配对 95% 区间为 [4.31, 14.69]。这与“主要收益体现在更稳定地采到正确答案”相符；仅凭 16 次未答对，不能证明 Base 不具备该题的潜在能力。当前结果也不能证明出现了训练前绝对不存在的推理能力。', '',
    '两个分片分别与自己的同题 Base 对照：第一批 pass@1 为 45.50% → 60.31%，第二批为 41.56% → 61.56%。两批方向一致；它们是同一模型、同一测试集的分片，不是两个独立训练重复实验。', '',
    '| 学科 | 题数 | Base pass@1 | RL pass@1 | 提升（百分点） |', '|---|---:|---:|---:|---:|'
]
subject_names = {'algebra': '代数', 'counting_and_probability': '计数与概率', 'geometry': '几何', 'intermediate_algebra': '中级代数', 'number_theory': '数论', 'prealgebra': '初等代数', 'precalculus': '预备微积分'}
for subject, values in subjects.items():
    m = values['metrics']
    lines.append(f'| {subject_names[subject]} | {values["question_count"]} | {m["base_pass_at_1"]["estimate"]*100:.2f}% | {m["rl_pass_at_1"]["estimate"]*100:.2f}% | {m["delta_pass_at_1"]["estimate"]*100:+.2f} |')
lines += ['',
    '七个学科的单次成功率均提高。中级代数仍然最难：从 15.16% 升到 30.31%。几何只有 9 题、预备微积分只有 11 题，不宜据此给学科提升幅度排出稳定名次。学科拆分属于探索性分析。', '',
    '| 输出诊断（每个模型 3,200 次） | Base | RL |', '|---|---:|---:|',
    '| 自然结束 EOS | 2,770（86.56%） | 2,742（85.69%） |',
    '| 超过 300 秒 | 219（6.84%） | 320（10.00%） |',
    '| 重复输出被终止 | 211（6.59%） | 138（4.31%） |',
    '| 答案提取失败 | 176（5.50%） | 92（2.88%） |',
    '| 答案冲突 / ambiguous | 41（1.28%） | 0 |',
    '| 平均输出 token 数 | 1,441.9 | 1,403.4 |', '',
    '提取失败和重复输出减少，但超时增加。平均输出长度仅减少约 2.7%，不能据此声称推理长度大幅缩短。仅看 EOS 回答，正确率为 50.29% → 71.12%；这个条件统计排除了未正常结束的回答，存在选择偏差，不能代替主结果。提取改善、回答内容变化和停止行为共同影响主指标，不能仅凭上述分类把它们分解为独立的因果贡献。', '',
    '计算速度也需要单独说明。Base 约 40.6 token/s，第一批 RL 约 20.0，第二批约 28.5（总生成 token 除以各回答耗时）。三台均为 RTX 4090，软件、线程设置、提示词、采样参数和终止规则已匹配，但实际吞吐不同。相同 300 秒限时下，超时回答的中位输出长度约为 12,255 / 6,040 / 8,558 token；因此当前对比衡量的是实际协议下的成功率，并非严格相同 token 预算下的纯推理能力。Base 的回答耗时合计约 31.57 小时；RL 两批回答耗时合计约 50.84 小时，分布在两台机器上。这里的“采样效率提高”不能理解为“运行更快”。', '',
    '评分重放与冻结的判分器一致，但这不代表所有数学判断都没有误差。第一批 math-l4-test-algebra-0739 的 sample_index 11、14 等回答受 #### 小标题影响；第二批 number_theory-0052 的 sample_index 1 正确得到个位数字 7，却被标题干扰判为提取失败；geometry-0022 的 sample_index 3 正确得到最大面积 2500，也因 Final Answer 后的文字触发提取失败。机械扫描另列出 18 条“正确数值候选与无效小标题候选并存”的提取失败记录，这只是待复核候选数，并非已穷尽的误判数量。具体记录见 quality_review.json。当前分析没有为得到更好的结果而修正分数。正式提交前，应单独版本化修复判分器，并对全部 Base/RL 原始回答统一重评分，同时保留本次冻结结果。', '',
    '有限人工抽查的 6 条正确标签，其最终数值答案均正确，但其中一条中间推导存在错误，随后以直接计算得到正确数值。因此这些指标评价最终答案成功率，并不验证每一步推理的正确性；本次也未对全部 6,400 条回答进行人工数学复核。', '',
    '重复停止的回答保留 unknown，在主指标中不提供已确认成功；300 秒超时按事先约定记失败。若把 unknown 的数学正确性在最坏/最好情况间变化，单次成功率差值仍为 +10.81 至 +21.72 个百分点；pass@16 差值范围为 −3 至 +14 个百分点。这些是敏感性上下界，不是置信区间，也不是取消停止规则后的能力估计。因此单次成功率改善更稳健，“内在推理边界扩大”需要更强证据。', '',
    '训练证据与测试结果一致：完成 384 次优化器更新，376 步有有效更新；192 个适配器参数张量发生变化，Base 保持冻结。开发集用于选择 checkpoint-384，测试结果未参与检查点选择。512 道训练题、128 道验证题和 200 道测试题通过既有数据校验，集合互不重叠；此前边界探索的题目也已排除。这里的隔离检查不涉及原始模型预训练语料污染。', '',
    '结论限于一次训练种子、MATH Level 4 数值答案筛选子集及当前判分器；不应直接推广到全部 MATH 或所有强化学习算法。', '',
    '合并保存为 experiment/outputs/math_l4_v1_fix1/test_rl_split_merged/。合并记录逐条等于两个原分片；原始下载文件未改动。本地 Base 权重与 tokenizer 等全部 7 个文件也已重新计算哈希，与评测身份一致。校验清单见 integrity_audit.json，统计机器可读结果见 experiment/outputs/math_l4_v1_fix1/main_analysis/summary.json，独立复算见 independent_stats.json；2×2 总览图见 figures/math_l4_base_vs_rl_summary.png，同目录含可导出的 PDF。', ''
]
(review / 'RESULTS_ZH.md').write_text('\n'.join(lines))
print('[REVIEW COMPLETE] 6400 test answers | 200 paired questions | independent statistics match | originals unchanged')
print('[SAVED] ' + str(review / 'RESULTS_ZH.md'))
print('[SAVED] ' + str(review / 'integrity_audit.json'))
