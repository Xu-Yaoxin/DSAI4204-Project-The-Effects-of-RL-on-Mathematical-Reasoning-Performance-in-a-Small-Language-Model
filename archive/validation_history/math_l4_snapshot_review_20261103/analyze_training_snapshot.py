"""Replay downloaded training rewards and summarize the completed runs on CPU."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

source = Path('/Users/xu./Downloads/math_l4_snapshot_20261103')
trusted = Path(__file__).resolve().parents[3] / 'math_l4_v1'
destination = Path(__file__).resolve().parent
run = source / 'outputs/math_l4_v1_fix1'
scorer_path = trusted / 'scripts/verify_answers.py'
scorer_bytes = scorer_path.read_bytes()
scorer_hash = hashlib.sha256(scorer_bytes).hexdigest()
assert hashlib.sha256((source / 'scripts/verify_answers.py').read_bytes()).hexdigest() == scorer_hash
scorer_code = compile(scorer_bytes, str(scorer_path), 'exec')
config = json.loads((source / 'configs/math_l4.json').read_text())
group_size = config['training']['num_generations']
questions_per_update = config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'] // group_size
diagnostic = {
    'source': str(source), 'run': 'outputs/math_l4_v1_fix1',
    'scorer_sha256': scorer_hash, 'gpu_used': False, 'model_loaded': False,
    'source_mutated': False, 'training_runs': {}, 'errors': [],
    'interpretation': 'Training rewards describe sampled answers on repeated training questions. Engineering gates and improved training reward do not establish held-out improvement.',
}
for mode in ('engineering', 'pilot', 'formal'):
    directory = run / mode
    report = json.loads((directory / 'report.json').read_text())
    logs = json.loads((directory / 'trainer_logs.json').read_text())
    split = 'train' if mode == 'formal' else mode
    data = [json.loads(line) for line in (source / 'data/processed' / (split + '.jsonl')).read_text().splitlines() if line.strip()]
    by_id = {row['id']: row for row in data}
    raw = (directory / 'reward_audit.jsonl').read_bytes()
    assert not raw or raw.endswith(b'\n'), mode
    groups = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rank_groups = [json.loads(line) for line in (directory / 'reward_audit.rank-0.jsonl').read_bytes().splitlines() if line.strip()]
    assert groups == rank_groups, mode
    inputs = []
    saved = []
    owners = []
    for group in groups:
        assert group['question_id'] in by_id, (mode, group['question_id'])
        assert len(group['samples']) == len(group['rewards']) == group_size
        assert group['mixed_rewards'] == (min(group['rewards']) < max(group['rewards']))
        for index, sample in enumerate(group['samples']):
            inputs.append({'text': sample['text'], 'answer': by_id[group['question_id']]['answer'], 'truncated': sample['truncated']})
            saved.append(sample)
            owners.append((group, index))
    replay_state = {'SCORE_INPUT': inputs, '__file__': str(scorer_path)}
    exec(scorer_code, replay_state)
    mismatches = []
    for index, replay in enumerate(replay_state['SCORE_OUTPUT']):
        if inputs[index]['truncated']:
            replay.update(correct=False, reward=0.0, status='unfinished', extracted=None, format_compliant=False, truncated=True)
        group, group_index = owners[index]
        if replay != saved[index]['verifier'] or replay['reward'] != group['rewards'][group_index]:
            mismatches.append({'index': index, 'question_id': group['question_id'], 'replayed': replay, 'saved': saved[index]['verifier']})
    step_groups = defaultdict(list)
    for group in groups:
        step_groups[group['optimizer_step_before_update']].append(group)
    observed_steps = sorted(step_groups)
    assert observed_steps == list(range(report['optimizer_steps'])), mode
    assert all(len(rows) == questions_per_update and len({row['question_id'] for row in rows}) == questions_per_update for rows in step_groups.values()), mode
    metric_rows = [row for row in logs if 'loss' in row]
    assert [row['step'] for row in metric_rows] == list(range(1, report['optimizer_steps'] + 1)), mode
    nonfinite = [{'step': row.get('step'), 'key': key} for row in logs for key, value in row.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value)]
    nonzero_grad = sum(row.get('grad_norm', 0) > 0 for row in metric_rows)
    counts = {
        'reward_groups': len(groups), 'completion_count': len(saved),
        'mixed_reward_groups': sum(group['mixed_rewards'] for group in groups),
        'correct_completion_count': sum(sample['verifier']['correct'] for sample in saved),
        'truncation_count': sum(sample['truncated'] for sample in saved),
        'format_compliant_count': sum(sample['verifier']['format_compliant'] for sample in saved),
        'effective_update_steps': nonzero_grad,
    }
    for key, value in counts.items():
        assert report[key] == value, (mode, key, value, report[key])
    assert nonfinite == report['nonfinite_metrics'] == [], mode
    log_audit_errors = []
    for row in metric_rows:
        step_samples = [sample for group in step_groups[row['step'] - 1] for sample in group['samples']]
        expected = {
            'reward': statistics.mean(sample['verifier']['reward'] for sample in step_samples),
            'completions/clipped_ratio': statistics.mean(sample['truncated'] for sample in step_samples),
            'completions/mean_length': statistics.mean(sample['completion_tokens'] for sample in step_samples),
        }
        for key, value in expected.items():
            if abs(row[key] - value) > 1e-6:
                log_audit_errors.append({'step': row['step'], 'field': key, 'log': row[key], 'audit': value})
    steps_per_epoch = len(data) // questions_per_update
    summaries = []
    question_epochs = defaultdict(dict)
    for epoch_index in range(math.ceil(report['optimizer_steps'] / steps_per_epoch)):
        lower = epoch_index * steps_per_epoch
        upper = min(report['optimizer_steps'], lower + steps_per_epoch)
        epoch_groups = [group for group in groups if lower <= group['optimizer_step_before_update'] < upper]
        epoch_samples = [sample for group in epoch_groups for sample in group['samples']]
        epoch_logs = [row for row in metric_rows if lower < row['step'] <= upper]
        correct_count = sum(sample['verifier']['correct'] for sample in epoch_samples)
        truncated_count = sum(sample['truncated'] for sample in epoch_samples)
        for group in epoch_groups:
            assert epoch_index + 1 not in question_epochs[group['question_id']], (mode, epoch_index, group['question_id'])
            question_epochs[group['question_id']][epoch_index + 1] = statistics.mean(group['rewards'])
        summaries.append({
            'epoch': epoch_index + 1, 'steps': [lower + 1, upper], 'questions': len(epoch_groups),
            'responses': len(epoch_samples), 'correct': correct_count,
            'reward_rate': correct_count / len(epoch_samples),
            'unfinished': truncated_count, 'unfinished_rate': truncated_count / len(epoch_samples),
            'mixed_reward_group_rate': statistics.mean(group['mixed_rewards'] for group in epoch_groups),
            'mean_output_tokens': statistics.mean(sample['completion_tokens'] for sample in epoch_samples),
            'median_output_tokens': statistics.median(sample['completion_tokens'] for sample in epoch_samples),
            'eos_correct_rate': correct_count / (len(epoch_samples) - truncated_count),
            'verifier_status_counts': dict(Counter(sample['verifier']['status'] for sample in epoch_samples)),
            'format_compliant_rate': statistics.mean(sample['verifier']['format_compliant'] for sample in epoch_samples),
            'entropy_mean': statistics.mean(row['entropy'] for row in epoch_logs),
            'grad_norm_mean': statistics.mean(row['grad_norm'] for row in epoch_logs),
            'positive_grad_steps': sum(row['grad_norm'] > 0 for row in epoch_logs),
            'step_time_mean_seconds': statistics.mean(row['step_time'] for row in epoch_logs),
            'logged_step_time_sum_seconds': sum(row['step_time'] for row in epoch_logs),
        })
    paired = None
    if len(summaries) == 3:
        assert set(question_epochs) == set(by_id)
        assert all(set(values) == {1, 2, 3} for values in question_epochs.values())
        differences = [values[3] - values[1] for values in question_epochs.values()]
        paired = {
            'questions': len(differences), 'mean_epoch3_minus_epoch1_reward': statistics.mean(differences),
            'improved_questions': sum(value > 0 for value in differences),
            'unchanged_questions': sum(value == 0 for value in differences),
            'worsened_questions': sum(value < 0 for value in differences),
            'note': 'Descriptive paired results for the same training questions, eight generated answers per question per epoch; not a held-out or Base comparison.',
        }
    windows = {}
    for name, window in [('first_32_updates', metric_rows[:32]), ('last_32_updates', metric_rows[-32:])]:
        windows[name] = {key: statistics.mean(row[key] for row in window) for key in ['reward', 'completions/clipped_ratio', 'completions/mean_length', 'entropy', 'grad_norm', 'step_time']}
        windows[name]['count'] = len(window)
    stats = {
        'status': report['status'], 'optimizer_steps': report['optimizer_steps'],
        'training_seconds': report['training_seconds'], 'training_hours': report['training_seconds'] / 3600,
        'epochs_completed': report['epochs_completed'], 'gates': report['gates'],
        'changed_adapter_tensor_count': report['changed_adapter_parameters'],
        'adapter_max_abs_delta': report['adapter_max_abs_delta'],
        'base_hash_unchanged_in_recorded_audit': all(audit['base_sha256_before'] == audit['base_sha256_after'] for audit in report['rank_audits']),
        'resume_history': report['resume_history'], 'peak_cuda_allocated_gib': report['peak_cuda_allocated_bytes'] / 2**30,
        'peak_cuda_reserved_gib': report['peak_cuda_reserved_bytes'] / 2**30,
        'counts': counts, 'unique_questions': len(question_epochs),
        'reward_replay_checked': len(saved), 'reward_replay_mismatches': mismatches,
        'trainer_log_vs_reward_audit_mismatches': log_audit_errors,
        'nonfinite_metrics': nonfinite,
        'zero_gradient_steps': [row['step'] for row in metric_rows if row['grad_norm'] == 0],
        'zero_gradient_steps_all_groups_uniform_reward': all(not group['mixed_rewards'] for row in metric_rows if row['grad_norm'] == 0 for group in step_groups[row['step'] - 1]),
        'loss_absolute_max': max(abs(row['loss']) for row in metric_rows),
        'epochs': summaries, 'paired_training_questions_epoch1_vs3': paired, 'windows': windows,
        'verifier_status_counts': dict(Counter(sample['verifier']['status'] for sample in saved)),
        'notes': ['Saved audit data and logs were replayed without loading model or optimizer files.', 'Recorded base hashes and update gates are cross-checked evidence, not an independent re-execution of GPU training.'],
    }
    diagnostic['training_runs'][mode] = stats
    if mismatches or log_audit_errors:
        diagnostic['errors'].append(mode)
    print(f"[TRAIN AUDIT] {mode} | steps={report['optimizer_steps']} | replay={len(saved)} | mismatches={len(mismatches)} | log_mismatches={len(log_audit_errors)}", flush=True)
diagnostic['training_reward_replays'] = sum(row['reward_replay_checked'] for row in diagnostic['training_runs'].values())
diagnostic['passed'] = not diagnostic['errors']
output = destination / 'training_diagnostic.json'
output.write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
print(f'[SAVED] {output}', flush=True)
