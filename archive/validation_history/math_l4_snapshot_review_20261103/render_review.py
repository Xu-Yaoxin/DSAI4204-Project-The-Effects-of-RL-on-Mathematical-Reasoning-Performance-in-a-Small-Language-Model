"""Create a descriptive training figure and an explicitly unofficial sensitivity check."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output = Path(__file__).resolve().parent
training = json.loads((output / 'training_diagnostic.json').read_text())['training_runs']['formal']
epochs = training['epochs']
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11, 'figure.facecolor': 'white'})
figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))
panels = [
    ('Correct reward on training answers', 'Answers (%)', [100 * e['reward_rate'] for e in epochs], '#2563eb', (0, 65)),
    ('Unfinished training answers', 'Answers (%)', [100 * e['unfinished_rate'] for e in epochs], '#c2410c', (0, 35)),
    ('Answer extraction failures', 'Answers (%)', [100 * e['verifier_status_counts'].get('extraction_failed', 0) / e['responses'] for e in epochs], '#7c3aed', (0, 25)),
    ('Mean generated length', 'Tokens per answer', [e['mean_output_tokens'] for e in epochs], '#0f766e', (0, 1150)),
]
for axis, (title, ylabel, values, color, limits) in zip(axes.flat, panels):
    axis.plot([1, 2, 3], values, marker='o', markersize=7, linewidth=2.2, color=color)
    for epoch, value in zip([1, 2, 3], values):
        label = f'{value:.1f}' if 'Tokens' in ylabel else f'{value:.2f}%'
        axis.annotate(label, (epoch, value), xytext=(0, 11), textcoords='offset points', ha='center', fontsize=11)
    axis.set(title=title, xlabel='Training epoch', ylabel=ylabel, ylim=limits, xlim=(0.7, 3.3))
    axis.set_xticks([1, 2, 3])
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', alpha=0.2)
figure.suptitle('MATH Level 4 / Qwen3.5-2B: completed formal training', fontsize=16, y=0.98)
figure.text(0.5, 0.02, '512 training questions; 4,096 answers per epoch. These are training metrics, not Base/RL test gains.', ha='center', fontsize=10)
figure.tight_layout(rect=[0, 0.055, 1, 0.94], h_pad=2.0, w_pad=2.0)
figure.savefig(output / 'training_overview.png', dpi=180)
figure.savefig(output / 'training_overview.pdf')
plt.close(figure)

source = Path('/Users/xu./Downloads/math_l4_snapshot_20261103')
rows = [json.loads(line) for line in (source / 'outputs/math_l4_v1_fix1/dev/checkpoint-128/samples.jsonl').read_text().splitlines()]
excluded = {'math-l4-train-precalculus-0660', 'math-l4-train-precalculus-0471'}
retained = [row for row in rows if row['id'] not in excluded]
correct = sum(row['correct'] is True for row in retained)
unknown = sum(row['correct'] is None for row in retained)
flagged = [row for row in retained if row['id'] == 'math-l4-train-number_theory-0194' and row['sample_index'] == 1]
assert len(flagged) == 1 and flagged[0]['correct'] is False
diagnostic = {
    'official_score_changed': False,
    'purpose': 'Sensitivity only; not a new selection rule or corrected experiment result.',
    'exclusion_basis': 'Two source-confirmed multi-value questions incompatible with the frozen scalar-answer design, including one with no observed answer.',
    'excluded_question_ids': sorted(excluded), 'removed_available_answers': len(rows) - len(retained),
    'remaining_answers': len(retained), 'original_correct_on_remaining': correct, 'remaining_unknown': unknown,
    'exclude_only_rate': correct / len(retained),
    'exclude_plus_one_manually_confirmed_markup_false_negative_rate': (correct + 1) / len(retained),
    'limits': ['No change to source labels, recorded scores, selection or scorer.',
               'Other ambiguities retain the frozen conservative policy.',
               'The partial ordered dev prefix is not representative proof of full-split accuracy.'],
}
(output / 'diagnostic_sensitivity.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
print(json.dumps(diagnostic, indent=2))
print('[SAVED] training_overview.png and training_overview.pdf')
