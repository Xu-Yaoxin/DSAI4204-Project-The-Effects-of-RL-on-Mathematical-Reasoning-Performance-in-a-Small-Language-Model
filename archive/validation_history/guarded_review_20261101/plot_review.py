"""Render the frozen calibration metrics and runtime diagnostics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
summary = json.loads((root / 'replayed_analysis/summary.json').read_text())
audit = json.loads((root / 'runtime_audit.json').read_text())
groups = summary['groups']
labels = ['GSM-Symbolic main', 'MATH Level 4']
colors = ['#24799C', '#AD5B26']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white',
    'savefig.facecolor': 'white'})
fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout='constrained')
fig.suptitle('Baseline boundary evaluation | Qwen3.5-2B-Base', fontsize=17, fontweight='bold')
ax = axes[0, 0]
estimates = np.array([g['observed_success_per_request']['estimate'] for g in groups]) * 100
low = np.array([g['observed_success_per_request']['ci_low'] for g in groups]) * 100
high = np.array([g['observed_success_per_request']['ci_high'] for g in groups]) * 100
ax.axhspan(40, 50, color='#D9E7BB', alpha=.8, label='Target: 40–50%', zorder=0)
ax.bar(labels, estimates, width=.5, color=colors, zorder=2)
ax.errorbar(labels, estimates, yerr=[estimates-low, high-estimates], fmt='none', color='#222222', capsize=5, linewidth=1.5, zorder=3)
for i, value in enumerate(estimates):
    ax.text(i, value-8, f'{value:.2f}%', ha='center', va='center', color='white', fontweight='bold', fontsize=13)
ax.set_ylim(0, 85)
ax.set_ylabel('Correct / all requested attempts (%)')
ax.set_title('A  Success under the fixed stopping policy', loc='left')
ax.legend(loc='upper right', frameon=False, fontsize=9)
ax.yaxis.grid(alpha=.2, zorder=0)

ax = axes[0, 1]
category_labels = ['Correct', 'Numeric mismatch', 'Parse / ambiguity', '300 s timeout', 'Repetition stop']
category_colors = ['#24799C', '#87939D', '#E2B85B', '#CC694B', '#876A9A']
counts = [[g['correct_count'], g['status_counts'].get('incorrect',0),
    g['status_counts'].get('extraction_failed',0)+g['status_counts'].get('ambiguous',0),
    g['timeout_count'], g['termination_counts'].get('repetition_guard',0)] for g in groups]
left = np.zeros(2)
for i, (name, color) in enumerate(zip(category_labels, category_colors)):
    values = np.array([row[i] for row in counts])
    ax.barh(labels, values, left=left, label=name, color=color, height=.45)
    for j, val in enumerate(values):
        if val >= 12:
            ax.text(left[j]+val/2, j, str(val), ha='center', va='center', fontsize=9, color='white' if i in (0,3,4) else '#17212B')
    left += values
ax.invert_yaxis()
ax.set_xlim(0,400)
ax.set_xlabel('Responses (400 per group)')
ax.set_title('B  Where unsuccessful attempts went', loc='left')
ax.legend(loc='upper center', bbox_to_anchor=(.5,-.20), ncol=2, frameon=False, fontsize=9)

ax = axes[1, 0]
for i, g in enumerate(groups):
    values = [audit['groups'][g['group_id']]['correct_per_question_histogram'][str(k)] for k in range(5)]
    x = np.arange(5)+(i-.5)*.34
    ax.bar(x, values, width=.32, label=labels[i], color=colors[i], zorder=2)
    for xx,val in zip(x,values):
        ax.text(xx,val+.7,str(val),ha='center',fontsize=9)
ax.set_xticks(range(5))
ax.set_xlabel('Successful responses among 4 attempts per question')
ax.set_ylabel('Questions')
ax.set_ylim(0,52)
ax.set_title('C  Question-level success distribution', loc='left')
ax.legend(frameon=False,fontsize=9)
ax.yaxis.grid(alpha=.2,zorder=0)

ax = axes[1, 1]
reasons = ['eos','answer_timeout','repetition_guard']
minutes = [sum(g['termination'].get(reason,{}).get('seconds_sum',0) for g in audit['groups'].values())/60 for reason in reasons]
shares = [value*60/audit['all_answer_seconds']*100 for value in minutes]
reason_labels = ['EOS completed (736)', '300 s timeout (33)', 'Repetition stop (31)']
ax.barh(reason_labels,minutes,color=['#24799C','#CC694B','#876A9A'],height=.5)
ax.invert_yaxis()
for i,(value,share) in enumerate(zip(minutes,shares)):
    ax.text(value+3,i,f'{value:.1f} min  ({share:.1f}%)',va='center',fontsize=9)
ax.set_xlim(0,225)
ax.set_xlabel('Summed per-answer generation time (minutes)')
ax.set_title('D  Long failures dominate runtime',loc='left')
fig.supxlabel('100 questions per group × 4 samples | 95% family-cluster bootstrap intervals | Original scores retained',fontsize=9)
for suffix in ('png','pdf'):
    fig.savefig(root/f'overview.{suffix}',dpi=180)
plt.close(fig)
print(root/'overview.png')
