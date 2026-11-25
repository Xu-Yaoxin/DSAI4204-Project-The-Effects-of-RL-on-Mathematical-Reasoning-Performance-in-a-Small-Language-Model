"""Render the frozen baseline calibration as a publication-style 2x2 figure."""
from collections import Counter
import json
import os
from pathlib import Path

out = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(out / '.matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

summary = json.loads((out / 'recomputed_analysis/summary.json').read_text())
questions = [json.loads(line) for line in (out / 'recomputed_analysis/per_question.jsonl').read_text().split('\n') if line.strip()]
groups = summary['groups']
labels = ['GSM main', 'GSM P1', 'GSM P2', 'MATH L1–3', 'MATH L4–5']
colors = ['#167D9A', '#3566A8', '#6D5CA6', '#9B7420', '#AF594B']
y = np.arange(5)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.titlesize':13,
                     'axes.labelsize':11, 'xtick.labelsize':10, 'ytick.labelsize':11,
                     'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
fig, axes = plt.subplots(2, 2, figsize=(14, 10.4))
fig.subplots_adjust(left=.11, right=.97, top=.88, bottom=.20, wspace=.38, hspace=.52)
fig.suptitle('Where does the 2B baseline become difficult?', x=.11, y=.97, ha='left', fontsize=20, weight='bold')
fig.text(.11,.93,'Qwen3.5-2B-Base · 100 questions · 4 responses per question · 1,024-token output limit',fontsize=12,color='#444444')

ax = axes[0,0]
ax.axvspan(40,50,color='#E4EFDF',zorder=0,label='Target: 40–50%')
ax.axvline(45,color='#768C67',lw=1,ls='--',zorder=1)
for i,g in enumerate(groups):
 p=g['pass_at_1']
 ax.errorbar(p['estimate']*100,i,xerr=[[100*(p['estimate']-p['ci_low'])],[100*(p['ci_high']-p['estimate'])]],fmt='o',color=colors[i],capsize=4,markersize=7,lw=2)
 ax.text(98,i,f"{p['estimate']:.2%}",ha='right',va='center',fontsize=10)
ax.set(yticks=y,yticklabels=labels,xlim=(0,100),xlabel='Single-attempt success (%)')
ax.invert_yaxis()
ax.set_title('A  pass@1 with 95% family-bootstrap intervals',loc='left',pad=12)
ax.grid(axis='x',alpha=.18)
# The target band is explained in the figure note.

ax = axes[0,1]
levels = json.loads((out / 'math_by_level.json').read_text())['groups']
ax.axvspan(40,50,color='#E4EFDF',zorder=0)
for i,g in enumerate(levels):
    p = g['pass1']
    lo,hi = g['ci95_pass1']
    color = '#167D9A' if g['level']==4 else '#8C7661'
    ax.errorbar(p*100,i,xerr=[[100*(p-lo)],[100*(hi-p)]],fmt='o',color=color,capsize=4,markersize=8 if g['level']==4 else 6,lw=2)
    ax.text(116,i,f"{p:.2%}",ha='right',va='center',fontsize=10)
ax.set(yticks=y,yticklabels=[f"Level {g['level']} (n={g['questions']})" for g in levels],xlim=(0,119),xticks=[0,25,50,75,100],xlabel='Single-attempt success (%)')
ax.invert_yaxis()
ax.set_title('B  MATH levels: exploratory breakdown',loc='left',pad=12)
ax.grid(axis='x',alpha=.18)

ax = axes[1,0]
left=np.zeros(5)
shades=['#C7CDD3','#ABC9DB','#72A6C3','#337DA4','#124F71']
for k in range(5):
 values=np.array([sum(q['correct_count']==k for q in questions if q['group_id']==g['group_id']) for g in groups])
 ax.barh(y,values,left=left,color=shades[k],height=.6,label=f'{k}/4 correct',edgecolor='white',linewidth=.6)
 for i,v in enumerate(values):
  if v: ax.text(left[i]+v/2,i,str(v),ha='center',va='center',color='white' if k>=3 else '#19232A',fontsize=10)
 left += values
ax.set(yticks=y,yticklabels=labels,xlim=(0,20),xticks=[0,5,10,15,20],xlabel='Number of questions (20 per group)')
ax.invert_yaxis()
ax.set_title('C  Repeated success varies across questions',loc='left',pad=12)
ax.legend(ncol=3,loc='upper left',bbox_to_anchor=(-.02,-.20),frameon=False,fontsize=9)

ax = axes[1,1]
for offset,key,color,label in [(-.18,'extraction_failure_rate','#AD6A42','Answer extraction failed'),(.18,'truncation_rate','#7B579B','Reached output limit')]:
 values=np.array([g[key]*100 for g in groups])
 ax.barh(y+offset,values,height=.32,color=color,label=label)
 for i,v in enumerate(values): ax.text(v+.65,i+offset,f'{v:.2f}%',va='center',fontsize=9)
ax.set(yticks=y,yticklabels=labels,xlim=(0,38),xticks=[0,10,20,30],xlabel='Share of generated responses (%)')
ax.invert_yaxis()
ax.set_title('D  Parsing and length limits affect the scores',loc='left',pad=12)
ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
ax.legend(loc='upper left',bbox_to_anchor=(-.02,-.20),frameon=False,fontsize=9)
fig.text(.11,.035,'Exploratory calibration; 20 questions per group. Repeats are not independent questions. Green band: 40–50% target.\nIntervals are marginal. Parsing failures and truncation overlap. MATH results cover scalar, text-only subsets.',fontsize=10,color='#555555')
for suffix in ('png','pdf'):
 fig.savefig(out/f'boundary_overview.{suffix}',dpi=190,facecolor='white')
plt.close(fig)
print('[SAVED] boundary_overview.png and boundary_overview.pdf')
