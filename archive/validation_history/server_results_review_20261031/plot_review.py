"""Plot archived primary results alongside a clearly labelled post-hoc diagnostic."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rlvr-review-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

archive = root.parents[3] / "server_results/run_20261031/code"
primary = json.loads((archive / "outputs/v1_2/main_analysis/summary.json").read_text())
secondary = json.loads((root / "secondary_comparison.json").read_text())
selection = json.loads((archive / "outputs/v1_2/selection/selection.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white"})
fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2))
blue, orange, gray = "#2864A5", "#D87620", "#586675"
axis = axes[0, 0]
x = np.arange(2)
width = 0.32
for index, (label, color) in enumerate((("base", blue), ("rl", orange))):
    values = [primary["metrics"][label]["pass_at_k"]["1"], secondary["metrics"][f"{label}_1"]["estimate"]]
    bars = axis.bar(x + (index - 0.5) * width, values, width, color=color, label=label.title())
    axis.bar_label(bars, labels=[f"{value:.2%}" for value in values], padding=4, fontsize=10)
axis.set_xticks(x, ["Original scorer", "Duplicate-marker\ndiagnostic"])
axis.set_ylim(0, 1)
axis.yaxis.set_major_formatter(PercentFormatter(1))
axis.set_ylabel("pass@1")
axis.set_title("A  |  Sensitivity of single-answer success", loc="left", fontweight="bold")
axis.legend(frameon=False, loc="upper left")

axis = axes[0, 1]
ks = [1, 2, 4, 8, 16]
x = np.arange(len(ks))
for label, color, marker in (("base", blue, "o"), ("rl", orange, "s")):
    items = [secondary["metrics"][f"{label}_{k}"] for k in ks]
    values = [item["estimate"] for item in items]
    axis.plot(x, values, color=color, marker=marker, linewidth=2, label=label.title())
    axis.fill_between(x, [item["ci_low"] for item in items], [item["ci_high"] for item in items], color=color, alpha=0.10)
axis.set_xticks(x, ks)
axis.set_ylim(0, 1.02)
axis.set_xlabel("Number of attempts, k")
axis.set_ylabel("Estimated pass@k")
axis.yaxis.set_major_formatter(PercentFormatter(1))
axis.set_title("B  |  Curves after the narrow diagnostic", loc="left", fontweight="bold")
axis.legend(frameon=False, loc="lower right")

axis = axes[1, 0]
rows = [
    ("Original: pass@1", primary["comparisons"]["delta_pass_at_k"]["1"], gray),
    ("Diagnostic: pass@1", secondary["metrics"]["delta_1"], blue),
    ("Original: pass@16", primary["comparisons"]["delta_pass_at_k"]["16"], gray),
    ("Diagnostic: pass@16", secondary["metrics"]["delta_16"], blue),
]
for y, (label, item, color) in enumerate(rows):
    value = item["estimate"] * 100
    axis.errorbar(value, y, xerr=[[value-item["ci_low"]*100], [item["ci_high"]*100-value]],
        fmt="o", color=color, elinewidth=2, capsize=3, markersize=6)
axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
axis.set_yticks(range(len(rows)), [item[0] for item in rows])
axis.set_xlim(-4, 40)
axis.set_ylim(3.6, -0.6)
axis.set_xlabel("RL minus Base (percentage points), 95% CI")
axis.set_title("C  |  Paired gains under each scorer", loc="left", fontweight="bold")

axis = axes[1, 1]
steps = [candidate["step"] for candidate in selection["candidates"]]
values = [candidate["pass_at_1"] for candidate in selection["candidates"]]
axis.plot(steps, values, "o-", color="#39816D", linewidth=2, markersize=7)
for step, value in zip(steps, values):
    axis.annotate(f"{value:.1%}", (step, value), xytext=(0, 9), textcoords="offset points", ha="center")
axis.set_xticks(steps)
axis.set_ylim(0.65, 0.88)
axis.set_xlim(100, 412)
axis.yaxis.set_major_formatter(PercentFormatter(1))
axis.set_xlabel("Formal training checkpoint (optimizer step)")
axis.set_ylabel("Development pass@1")
axis.set_title("D  |  Original development selection", loc="left", fontweight="bold")
axis.text(0.04, 0.08, "Checkpoint 384 selected using development data\nOriginal scorer; 200 questions, 4 samples each",
    transform=axis.transAxes, fontsize=9, color=gray)
for axis in axes.flat:
    axis.grid(axis="y", alpha=0.16)
    axis.set_axisbelow(True)
fig.suptitle("2B RLVR result review: scoring sensitivity and checkpoint selection", fontsize=15, fontweight="bold", y=0.985)
fig.subplots_adjust(left=0.16, right=0.975, top=0.91, bottom=0.13, hspace=0.48, wspace=0.32)
fig.text(0.5, 0.042, "Test: 200 questions, 16 samples/model. Intervals: 2,000 paired question bootstraps; one fixed training run.\n"
    "The duplicate-marker result is post hoc, fixes one parser issue only, and does not replace the original result or training reward.",
    ha="center", va="center", fontsize=9, color=gray)
fig.savefig(root / "review_overview.png", dpi=180)
plt.close(fig)
print(f"[PLOT SAVED] {root / 'review_overview.png'}")
