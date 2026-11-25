#!/usr/bin/env python3
"""Render a 2x2 summary figure from validated analysis output."""

import argparse
import json
import os
from pathlib import Path
import tempfile


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--summary", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
args.summary = (root / args.summary).resolve()
args.output = (root / args.output).resolve()
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dsai4207-matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "dsai4207-font-cache"))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except ImportError as exc:
    raise SystemExit("Plotting needs matplotlib. Install the project requirements first.") from exc

with args.summary.open(encoding="utf-8") as handle:
    summary = json.load(handle)
if summary.get("schema_version") != 1:
    raise SystemExit("Unsupported summary schema_version")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "svg.fonttype": "none"})
figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
colors = {"base": "#2864A5", "rl": "#D57928"}
ks = summary["sampling"]["k_values"]
for label, marker in (("base", "o"), ("rl", "s")):
    values = [summary["metrics"][label]["pass_at_k"][str(k)] for k in ks]
    low = [summary["metrics"][label]["pass_at_k_ci"][str(k)]["ci_low"] for k in ks]
    high = [summary["metrics"][label]["pass_at_k_ci"][str(k)]["ci_high"] for k in ks]
    axes[0, 0].plot(ks, values, marker=marker, markersize=6, linewidth=2, label=label.upper() if label == "rl" else "Base", color=colors[label])
    axes[0, 0].fill_between(ks, low, high, color=colors[label], alpha=0.12)
axes[0, 0].set_xscale("log", base=2)
axes[0, 0].set_xticks(ks, [str(k) for k in ks])
axes[0, 0].set_ylim(-0.02, 1.02)
axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1))
axes[0, 0].set_xlabel("Number of attempts, k")
axes[0, 0].set_ylabel("Estimated pass@k")
axes[0, 0].set_title("A  Sampling success and coverage", loc="left", pad=12)
axes[0, 0].grid(axis="y", alpha=0.18)
axes[0, 0].legend(frameon=False, loc="lower right")

contrast_names = [f"Delta pass@{k}" for k in ks]
contrasts = [summary["comparisons"]["delta_pass_at_k"][str(k)] for k in ks]
if summary["comparisons"]["delta1_minus_delta16"] is not None:
    contrast_names.append("Delta1 - Delta16")
    contrasts.append(summary["comparisons"]["delta1_minus_delta16"])
if summary["structural"]["relational_minus_numerical"] is not None:
    contrast_names.append("Relational - numerical")
    contrasts.append(summary["structural"]["relational_minus_numerical"])
for index, contrast in enumerate(contrasts):
    color = "#23465C" if index < len(ks) else "#8B4D86"
    axes[0, 1].plot([100 * contrast["ci_low"], 100 * contrast["ci_high"]], [index, index], color=color, linewidth=2)
    axes[0, 1].scatter([100 * contrast["estimate"]], [index], color=color, s=36, zorder=3)
axes[0, 1].axvline(0, color="#838383", linestyle="--", linewidth=1)
axes[0, 1].set_yticks(range(len(contrast_names)), contrast_names)
axes[0, 1].invert_yaxis()
axes[0, 1].set_xlabel("Difference (percentage points), with 95% CI")
axes[0, 1].set_title("B  Paired differences", loc="left", pad=12)
axes[0, 1].grid(axis="x", alpha=0.18)
axes[0, 1].margins(y=0.15)

categories = ["both", "base_only", "rl_only", "neither"]
category_labels = ["Both", "Base only", "RL only", "Neither"]
counts = [summary["coverage"][category] for category in categories]
bar_colors = ["#568D78", colors["base"], colors["rl"], "#A8ADB2"]
bars = axes[1, 0].bar(category_labels, counts, color=bar_colors, width=0.64)
axes[1, 0].bar_label(bars, labels=[f"{count}\n({count / summary['coverage']['total']:.0%})" for count in counts], padding=4, fontsize=9)
axes[1, 0].set_ylim(0, max(1, max(counts)) * 1.3)
axes[1, 0].set_ylabel("Number of questions")
axes[1, 0].set_title(f"C  Observed coverage across n = {summary['sampling']['n_samples']}", loc="left", pad=12)
axes[1, 0].grid(axis="y", alpha=0.18)
axes[1, 0].set_axisbelow(True)

diagnostic_fields = ["format_compliance_rate", "extraction_failure_rate", "ambiguous_answer_rate", "truncation_rate"]
diagnostic_labels = ["Format\ncompliance", "Extraction\nfailure", "Ambiguous\nanswer", "Truncation"]
for label, offset in (("base", -0.18), ("rl", 0.18)):
    values = [summary["diagnostics"][label][field] for field in diagnostic_fields]
    bars = axes[1, 1].bar([index + offset for index in range(4)], values, width=0.34, label=label.upper() if label == "rl" else "Base", color=colors[label])
    axes[1, 1].bar_label(bars, labels=[f"{value:.1%}" for value in values], fontsize=7.5, padding=3 if label == "base" else 12)
axes[1, 1].set_xticks(range(4), diagnostic_labels)
axes[1, 1].set_ylim(0, 1.17)
axes[1, 1].yaxis.set_major_formatter(PercentFormatter(1))
axes[1, 1].set_ylabel("Fraction of generated responses")
axes[1, 1].set_title("D  Scoring and generation diagnostics", loc="left", pad=12)
axes[1, 1].grid(axis="y", alpha=0.18)
axes[1, 1].set_axisbelow(True)
axes[1, 1].legend(frameon=False, loc="upper right")

title = "RLVR: mathematical success and finite-sample coverage"
if summary["synthetic"]:
    title = "SYNTHETIC PIPELINE TEST — not model measurements"
figure.suptitle(title, fontsize=15, fontweight="bold")
subtitle = f"{summary['sampling']['question_count']} questions | n={summary['sampling']['n_samples']} per model | {summary['sampling']['bootstrap_replicates']} paired bootstrap replicates by {summary['sampling']['bootstrap_unit']}"
figure.supxlabel(subtitle + "\n95% marginal intervals; uncertainty across training seeds is not included.", fontsize=9, color="#555555")
args.output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(args.output, dpi=220, bbox_inches="tight")
plt.close(figure)
print(f"[PLOT 100%] [SAVED] {args.output.resolve()}")
