"""Inspect final v1.4 saved results and export a static research figure."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

destination = Path(__file__).resolve().parent
root = destination.parents[2]
source = root / "outputs/rescore_v1_4"
ready = json.loads((source / "ready.json").read_text())
assert ready["complete"] is True
for relative, expected in ready["artifacts_sha256"].items():
    assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected, relative
summary = json.loads((source / "main_analysis/summary.json").read_text())
diagnostics = json.loads((source / "diagnostics.json").read_text())
questions = [json.loads(line) for line in (source / "main_analysis/per_question.jsonl").read_text().split("\n") if line.strip()]
outcomes = Counter("improved" if row["rl"]["correct_count"] > row["base"]["correct_count"] else "declined" if row["rl"]["correct_count"] < row["base"]["correct_count"] else "tied" for row in questions)
perfect = {label: sum(row[label]["correct_count"] == 16 for row in questions) for label in ("base", "rl")}
pair_counts = Counter((row["base"]["correct_count"], row["rl"]["correct_count"]) for row in questions)
unresolved_base = sum(diagnostics["test_base"]["revised_status_counts"].get(status, 0) for status in ("extraction_failed", "ambiguous"))
base_max_if_unresolved_correct = (diagnostics["test_base"]["revised_correct"] + unresolved_base) / 3200
adversarial_gap = summary["metrics"]["rl"]["pass_at_k"]["1"] - base_max_if_unresolved_correct
review = {
    "analysis_kind": "posthoc_descriptive_interpretation",
    "source_ready_sha256": hashlib.sha256((source / "ready.json").read_bytes()).hexdigest(),
    "question_outcomes": dict(outcomes), "all_16_correct_questions": perfect,
    "observed_coverage": summary["coverage"],
    "correct_count_histograms": {label: dict(sorted(Counter(row[label]["correct_count"] for row in questions).items())) for label in ("base", "rl")},
    "ceiling": {"maximum_possible_delta_pass16_pp": 5.0, "minimum_D_given_observed_delta1_pp": 9.25},
    "unresolved_score_sensitivity": {"base_unresolved": unresolved_base, "base_upper_if_all_unresolved_correct": base_max_if_unresolved_correct, "rl_unchanged": summary["metrics"]["rl"]["pass_at_k"]["1"], "remaining_sample_gap": adversarial_gap,
        "assumptions": "Only unresolved (extraction_failed or ambiguous) rows are changed. All parsed numeric scores, gold answers and current RL scores are assumed valid. This descriptive sample bound has no bootstrap interval and is not a lower bound on reasoning improvement."},
    "examples": [row for row in questions if row["id"] in ("gsm8k-test-000002", "gsm8k-test-000419")],
}
assert outcomes == {"improved": 150, "tied": 31, "declined": 19}
assert perfect == {"base": 16, "rl": 109}
assert abs(adversarial_gap - 159 / 3200) < 1e-12
(destination / "additional_statistics.json").write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")

os.environ.setdefault("MPLCONFIGDIR", str(root / "outputs/matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
                     "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
blue, orange, green, red = "#2563A6", "#C55C16", "#27765D", "#A94442"
fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.1))
fig.subplots_adjust(left=0.075, right=0.97, top=0.87, bottom=0.15, hspace=0.46, wspace=0.27)
fig.suptitle("Qwen3.5-2B-Base: saved-answer rescoring", fontsize=20, fontweight="bold", y=0.965)
fig.text(0.5, 0.92, "200 test questions | 16 samples per model | frozen checkpoint 384 | posthoc v1.4 scoring", ha="center", fontsize=11, color="#4B5563")
positions = list(range(5))
ks = ["1", "2", "4", "8", "16"]
ax = axes[0, 0]
for label, color, marker in (("base", blue, "o"), ("rl", orange, "s")):
    values = [summary["metrics"][label]["pass_at_k"][k] * 100 for k in ks]
    lows = [summary["metrics"][label]["pass_at_k_ci"][k]["ci_low"] * 100 for k in ks]
    highs = [summary["metrics"][label]["pass_at_k_ci"][k]["ci_high"] * 100 for k in ks]
    ax.plot(positions, values, color=color, marker=marker, linewidth=2.3, markersize=6, label="Base" if label == "base" else "RL")
    ax.fill_between(positions, lows, highs, color=color, alpha=0.10, linewidth=0)
ax.set(title="A  Success within k sampled answers", xlabel="Sampling budget k", ylabel="pass@k", xticks=positions, xticklabels=ks, ylim=(50, 101))
ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.grid(axis="y", alpha=0.18)
ax.legend(loc="lower right", frameon=False)
ax.annotate("64.81%", (0, 64.8125), xytext=(10, -19), textcoords="offset points", color=blue, fontsize=10)
ax.annotate("79.06%", (0, 79.0625), xytext=(10, -20), textcoords="offset points", color=orange, fontsize=10)

ax = axes[0, 1]
points = [summary["comparisons"]["delta_pass_at_k"][k]["estimate"] * 100 for k in ks]
low_errors = [(summary["comparisons"]["delta_pass_at_k"][k]["estimate"] - summary["comparisons"]["delta_pass_at_k"][k]["ci_low"]) * 100 for k in ks]
high_errors = [(summary["comparisons"]["delta_pass_at_k"][k]["ci_high"] - summary["comparisons"]["delta_pass_at_k"][k]["estimate"]) * 100 for k in ks]
ax.axhline(0, color="#6B7280", linewidth=1, linestyle="--")
ax.errorbar(positions, points, yerr=[low_errors, high_errors], color=green, fmt="o-", linewidth=2, capsize=5)
for position, value in zip(positions, points):
    ax.annotate(f"{value:+.2f}", (position, value), xytext=(0, 22), textcoords="offset points", ha="center", fontsize=10)
ax.set(title="B  RL gain and 95% paired intervals", xlabel="Sampling budget k", ylabel="RL minus Base (percentage points)", xticks=positions, xticklabels=ks, xlim=(-0.4, 4.4), ylim=(-4, 21))
ax.grid(axis="y", alpha=0.18)
ax.text(0.97, 0.94, "D = gain@1 minus gain@16\n+13.75 pp [10.50, 16.72]", transform=ax.transAxes, ha="right", va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})

ax = axes[1, 0]
for (base_count, rl_count), count in sorted(pair_counts.items()):
    color = green if rl_count > base_count else red if rl_count < base_count else "#8C98A8"
    ax.scatter(base_count, rl_count, s=20 + 12 * count, color=color, alpha=0.68, linewidth=0.5, edgecolor="white")
ax.plot([0, 16], [0, 16], color="#6B7280", linestyle="--", linewidth=1, zorder=0)
ax.set(title="C  Correct answers per question", xlabel="Base correct answers out of 16", ylabel="RL correct answers out of 16", xlim=(-0.8, 16.8), ylim=(-0.8, 16.8), xticks=[0, 4, 8, 12, 16], yticks=[0, 4, 8, 12, 16])
ax.grid(alpha=0.13)
ax.text(0.97, 0.03, "150 improve / 31 tie / 19 decline\nAll 16 correct: Base 16 / RL 109\nBubble area grows with question count", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.93})

ax = axes[1, 1]
bar_positions = [0, 1]
for offset, label, color in ((-0.18, "base", blue), (0.18, "rl", orange)):
    values = [diagnostics[f"test_{label}"]["original_correct"] / 32, diagnostics[f"test_{label}"]["revised_correct"] / 32]
    bars = ax.bar([position + offset for position in bar_positions], values, width=0.32, color=color, label="Base" if label == "base" else "RL")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.1, f"{value:.2f}%", ha="center", fontsize=10)
ax.set(title="D  Sensitivity to the scoring rules", ylabel="pass@1", xticks=bar_positions, xticklabels=["Original scorer", "Revised scorer"], ylim=(0, 100))
ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.grid(axis="y", alpha=0.15)
ax.set_axisbelow(True)
ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
fig.text(0.075, 0.045, "Intervals: 2,000 question-paired bootstrap draws, conditional on one trained model. No new generation or retraining.\nFinite-sample coverage and numeric answer scores do not identify reasoning mechanisms; Base pass@16 is already 95%.", fontsize=9.2, color="#4B5563", linespacing=1.5)
for extension in ("png", "pdf"):
    fig.savefig(destination / f"results_overview.{extension}", dpi=190)
plt.close(fig)
print("[ANALYSIS COMPLETE] Additional descriptive counts verified; original results unchanged")
print(f"[SAVED] {destination / 'additional_statistics.json'}")
print(f"[SAVED] {destination / 'results_overview.png'}")
