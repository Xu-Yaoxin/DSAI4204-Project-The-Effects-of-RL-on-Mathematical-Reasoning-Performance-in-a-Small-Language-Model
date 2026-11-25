"""Render the verified MATH Level 4 Base/RL analysis as a four-panel figure."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "math_l4_review_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "math_l4_review_cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--analysis", type=Path, default=Path(__file__).resolve().parent / "experiment/outputs/math_l4_v1_fix1/main_analysis")
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures")
args = parser.parse_args()
analysis_dir = args.analysis.resolve()
output_dir = args.output_dir.resolve()
ready = json.loads((analysis_dir / "ready.json").read_text())
if ready.get("complete") is not True:
    raise ValueError("Primary analysis is not complete.")
for filename in ("summary.json", "per_question.jsonl"):
    digest = hashlib.sha256((analysis_dir / filename).read_bytes()).hexdigest()
    if digest != ready["artifacts_sha256"][filename]:
        raise ValueError(f"Primary analysis checksum mismatch: {filename}")
summary = json.loads((analysis_dir / "summary.json").read_text())
questions = [json.loads(line) for line in (analysis_dir / "per_question.jsonl").read_text().splitlines() if line.strip()]
if summary.get("synthetic") is not False or summary["experiment_type"] != "math_l4_base_rl":
    raise ValueError("Expected the non-synthetic MATH Level 4 comparison.")
if summary["identity"] != ready["identity"]:
    raise ValueError("Summary identity differs from the completion receipt.")
if len(questions) != summary["question_count"] or len({q["id"] for q in questions}) != len(questions):
    raise ValueError("Per-question records do not match the summary.")
if summary["settings"]["generation_guard"]["answer_timeout_seconds"] != 300:
    raise ValueError("The figure caption requires the prespecified 300-second deadline.")
ks = [1, 4, 16]
n_samples = summary["n_samples"]
metrics = summary["metrics"]
diagnostics = summary["diagnostics"]
for label in ("base", "rl"):
    for k in ks:
        values = [q[label][f"pass_at_{k}"]["estimate"] for q in questions]
        if not np.isclose(np.mean(values), metrics[f"{label}_pass_at_{k}"]["estimate"], rtol=0, atol=1e-12):
            raise ValueError(f"Per-question estimate mismatch: {label}, k={k}")
    terminations = Counter()
    for q in questions:
        terminations.update(q[label]["termination_counts"])
    if dict(terminations) != diagnostics[label]["termination_counts"]:
        raise ValueError(f"Termination totals differ: {label}")
    if sum(terminations.values()) != len(questions) * n_samples:
        raise ValueError(f"Incomplete requested-attempt denominator: {label}")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.titlesize": 13, "axes.titleweight": "semibold",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#6A737D", "axes.linewidth": 0.7,
    "grid.color": "#DFE4EA", "grid.linewidth": 0.7,
    "text.color": "#17222D", "axes.labelcolor": "#17222D",
    "xtick.color": "#344455", "ytick.color": "#344455",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
base_color = "#4778A8"
rl_color = "#1F876F"
neutral_color = "#AAB4BF"
fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
fig.subplots_adjust(left=0.075, right=0.965, top=0.865, bottom=0.185, wspace=0.27, hspace=0.48)
fig.suptitle("MATH Level 4 | Base vs. RL", x=0.075, y=0.972, ha="left", fontsize=22, fontweight="bold")
fig.text(0.075, 0.936,
         f"{summary['question_count']} paired questions  |  {n_samples} attempts per model per question  |  "
         f"RL checkpoint: step {summary['selected_step']}", fontsize=11, color="#526171")

ax = axes[0, 0]
positions = np.arange(len(ks))
bar_width = 0.32
for label, display_name, color, offset in (("base", "Base", base_color, -bar_width / 2), ("rl", "RL", rl_color, bar_width / 2)):
    estimates = np.array([metrics[f"{label}_pass_at_{k}"]["estimate"] * 100 for k in ks])
    intervals = np.array([metrics[f"{label}_pass_at_{k}"]["ci95"] for k in ks]) * 100
    errors = np.vstack((estimates - intervals[:, 0], intervals[:, 1] - estimates))
    ax.bar(positions + offset, estimates, width=bar_width, color=color, label=display_name, zorder=3)
    ax.errorbar(positions + offset, estimates, yerr=errors, fmt="none", ecolor="#263543", elinewidth=1, capsize=3, zorder=4)
    for xpos, estimate, upper in zip(positions + offset, estimates, intervals[:, 1]):
        ax.text(xpos, upper + 2.4, f"{estimate:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="medium")
ax.set_title("A  Operational pass@k", loc="left", pad=14)
ax.set_xticks(positions, [f"pass@{k}" for k in ks])
ax.set_ylim(0, 108)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel("Estimated success (%)")
ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.grid(axis="y", zorder=0)
ax.legend(frameon=False, loc="upper left", ncols=2, fontsize=10)

ax = axes[0, 1]
delta_estimates = np.array([metrics[f"delta_pass_at_{k}"]["estimate"] for k in ks]) * 100
delta_intervals = np.array([metrics[f"delta_pass_at_{k}"]["ci95"] for k in ks]) * 100
delta_errors = np.vstack((delta_estimates - delta_intervals[:, 0], delta_intervals[:, 1] - delta_estimates))
rows = np.arange(len(ks))[::-1]
ax.axvline(0, color="#77828D", linewidth=1, linestyle="--", zorder=1)
ax.errorbar(delta_estimates, rows, xerr=delta_errors, fmt="o", markersize=7, color=rl_color, elinewidth=2, capsize=4, zorder=3)
for ypos, estimate, interval in zip(rows, delta_estimates, delta_intervals):
    ax.text(0.99, ypos + 0.27, f"{estimate:+.2f} pp  [{interval[0]:+.2f}, {interval[1]:+.2f}]",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=10)
left_limit = min(0, float(delta_intervals.min()))
right_limit = max(0, float(delta_intervals.max()))
delta_span = max(right_limit - left_limit, 4)
ax.set_xlim(left_limit - 0.10 * delta_span, right_limit + 0.10 * delta_span)
ax.set_ylim(-0.4, len(ks) - 0.20)
ax.set_yticks(rows, [f"pass@{k}" for k in ks])
ax.set_xlabel("Paired difference: RL - Base (percentage points)")
ax.set_title("B  Paired differences and 95% CIs", loc="left", pad=14)
ax.grid(axis="x", zorder=0)

ax = axes[1, 0]
changes = np.array([q["rl"]["correct_count"] - q["base"]["correct_count"] for q in questions])
support = np.arange(-n_samples, n_samples + 1)
counts = np.array([int(np.sum(changes == value)) for value in support])
colors = [base_color if value < 0 else rl_color if value > 0 else neutral_color for value in support]
ax.bar(support, counts, width=0.84, color=colors, zorder=3)
ax.axvline(0, color="#5C6875", linewidth=0.8, linestyle="--", zorder=2)
ax.set_xlim(-n_samples - 1, n_samples + 1)
ax.set_xticks(np.arange(-n_samples, n_samples + 1, 4))
ax.set_xlabel(f"Change in confirmed successes out of {n_samples} (RL - Base)")
ax.set_ylabel("Questions")
ax.set_ylim(0, max(counts) * 1.24)
ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
ax.grid(axis="y", zorder=0)
ax.set_title("C  Per-question success-count changes", loc="left", pad=14)
declined, tied, improved = [int(np.sum(changes < 0)), int(np.sum(changes == 0)), int(np.sum(changes > 0))]
ax.text(0.02, 0.96, f"Declined: {declined}    Tied: {tied}    Improved: {improved}",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

ax = axes[1, 1]
termination_styles = {
    "eos": ("EOS completion", "#5C86B2"),
    "repetition_guard": ("Repetition stop", "#DAA446"),
    "answer_timeout": ("300 s timeout", "#BA6169"),
    "no_token_timeout": ("No-token stop", "#9379A8"),
    "context_limit": ("Context limit", "#7B8794"),
}
observed_terminations = set(diagnostics["base"]["termination_counts"]) | set(diagnostics["rl"]["termination_counts"])
if observed_terminations - set(termination_styles):
    raise ValueError(f"Unmapped termination reasons: {observed_terminations - set(termination_styles)}")
lefts = np.zeros(2)
model_rows = np.array([1, 0])
for reason, (display_name, color) in termination_styles.items():
    if reason not in observed_terminations:
        continue
    percentages = np.array([100 * diagnostics[label]["termination_counts"].get(reason, 0) / diagnostics[label]["requested_count"] for label in ("base", "rl")])
    ax.barh(model_rows, percentages, left=lefts, height=0.47, color=color, label=display_name, edgecolor="white", linewidth=0.8, zorder=3)
    for ypos, start, percentage in zip(model_rows, lefts, percentages):
        if percentage >= 15:
            ax.text(start + percentage / 2, ypos, f"{percentage:.2f}%", ha="center", va="center", fontsize=10,
                    color="white" if reason != "repetition_guard" else "#27313B", fontweight="medium")
    lefts += percentages
timeout_rates = {label: 100 * diagnostics[label]["termination_counts"].get("answer_timeout", 0) / diagnostics[label]["requested_count"] for label in ("base", "rl")}
repetition_rates = {label: 100 * diagnostics[label]["termination_counts"].get("repetition_guard", 0) / diagnostics[label]["requested_count"] for label in ("base", "rl")}
ax.text(0, 0.97, f"Timeout rate: Base {timeout_rates['base']:.2f}%  |  RL {timeout_rates['rl']:.2f}%",
        transform=ax.transAxes, ha="left", va="top", fontsize=10)
ax.text(0, 0.88, f"Repetition stops: Base {repetition_rates['base']:.2f}%  |  RL {repetition_rates['rl']:.2f}%",
        transform=ax.transAxes, ha="left", va="top", fontsize=10, color="#526171")
ax.set_yticks(model_rows, ["Base", "RL"])
ax.set_ylim(-0.55, 1.85)
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.set_xlabel("Share of all requested attempts (%)")
ax.set_title("D  Termination composition", loc="left", pad=14)
ax.grid(axis="x", zorder=0)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22), ncols=2, fontsize=9.5,
          borderaxespad=0, columnspacing=1.8, handlelength=1.4)

fig.text(0.075, 0.087,
         "Policy: fixed 300 s deadline and repetition guard; unknown stops contribute no confirmed successes.",
         ha="left", fontsize=9.5, color="#526171")
fig.text(0.075, 0.065,
         f"Error bars: marginal 95% paired family-bootstrap CIs ({summary['bootstrap']:,} resamples; "
         f"{summary['family_count']} families). pp = percentage points.",
         ha="left", fontsize=9.5, color="#526171")
fig.text(0.075, 0.043,
         "Pass@16 reports coverage observed with 16 attempts; one training seed does not establish a general RL effect.",
         ha="left", fontsize=9.5, color="#526171")

output_dir.mkdir(parents=True, exist_ok=True)
output_stem = output_dir / "math_l4_base_vs_rl_summary"
for extension in ("png", "pdf"):
    fig.savefig(output_stem.with_suffix(f".{extension}"), dpi=220, metadata={"Creator": "Matplotlib; plot_summary.py"})
plt.close(fig)
receipt = {
    "analysis_directory": str(analysis_dir),
    "source_sha256": {filename: hashlib.sha256((analysis_dir / filename).read_bytes()).hexdigest() for filename in ("summary.json", "per_question.jsonl", "ready.json")},
    "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "outputs_sha256": {extension: hashlib.sha256(output_stem.with_suffix(f".{extension}").read_bytes()).hexdigest() for extension in ("png", "pdf")},
    "question_changes": {"declined": declined, "tied": tied, "improved": improved},
    "timeout_rates_percent": timeout_rates,
    "matplotlib_version": matplotlib.__version__,
    "numpy_version": np.__version__,
}
(output_dir / "figure_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(f"[SAVED] {output_stem.with_suffix('.png')}")
print(f"[SAVED] {output_stem.with_suffix('.pdf')}")
print(f"[SAVED] {output_dir / 'figure_receipt.json'}")
