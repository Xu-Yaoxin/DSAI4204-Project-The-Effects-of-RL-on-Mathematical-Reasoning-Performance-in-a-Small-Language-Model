"""Plot Base and LoRA-GRPO pass@k with separate fits through k=1 and k=16."""

import argparse
import json
import os
import tempfile
from math import comb
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dsai4207-figure2-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
KS = (1, 2, 4, 8, 16)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=HERE.parent / "results/final/per_question.jsonl")
parser.add_argument("--output-dir", type=Path, default=HERE)
args = parser.parse_args()
with args.data.open(encoding="utf-8") as handle:
    problems = [json.loads(line) for line in handle if line.strip()]
if len({p["id"] for p in problems}) != 200 or len(problems) != 200:
    raise ValueError("Expected 200 distinct test problems.")
if any(p["n_samples"] != 16 for p in problems):
    raise ValueError("Expected 16 attempts per model on each problem.")

scores = {}
for model in ("base", "rl"):
    counts = [p[model]["correct_count"] for p in problems]
    if any(type(c) is not int or not 0 <= c <= 16 for c in counts):
        raise ValueError(f"Invalid correct_count for {model}.")
    scores[model] = [
        100 * sum(comb(16, k) - comb(16 - c, k) for c in counts)
        / (len(counts) * comb(16, k))
        for k in KS
    ]

colors = {"base": "#356DAC", "rl": "#C95D27"}
models = {"base": ("Base", "o", -25), "rl": ("RL (LoRA-GRPO)", "s", 15)}
k_grid = np.geomspace(KS[0], KS[-1], 321)
args.output_dir.mkdir(parents=True, exist_ok=True)
for scale in ("log2", "linear_k"):
    predictor = np.log2(k_grid) if scale == "log2" else k_grid
    fit_label = r"Endpoint fits in $\log_2(k)$" if scale == "log2" else r"Endpoint fits in $k$"
    fig, ax = plt.subplots(figsize=(10.05, 6.45), dpi=200)
    for model, (label, marker, offset) in models.items():
        y = scores[model]
        # Two distinct endpoints determine the least-squares line exactly.
        slope = (y[-1] - y[0]) / (predictor[-1] - predictor[0])
        fitted = y[0] + slope * (predictor - predictor[0])
        ax.plot(k_grid, fitted, linestyle=(0, (5, 3)), linewidth=1.8,
                color=colors[model], alpha=0.85, zorder=2)
        ax.plot(KS, y, marker=marker, markersize=9, linewidth=2.4,
                color=colors[model], label=label, zorder=3)
        for k, value in zip(KS, y):
            ax.annotate(f"{value:.2f}", (k, value), xytext=(0, offset),
                        textcoords="offset points", ha="center", color=colors[model],
                        fontsize=14, bbox=dict(facecolor="white", edgecolor="none", pad=0.5))

    ax.set_xscale("log", base=2)
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.set_xlim(0.85, 18.7)
    ax.set_ylim(35, 102)
    ax.set_yticks(range(40, 101, 10))
    ax.set_xlabel("Number of attempts (k)", fontsize=17)
    ax.set_ylabel("pass@k (%)", fontsize=17)
    ax.tick_params(labelsize=16)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    model_legend = ax.legend(loc="upper left", frameon=False, fontsize=17, handlelength=1.9)
    ax.add_artist(model_legend)
    fit_key = Line2D([], [], linestyle=(0, (5, 3)), color="#555555", linewidth=1.8)
    ax.legend([fit_key], [fit_label], loc="lower right", frameon=False, fontsize=14)

    fig.tight_layout()
    output = args.output_dir / f"figure2_pass_at_k_endpoint_fit_{scale}.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")

print("k:", *KS)
for model, values in scores.items():
    print(model + ":", *(f"{value:.2f}" for value in values))
