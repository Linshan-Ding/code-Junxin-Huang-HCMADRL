#!/usr/bin/env python3
"""Distribution of both objectives over the 135 (instance, weight) settings.

Each value is the mean over 20 evaluation seeds. Values are reported relative
to the best method in that setting so that instances of very different scale
contribute comparably; a value of 1.0 means "best in this setting".
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scifig_style import FULL_MM, data_dir, save, style, use_scifig  # noqa: E402

ORDER = ["HCMAGRL", "w/o Graph", "w/o Hierarchy", "w/o EdgeAttn",
         "w/o Heterogeneity", "EDQN", "D-DRL", "DDQN", "SAC"]
SHORT = {"w/o Heterogeneity": "w/o Hetero."}


def panel(ax, df, col, ylabel):
    best = df.groupby(["instance_id", "alpha_t"])[col].transform("min")
    df = df.assign(ratio=df[col] / best)
    data = [df[df.method == m]["ratio"].to_numpy() for m in ORDER]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=0.8),
                    whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                    boxprops=dict(lw=0.5))
    for patch, m in zip(bp["boxes"], ORDER):
        patch.set_facecolor(style(m)["color"])
        patch.set_alpha(0.30)
        patch.set_edgecolor(style(m)["color"])
    ax.axhline(1.0, color="black", lw=0.5, ls=":", zorder=0)
    ax.set_xticks(range(1, len(ORDER) + 1))
    ax.set_xticklabels([SHORT.get(m, m) for m in ORDER], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def main():
    df = pd.read_csv(data_dir() / "eval_cells.csv")
    use_scifig(FULL_MM, FULL_MM * 0.40)
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    panel(axes[0], df, "makespan_mean", "Makespan ratio to best method")
    panel(axes[1], df, "cost_mean", "Reconfiguration cost ratio to best method")
    save(fig, "fig_objective_boxplots")


if __name__ == "__main__":
    main()
