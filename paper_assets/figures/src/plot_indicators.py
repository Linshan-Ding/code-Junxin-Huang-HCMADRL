#!/usr/bin/env python3
"""Multi-objective indicators (IGD, Spread, HV) over the 27 test instances.

Boxes summarize the per-instance indicator values; the companion panel reports
the per-indicator win rate, i.e. the fraction of the 27 instances on which each
method attains the best value.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scifig_style import FULL_MM, HATCH, data_dir, save, style, use_scifig  # noqa: E402

GROUPS = {
    "rl": ["HCMAGRL", "DDQN", "EDQN", "SAC", "D-DRL"],
    "ablation": ["HCMAGRL", "w/o EdgeAttn", "w/o Hierarchy", "w/o Heterogeneity", "w/o Graph"],
}
INDICATORS = ["IGD", "Spread", "HV"]
ARROW = {"IGD": r"$\downarrow$", "Spread": r"$\downarrow$", "HV": r"$\uparrow$"}


def figure(ind_df, win_df, group, stem):
    methods = GROUPS[group]
    use_scifig(FULL_MM, FULL_MM * 0.30)
    fig, axes = plt.subplots(1, 4, constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})

    for ax, ind in zip(axes, INDICATORS):
        data = [ind_df[(ind_df.group == group) & (ind_df.indicator == ind)
                       & (ind_df.method == m)]["value"].dropna().to_numpy()
                for m in methods]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=0.8),
                        whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5),
                        boxprops=dict(lw=0.5))
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(style(m)["color"])
            patch.set_alpha(0.30)
            patch.set_edgecolor(style(m)["color"])
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([str(i + 1) for i in range(len(methods))])
        ax.set_xlabel("Method")
        ax.set_ylabel(f"{ind} {ARROW[ind]}")
        ax.grid(axis="x", visible=False)

    ax = axes[3]
    w = 0.26
    x = np.arange(len(INDICATORS))
    for i, m in enumerate(methods):
        vals = [win_df[(win_df.group == group) & (win_df.indicator == ind)
                       & (win_df.method == m)]["win_rate"].iloc[0] for ind in INDICATORS]
        ax.bar(x + (i - (len(methods) - 1) / 2) * w / 1.6, vals, w / 1.6,
               label=f"{i + 1}. {m}", color=style(m)["color"], alpha=0.85,
               edgecolor=style(m)["color"], lw=0.4, hatch=HATCH.get(m, ""))
    ax.set_xticks(x)
    ax.set_xticklabels(INDICATORS)
    ax.set_ylabel("Win rate (\\%)")
    ax.set_xlabel("Indicator")
    ax.grid(axis="x", visible=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, columnspacing=1.2)
    save(fig, stem)


def main():
    ind = pd.read_csv(data_dir() / "indicators.csv")
    win = pd.read_csv(data_dir() / "indicator_winrate.csv")
    figure(ind, win, "rl", "fig_indicators_rl")
    figure(ind, win, "ablation", "fig_indicators_ablation")


if __name__ == "__main__":
    main()
