#!/usr/bin/env python3
"""Training convergence: HCMAGRL against its ablated variants and against the
deep reinforcement learning baselines.

Curves are normalized per run by that run's own first-episode value before being
averaged over the 27 instances, because raw makespans span 960-3000+ and an
unnormalized mean would simply track the largest instances. The shaded band is
+/- 1 standard deviation across those 27 instances.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scifig_style import FULL_MM, data_dir, save, style, use_scifig  # noqa: E402

ABLATION = ["HCMAGRL", "w/o EdgeAttn", "w/o Hierarchy", "w/o Heterogeneity", "w/o Graph"]
RL = ["HCMAGRL", "DDQN", "EDQN", "SAC", "D-DRL"]
SMOOTH = 15  # rolling window, episodes


def panel(ax, df, methods, metric, ylabel):
    for m in methods:
        d = df[(df.method == m) & (df.metric == metric)].sort_values("episode")
        if d.empty:
            continue
        mu = d["mean"].rolling(SMOOTH, min_periods=1, center=True).mean()
        sd = d["std"].rolling(SMOOTH, min_periods=1, center=True).mean()
        st = style(m)
        ax.plot(d.episode, mu, label=m, color=st["color"], ls=st["ls"], lw=1.0, zorder=3)
        ax.fill_between(d.episode, mu - sd, mu + sd, color=st["color"], alpha=0.08, lw=0, zorder=1)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, 500)


def figure(df, methods, stem):
    use_scifig(FULL_MM, FULL_MM * 0.36)
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    panel(axes[0], df, methods, "makespan_mean", "Normalized makespan")
    panel(axes[1], df, methods, "cost_mean", "Normalized reconfiguration cost")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5,
               handlelength=2.6, columnspacing=1.6)
    save(fig, stem)


def main():
    df = pd.read_csv(data_dir() / "convergence.csv")
    df = df[df.weight == "balanced"]
    figure(df, ABLATION, "fig_convergence_ablation")
    figure(df, RL, "fig_convergence_rl")


if __name__ == "__main__":
    main()
