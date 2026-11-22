#!/usr/bin/env python3
"""Learning curves of HCMAGRL against the deep reinforcement learning baselines
(fig_convergence_rl) and against its ablated variants
(fig_convergence_ablation), at equal weights.

The statistics come from figures/data/convergence.csv, which
scripts/build_data.py writes. There every run is divided by the best final
value that any of the nine methods reaches on its instance, so each curve keeps
the level its method starts from. Per episode the thin line is the median over
the 27 instances, the thick line its 21-episode centred moving average and the
band the interquartile range. Every line is solid (see scifig_style). The
insets of the ablation figure enlarge episodes 100 to 500, where the variants
differ by a few percent.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scifig_style import (FULL_MM, data_dir, episode_axis, learning_curves,  # noqa: E402
                          save, style, use_scifig)

OURS = "HCMAGRL"
RL = [OURS, "DDQN", "EDQN", "SAC", "D-DRL"]
ABLATION = [OURS, "w/o EdgeAttn", "w/o Hierarchy", "w/o Heterogeneity", "w/o Graph"]
MS, COST = "makespan_mean", "cost_mean"
COST_TICKS = [1, 1.5, 2, 3, 4, 6, 8, 12]
# Insets of the ablation figure: y limits and y ticks per objective.
ZOOM_EPISODES = (100, 500)
ZOOM = {MS: ((0.98, 1.13), [1.0, 1.05, 1.1]),
        COST: ((0.97, 1.45), [1.0, 1.1, 1.2, 1.3, 1.4])}


def inset(ax, df, methods, metric):
    """Late training enlarged, thin and thick lines only."""
    ylim, yticks = ZOOM[metric]
    ins = ax.inset_axes([0.40, 0.42, 0.56, 0.52])
    order = [m for m in methods if m != OURS] + [OURS]
    for k, m in enumerate(order):
        d = df[(df.method == m) & (df.metric == metric)].sort_values("episode")
        c = style(m)["color"]
        ins.plot(d.episode, d["median"], color=c, lw=0.35, alpha=0.40, zorder=2)
        ins.plot(d.episode, d.median_smooth, color=c, lw=1.1, zorder=3 + k)
    ins.axhline(1.0, color="black", lw=0.5, zorder=0)
    ins.set(xlim=ZOOM_EPISODES, ylim=ylim)
    ins.set_xticks([200, 300, 400, 500])
    ins.set_yticks(yticks)
    ins.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ins.tick_params(labelsize=6, length=1.8, pad=1.5)
    ins.set_facecolor("white")
    ins.grid(True, lw=0.3, alpha=0.7)
    ax.indicate_inset_zoom(ins, edgecolor="#555555", lw=0.5, alpha=0.9)


def figure(df, methods, stem, zoom=False):
    use_scifig(FULL_MM, FULL_MM * 0.40)
    fig, (a, b) = plt.subplots(1, 2, constrained_layout=True)
    n = int(df.episode.max())

    learning_curves(a, df, methods, MS)
    episode_axis(a, n)
    a.set_ylim(0.95, 2.45)
    a.set_ylabel("Makespan / best final value")
    a.set_title("(a) Makespan", loc="left")

    learning_curves(b, df, methods, COST)
    episode_axis(b, n)
    b.set_yscale("log")
    b.yaxis.set_major_locator(FixedLocator(COST_TICKS))
    b.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    b.yaxis.set_minor_locator(NullLocator())
    b.set_ylim(0.9, 13)
    b.set_ylabel("Reconfiguration cost / best final value")
    b.set_title("(b) Reconfiguration cost", loc="left")

    if zoom:
        inset(a, df, methods, MS)
        inset(b, df, methods, COST)
    handles, labels = a.get_legend_handles_labels()
    idx = [labels.index(m) for m in methods]
    fig.legend([handles[i] for i in idx], [labels[i] for i in idx], loc="outside lower center",
               ncol=len(methods), handlelength=2.4, columnspacing=1.5)
    save(fig, stem)


def main():
    df = pd.read_csv(data_dir() / "convergence.csv")
    df = df[df.weight == "balanced"]
    figure(df, RL, "fig_convergence_rl")
    figure(df, ABLATION, "fig_convergence_ablation", zoom=True)


if __name__ == "__main__":
    main()
