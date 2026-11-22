#!/usr/bin/env python3
"""Computational cost of training (fig_efficiency).

(a) Mean wall-clock seconds per training episode against the cell count, with
the standard deviation over the nine instances of each count. It isolates the
price paid for the edge-conditioned attention and for the heterogeneous graph.
(b) Decision epochs per episode over training, from convergence.csv and drawn
like the learning curves of plot_convergence.py. The episode axis is
logarithmic because the learned policies shed most of their surplus decision
epochs within the first few dozen episodes.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scifig_style import (FULL_MM, OI, data_dir, episode_axis, learning_curves,  # noqa: E402
                          save, style, use_scifig)

METHODS = ["HCMAGRL", "w/o Heterogeneity", "w/o Hierarchy", "w/o EdgeAttn",
           "w/o Graph", "SAC"]

# The shared palette gives the ablations and the baselines the same four
# identities, which is unambiguous as long as a figure shows one family at a
# time. This figure is the exception: it puts SAC next to the ablations as a
# cost reference, so SAC gets a local identity here rather than colliding with
# "w/o Heterogeneity". Every line of this figure is solid.
LOCAL = {"SAC": dict(color=OI["sky"], ls="-", marker="X")}


def estyle(method: str) -> dict:
    return {**style(method), **LOCAL.get(method, {})}


def main():
    rt = pd.read_csv(data_dir() / "runtime.csv")
    rt = rt[rt.method.isin(METHODS)]
    agg = (rt.groupby(["method", "M"])
             .agg(sec=("sec_per_episode", "mean"), sd=("sec_per_episode", "std"))
             .reset_index())
    conv = pd.read_csv(data_dir() / "convergence.csv")
    conv = conv[conv.weight == "balanced"]

    use_scifig(FULL_MM, FULL_MM * 0.38)
    fig, (a, b) = plt.subplots(1, 2, constrained_layout=True)

    centre = (len(METHODS) - 1) / 2
    for k, m in enumerate(METHODS):
        d = agg[agg.method == m].sort_values("M")
        st = estyle(m)
        # A small horizontal offset keeps the error bars at one cell count apart.
        a.errorbar(d.M + (k - centre) * 0.12, d.sec, yerr=d.sd, capsize=1.5, elinewidth=0.5,
                   color=st["color"], ls="-", marker=st["marker"], lw=1.1, markersize=3.2,
                   markeredgewidth=0.4, markeredgecolor="white", label=m)
    a.set_xlabel("Number of cells $M$")
    a.set_ylabel("Training time per episode (s)")
    a.set_xticks([4, 8, 12])
    a.set_yscale("log")
    a.set_title("(a) Cost of one training episode", loc="left")

    learning_curves(b, conv, METHODS, "steps_mean", styles=LOCAL, logx=True)
    episode_axis(b, int(conv.episode.max()), logx=True)
    b.set_ylim(0.9, 1.62)
    b.set_ylabel("Decision epochs / best final value")
    b.set_title("(b) Decision epochs per episode", loc="left")

    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(METHODS),
               handlelength=2.4, columnspacing=1.3)
    save(fig, "fig_efficiency")


if __name__ == "__main__":
    main()
