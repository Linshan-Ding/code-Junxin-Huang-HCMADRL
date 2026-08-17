#!/usr/bin/env python3
"""Computational cost of training.

Left: mean wall-clock seconds per training episode against instance size, which
isolates the price paid for edge-conditioned attention and for the heterogeneous
graph. Right: the number of decision epochs per episode over training, showing
that the learned policy also reaches a terminal state in fewer decisions.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scifig_style import FULL_MM, OI, data_dir, save, style, use_scifig  # noqa: E402

METHODS = ["HCMAGRL", "w/o Heterogeneity", "w/o Hierarchy", "w/o EdgeAttn",
           "w/o Graph", "SAC"]
SMOOTH = 15

# The shared palette gives the ablations and the baselines the same four
# identities, which is unambiguous as long as a figure shows one family at a
# time. This figure is the exception: it puts SAC next to the ablations as a
# cost reference, so SAC gets a local identity here rather than colliding with
# "w/o Heterogeneity".
LOCAL = {"SAC": dict(color=OI["sky"], ls=(0, (1, 1)), marker="X")}


def estyle(method: str) -> dict:
    return LOCAL.get(method, style(method))


def main():
    rt = pd.read_csv(data_dir() / "runtime.csv")
    rt = rt[rt.method.isin(METHODS)]
    agg = (rt.groupby(["method", "M"])
             .agg(sec=("sec_per_episode", "mean"), sd=("sec_per_episode", "std"))
             .reset_index())

    conv = pd.read_csv(data_dir() / "convergence.csv")
    conv = conv[(conv.weight == "balanced") & (conv.metric == "steps_mean")
                & (conv.method.isin(METHODS))]

    use_scifig(FULL_MM, FULL_MM * 0.34)
    fig, axes = plt.subplots(1, 2, constrained_layout=True)

    for m in METHODS:
        d = agg[agg.method == m].sort_values("M")
        st = estyle(m)
        axes[0].errorbar(d.M, d.sec, yerr=d.sd, capsize=1.5, elinewidth=0.5,
                         color=st["color"], ls=st["ls"], marker=st["marker"],
                         lw=1.0, label=m)
        c = conv[conv.method == m].sort_values("episode")
        mu = c["mean"].rolling(SMOOTH, min_periods=1, center=True).mean()
        axes[1].plot(c.episode, mu, color=st["color"], ls=st["ls"], lw=1.0, label=m)

    axes[0].set_xlabel("Number of cells $M$")
    axes[0].set_ylabel("Training time per episode (s)")
    axes[0].set_xticks([4, 8, 12])
    axes[0].set_yscale("log")
    axes[1].set_xlabel("Training episode")
    axes[1].set_ylabel("Normalized decision epochs")
    axes[1].set_xlim(1, 500)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6,
               columnspacing=1.2, handlelength=2.2)
    save(fig, "fig_efficiency")


if __name__ == "__main__":
    main()
