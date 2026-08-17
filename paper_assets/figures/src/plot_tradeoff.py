#!/usr/bin/env python3
"""Objective trade-off induced by the preference weight.

Left and centre: how each objective responds as the makespan weight alpha_t is
swept from 0 to 1, averaged over the 27 instances after normalizing each
instance by its own value at alpha_t = 0.5. Right: the resulting trade-off curve
in the (makespan, cost) plane, which is the scalarization's approximation of the
Pareto front aggregated over all instances.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scifig_style import FULL_MM, data_dir, save, style, use_scifig  # noqa: E402

METHODS = ["HCMAGRL", "w/o Hierarchy", "w/o Graph", "DDQN", "SAC"]


def normalize(df, col):
    ref = (df[df.alpha_t == 0.5].set_index(["method", "instance_id"])[col]
           .rename("ref"))
    out = df.join(ref, on=["method", "instance_id"])
    return out[col] / out["ref"]


def main():
    df = pd.read_csv(data_dir() / "eval_cells.csv")
    df = df[df.method.isin(METHODS)].copy()
    df["m_rel"] = normalize(df, "makespan_mean")
    df["c_rel"] = normalize(df, "cost_mean")
    agg = (df.groupby(["method", "alpha_t"])
             .agg(m=("m_rel", "mean"), ms=("m_rel", "std"),
                  c=("c_rel", "mean"), cs=("c_rel", "std"))
             .reset_index())

    use_scifig(FULL_MM, FULL_MM * 0.32)
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    for m in METHODS:
        d = agg[agg.method == m].sort_values("alpha_t")
        st = style(m)
        kw = dict(color=st["color"], ls=st["ls"], marker=st["marker"], lw=1.0, label=m)
        axes[0].errorbar(d.alpha_t, d.m, yerr=d.ms, capsize=1.5, elinewidth=0.5, **kw)
        axes[1].errorbar(d.alpha_t, d.c, yerr=d.cs, capsize=1.5, elinewidth=0.5, **kw)
        axes[2].plot(d.m, d.c, **kw)

    axes[0].set_xlabel(r"Makespan weight $\alpha_t$")
    axes[0].set_ylabel("Relative makespan")
    axes[1].set_xlabel(r"Makespan weight $\alpha_t$")
    axes[1].set_ylabel("Relative reconfiguration cost")
    axes[2].set_xlabel("Relative makespan")
    axes[2].set_ylabel("Relative reconfiguration cost")
    for ax in axes[:2]:
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, columnspacing=1.4)
    save(fig, "fig_weight_tradeoff")


if __name__ == "__main__":
    main()
