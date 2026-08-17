#!/usr/bin/env python3
"""Approximate Pareto fronts in the makespan-cost plane, six instances a figure.

These are the appendix grids. The main-text figures are drawn by
`plot_pareto_joint.py`, which is a port of the original code base's own Pareto
script and carries that script's style; this one is in the house style of
`scifig_style.py` and trades the marginal panels for five more instances.

Each panel is one instance. Every method contributes the 100 evaluations behind
its weight sweep (5 preference weights x 20 evaluation seeds); the dominated
ones are drawn as a faint cloud and the non-dominated ones as filled markers
joined by the attainment staircase. The cloud is what makes the front readable:
several baselines have a front of a single point, and a lone marker on empty
axes says nothing about where that method's solutions actually lie.

Both objectives are minimized, so sorting a front by makespan puts it in
descending cost, and `step(where="post")` traces the boundary of the region the
method attained.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scifig_style import FULL_MM, data_dir, save, style, use_scifig  # noqa: E402

RL = ["DDQN", "EDQN", "SAC", "D-DRL", "HCMAGRL"]
ABLATION = ["w/o EdgeAttn", "w/o Hierarchy", "w/o Heterogeneity", "w/o Graph",
            "HCMAGRL"]
# Ours is drawn last so that it sits on top of the other clouds; the legend is
# reordered to put it first.

RL_CIDS = ["C11", "C18", "C21", "C23", "C24", "C25"]
ABL_CIDS = ["C3", "C5", "C17", "C19", "C20", "C21"]

OURS = "HCMAGRL"
PAD = 0.08


def panel(ax, df, methods, cid, dims):
    d = df[(df.cid == cid) & (df.method.isin(methods))]
    # Baselines that converge on the same solution land on exactly the same
    # coordinates, so the markers are drawn in slightly decreasing size: a
    # coincident stack then nests instead of hiding everything but the last.
    for k, m in enumerate(methods):
        g = d[d.method == m]
        if g.empty:
            continue
        ms = 3.4 if m == OURS else 4.2 - 0.3 * k
        st = style(m)
        dom = g[~g.nondominated]
        ax.scatter(dom.makespan, dom.cost, s=2.2, lw=0, alpha=0.25,
                   color=st["color"], zorder=1)
        front = g[g.nondominated].sort_values("makespan")
        if len(front) > 1:
            ax.step(front.makespan, front.cost, where="post", color=st["color"],
                    ls=st["ls"], lw=0.8, zorder=3)
        ax.plot(front.makespan, front.cost, ls="none", marker=st["marker"],
                ms=ms, mfc=st["color"], mec="white", mew=0.35, label=m, zorder=4)

    # The clouds reach much further than the fronts, so letting them set the
    # limits would push every front into a corner. The window always contains
    # every non-dominated point and the bulk of the cloud; the sparse tail is
    # clipped away.
    fronts = d[d.nondominated]
    for axis, col in ((ax.set_xlim, "makespan"), (ax.set_ylim, "cost")):
        lo = min(fronts[col].min(), d[col].quantile(0.02))
        hi = max(fronts[col].max(), d[col].median())
        pad = (hi - lo) * PAD or max(abs(hi), 1.0) * PAD
        axis(lo - pad, hi + pad)

    M, A, R = dims
    ax.set_title(f"{cid} ($M{{=}}{M}$, $A{{=}}{A}$, $R{{=}}{R}$)", pad=3)


def figure(df, methods, cids, dims, stem):
    use_scifig(FULL_MM, FULL_MM * 0.66)
    fig, axes = plt.subplots(2, 3, constrained_layout=True)
    for ax, cid in zip(axes.ravel(), cids):
        panel(ax, df, methods, cid, dims[cid])
    for r in range(2):
        for c in range(3):
            if r == 1:
                axes[r][c].set_xlabel("Makespan")
            if c == 0:
                axes[r][c].set_ylabel("Reconfiguration cost")

    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [labels.index(OURS)] + [i for i, l in enumerate(labels)
                                         if l != OURS]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="outside lower center", ncol=5, columnspacing=1.4,
               handlelength=1.2)
    save(fig, stem)


def main():
    df = pd.read_csv(data_dir() / "pareto_points.csv")
    inst = pd.read_csv(data_dir() / "instances.csv").set_index("cid")
    dims = {c: (inst.at[c, "M"], inst.at[c, "A"], inst.at[c, "R"])
            for c in inst.index}
    figure(df, RL, RL_CIDS, dims, "fig_pareto_rl")
    figure(df, ABLATION, ABL_CIDS, dims, "fig_pareto_ablation")


if __name__ == "__main__":
    main()
