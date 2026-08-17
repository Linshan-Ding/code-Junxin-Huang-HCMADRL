#!/usr/bin/env python3
"""Pareto fronts for the main text, in the style of the original code base.

This is a port of `plot_pareto_front.py` from the HCMADRL repository -- the
script `run_weight_sweep.py` calls once the five weight runs of an instance have
finished. The plotting style is that script's and is deliberately not the house
style of `scifig_style.py`: seaborn `JointGrid`, `Set1`, Times New Roman,
translucent filled kernel densities on both margins, dashed lines through the
non-dominated points, and the legend inside the joint panel.

What changed in the port, and nothing else did:

  hue      The original colours by preference weight, because it draws one
           method at a time. Figures 11 and 12 compare methods, so the hue is
           the method. Same five levels, same palette, same order of assignment.
  layout   Each figure carries two instances. `JointGrid` builds its own
           `Figure`, so two of them cannot share a page; `joint_axes` below
           rebuilds its gridspec on a `SubFigure` instead. Everything drawn into
           those axes is still the original's seaborn calls.
  sizes    `height=8` is 203 mm and a panel here is half of a 164.6 mm text
           block, so the figure is sized to the text block and the point sizes
           come down with it. Scaling them by the same factor would put them
           under 7 pt, which is the floor for this journal, so they stop there.
  front    The original computes the non-dominated points and has the line that
           joins them commented out. Figures 11 and 12 are about those fronts,
           so the line is switched back on, drawn once per method exactly as the
           commented-out block draws it once for the pooled points.

The font is requested as Times New Roman first and falls back to Liberation
Serif, which is metric-compatible with it, on machines that do not have the
Microsoft font installed.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scifig_style import FULL_MM, MM, data_dir, save  # noqa: E402

RL = ["DDQN", "EDQN", "SAC", "D-DRL"]
ABLATION = ["w/o EdgeAttn", "w/o Hierarchy", "w/o Heterogeneity", "w/o Graph"]
OURS = "HCMAGRL"

# Two instances a figure, the richest fronts of each family: 19 and 25
# non-dominated points against the baselines, 46 and 20 against the ablations.
RL_CIDS = ["C23", "C24"]
ABL_CIDS = ["C19", "C21"]

# `JointGrid(ratio=5, space=0.2)`, the values the original passes.
RATIO = 5
SPACE = 0.2

# `bbox_inches="tight"` grows the canvas past `figsize`; this trims it back so
# that the saved PDF is exactly the 164.6 mm text block and needs no scaling.
TIGHT_SHRINK = 1.0


def is_pareto_efficient(points):
    """
    计算非支配点。
    points: [(makespan, cost), ...]
    两个目标都是越小越好。
    """
    efficient = []

    for i, (m_i, c_i) in enumerate(points):
        dominated = False

        for j, (m_j, c_j) in enumerate(points):
            if i == j:
                continue

            if (
                m_j <= m_i
                and c_j <= c_i
                and (m_j < m_i or c_j < c_i)
            ):
                dominated = True
                break

        efficient.append(not dominated)

    return efficient


def load_records(cid: str, methods):
    """The evaluations of one instance, named as the original script names them.

    The original reads the five weight checkpoints of an instance straight out
    of `.pt` files; here the same records come from `pareto_points.csv`, which
    `build_data.py` already aggregated over the weight sweep and the evaluation
    seeds, so that this figure and the reported dominance counts are built from
    one table.
    """
    df = pd.read_csv(data_dir() / "pareto_points.csv")
    df = df[(df.cid == cid) & (df.method.isin(methods))]
    return df.rename(columns={"makespan": "Makespan", "cost": "Cost",
                              "method": "Method"})


def joint_axes(sf):
    """`JointGrid`'s three axes, on a `SubFigure` so that two fit in one figure.

    Reproduced from `JointGrid.__init__`: the same square gridspec, the same
    slices, the same hiding of the ticks and labels on the density axes, the
    same despining, and the same `subplots_adjust(space)` at the end. Only the
    container differs, because `JointGrid` always makes a whole new `Figure`.
    """
    gs = sf.add_gridspec(RATIO + 1, RATIO + 1)
    ax_joint = sf.add_subplot(gs[1:, :-1])
    ax_marg_x = sf.add_subplot(gs[0, :-1], sharex=ax_joint)
    ax_marg_y = sf.add_subplot(gs[1:, -1], sharey=ax_joint)

    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_x.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_marg_x.yaxis.grid(False)
    ax_marg_y.xaxis.grid(False)

    sns.despine(sf)
    sns.despine(ax=ax_marg_x, left=True)
    sns.despine(ax=ax_marg_y, bottom=True)
    for axes in (ax_marg_x, ax_marg_y):
        for axis in (axes.xaxis, axes.yaxis):
            axis.label.set_visible(False)

    sf.subplots_adjust(hspace=SPACE, wspace=SPACE)
    return ax_joint, ax_marg_x, ax_marg_y


def panel(sf, cid, methods, dims, legend):
    df = load_records(cid, methods)

    # ======================
    # 计算非支配 Pareto 前沿
    # ======================
    # Per method, because the paper counts dominance between one method's front
    # and another's. `nondominated` was computed by build_data.py; the original
    # routine above is run over the same points as a cross-check.
    for m in methods:
        g = df[df.Method == m]
        assert list(g.nondominated) == is_pareto_efficient(
            list(zip(g.Makespan.values, g.Cost.values))), m

    ax_joint, ax_marg_x, ax_marg_y = joint_axes(sf)

    sns.scatterplot(
        data=df,
        x="Makespan",
        y="Cost",
        hue="Method",
        hue_order=methods,
        palette="Set1",
        alpha=0.75,
        s=10,
        linewidth=0,
        ax=ax_joint,
    )

    # 画 Pareto 非支配前沿线
    palette = dict(zip(methods, sns.color_palette("Set1", len(methods))))
    for m in methods:
        front = df[(df.Method == m) & df.nondominated].sort_values("Makespan")
        if len(front) >= 2:
            ax_joint.plot(
                front["Makespan"],
                front["Cost"],
                linestyle="--",
                linewidth=1.2,
                marker="o",
                markersize=3.4,
                color=palette[m],
            )

    # 边缘分布：样本太少或重复点过多时 kde 可能失败，所以单独 try
    for axis, marg in (("x", ax_marg_x), ("y", ax_marg_y)):
        try:
            sns.kdeplot(
                data=df,
                **{axis: "Makespan" if axis == "x" else "Cost"},
                hue="Method",
                hue_order=methods,
                palette="Set1",
                fill=True,
                alpha=0.35,
                linewidth=0.8,
                cut=0,
                legend=False,
                ax=marg,
                warn_singular=False,
            )
        except Exception as e:
            print(f"{axis} KDE 绘制失败，跳过：{e}")

    if legend:
        handles, labels = ax_joint.get_legend_handles_labels()
        ax_joint.legend(
            handles=handles,
            labels=labels,
            title="Method",
            loc="upper right",
            fontsize=7,
            title_fontsize=7.5,
            frameon=True,
            fancybox=True,
            shadow=True,
            handletextpad=0.3,
            borderpad=0.4,
            labelspacing=0.3,
        )
    elif ax_joint.get_legend() is not None:
        ax_joint.get_legend().remove()

    M, A, R = dims
    ax_marg_x.set_title(f"{cid}: M={M}, A={A}, R={R}", fontsize=9.5, pad=4)
    ax_joint.set_xlabel("Makespan", fontsize=9)
    ax_joint.set_ylabel("Cost", fontsize=9)


def figure(cids, methods, dims, stem):
    # ======================
    # 绘图
    # ======================
    sns.set_style("white")

    # After `set_style`, not before it. The original sets these first, and
    # `set_style` then overwrites `font.family` with its own sans-serif stack --
    # which is headed by Arial -- so the Times New Roman the original asks for
    # never reaches the page. Applying the block afterwards is what makes the
    # figure come out in the font the original names.
    plt.rcParams.update({
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.titlesize": 9.5,
        "figure.facecolor": "w",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.bottom": False,
        "ytick.left": False,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })

    width = FULL_MM * MM * TIGHT_SHRINK
    fig = plt.figure(figsize=(width, width / 2))
    subfigs = fig.subfigures(1, 2, wspace=0.02)
    for k, (sf, cid) in enumerate(zip(subfigs, cids)):
        panel(sf, cid, methods, dims[cid], legend=(k == 0))
    save(fig, stem)


def main():
    inst = pd.read_csv(data_dir() / "instances.csv").set_index("cid")
    dims = {c: (inst.at[c, "M"], inst.at[c, "A"], inst.at[c, "R"])
            for c in inst.index}
    figure(RL_CIDS, [OURS] + RL, dims, "fig_pareto_joint_rl")
    figure(ABL_CIDS, [OURS] + ABLATION, dims, "fig_pareto_joint_abl")


if __name__ == "__main__":
    main()
