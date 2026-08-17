#!/usr/bin/env python3
"""Triangular fuzzy reconfiguration times.

Left: the membership function of a triangular fuzzy number, its graded-mean
integration representation used to drive the simulator, and the spread that the
policy observes as an explicit uncertainty feature. Right: the distribution of
both quantities over the 27 benchmark instances.

The numbers on the left are the module-0 add and remove times of instance
M4_A3_R4_J2, taken verbatim from data/M4_A3_R4_J2/module_data.csv.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scifig_style import FULL_MM, OI, data_dir, save, use_scifig  # noqa: E402

ADD = (15, 41, 60)   # module 0, time_add_fuzzy
REM = (19, 33, 49)   # module 0, time_rem_fuzzy


def graded_mean(t):
    l, m, u = t
    return (l + 4 * m + u) / 6.0


BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)


def draw_tfn(ax, t, color, label, x_text, y_text):
    l, m, u = t
    ax.plot([l, m, u], [0, 1, 0], color=color, lw=1.1, label=label, zorder=3)
    ax.fill_between([l, m, u], [0, 1, 0], color=color, alpha=0.13, lw=0, zorder=1)
    g = graded_mean(t)
    ax.plot([g, g], [0, 1.0], color=color, ls="--", lw=0.8, zorder=3)
    # Annotation is placed off the triangle and given a white backing so the
    # membership curves never run through the text.
    ax.annotate(f"$\\tilde{{t}}^{{\\,\\mathrm{{gm}}}}={g:.1f}$", xy=(x_text, y_text),
                fontsize=7, color=color, va="center", ha="center",
                bbox=BOX, zorder=6)


def main():
    use_scifig(FULL_MM, FULL_MM * 0.34)
    fig, axes = plt.subplots(1, 3, constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.3, 1, 1]})

    ax = axes[0]
    draw_tfn(ax, ADD, OI["blue"], r"mount $\tilde{t}^{+}_{\alpha}$", 48.5, 0.62)
    draw_tfn(ax, REM, OI["vermilion"], r"dismount $\tilde{t}^{-}_{\alpha}$", 25.0, 0.62)
    l, m, u = ADD
    ax.annotate("", xy=(l, 0.16), xytext=(u, 0.16),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=OI["gray"]),
                zorder=4)
    ax.text((l + u) / 2, 0.16, r"spread $u-\ell$", ha="center", va="center",
            fontsize=7, color=OI["gray"], bbox=BOX, zorder=6)
    ax.set_xlabel("Reconfiguration time")
    ax.set_ylabel(r"Membership $\mu(t)$")
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    # Upper left is the only region free of both triangles and both
    # annotations, so the legend goes there.
    ax.legend(loc="upper left", fontsize=7, ncol=1, handlelength=1.8,
              borderaxespad=0.3)

    inst = pd.read_csv(data_dir() / "instances.csv")
    for ax, col, xlabel in ((axes[1], "avg_reconfig_time", "Mean defuzzified time"),
                            (axes[2], "avg_fuzzy_spread", "Mean spread $u-\\ell$")):
        ax.hist(inst[col], bins=9, color=OI["blue"], alpha=0.6,
                edgecolor=OI["blue"], lw=0.6, zorder=3)
        mu = inst[col].mean()
        top = ax.get_ylim()[1] * 1.18
        ax.set_ylim(0, top)
        ax.axvline(mu, color=OI["vermilion"], ls="--", lw=0.9, zorder=4)
        ax.text(mu, top * 0.97, f"mean {mu:.1f}", fontsize=7,
                color=OI["vermilion"], ha="center", va="top", bbox=BOX, zorder=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of instances")

    for ax in axes:
        ax.set_axisbelow(True)   # grid behind the data
    save(fig, "fig_fuzzy_times")


if __name__ == "__main__":
    main()
