#!/usr/bin/env python3
"""Public RMSSP benchmark: makespan against four metaheuristic baselines.

The baselines were run under two termination criteria reported by their source
(m*n*1000 evaluations, and m*n*8 s of wall-clock time); the learned policy is a
single set of results compared against both. Error bars on our method are the
standard deviation over the five independent benchmark cases.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scifig_style import FULL_MM, data_dir, save, style, use_scifig  # noqa: E402

BASELINES = ["SADE", "MSELS", "IGA", "DABC"]
TITLE = {"evaluations": r"Termination: $m\,n\,{\times}\,1000$ evaluations",
         "runtime": r"Termination: $m\,n\,{\times}\,8$ s"}


def main():
    df = pd.read_csv(data_dir() / "benchmark_rmssp.csv")
    sizes = df[df.criterion == "evaluations"]["size"].unique().tolist()
    cases = df[df.criterion == "cases"]
    ours_sd = cases.groupby("size")["makespan"].std().reindex(sizes)

    use_scifig(FULL_MM, FULL_MM * 0.34)
    fig, axes = plt.subplots(1, 2, constrained_layout=True, sharey=True)
    x = np.arange(len(sizes))

    for ax, crit in zip(axes, ("evaluations", "runtime")):
        sub = df[df.criterion == crit]
        for m in BASELINES + ["HCMAGRL"]:
            d = sub[sub.algorithm == m].set_index("size").reindex(sizes)
            st = style(m)
            if m == "HCMAGRL":
                ax.errorbar(x, d.makespan, yerr=ours_sd, capsize=1.5, elinewidth=0.5,
                            color=st["color"], ls=st["ls"], marker=st["marker"],
                            lw=1.2, label=m, zorder=5)
            else:
                ax.plot(x, d.makespan, color=st["color"], ls=st["ls"],
                        marker=st["marker"], lw=0.9, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=45, ha="right")
        ax.set_xlabel(r"Instance size ($m \times n$)")
        ax.set_title(TITLE[crit], fontsize=7)
    axes[0].set_ylabel("Makespan")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, columnspacing=1.4)
    save(fig, "fig_rmssp_benchmark")


if __name__ == "__main__":
    main()
