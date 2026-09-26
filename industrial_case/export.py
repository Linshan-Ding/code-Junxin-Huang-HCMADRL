"""Vector figures and blue revision tables, with a strict formal-result gate."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
import numpy as np

from paper_assets.figures.scifig_style import use_scifig
from .analysis import aggregate, nondominated
from .common import DATA, METHODS, ROOT, read_json
from .data import PUBLISHED

WIDTH = 164.6 / 25.4
COLORS = dict(zip(METHODS, ["#0072B2", "#009E73", "#CC79A7", "#6E6E6E", "#E69F00", "#56B4E9", "#D55E00"]))
MARKERS = dict(zip(METHODS, ["o", "^", "v", "s", "D", "P", "x"]))
CAPTION = r"\colorlet{scolor}{blue}\captionsetup{labelfont={color=blue},textfont={color=blue}}"


def tex(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # CAS resets the caption text colour and bypasses caption package fonts.
    value = value.replace(r"\caption{", r"\caption{\color{blue}")
    path.write_text(value.strip() + "\n", encoding="utf-8")


def save_figure(fig, paper, name):
    target = paper / "figures" / (name + ".pdf")
    target.parent.mkdir(parents=True, exist_ok=True)
    # No tight bounding box: exact final printed width is part of the manuscript contract.
    fig.savefig(target)
    proofs = ROOT / "paper_assets/figures/_proofs/industrial"
    proofs.mkdir(parents=True, exist_ok=True)
    fig.savefig(proofs / (name + ".png"), dpi=180)
    plt.close(fig)


def schematic(paper):
    use_scifig()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.85))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.04, top=0.97)
    ax.set(xlim=(-0.6, 4.6), ylim=(-1.55, 2.45))
    ax.axis("off")
    ax.text(-0.5, 2.15, "Published: four motor types A-D; demand 5 / 4 / 4 / 3; five-stage processing times", fontsize=8)
    ax.text(-0.5, 1.75, "Reconstruction: independent station changeovers and product-specific logical modules", fontsize=8, color="#555555")
    shades = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    for j in range(5):
        ax.add_patch(FancyBboxPatch((j-0.35, 0.35), .70, .80, boxstyle="round,pad=0.03", facecolor="#eef4f7", edgecolor="#334455"))
        ax.text(j, .85, f"S{j+1}", ha="center", fontsize=11)
        ax.text(j, .55, "one installed\nmodule", ha="center", va="center", fontsize=7)
        for r in range(4):
            x = j-.36+r*.18
            ax.add_patch(Rectangle((x, -.35), .16, .31, facecolor=shades[r], alpha=.85))
            ax.text(x+.08, -.195, "ABCD"[r], color="white" if r != 1 else "black", ha="center", va="center", fontsize=8)
        ax.annotate("", (j, .30), (j, -.01), arrowprops=dict(arrowstyle="<->", color="#555555", linestyle="--"))
        if j < 4:
            ax.annotate("", (j+.61, .8), (j+.39, .8), arrowprops=dict(arrowstyle="->", color="#334455"))
    ax.text(2, -.70, "Twenty stage-specific logical modules; dashed links denote assumed replaceability", ha="center", fontsize=7)
    ax.text(2, -1.17, "Upper policy: choose station/module       Lower policy: dispatch an available operation", ha="center", fontsize=8)
    save_figure(fig, paper, "fig_industrial_system")


def source_tables(paper):
    info = read_json(DATA / "derived.json")
    rows = [" & ".join([label, product, str(q), *map(str, times)]) + r" \\" for label, product, q, times in
            zip(PUBLISHED["product_labels"], PUBLISHED["products"], PUBLISHED["demand"], PUBLISHED["processing"])]
    tex(paper / "tables/industrial_case.tex", r"""
\begin{table}
\color{blue}\centering
""" + CAPTION + r"""
\caption{Literature-reported motor-case data. Processing times retain the source time-unit.}
\label{tab:industrial-data}
\small
\begin{tabular}{llrrrrrr}\toprule
Type & Motor model & Quantity & S1 & S2 & S3 & S4 & S5\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
\par\smallskip\raggedright\footnotesize
Published: product identities, quantities and processing times (Zhao et al., 2026, Section 5 and Table 3).
Derived: additive installation/removal costs fitted to their Table 1.
Assumed: station-specific logical modules, equal station cost shares, empty initial cells,
unlimited buffers and triangular installation/removal times.
\end{table}
""")
    lines = []
    for r in range(4):
        raw = " & ".join(f"{x:g}" for x in PUBLISHED["switch_cost"][r])
        fitted = " & ".join(f"{0 if r == s else info['cost_removal_line'][r]+info['cost_installation_line'][s]:.3f}" for s in range(4))
        lines.append(f"{'ABCD'[r]} & {raw} & {fitted}" + r" \\")
    tex(paper / "tables/industrial_cost_fit.tex", r"""
\begin{table}\color{blue}\centering
""" + CAPTION + r"""
\caption{Published and reconstructed line-level switching costs (thousand CNY). The diagonal is zero because no changeover is performed.}
\label{tab:industrial-cost-fit}\small
\begin{tabular}{lrrrrrrrr}\toprule
&\multicolumn{4}{c}{Published destination}&\multicolumn{4}{c}{Additive approximation}\\
From & A & B & C & D & A & B & C & D\\\midrule
""" + "\n".join(lines) + r"""\bottomrule\end{tabular}\end{table}""")
    module_lines = []
    for m in info["modules"]:
        triple = lambda v: "(" + ",".join(f"{x:g}" for x in v) + ")"
        module_lines.append(f"S{m['station']+1} & {'ABCD'[m['product']]} & {triple(m['mount_time'])} & {triple(m['remove_time'])} & {m['mount_cost_cny']} & {m['remove_cost_cny']}" + r" \\")
    tex(paper / "tables/industrial_modules.tex", r"""
\begin{table}\color{blue}\centering
""" + CAPTION + r"""
\caption{Reconstructed module parameters. Times are assumed triangular numbers in source time-units; costs are derived CNY values.}
\label{tab:industrial-modules}\small
\begin{tabular}{llrrrr}\toprule
Cell & Type & Installation time & Removal time & Install cost & Remove cost\\\midrule
""" + "\n".join(module_lines) + r"""\bottomrule\end{tabular}\end{table}""")
    return info


def fmt(row, key):
    if row.get(key + "_mean") is None:
        return "--"
    value = f"{row[key + '_mean']:.2f}"
    if row["independent_replicates"] > 1:
        value += r" $\pm$ " + f"{row[key + '_std']:.2f}"
    return value


def comparison_table(paper, summary):
    rows = []
    for row in summary:
        rows.append(" & ".join([row["method"], f"{row['feasible']}/{row['attempted']}", fmt(row, "makespan"),
                                 fmt(row, "cost"), fmt(row, "reconfigurations"), fmt(row, "hv"), fmt(row, "runtime_seconds")]) + r" \\")
    tex(paper / "tables/industrial_results.tex", r"""
\begin{table}\color{blue}\centering
""" + CAPTION + r"""
\caption{Industrial-case comparison. Objectives and changes use equal weights; HV uses each seed's five-preference portfolio.}
\label{tab:industrial-results}\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrrr}\toprule
Method & Feasible & Makespan & Cost (CNY) & Changes & HV & Exec. (s)\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
\par\smallskip\raggedright\footnotesize
Mean $\pm$ sample SD for independent training/search seeds; deterministic rules appear once.
Objective means include feasible schedules only, with the denominator reported explicitly.
Initial installations are excluded from the change count but included in time and cost.
\end{table}
""")
    rows = [" & ".join([r["method"], fmt(r, "offline_seconds"), fmt(r, "runtime_seconds"),
                         fmt(r, "original_matrix_cost")]) + r" \\" for r in summary]
    tex(paper / "tables/industrial_runtime.tex", r"""
\begin{table}\color{blue}\centering
""" + CAPTION + r"""
\caption{Industrial-case computation time and cost-model sensitivity. Offline time covers an entire preference portfolio; execution time measures one greedy rollout or rule decoding.}
\label{tab:industrial-runtime}\small
\begin{tabular}{lrrr}\toprule
Method & Offline/search (s) & Execution (s) & Original-matrix cost (CNY)\\\midrule
""" + "\n".join(rows) + r"""\bottomrule\end{tabular}\end{table}""")


def result_figures(paper, records, sensitivity):
    use_scifig()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    fig.subplots_adjust(left=.085, right=.985, bottom=.26, top=.9, wspace=.4)
    for method in METHODS:
        rows = [r for r in records if r["method"] == method and r["feasible"]]
        if rows:
            points = nondominated([[r["makespan"], r["cost"]] for r in rows])
            axes[0].scatter(points[:, 0], points[:, 1]/1000, s=19, marker=MARKERS[method], color=COLORS[method], label=method)
        rows = [r for r in sensitivity if r["method"] == method]
        if rows:
            means = [np.mean([r["makespan"] for r in rows if r["multiplier"] == f]) for f in [.5, 1, 2]]
            axes[1].plot([.5, 1, 2], means, marker=MARKERS[method], color=COLORS[method], markersize=3, lw=1, label=method)
    axes[0].set(xlabel="Makespan (source time-unit)", ylabel="Cost (thousand CNY)", title="(a) Pooled non-dominated representatives")
    axes[1].set(xlabel="Installation/removal time multiplier", ylabel="Replayed makespan", title="(b) Fixed-order schedule sensitivity", xticks=[.5, 1, 2])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    save_figure(fig, paper, "fig_industrial_performance")
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 3.55), sharex=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.19, top=.93, hspace=.36)
    product_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    for ax, method, seed in zip(axes, ["HCMAGRL", "TwoStage"], [42, -1]):
        row = next(r for r in records if r["method"] == method and r["seed"] == seed and r["weight"] == .5)
        if not row["feasible"]:
            ax.text(.5, .5, "No feasible schedule in the prespecified run", transform=ax.transAxes, ha="center")
        else:
            for e in row["events"]:
                color = product_colors[e["product"]] if e["kind"] == "process" else "#aaaaaa" if e["kind"] == "install" else "white"
                ax.barh(e["machine"], e["end"]-e["start"], left=e["start"], height=.65, color=color,
                        edgecolor="#555555", linewidth=.25, hatch="///" if e["kind"] == "remove" else None)
        ax.set(yticks=range(5), yticklabels=[f"S{j}" for j in range(1, 6)], title=f"{method}: fixed seed {seed if seed >= 0 else 'deterministic'}, equal preference")
        ax.invert_yaxis()
    axes[-1].set_xlabel("Time (source time-unit)")
    handles = [Patch(facecolor=c, label=l) for c, l in zip(product_colors, "ABCD")]
    handles += [Patch(facecolor="#aaaaaa", label="Install"), Patch(facecolor="white", edgecolor="#555555", hatch="///", label="Remove")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    save_figure(fig, paper, "fig_industrial_gantt")


def export_paper(name, output=None, paper_path=None, draft=False):
    if name != "full":
        raise ValueError("Paper export requires --profile full. Smoke results cannot populate the manuscript.")
    paper = Path(paper_path or os.environ.get("HCMAGRL_PAPER", ROOT.parent / "Junxin_Huang_HCMAGRL_RMS_FRT")).resolve()
    if not (paper / "main.tex").exists():
        raise ValueError("--paper must identify the existing manuscript repository.")
    # Check all formal inputs BEFORE mutating any manuscript artifact.
    result = None if draft else aggregate(name, output)
    existing = paper / "macros/industrial-results.tex"
    if draft and existing.exists() and r"\IndustrialResultsReadytrue" in existing.read_text(encoding="utf-8"):
        raise ValueError("Refusing to replace completed formal results with a draft.")
    info = source_tables(paper)
    schematic(paper)
    macros = r"\newif\ifIndustrialResultsReady" + "\n" + (r"\IndustrialResultsReadyfalse" if draft else r"\IndustrialResultsReadytrue")
    macros += "\n" + "\n".join("\\newcommand{\\" + key + "}{" + f"{value:.3f}" + "}" for key, value in
                                     [("IndustrialCostMAE", info["cost_fit"]["mae_thousand_cny"]),
                                      ("IndustrialCostRMSE", info["cost_fit"]["rmse_thousand_cny"])])
    tex(existing, macros)
    if draft:
        tex(paper / "tables/industrial_results.tex", r"""
\begin{table}\color{blue}\centering
""" + CAPTION + r"""
\caption{Industrial-case comparison: full experiments pending.}
\label{tab:industrial-results}
\begin{tabular}{ll}\toprule Method & Result status\\\midrule
HCMAGRL, Flat, MLP & Pending full training and evaluation\\
SPT, SetupGreedy, TwoStage, NSGA-II & Pending full-profile evaluation\\
\bottomrule\end{tabular}\end{table}
""")
        tex(paper / "tables/industrial_result_figures.tex", r"""
\par\noindent\textbf{Full experiments pending.}
The Pareto/sensitivity and Gantt figures will be generated from audited full-profile runs.
Smoke tests are excluded from manuscript results.
""")
        tex(paper / "tables/industrial_runtime.tex", r"""
\begin{table}\color{blue}\centering
""" + CAPTION + r"""
\caption{Industrial-case runtime and cost-model comparison: full experiments pending.}
\label{tab:industrial-runtime}
\begin{tabular}{ll}\toprule Item & Status\\\midrule
Training/search and execution time & Pending full-profile runs\\
Original-matrix cost comparison & Pending audited full-profile schedules\\
\bottomrule\end{tabular}\end{table}
""")
        tex(paper / "tables/industrial_findings.tex", r"\par\noindent No performance or superiority claim is made before the full experiments are completed.")
    else:
        out, manifest, records, summary, sensitivity = result
        comparison_table(paper, summary)
        result_figures(paper, records, sensitivity)
        tex(paper / "tables/industrial_result_figures.tex", r"""
Figure~\ref{fig:industrial-performance} reports the observed objective trade-offs and
fixed-order sensitivity; Figure~\ref{fig:industrial-gantt} shows the prespecified schedules.
\begin{figure}\centering\color{blue}
""" + CAPTION + r"""
\includegraphics{figures/fig_industrial_performance.pdf}
\caption{Industrial-case non-dominated representatives and fixed-order schedule replay. The replay varies assumed changeover times; it is not a stochastic-worker robustness test.}
\label{fig:industrial-performance}\end{figure}
\begin{figure}\centering\color{blue}
""" + CAPTION + r"""
\includegraphics{figures/fig_industrial_gantt.pdf}
\caption{Prespecified industrial schedules: HCMAGRL seed 42 at equal weights and the deterministic cost-first two-stage method. Unfilled spans denote idle time.}
\label{fig:industrial-gantt}\end{figure}
""")
        ours = next(r for r in summary if r["method"] == "HCMAGRL")
        base = next(r for r in summary if r["method"] == "TwoStage")
        if ours["makespan_mean"] is None:
            finding = "HCMAGRL produced no feasible balanced-preference schedule in the prescribed runs; no objective improvement is asserted."
        else:
            dt = 100*(ours["makespan_mean"]/base["makespan_mean"]-1)
            dc = 100*(ours["cost_mean"]/base["cost_mean"]-1)
            finding = (f"Relative to the cost-first two-stage schedule, the mean balanced-preference HCMAGRL makespan changes by {dt:+.2f}\\% "
                       f"and its additive reconfiguration cost by {dc:+.2f}\\%. These signed changes use feasible runs; "
                       "negative values indicate reductions, and the reported feasibility counts must be read alongside them.")
        original_rank = sorted([r for r in summary if r["cost_mean"] is not None], key=lambda r: r["original_matrix_cost_mean"])
        fitted_rank = sorted([r for r in summary if r["cost_mean"] is not None], key=lambda r: r["cost_mean"])
        finding += " The ordering of balanced-preference mean costs " + ("is unchanged" if [r["method"] for r in original_rank] == [r["method"] for r in fitted_rank] else "changes") + " when the published switching matrix is used for post-hoc accounting."
        tex(paper / "tables/industrial_findings.tex", finding)
        destination = ROOT / "paper_assets/figures/data/industrial"
        destination.mkdir(parents=True, exist_ok=True)
        for path in (out / "aggregate").glob("*"):
            shutil.copy2(path, destination / path.name)
    print(f"Exported {'pending draft' if draft else 'audited full results'} to {paper}")
