"""Vector figures, tables and result macros of the motor case, with a strict formal-result gate.

The products are uncoloured by default, as a fresh submission needs them. Pass
a revision colour (``--revision-color blue``) to mark them as changed text in a
revised manuscript. The derived result macros come from industrial_case.macros,
which paper_assets/scripts/make_industrial_macros.py shares.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from paper_assets.figures.scifig_style import use_scifig
from .common import DATA, METHODS, ROOT, read_json
from .data import PUBLISHED
from .macros import TABLE_LABEL, derive, render

WIDTH = 164.6 / 25.4
COLORS = dict(zip(METHODS, ["#0072B2", "#009E73", "#CC79A7", "#6E6E6E", "#E69F00", "#56B4E9", "#D55E00"]))
MARKERS = dict(zip(METHODS, ["o", "^", "v", "s", "D", "P", "x"]))
# Hollow markers of graded size, so that methods sharing a point stay visible.
SIZES = dict(zip(METHODS, [46, 34, 34, 30, 26, 40, 34]))
# Legend names: the main study's figure names for the two ablated variants and
# the words of the text for the rules.
FIGURE_LABEL = {"HCMAGRL": "HCMAGRL", "Flat": "w/o Hierarchy", "MLP": "w/o Graph", "SPT": "SPT",
                "SetupGreedy": "Greedy", "TwoStage": "Two-stage", "NSGA-II": "NSGA-II"}
PRODUCT_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]   # motor types A-D, also in Figure 15
MULTIPLIERS = [0.5, 1, 2]
# The two schedules of Figure 17, fixed before the results were seen.
GANTT_PANELS = (("(a) HCMAGRL", "HCMAGRL", 42), ("(b) Two-stage construction", "TwoStage", -1))
MM = 1 / 25.4
SCHEMATIC = Path(__file__).resolve().parent / "figures" / "fig_industrial_system.tex"
REVISION_COLOR = None   # e.g. "blue": colour the exported tables, captions and figure text as revised


def float_open(environment):
    """'\\begin{table}\\centering', coloured with the caption set-up when a revision colour is on."""
    if not REVISION_COLOR:
        return "\\begin{%s}\\centering" % environment
    return ("\\begin{%s}\\color{%s}\\centering\n\\colorlet{scolor}{%s}"
            "\\captionsetup{labelfont={color=%s},textfont={color=%s}}"
            % ((environment,) + (REVISION_COLOR,) * 4))


def tex(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if REVISION_COLOR:
        # CAS resets the caption text colour and bypasses caption package fonts.
        value = value.replace(r"\caption{", r"\caption{\color{%s}" % REVISION_COLOR)
    path.write_text(value.strip() + "\n", encoding="utf-8")


def save_figure(fig, paper, name):
    target = paper / "figures" / (name + ".pdf")
    target.parent.mkdir(parents=True, exist_ok=True)
    proofs = ROOT / "paper_assets/figures/_proofs/industrial"
    proofs.mkdir(parents=True, exist_ok=True)
    # No tight bounding box: the exact printed width is part of the manuscript
    # contract. use_scifig turns tight cropping on for the main study's figures,
    # so it is switched off here explicitly.
    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(target)
        fig.savefig(proofs / (name + ".png"), dpi=180)
    plt.close(fig)


def schematic(paper):
    """Figure 15: compile the TikZ drawing of the line at the printed width.

    The drawing (figures/fig_industrial_system.tex) uses the shared TikZ style
    of the paper and states in its header what is published, what is
    reconstructed and what is illustrative.
    """
    compiler = shutil.which("pdflatex")
    if compiler is None:
        raise FileNotFoundError("pdflatex is needed to compile " + SCHEMATIC.name)
    with tempfile.TemporaryDirectory() as build:
        run = subprocess.run([compiler, "-interaction=nonstopmode", "-halt-on-error",
                              "-output-directory=" + build, SCHEMATIC.name],
                             cwd=SCHEMATIC.parent, capture_output=True, text=True)
        if run.returncode != 0:
            raise ValueError(SCHEMATIC.name + " did not compile:\n" + run.stdout[-1500:])
        target = paper / "figures" / "fig_industrial_system.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(build) / "fig_industrial_system.pdf", target)


def source_tables(paper):
    info = read_json(DATA / "derived.json")
    rows = [" & ".join([label, product, str(q), *map(str, times)]) + r" \\" for label, product, q, times in
            zip(PUBLISHED["product_labels"], PUBLISHED["products"], PUBLISHED["demand"], PUBLISHED["processing"])]
    tex(paper / "tables/industrial_case.tex", float_open("table") + r"""
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
    tex(paper / "tables/industrial_cost_fit.tex", float_open("table") + r"""
\caption{Published and reconstructed line-level switching costs (thousand CNY). The diagonal is zero because no reconfiguration is performed.}
\label{tab:industrial-cost-fit}\small
\begin{tabular}{lrrrrrrrr}\toprule
&\multicolumn{4}{c}{Published destination}&\multicolumn{4}{c}{Additive approximation}\\
From & A & B & C & D & A & B & C & D\\\midrule
""" + "\n".join(lines) + r"""\bottomrule\end{tabular}\end{table}""")
    module_lines = []
    for m in info["modules"]:
        triple = lambda v: "(" + ",".join(f"{x:g}" for x in v) + ")"
        module_lines.append(f"S{m['station']+1} & {'ABCD'[m['product']]} & {triple(m['mount_time'])} & {triple(m['remove_time'])} & {m['mount_cost_cny']} & {m['remove_cost_cny']}" + r" \\")
    tex(paper / "tables/industrial_modules.tex", float_open("table") + r"""
\caption{Reconstructed module parameters. Times are assumed triangular numbers in the source time unit, and costs are derived values in CNY.}
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
        rows.append(" & ".join([TABLE_LABEL[row["method"]], f"{row['feasible']}/{row['attempted']}", fmt(row, "makespan"),
                                 fmt(row, "cost"), fmt(row, "reconfigurations"), fmt(row, "hv"), fmt(row, "runtime_seconds")]) + r" \\")
    tex(paper / "tables/industrial_results.tex", float_open("table") + r"""
\caption{Results on the motor-assembly case. Objectives and change counts are reported at equal weights, and the hypervolume (HV) is computed over the five-weight portfolio of each seed.}
\label{tab:industrial-results}\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrrr}\toprule
Method & Feasible & Makespan & Cost (CNY) & Changes & HV & Exec. (s)\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
\par\smallskip\raggedright\footnotesize
Mean $\pm$ sample SD over independent training or search seeds. Deterministic rules appear once.
Objective means include feasible schedules only, with the denominator reported explicitly.
Initial installations are excluded from the change count but included in time and cost.
\end{table}
""")
    rows = [" & ".join([TABLE_LABEL[r["method"]], fmt(r, "offline_seconds"), fmt(r, "runtime_seconds"),
                         fmt(r, "original_matrix_cost")]) + r" \\" for r in summary]
    tex(paper / "tables/industrial_runtime.tex", float_open("table") + r"""
\caption{Computation time and cost-model sensitivity on the motor-assembly case. Offline time covers an entire preference portfolio, and execution time measures one greedy rollout or one rule decoding.}
\label{tab:industrial-runtime}\small
\begin{tabular}{lrrr}\toprule
Method & Offline/search (s) & Execution (s) & Original-matrix cost (CNY)\\\midrule
""" + "\n".join(rows) + r"""\bottomrule\end{tabular}\end{table}""")


def _scatter(ax, method, points, scale=1.0):
    kwargs = dict(s=SIZES[method] * scale, marker=MARKERS[method], linewidths=0.9, zorder=3)
    if MARKERS[method] in ("x", "+"):
        kwargs["color"] = COLORS[method]
    else:
        kwargs.update(facecolors="none", edgecolors=COLORS[method])
    ax.scatter(points[:, 0], points[:, 1] / 1000, **kwargs)


def performance_figure(paper, fronts, replay):
    """Figure 16.

    fronts: method -> non-dominated (makespan, cost in CNY) points pooled over
    the weights and the seeds. replay: method -> mean makespan of the equal-weight
    schedules replayed at each multiplier of MULTIPLIERS. The inset of (a)
    enlarges the points within 15% of the lowest cost, so the outlying SPT point
    does not squeeze the others into a corner.
    """
    use_scifig()
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 59 * MM))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.265, top=0.895, wspace=0.27)
    points = {m: np.asarray(fronts[m], dtype=float) for m in METHODS if m in fronts}
    every = np.vstack(list(points.values()))
    near = every[every[:, 1] <= 1.15 * every[:, 1].min()]
    inset = a.inset_axes([0.14, 0.30, 0.58, 0.62])
    for method, p in points.items():
        _scatter(a, method, p)
        _scatter(inset, method, p, scale=1.3)
    a.set_xlim(every[:, 0].min() - 1.5, every[:, 0].max() + 2.5)
    a.set_ylim(every[:, 1].min() / 1000 - 0.5, every[:, 1].max() / 1000 + 0.6)
    inset.set_xlim(near[:, 0].min() - 0.9, near[:, 0].max() + 0.9)
    inset.set_ylim(near[:, 1].min() / 1000 - 0.07, near[:, 1].max() / 1000 + 0.09)
    inset.tick_params(labelsize=6, length=1.8, pad=1.5)
    inset.set_facecolor("white")
    a.indicate_inset_zoom(inset, edgecolor="#555555", lw=0.5, alpha=0.9)
    a.set(xlabel="Makespan (source time unit)", ylabel="Cost (thousand CNY)")
    a.set_title("(a) Pooled non-dominated points", loc="left")
    for method in METHODS:
        if method in replay:
            b.plot(MULTIPLIERS, replay[method], color=COLORS[method], lw=1.0, marker=MARKERS[method],
                   markersize=4.2, markerfacecolor="none", markeredgecolor=COLORS[method],
                   markeredgewidth=0.9, label=FIGURE_LABEL[method])
    b.set(xlabel="Installation and removal time multiplier", ylabel="Replayed makespan",
          xticks=MULTIPLIERS, xticklabels=["0.5", "1", "2"])
    b.set_title("(b) Fixed-order replay", loc="left")
    handles, labels = b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               handlelength=2.2, columnspacing=1.3)
    save_figure(fig, paper, "fig_industrial_performance")


def gantt_figure(paper, schedules):
    """Figure 17. schedules: (panel title, events or None) for each panel."""
    use_scifig()
    fig, axes = plt.subplots(len(schedules), 1, figsize=(WIDTH, 86 * MM), sharex=True)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.165, top=0.935, hspace=0.42)
    longest = max((max(e["end"] for e in ev) for _, ev in schedules if ev), default=0)
    for ax, (title, events) in zip(axes, schedules):
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        if not events:
            ax.text(.5, .5, "No feasible schedule in the prespecified run", transform=ax.transAxes, ha="center")
            continue
        for e in events:
            face, hatch = {"process": (None, None), "install": ("#AAAAAA", None),
                           "remove": ("white", "////")}[e["kind"]]
            ax.barh(e["machine"], e["end"] - e["start"], left=e["start"], height=.62,
                    color=face or PRODUCT_COLORS[e["product"]], edgecolor="#555555",
                    linewidth=.25, hatch=hatch)
        makespan = max(e["end"] for e in events)
        ax.axvline(makespan, color="black", lw=0.7, zorder=4)
        ax.text(makespan, 1.02, r"$C_{\max} = %.2f$" % makespan, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=7)
        ax.set(yticks=range(5), yticklabels=[f"S{j}" for j in range(1, 6)], ylim=(4.6, -0.6))
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlim(0, 5 * np.ceil(1.04 * longest / 5))
    axes[-1].set_xlabel("Time (source time unit)")
    handles = [Patch(facecolor=c, edgecolor="#555555", linewidth=.25, label=l)
               for c, l in zip(PRODUCT_COLORS, "ABCD")]
    handles += [Patch(facecolor="#AAAAAA", edgecolor="#555555", linewidth=.25, label="Installation"),
                Patch(facecolor="white", edgecolor="#555555", linewidth=.25, hatch="////", label="Removal")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    save_figure(fig, paper, "fig_industrial_gantt")


def result_figures(paper, records, sensitivity):
    """Figures 16 and 17 from the audited records of a formal run."""
    from .analysis import nondominated   # needs the simulation stack, which drawing alone does not
    fronts, replay = {}, {}
    for method in METHODS:
        rows = [r for r in records if r["method"] == method and r["feasible"]]
        if rows:
            fronts[method] = nondominated([[r["makespan"], r["cost"]] for r in rows])
        rows = [r for r in sensitivity if r["method"] == method]
        if rows:
            replay[method] = [float(np.mean([r["makespan"] for r in rows if r["multiplier"] == f]))
                              for f in MULTIPLIERS]
    performance_figure(paper, fronts, replay)
    schedules = []
    for title, method, seed in GANTT_PANELS:
        row = next(r for r in records if r["method"] == method and r["seed"] == seed and r["weight"] == .5)
        schedules.append((title, row["events"] if row["feasible"] else None))
    gantt_figure(paper, schedules)


def export_paper(name, output=None, paper_path=None, draft=False, color=None):
    global REVISION_COLOR
    REVISION_COLOR = color or None
    if name != "full":
        raise ValueError("Paper export requires --profile full. Smoke results cannot populate the manuscript.")
    paper = Path(paper_path or os.environ.get("HCMAGRL_PAPER", ROOT.parent / "Junxin_Huang_HCMAGRL_RMS_FRT")).resolve()
    if not (paper / "main.tex").exists():
        raise ValueError("--paper must identify the existing manuscript repository.")
    from .analysis import aggregate
    # Check all formal inputs BEFORE mutating any manuscript artifact.
    result = None if draft else aggregate(name, output)
    existing = paper / "macros/industrial-results.tex"
    if draft and existing.exists() and r"\IndustrialResultsReadytrue" in existing.read_text(encoding="utf-8"):
        raise ValueError("Refusing to replace completed formal results with a draft.")
    info = source_tables(paper)
    schematic(paper)
    fit = [("IndustrialCostMAE", info["cost_fit"]["mae_thousand_cny"]),
           ("IndustrialCostRMSE", info["cost_fit"]["rmse_thousand_cny"])]
    if draft:
        tex(existing, render(fit, {}, ready=False))
        tex(paper / "tables/industrial_results.tex", float_open("table") + r"""
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
        tex(paper / "tables/industrial_runtime.tex", float_open("table") + r"""
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
        tex(existing, render(fit, derive(summary)))
        result_figures(paper, records, sensitivity)
        tex(paper / "tables/industrial_result_figures.tex", r"""
Figure~\ref{fig:industrial-performance} reports the observed objective trade-offs and
the fixed-order sensitivity. Figure~\ref{fig:industrial-gantt} shows the prespecified schedules.
""" + float_open("figure") + r"""
\includegraphics{figures/fig_industrial_performance.pdf}
\caption{Objective trade-offs on the motor-assembly case. (a) Pooled non-dominated points of every method over the five preference weights and the seeds. (b) Makespan of each schedule at equal weights, replayed with the operation sequence of every station fixed, under installation and removal times scaled by $0.5$, $1$ and $2$. The replay varies assumed durations and is not a test of robustness to operator variability.}
\label{fig:industrial-performance}\end{figure}
""" + float_open("figure") + r"""
\includegraphics{figures/fig_industrial_gantt.pdf}
\caption{Schedules at equal weights on the motor-assembly case, HCMAGRL with seed $42$ and the deterministic two-stage construction, both fixed before the results were seen. Colors are the four motor types, gray spans are installations, hatched spans are removals and unfilled spans are idle time.}
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
                       f"and its additive reconfiguration cost by {dc:+.2f}\\%. These signed changes use feasible runs. "
                       "Negative values indicate reductions, and the reported feasibility counts must be read alongside them.")
        original_rank = sorted([r for r in summary if r["cost_mean"] is not None], key=lambda r: r["original_matrix_cost_mean"])
        fitted_rank = sorted([r for r in summary if r["cost_mean"] is not None], key=lambda r: r["cost_mean"])
        finding += " The ordering of balanced-preference mean costs " + ("is unchanged" if [r["method"] for r in original_rank] == [r["method"] for r in fitted_rank] else "changes") + " when the published switching matrix is used for post-hoc accounting."
        tex(paper / "tables/industrial_findings.tex", finding)
        destination = ROOT / "paper_assets/figures/data/industrial"
        destination.mkdir(parents=True, exist_ok=True)
        for path in (out / "aggregate").glob("*"):
            shutil.copy2(path, destination / path.name)
    print(f"Exported {'pending draft' if draft else 'audited full results'} to {paper}")
