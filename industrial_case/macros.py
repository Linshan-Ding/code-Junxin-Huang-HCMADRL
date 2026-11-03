"""Derived result macros of the motor case, computed by one formula from two sources.

The manuscript types no experimental number by hand. The numbers that its
case-study prose needs beyond the tables (relative gaps, ratios, hours) are
LaTeX macros written to macros/industrial-results.tex:

* export.py derives them from the aggregate summary when the paper is
  exported after a formal run (run_07);
* paper_assets/scripts/make_industrial_macros.py derives the same macros from
  the two exported result tables of the paper repository when the aggregate
  CSVs are not at hand, so that a checkout holding only the audited products
  still builds a manuscript whose numbers trace back to the pipeline.

Both paths call `derive`, so they cannot disagree on a formula. Signed gaps are
emitted next to their magnitudes so that the prose can state a direction
without a number typed by hand.
"""
from __future__ import annotations

import re

ROUND = {"makespan": 1, "cost": 0, "original_matrix_cost": 0, "reconfigurations": 0, "hv": 2,
         "runtime_seconds": 2, "offline_seconds": 0}
SLUG = {"HCMAGRL": "Ours", "Flat": "Flat", "MLP": "MLP", "SPT": "SPT", "SetupGreedy": "SetupGreedy",
        "TwoStage": "TwoStage", "NSGA-II": "NSGA"}


def _num(value, digits):
    return f"{value:.{digits}f}"


def _pct(numerator, denominator):
    return 100.0 * (numerator / denominator - 1.0)


def derive(summary):
    """Macro name -> LaTeX text, from summary rows (method, feasible, attempted, <field>_mean/_std)."""
    rows = {r["method"]: r for r in summary}
    ours, two, nsga, greedy = rows["HCMAGRL"], rows["TwoStage"], rows["NSGA-II"], rows["SetupGreedy"]
    m = {}
    for method, slug in SLUG.items():
        r = rows.get(method)
        if r is None or r.get("makespan_mean") is None:
            continue
        m[f"IndustrialFeasible{slug}"] = f"{r['feasible']}/{r['attempted']}"
        for field, digits in ROUND.items():
            mean = r.get(f"{field}_mean")
            if mean is None:
                continue
            m[f"Industrial{_camel(field)}{slug}"] = _num(mean, digits)
            std = r.get(f"{field}_std")
            if std is not None:
                m[f"Industrial{_camel(field)}SD{slug}"] = _num(std, digits)
    # relative gaps at equal weights, signed (direction) and unsigned (magnitude)
    for name, (a, b) in {"MsTwoStage": (ours["makespan_mean"], two["makespan_mean"]),
                         "CostTwoStage": (ours["cost_mean"], two["cost_mean"]),
                         "MsNSGA": (nsga["makespan_mean"], ours["makespan_mean"]),
                         "CostNSGA": (nsga["cost_mean"], ours["cost_mean"])}.items():
        gap = _pct(a, b)
        m[f"IndustrialGap{name}Signed"] = f"${gap:+.1f}$\\%"
        m[f"IndustrialGap{name}"] = f"{abs(gap):.1f}\\%"
    # offline and online computation
    m["IndustrialOfflineOursHours"] = _num(ours["offline_seconds_mean"] / 3600.0, 1)
    m["IndustrialSearchNSGA"] = _num(nsga["offline_seconds_mean"], 1)
    m["IndustrialExecToSearchRatio"] = f"{nsga['offline_seconds_mean'] / ours['runtime_seconds_mean']:.0f}"
    # cost-model sensitivity
    m["IndustrialOrigCostSDRatio"] = _num(ours["original_matrix_cost_std"] / ours["cost_std"], 1) \
        if ours.get("cost_std") else "--"
    m["IndustrialOrigCostGapSetupGreedySigned"] = f"${_pct(ours['original_matrix_cost_mean'], greedy['original_matrix_cost_mean']):+.1f}$\\%"
    return m


def _camel(field):
    return {"makespan": "Makespan", "cost": "Cost", "original_matrix_cost": "OrigCost",
            "reconfigurations": "Changes", "hv": "HV", "runtime_seconds": "Exec",
            "offline_seconds": "Offline"}[field]


def render(fit, macros, ready=True):
    """The complete macros file: readiness flag, cost-fit errors, derived macros."""
    lines = [r"\newif\ifIndustrialResultsReady",
             r"\IndustrialResultsReadytrue" if ready else r"\IndustrialResultsReadyfalse"]
    lines += ["\\newcommand{\\%s}{%.3f\\xspace}" % (key, value) for key, value in fit]
    lines += ["\\newcommand{\\%s}{%s\\xspace}" % (key, value) for key, value in macros.items()]
    return "\n".join(lines) + "\n"


def fit_from_macros(text):
    """The two cost-fit macros already in a macros file, so a re-render keeps them."""
    pairs = []
    for key in ("IndustrialCostMAE", "IndustrialCostRMSE"):
        match = re.search(r"\\newcommand\{\\" + key + r"\}\{([0-9.]+)(?:\\xspace)?\}", text)
        if not match:
            raise ValueError(f"{key} missing from the existing macros file")
        pairs.append((key, float(match.group(1))))
    return pairs


def _cell(text):
    """'196.54 $\\pm$ 3.41' -> (196.54, 3.41); '212.75' -> (212.75, None); '--' -> (None, None)."""
    text = text.strip()
    if text in {"--", ""}:
        return None, None
    parts = [p.strip() for p in text.split(r"$\pm$")]
    return float(parts[0]), (float(parts[1]) if len(parts) > 1 else None)


def _rows(tex, columns):
    rows = {}
    for line in tex.splitlines():
        if "&" not in line or line.strip().startswith(("Method", "%", "\\", "From", "&")):
            continue
        # the export joins the last row and \bottomrule on one line
        cells = [c.strip() for c in line.split(r"\bottomrule")[0].replace(r"\\", "").split("&")]
        if len(cells) != columns:
            continue
        rows[cells[0]] = cells[1:]
    if not rows:
        raise ValueError("no data rows found in the table")
    return rows


def summary_from_tables(results_tex, runtime_tex):
    """Rebuild the summary rows that `derive` needs from the two exported result tables."""
    results, runtime = _rows(results_tex, 7), _rows(runtime_tex, 4)
    summary = []
    for method, cells in results.items():
        feasible, attempted = (int(x) for x in cells[0].split("/"))
        row = dict(method=method, feasible=feasible, attempted=attempted)
        for field, cell in zip(["makespan", "cost", "reconfigurations", "hv", "runtime_seconds"], cells[1:]):
            row[f"{field}_mean"], row[f"{field}_std"] = _cell(cell)
        offline, _exec, original = runtime[method]
        row["offline_seconds_mean"], row["offline_seconds_std"] = _cell(offline)
        row["original_matrix_cost_mean"], row["original_matrix_cost_std"] = _cell(original)
        row["independent_replicates"] = attempted
        summary.append(row)
    return summary
