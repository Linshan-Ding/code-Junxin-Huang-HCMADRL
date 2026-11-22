#!/usr/bin/env python3
"""Redraw the motor-case figures and relabel its tables without the result tree.

    cd paper_assets && python3 scripts/make_industrial_figures.py

run_07_export_paper.py writes the case products from the aggregate of a formal
run. This script is for a checkout that holds the audited products of that
run but not the result tree, the counterpart of make_industrial_macros.py:

  * Figure 15 is compiled from industrial_case/figures/fig_industrial_system.tex.
  * Tables 8 and 9 are re-emitted from their own parsed values with the method
    names of the paper, so every number stays the audited one.
  * Figures 16 and 17 are drawn from figures/data/industrial/plotted.json,
    the plotted data of the audited exports (recover_industrial_plots.py).

Every product goes through the drawing and table functions of
industrial_case/export.py, so a later formal export draws the same figures.
The paper repository is the sibling checkout by default, or $HCMAGRL_PAPER.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from industrial_case.export import (GANTT_PANELS, MULTIPLIERS, comparison_table, gantt_figure,  # noqa: E402
                                    performance_figure, schematic)
from industrial_case.macros import summary_from_tables  # noqa: E402

PAPER = Path(os.environ.get("HCMAGRL_PAPER", ROOT.parent / "Junxin_Huang_HCMAGRL_RMS_FRT")).resolve()
PLOTTED = ROOT / "paper_assets" / "figures" / "data" / "industrial" / "plotted.json"


def main():
    if not (PAPER / "main.tex").exists():
        sys.exit(f"[FAIL] no manuscript at {PAPER}")
    schematic(PAPER)
    print("  fig_industrial_system.pdf")

    tables = [(PAPER / "tables" / f"industrial_{n}.tex").read_text(encoding="utf-8") for n in ("results", "runtime")]
    comparison_table(PAPER, summary_from_tables(*tables))
    print("  tables/industrial_results.tex, tables/industrial_runtime.tex")

    data = json.loads(PLOTTED.read_text(encoding="utf-8"))
    if data["multipliers"] != MULTIPLIERS:
        sys.exit("[FAIL] plotted.json was recovered for other multipliers")
    performance_figure(PAPER, {m: np.asarray(p, dtype=float) for m, p in data["fronts"].items()}, data["replay"])
    print("  fig_industrial_performance.pdf")
    gantt_figure(PAPER, [(title, data["schedules"][method]) for title, method, _ in GANTT_PANELS])
    print("  fig_industrial_gantt.pdf")


if __name__ == "__main__":
    main()
