#!/usr/bin/env python3
"""Rebuild macros/industrial-results.tex from the exported case tables.

    cd paper_assets && python3 scripts/make_industrial_macros.py

run_07_export_paper.py writes the same file from the aggregate CSVs of a formal
run. This script is for a checkout that holds the audited products of that run
(the two result tables in the paper repository) but not the result tree: it
parses tables/industrial_results.tex and tables/industrial_runtime.tex, keeps
the two cost-fit macros already in the file, and derives the rest with the
formula export.py uses (industrial_case/macros.py). Nothing is typed by hand.

The paper repository is the sibling checkout by default, or $HCMAGRL_PAPER.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from industrial_case.macros import derive, fit_from_macros, render, summary_from_tables  # noqa: E402

PAPER = Path(os.environ.get("HCMAGRL_PAPER", ROOT.parent / "Junxin_Huang_HCMAGRL_RMS_FRT")).resolve()
target = PAPER / "macros" / "industrial-results.tex"
results = (PAPER / "tables" / "industrial_results.tex").read_text(encoding="utf-8")
runtime = (PAPER / "tables" / "industrial_runtime.tex").read_text(encoding="utf-8")
existing = target.read_text(encoding="utf-8")
if r"\IndustrialResultsReadytrue" not in existing:
    sys.exit("[FAIL] the macros file is not marked ready: export the formal results first")
macros = derive(summary_from_tables(results, runtime))
target.write_text(render(fit_from_macros(existing), macros), encoding="utf-8")
print(f"macros: {len(macros)} derived definitions -> {target}")
