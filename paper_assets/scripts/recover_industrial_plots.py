#!/usr/bin/env python3
"""Recover the plotted data of the motor case from its audited figure exports.

    cd paper_assets && python3 scripts/recover_industrial_plots.py [--revision 48f3cfb]

Why this exists. run_07_export_paper.py drew Figures 16 and 17 from the
aggregate of the formal run, and that result tree is not in this repository.
The audited exports are: the two result tables, which
make_industrial_macros.py already reads, and the two vector figures. A vector
PDF stores every plotted coordinate exactly, so this script reads the figures
of the given paper revision and writes what they plot to
figures/data/industrial/plotted.json, from which make_industrial_figures.py
redraws them in the current style. Nothing is estimated or typed by hand.

What is read
  * fig_industrial_performance.pdf: the pooled non-dominated points of panel
    (a) and the mean replayed makespans of panel (b), by marker colour.
  * fig_industrial_gantt.pdf: every span of the two schedules, a process span
    by its type colour, an installation by its grey, a removal by its hatched
    outline.
Coordinates are mapped to data through the grid lines, matched to the numeric
tick labels.

What is checked before anything is written
  * Each schedule has the published demand of every type at every station and
    every process span lasts the published processing time.
  * Every installation and removal lasts the modal time of its station in
    data/industrial_motor/derived.json.
  * The replayed makespan at multiplier 1 is the equal-weight makespan of
    every method in tables/industrial_results.tex, and the points of SPT, of
    the two-stage construction and of NSGA-II are those of the same table.
  * Every recovered value lies on the 0.001 time grid of the case.

Requires PyMuPDF (pip install pymupdf), needed by this script only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from industrial_case.common import DATA, read_json  # noqa: E402
from industrial_case.data import PUBLISHED  # noqa: E402
from industrial_case.export import COLORS, MULTIPLIERS, PRODUCT_COLORS  # noqa: E402
from industrial_case.macros import summary_from_tables  # noqa: E402

try:
    import pymupdf
except ImportError:  # pragma: no cover
    sys.exit("[FAIL] PyMuPDF is required: python3 -m pip install pymupdf")

PAPER = Path(os.environ.get("HCMAGRL_PAPER", ROOT.parent / "Junxin_Huang_HCMAGRL_RMS_FRT")).resolve()
OUT = ROOT / "paper_assets" / "figures" / "data" / "industrial" / "plotted.json"
GREY = (0.667, 0.667, 0.667)


def rgb(value):
    """'#0072B2' or a PDF colour tuple -> rounded RGB triple."""
    if value is None:
        return None
    if isinstance(value, str):
        value = tuple(int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return tuple(round(c, 3) for c in value)


def frames(page):
    """The white axes backgrounds, largest first by position."""
    out = []
    for d in page.get_drawings():
        r = d["rect"]
        if (d["type"] == "f" and rgb(d.get("fill")) == (1.0, 1.0, 1.0) and r.height > 30 and r.width > 100
                and r.x0 > 1 and r.x1 < page.rect.width - 0.5 and not (r.width == 72 and r.height == 72)):
            out.append(r)
    return out


def grid_lines(page, frame):
    xs, ys = set(), set()
    for d in page.get_drawings():
        if d["type"] != "s":
            continue
        for item in d["items"]:
            if item[0] != "l":
                continue
            p, q = item[1], item[2]
            if abs(p.x - q.x) < 1e-6 and abs(abs(p.y - q.y) - frame.height) < 0.5:
                xs.add(round(p.x, 6))
            if abs(p.y - q.y) < 1e-6 and abs(abs(p.x - q.x) - frame.width) < 0.5:
                ys.add(round(p.y, 6))
    keep = lambda v, lo, hi: lo - 0.5 <= v <= hi + 0.5
    return (sorted(v for v in xs if keep(v, frame.x0, frame.x1)),
            sorted(v for v in ys if keep(v, frame.y0, frame.y1)))


def tick_labels(page, frame, axis):
    out = []
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        try:
            value = float(text)
        except ValueError:
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if axis == "x" and frame.y1 < cy < frame.y1 + 14 and frame.x0 - 8 < cx < frame.x1 + 8:
            out.append((cx, value))
        if axis == "y" and frame.x0 - 30 < cx < frame.x0 and frame.y0 - 6 < cy < frame.y1 + 6:
            out.append((cy, value))
    return sorted(out)


def axis_map(lines, labels, edges):
    """Least-squares map PDF -> data through the grid lines under the tick labels.

    The spines are drawn like grid lines. A label goes to the nearest interior
    line, and to a spine only when no interior line lies within 2 pt, as for a
    tick at the axis limit."""
    lines = np.asarray(lines)
    inner = np.asarray([g for g in lines if min(abs(g - e) for e in edges) > 0.05])
    matched = []
    for pos, _ in labels:
        pool = inner if inner.size and np.min(abs(inner - pos)) < 2 else lines
        matched.append(pool[np.argmin(abs(pool - pos))])
    design = np.vstack([matched, np.ones(len(matched))]).T
    values = np.asarray([v for _, v in labels])
    coef = np.linalg.lstsq(design, values, rcond=None)[0]
    error = float(np.max(abs(design @ coef - values)))
    if error > 1e-4:
        raise ValueError(f"tick labels do not sit on the grid lines (error {error:.2e})")
    return lambda v: coef[0] * v + coef[1]


def on_grid(value, step=0.001, tol=1e-4):
    snapped = round(round(value / step) * step, 6)
    if abs(snapped - value) > tol:
        raise ValueError(f"{value} is not on the {step} grid")
    return snapped


def performance(pdf):
    page = pymupdf.open(stream=pdf, filetype="pdf")[0]
    colour = {rgb(c): m for m, c in COLORS.items()}
    panels = sorted(frames(page), key=lambda r: r.x0)
    if len(panels) != 2:
        raise ValueError("expected two panels in the performance figure")
    found = []
    for frame in panels:
        gx, gy = grid_lines(page, frame)
        fx = axis_map(gx, tick_labels(page, frame, "x"), (frame.x0, frame.x1))
        fy = axis_map(gy, tick_labels(page, frame, "y"), (frame.y0, frame.y1))
        points = set()
        for d in page.get_drawings():
            c = rgb(d.get("fill")) if d["type"] in ("f", "fs") else rgb(d.get("color"))
            r = d["rect"]
            if c not in colour or r.width > 8 or r.height > 8:
                continue
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            if frame.x0 - 3 <= cx <= frame.x1 + 3 and frame.y0 - 3 <= cy <= frame.y1 + 3:
                points.add((colour[c], round(fx(cx), 6), round(fy(cy), 6)))
        found.append(sorted(points))
    fronts = {}
    for method, x, y in found[0]:
        fronts.setdefault(method, []).append([on_grid(x), int(round(on_grid(y) * 1000))])
    replay = {}
    for method, x, y in found[1]:
        replay.setdefault(method, {})[min(MULTIPLIERS, key=lambda f: abs(f - x))] = round(y, 4)
    replay = {m: [v[f] for f in MULTIPLIERS] for m, v in replay.items()}
    return fronts, replay


def gantt(pdf):
    page = pymupdf.open(stream=pdf, filetype="pdf")[0]
    product = {rgb(c): k for k, c in enumerate(PRODUCT_COLORS)}
    panels = sorted(frames(page), key=lambda r: r.y0)
    if len(panels) != 2:
        raise ValueError("expected two panels in the Gantt figure")
    gx, _ = grid_lines(page, panels[0])
    fx = axis_map(gx, tick_labels(page, panels[-1], "x"), (panels[0].x0, panels[0].x1))
    schedules = []
    for frame in panels:
        spans = []
        for d in page.get_drawings():
            for item in d["items"]:
                if item[0] != "re":
                    continue
                r = item[1]
                if not (frame.y0 - 0.1 <= r.y0 and r.y1 <= frame.y1 + 0.1 and 5 < r.height < 20):
                    continue
                fill = rgb(d.get("fill"))
                if d["type"] == "fs" and fill in product:
                    kind, prod = "process", product[fill]
                elif d["type"] == "fs" and fill == GREY:
                    kind, prod = "install", None
                elif d["type"] == "s" and abs((d.get("width") or 0) - 0.25) < 1e-6:
                    kind, prod = "remove", None
                else:
                    continue
                spans.append((kind, prod, r))
        rows = sorted({round((r.y0 + r.y1) / 2, 3) for *_, r in spans})
        if len(rows) != 5:
            raise ValueError("expected five stations in a Gantt panel")
        events = {}
        for kind, prod, r in spans:
            e = dict(machine=rows.index(round((r.y0 + r.y1) / 2, 3)), kind=kind, product=prod,
                     start=on_grid(fx(r.x0)), end=on_grid(fx(r.x1)))
            events[(e["machine"], e["kind"], e["start"], e["end"])] = e
        schedules.append(sorted(events.values(), key=lambda e: (e["machine"], e["start"])))
    return schedules


def check(fronts, replay, schedules, summary):
    processing = PUBLISHED["processing"]
    demand = PUBLISHED["demand"]
    modules = read_json(DATA / "derived.json")["modules"]
    install = {m["station"]: m["mount_time"][1] for m in modules}
    remove = {m["station"]: m["remove_time"][1] for m in modules}
    for events in schedules:
        count = Counter((e["machine"], e["product"]) for e in events if e["kind"] == "process")
        assert all(count[(j, r)] == demand[r] for j in range(5) for r in range(4)), "demand per station"
        for e in events:
            length = round(e["end"] - e["start"], 6)
            expected = {"process": lambda: processing[e["product"]][e["machine"]],
                        "install": lambda: install[e["machine"]],
                        "remove": lambda: remove[e["machine"]]}[e["kind"]]()
            assert abs(length - expected) < 1e-6, f"{e['kind']} at S{e['machine'] + 1} lasts {length}"
    rows = {r["method"]: r for r in summary}
    two_stage = max(e["end"] for e in schedules[1])
    assert abs(two_stage - rows["TwoStage"]["makespan_mean"]) < 1e-6, "two-stage makespan"
    for method, values in replay.items():
        assert abs(values[MULTIPLIERS.index(1)] - rows[method]["makespan_mean"]) < 0.006, f"replay of {method}"
    for method in ("SPT", "TwoStage"):
        assert fronts[method] == [[rows[method]["makespan_mean"], int(rows[method]["cost_mean"])]], method
    assert [rows["NSGA-II"]["makespan_mean"], int(rows["NSGA-II"]["cost_mean"])] in fronts["NSGA-II"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--revision", default="48f3cfb",
                    help="paper revision that holds the audited exports (default: the last one before the redraw)")
    args = ap.parse_args()
    show = lambda name: subprocess.run(["git", "-C", str(PAPER), "show", f"{args.revision}:figures/{name}"],
                                       check=True, capture_output=True).stdout
    perf_pdf, gantt_pdf = show("fig_industrial_performance.pdf"), show("fig_industrial_gantt.pdf")
    fronts, replay = performance(perf_pdf)
    schedules = gantt(gantt_pdf)
    tables = [(PAPER / "tables" / f"industrial_{n}.tex").read_text(encoding="utf-8") for n in ("results", "runtime")]
    check(fronts, replay, schedules, summary_from_tables(*tables))
    record = {
        "source": {"paper_revision": args.revision,
                   "sha256": {"fig_industrial_performance.pdf": hashlib.sha256(perf_pdf).hexdigest(),
                              "fig_industrial_gantt.pdf": hashlib.sha256(gantt_pdf).hexdigest()},
                   "note": "Plotted data of the audited exports of the formal run (run_07), read from their "
                           "vector paths because the result tree is not in the repository."},
        "multipliers": MULTIPLIERS,
        "fronts": {m: sorted(v) for m, v in sorted(fronts.items())},
        "replay": dict(sorted(replay.items())),
        "schedules": {"HCMAGRL": schedules[0], "TwoStage": schedules[1]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    spans = sum(len(s) for s in schedules)
    print(f"recovered {sum(len(v) for v in fronts.values())} front points, {len(replay)} replay curves "
          f"and {spans} schedule spans; all checks passed -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
