#!/usr/bin/env python3
"""Consistency checks for the manuscript.

Run after `latexmk -pdf main.tex`:  python3 scripts/check.py

Checks
  1. every \\ref / \\eqref target has a matching \\label
  2. every float that defines a label is cited by \\ref somewhere
  3. every \\includegraphics target exists on disk
  4. every \\cite key exists in the bibliography
  5. every bibliography entry is cited
  6. newly generated figures are included without a scaling key
  7. no undefined references or citations remain in main.log
  8. overfull boxes worse than 2 pt
  9. no float carries a position specifier
 10. the abstract is within the venue's word limit
 11. the highlights are within the venue's count and character limits
 12. the keyword count is within the venue's range
 13. the title is not overlong and does not repeat a word root

Checks 10-13 are the venue's own hard limits. They are cheap to run and
expensive to miss: this manuscript once carried a 388-word abstract against a
250-word limit and two highlights over the 85-character limit, none of which is
visible by eye.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Submission limits, per the venues' guides for authors. Confirm against the
# guide at submission time -- publishers do revise these.
VENUE = {
    "elsevier":  dict(abstract=250, highlights=(3, 5, 85), keywords=(3, 8)),
    "rcim":      dict(abstract=250, highlights=(3, 5, 85), keywords=(3, 8)),
    "ieee-trans": dict(abstract=250, highlights=None, keywords=(3, 8)),
}
TARGET = "rcim"

# Titles in this field run 9-15 words; a longer one is a warning, not a failure.
TITLE_WORDS = 18

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--paper", type=Path,
                default=Path(os.environ.get(
                    "HCMAGRL_PAPER",
                    Path(__file__).resolve().parents[3] / "Junxin_Huang_HCMAGRL_RMS_FRT")),
                help="the manuscript directory (default: the sibling checkout)")
ap.add_argument("--venue", default=TARGET, choices=sorted(VENUE))
args = ap.parse_args()

ROOT = args.paper.resolve()
LIMITS = VENUE[args.venue]
SOURCES = [ROOT / "main.tex"] + sorted((ROOT / "sections").glob("*.tex")) \
          + sorted((ROOT / "appendix").glob("*.tex"))
TABLES = sorted((ROOT / "tables").glob("*.tex"))

problems: list[str] = []
notes: list[str] = []


def read(paths) -> str:
    return "\n".join(p.read_text() for p in paths)


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


body = strip_comments(read(SOURCES))
alltex = strip_comments(read(SOURCES + TABLES))

# ---- 1 & 2: labels and references ---------------------------------------
labels = set(re.findall(r"\\label\{([^}]+)\}", alltex))
refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", alltex))

for r in sorted(refs - labels):
    problems.append(f"reference to a missing label: {r}")

floats = {l for l in labels if l.split(":")[0] in {"fig", "tab", "alg"}}
for l in sorted(floats - refs):
    problems.append(f"float never referenced in the text: {l}")

# ---- 3: figure files ------------------------------------------------------
included = re.findall(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", body)
for opts, target in included:
    if not (ROOT / target).exists():
        problems.append(f"missing figure file: {target}")

# ---- 6: scaling keys ------------------------------------------------------
# Every figure is generated at its final printed width, so a scaling key would
# mean its labels no longer print at the size they were designed for.
for opts, target in included:
    if opts and re.search(r"\b(width|height|scale)\s*=", opts):
        problems.append(f"figure included with a scaling key: {target} {opts}")

# ---- 9: float position specifiers ----------------------------------------
# Every float is left to LaTeX's own placement algorithm. A [!t]-style
# specifier is what pushed the whole float set past the bibliography once
# already, so the rule is enforced here rather than remembered.
for src in SOURCES + TABLES:
    text = strip_comments(src.read_text())
    for env in ("figure", "table", "algorithm"):
        for m in re.finditer(r"\\begin\{" + env + r"\*?\}\s*\[", text):
            line = text[: m.start()].count("\n") + 1
            problems.append(
                f"float position specifier at {src.relative_to(ROOT)}:{line} "
                f"(\\begin{{{env}}} must carry no [])"
            )

# ---- 4 & 5: citations -----------------------------------------------------
bib = (ROOT / "cas-refs.bib").read_text()
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited: set[str] = set()
for group in re.findall(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", body):
    cited.update(k.strip() for k in group.split(","))

for k in sorted(cited - bibkeys):
    problems.append(f"citation with no bibliography entry: {k}")
for k in sorted(bibkeys - cited):
    notes.append(f"bibliography entry never cited: {k}")

# ---- 7 & 8: the build log -------------------------------------------------
log_path = ROOT / "main.log"
if not log_path.exists():
    problems.append("main.log not found; run latexmk first")
else:
    log = log_path.read_text(errors="replace")
    for m in set(re.findall(r"Warning: (?:Reference|Citation) `([^']+)' undefined", log)):
        problems.append(f"undefined in the last pass: {m}")
    overfull = [float(m) for m in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide", log)]
    bad = [o for o in overfull if o > 2.0]
    if bad:
        notes.append(f"{len(bad)} overfull hboxes worse than 2pt "
                     f"(largest {max(bad):.1f}pt)")

# ---- 10-13: the venue's hard limits --------------------------------------
main = strip_comments((ROOT / "main.tex").read_text())


def environment(name: str) -> str | None:
    m = re.search(r"\\begin\{%s\}(.*?)\\end\{%s\}" % (name, name), main, re.S)
    return m.group(1) if m else None


abstract = environment("abstract")
if abstract is None:
    problems.append("no abstract found in main.tex")
else:
    # A macro stands for the one number it expands to, so it counts as one word.
    words = re.sub(r"[{}\\~]|--", " ", re.sub(r"\\[A-Za-z]+", "X", abstract)).split()
    n = len([w for w in words if re.search(r"\w", w)])
    if n > LIMITS["abstract"]:
        problems.append(f"abstract is {n} words, over the {LIMITS['abstract']}-word limit")
    else:
        notes.append(f"abstract {n}/{LIMITS['abstract']} words")

hl = environment("highlights")
if LIMITS["highlights"] and hl is not None:
    lo, hi, chars = LIMITS["highlights"]
    items = [i.strip() for i in re.findall(r"\\item (.*)", hl)]
    if not lo <= len(items) <= hi:
        problems.append(f"{len(items)} highlights, outside the {lo}-{hi} range")
    for it in items:
        if len(it) > chars:
            problems.append(f"highlight is {len(it)} characters, over {chars}: {it[:48]}...")
    if items:
        notes.append(f"highlights {len(items)} items, longest {max(len(i) for i in items)}/{chars} chars")

kw = environment("keywords")
if kw is not None and LIMITS["keywords"]:
    lo, hi = LIMITS["keywords"]
    n = len([k for k in kw.split(r"\sep") if k.strip()])
    if not lo <= n <= hi:
        problems.append(f"{n} keywords, outside the {lo}-{hi} range")
    else:
        notes.append(f"keywords {n} (range {lo}-{hi})")

title = re.search(r"\\title\[mode = title\]\{(.*?)\}\n", main, re.S)
if title:
    t = title.group(1)
    if len(t.split()) > TITLE_WORDS:
        notes.append(f"title is {len(t.split())} words; this field runs 9-15")
    # A root repeated three times reads as clumsy even when each use is correct.
    roots: dict[str, int] = {}
    for w in re.findall(r"[A-Za-z]{6,}", t.lower()):
        roots[w[:8]] = roots.get(w[:8], 0) + 1
    for root, c in sorted(roots.items()):
        if c >= 3:
            notes.append(f"title repeats the root '{root}-' {c} times")

# ---- report ---------------------------------------------------------------
print(f"sources : {len(SOURCES)} files")
print(f"labels  : {len(labels)}   references: {len(refs)}")
print(f"figures : {len(included)} inclusions")
print(f"bib     : {len(cited)} cited of {len(bibkeys)} entries")
print()
for n in notes:
    print(f"  note   {n}")
for p in problems:
    print(f"  FAIL   {p}")
print()
if problems:
    print(f"{len(problems)} problem(s) found.")
    sys.exit(1)
print("all checks passed.")
