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
 14. for a double-anonymized venue, the reviewer copy carries no author details
 15. the submission package (submission.yaml, Title_Page, Cover_Letter) agrees
     with the manuscript: same title, same author order, corresponding authors
     starred, letter within one page with the required statements, templates
     free of identity literals, anonymous flag consistent with the class option

Checks 10-15 are the venue's own hard limits. They are cheap to run and
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
    "aei":       dict(abstract=250, highlights=(3, 5, 85), keywords=(3, 8)),
    "jms":       dict(abstract=250, highlights=(3, 5, 85), keywords=(3, 8)),
    "cie":       dict(abstract=250, highlights=(3, 5, 85), keywords=(3, 8)),
    "rcim":      dict(abstract=250, highlights=(3, 5, 85), keywords=(3, 8)),
    # Computers in Industry reviews double-anonymized: author details go on a
    # separate title page and the reviewer copy must carry none of them.
    "cii":       dict(abstract=250, highlights=(3, 5, 85), keywords=(3, 8), anonymized=True),
    "ieee-trans": dict(abstract=250, highlights=None, keywords=(3, 8)),
}
TARGET = "cii"

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
included = re.findall(r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}", alltex)
for opts, target in included:
    if not (ROOT / target).exists():
        problems.append(f"missing figure file: {target}")

# ---- 6: scaling keys ------------------------------------------------------
# Every data figure is generated at its final printed width, so a scaling key
# would mean its labels no longer print at the size they were designed for.
# The TikZ schematics are the documented exception: they are drawn at natural
# size, re-exported from Overleaf, and scaled to the text width in the body.
SCALED_OK = {
    "figures/fig-system.pdf",
    "figures/fig-framework.pdf",
    "figures/fig-graph-state.pdf",
    "figures/fig-action-mask.pdf",
    "figures/fig-attention.pdf",
}
for opts, target in included:
    if target in SCALED_OK:
        continue
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
# The cover letter's journal-fit citations live in the same .bib so that they
# are real entries, but they are cited by the letter, not by the body.
SUBMISSION_YAML = ROOT / "submission.yaml"
submission_cfg = None
fit_keys: set[str] = set()
if SUBMISSION_YAML.exists():
    try:
        import yaml as _yaml
        submission_cfg = _yaml.safe_load(SUBMISSION_YAML.read_text(encoding="utf-8"))
        fit_keys = set(submission_cfg.get("cover_letter", {}).get("fit_citekeys", []) or [])
    except ImportError:
        notes.append("submission: pyyaml not installed, yaml-based checks skipped")
for k in sorted(bibkeys - cited - fit_keys):
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

# ---- 14: double-anonymized review ----------------------------------------
# The class's doubleblind option hides the author block, and the
# \ifnum\theblind>0 branches in main.tex hide the acknowledgments and the
# repository address. Read the manuscript the way the reviewer copy prints it
# and look for what must not be there.
if LIMITS.get("anonymized"):
    before = len(problems)
    opts = re.search(r"\\documentclass\[([^\]]*)\]", main)
    if not (opts and "doubleblind" in opts.group(1)):
        problems.append("double-anonymized venue, but main.tex lacks the doubleblind class option")
    reviewer_copy = re.sub(r"\\ifnum\\theblind>0\\relax(.*?)\\else(.*?)\\fi",
                           r"\1", main, flags=re.S)
    # what the class option itself withholds
    reviewer_copy = re.sub(
        r"^\s*\\(author|ead|credit|affiliation|cormark|cortext|shortauthors)\b.*$",
        "", reviewer_copy, flags=re.M)
    prose = reviewer_copy + "\n" + strip_comments(read(SOURCES[1:]))
    for pat, what in [(r"github\.com/", "a repository address"),
                      (r"[Aa]cknowledg", "an acknowledgment"),
                      (r"supported by", "a funding statement"),
                      (r"\b[\w.]+@[\w.]+\.(edu|cn|com|org)\b", "an e-mail address"),
                      (r"\bour (previous|earlier|prior) work\b", "a self-identifying citation")]:
        m = re.search(pat, prose)
        if m:
            problems.append(f"reviewer copy still carries {what}: '{m.group(0)}'")
    if not (ROOT / "submission" / "Title_Page.tex").exists():
        problems.append("double-anonymized venue, but submission/Title_Page.tex is missing")
    if len(problems) == before:
        notes.append("anonymized: doubleblind option set, author details confined to submission/Title_Page.tex")

# ---- 15: the submission package agrees with the manuscript ----------------
# submission.yaml is the only place author facts are typed; make_submission.py
# renders the title page, the cover letter and the manuscript's author block
# from it. This check reads back what was rendered.
if SUBMISSION_YAML.exists():
    before = len(problems)
    sub = ROOT / "submission"
    tp, cl = sub / "Title_Page.tex", sub / "Cover_Letter.tex"
    for f in (tp, cl):
        if not f.exists():
            problems.append(f"submission: {f.name} missing (run make_submission.py)")
    if sub.exists():
        extra = sorted(p.name for p in sub.iterdir() if p.suffix not in {".tex", ".pdf"})
        if extra:
            problems.append(f"submission: stray files {extra} -- the folder holds only .tex and .pdf")
        stems = sorted({p.stem for p in sub.iterdir() if p.suffix in {".tex", ".pdf"}})
        if stems and stems != ["Cover_Letter", "Title_Page"]:
            problems.append(f"submission: expected exactly Title_Page and Cover_Letter, found {stems}")
    if not (ROOT / "frontmatter-authors.tex").exists():
        problems.append("submission: frontmatter-authors.tex missing (run make_submission.py)")
    if tp.exists() and cl.exists() and title:
        main_title = re.sub(r"\s+", " ", title.group(1)).strip().replace("\\", "")
        for f in (tp, cl):
            txt = re.sub(r"\s+", " ", f.read_text(encoding="utf-8")).replace("\\", "")
            if main_title not in txt:
                problems.append(f"submission: {f.name} title differs from main.tex")
        body = cl.read_text(encoding="utf-8")
        m = re.search(r"Dear .*?,(.*?)Yours sincerely", body, re.S)
        if m:
            letter = re.sub(r"%.*", "", m.group(1))
            letter = re.sub(r"\\[A-Za-z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", letter)
            words = len(re.findall(r"[A-Za-z]{2,}", letter))
            if words > 420:
                problems.append(f"submission: cover letter body is {words} words (keep it to one page, <= 420)")
            else:
                notes.append(f"submission: cover letter body {words} words")
            for need, label in (("original", "originality"), ("approved the manuscript", "all-authors-approved"),
                                ("competing", "competing-interests")):
                if need not in m.group(1).lower():
                    problems.append(f"submission: cover letter lacks the {label} statement")
        if submission_cfg:
            names = [a["name"] for a in submission_cfg["authors"]]
            tptxt = tp.read_text(encoding="utf-8")
            pos = [tptxt.find(n) for n in names]
            if -1 in pos:
                problems.append(f"submission: author missing from Title_Page: {names[pos.index(-1)]}")
            elif pos != sorted(pos):
                problems.append("submission: author order in Title_Page differs from submission.yaml")
            for n in [a["name"] for a in submission_cfg["authors"] if a.get("corresponding")]:
                if not re.search(re.escape(n) + r"\$\^\{\*\}\$", tptxt):
                    problems.append(f"submission: corresponding author {n} not starred on Title_Page")
            opts = re.search(r"\\documentclass\[([^\]]*)\]", main)
            blind = bool(opts and "doubleblind" in opts.group(1))
            if bool(submission_cfg.get("anonymous")) != blind:
                problems.append("submission: `anonymous:` in submission.yaml and the doubleblind class option disagree")
    # identity must live in submission.yaml only: templates carry no e-mail or ORCID literals
    for tpl in (Path(__file__).resolve().parents[1] / "templates").glob("*.tpl.tex"):
        tt = tpl.read_text(encoding="utf-8")
        if re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", tt) or re.search(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]", tt):
            problems.append(f"submission: template {tpl.name} contains an e-mail or ORCID literal -- identity belongs in submission.yaml")
    if len(problems) == before:
        notes.append("submission: Title_Page and Cover_Letter agree with submission.yaml and main.tex")

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
