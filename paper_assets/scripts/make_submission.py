#!/usr/bin/env python3
"""Render Title_Page.tex and Cover_Letter.tex from the paper's submission.yaml, compile, clean.

    cd paper_assets
    python3 scripts/make_submission.py            # renders + compiles
    python3 scripts/make_submission.py --no-pdf   # renders only

Single source of truth: submission.yaml at the root of the paper repository holds
every author fact; the title comes from main.tex; the numbers in the letter come
from macros/results.tex; the journal-fit citations come from cas-refs.bib. Nothing
about the authors is typed twice. The author block of main.tex is written to
frontmatter-authors.tex in the CAS syntax, or left empty when `anonymous: true`
(the manuscript then also carries the class's doubleblind option, which check.py
verifies).

The paper repository is the sibling checkout by default, or $HCMAGRL_PAPER.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent                                   # paper_assets/
PAPER = Path(os.environ.get("HCMAGRL_PAPER",
                            ASSETS.parents[1] / "Junxin_Huang_HCMAGRL_RMS_FRT")).resolve()
OUT = PAPER / "submission"
TPL = ASSETS / "templates"
BIB = PAPER / "cas-refs.bib"
COMPILER = "pdflatex"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--no-pdf", action="store_true", help="render .tex only")
args = ap.parse_args()

# ---------------------------------------------------------------- helpers
ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
       "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s: str) -> str:
    """Escape LaTeX specials in plain text. Macros such as \\GapMsEDQN{} pass through."""
    out, i = [], 0
    for m in re.finditer(r"\\[A-Za-z]+(?:\{\})?", s):
        out.append("".join(ESC.get(c, c) for c in s[i:m.start()]))
        out.append(m.group(0))
        i = m.end()
    out.append("".join(ESC.get(c, c) for c in s[i:]))
    return "".join(out)


def fill(template: str, mapping: dict) -> str:
    for k, v in mapping.items():
        template = template.replace(f"<<{k}>>", v)
    left = re.findall(r"<<[a-z_]+>>", template)
    if left:
        sys.exit(f"[FAIL] unfilled slots {sorted(set(left))}")
    return template


def title_from_main() -> str:
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    m = re.search(r"\\title(?:\[[^\]]*\])?\{(.*?)\}\s*\n", main, re.S)
    if not m:
        sys.exit("[FAIL] no \\title{...} in main.tex")
    return re.sub(r"\s+", " ", m.group(1)).strip()


def bib_entries(keys: list[str]) -> list[dict]:
    """Minimal .bib reader for the journal-fit citations: author, year, title, journal, volume, pages."""
    bib = BIB.read_text(encoding="utf-8")
    found = []
    for key in keys:
        m = re.search(r"@\w+\{" + re.escape(key) + r",(.*?)\n\}", bib, re.S)
        if not m:
            sys.exit(f"[FAIL] citekey {key} not in {BIB.name} (journal-fit citations must be real)")
        f = {k.lower(): re.sub(r"\s+", " ", v.strip("{} ")) for k, v in
             re.findall(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?\s*\n", m.group(1) + "\n", re.S)}
        f["key"] = key
        found.append(f)
    return found


def surname(a: str) -> str:
    a = a.strip()
    return a.split(",")[0].strip() if "," in a else a.split()[-1]


def harvard_inline(e: dict) -> str:
    names = [x for x in e["author"].split(" and ")]
    if len(names) == 1:
        return f"{surname(names[0])}, {e['year']}"
    if len(names) == 2:
        return f"{surname(names[0])} and {surname(names[1])}, {e['year']}"
    return f"{surname(names[0])} et al., {e['year']}"


def harvard_ref(e: dict) -> str:
    names = []
    for a in e["author"].split(" and "):
        a = a.strip()
        if "," in a:
            sur, given = [x.strip() for x in a.split(",", 1)]
        else:
            parts = a.split()
            sur, given = parts[-1], " ".join(parts[:-1])
        initials = "".join(g[0] + "." for g in given.replace("-", " ").split())
        names.append(f"{sur}, {initials}")
    vol = e.get("volume", "")
    pages = e.get("pages", "")
    tail = ", ".join(x for x in (vol, pages) if x)
    # bib fields are LaTeX already (accents, protected braces), so they are not escaped
    return f"{', '.join(names)}, {e['year']}. {e['title']}. {e.get('journal', e.get('booktitle', ''))} {tail}.".replace(" ,", ",").replace(" .", ".")


def join_and(items: list[str]) -> str:
    return ", ".join(items[:-1]) + (" and " if len(items) > 1 else "") + items[-1] if items else ""


# ---------------------------------------------------------------- load
cfg = yaml.safe_load((PAPER / "submission.yaml").read_text(encoding="utf-8"))
title = title_from_main()
if cfg.get("title") and re.sub(r"\s+", " ", cfg["title"]).strip() != title:
    sys.exit(f"[FAIL] submission.yaml title differs from main.tex:\n  yaml: {cfg['title']}\n  main: {title}")
aff = cfg["affiliations"]
authors = cfg["authors"]
corr = [a for a in authors if a.get("corresponding")]
if not corr:
    sys.exit("[FAIL] no corresponding author flagged")
signatory = next((a for a in authors if a["name"] == cfg.get("signatory")), corr[0])
elsevier = cfg.get("venue_family", "elsevier") == "elsevier"
anonymous = bool(cfg.get("anonymous"))
date = cfg.get("date") or dt.date.today().isoformat()
journal = esc(cfg["journal"])
article_type = esc(cfg.get("article_type", "research article"))


def aff_of(a: dict) -> dict:
    return aff[a["affiliations"][0]]


def place(f: dict) -> str:
    """'<institution>, <address>, <city> <postcode>, <country>' with empty parts dropped."""
    head = ", ".join(esc(x) for x in (f["institution"], f.get("address", "")) if x)
    city = " ".join(esc(str(x)) for x in (f.get("city", ""), f.get("postcode", "")) if x)
    return ", ".join(x for x in (head, city, esc(f.get("country", ""))) if x)


# ---------------------------------------------------------------- title page
blocks = []
for a in authors:
    f = aff_of(a)
    orcid_line = f"ORCID: {esc(a['orcid'])}\\par\n" if a.get("orcid") else "\\par\n"
    blocks.append(
        f"{{\\bfseries {esc(a['name'])}{'$^{*}$' if a.get('corresponding') else ''}\\par}}\n"
        f"{esc(f['department'])},\\\\\n{place(f)}.\\\\\n"
        + (f"Tel: {esc(a['tel'])}\\\\\n" if a.get("tel") else "")
        + f"E-mail: \\href{{mailto:{esc(a['email'])}}}{{{esc(a['email'])}}}\\\\\n"
        + orcid_line + "\\vspace{1.0em}\n")
credit = ""
if any(a.get("credit") for a in authors):
    credit = ("\\section*{CRediT authorship contribution statement}\n"
              + " ".join(f"\\textbf{{{esc(a['name'])}:}} {esc(a['credit'])}." for a in authors if a.get("credit"))
              + "\n")
funding = cfg.get("funding") or []
ack = ("\\section*{Acknowledgments}\nThis work was supported by " + esc(join_and(funding)) + ".\n") if funding else ""
d = cfg.get("declarations", {})
statements = ""
if d.get("competing_interests_title_page") or d.get("competing_interests"):
    statements += "\\section*{Declaration of competing interest}\n" + esc(d.get("competing_interests_title_page") or d["competing_interests"]) + "\n"
if d.get("data_availability_title_page") or d.get("data_availability"):
    statements += "\\section*{Data availability}\n" + esc(d.get("data_availability_title_page") or d["data_availability"]) + "\n"
bios = [a for a in authors if a.get("bio")] if elsevier else []
bio_page = ("\\clearpage\n" + "".join(f"{{\\bfseries {esc(a['name'])}}} {esc(a['bio'])}\\par\n\\vspace{{0.6em}}\n" for a in bios)) if bios else ""
title_tex = fill((TPL / "Title_Page.tpl.tex").read_text(encoding="utf-8"), dict(
    title=esc(title), journal=journal, article_type=article_type[:1].upper() + article_type[1:],
    review_note=" for double-anonymized review" if anonymous else "",
    author_blocks="".join(blocks), corr_plural="s" if len(corr) > 1 else "",
    credit=credit, acknowledgments=ack, statements=statements, bio_page=bio_page))

# ---------------------------------------------------------------- cover letter
cl = cfg["cover_letter"]
decl = []
if anonymous:
    decl.append("A separate title page carries the author details, as the double-anonymized review requires.")
if d.get("originality", True):
    decl.append("We confirm that this work is original, has not been published elsewhere and is not under consideration by another journal.")
if d.get("all_authors_approved", True):
    decl.append(f"All authors have approved the manuscript and agree with its submission to \\journalname{{{journal}}}.")
for k in ("competing_interests", "data_availability", "ai_use", "ethics"):
    if d.get(k):
        decl.append(esc(d[k]))
entries = bib_entries(cl.get("fit_citekeys", []))
if not entries:
    sys.exit("[FAIL] cover_letter.fit_citekeys is empty: name 3-5 recent papers from the target journal")
reviewers = ""
if cl.get("suggested_reviewers"):
    items = "".join(f"  \\item {esc(r['name'])}, {esc(r['affiliation'])}, {esc(r['email'])}. {esc(r.get('reason', ''))}\n" for r in cl["suggested_reviewers"])
    reviewers = "We suggest the following reviewers:\n\\begin{itemize}\n" + items + "\\end{itemize}\n\n"
if cl.get("excluded_reviewers"):
    items = "; ".join(f"{esc(r['name'])} ({esc(r.get('reason', ''))})" for r in cl["excluded_reviewers"])
    reviewers += f"We request that the following individuals not be invited to review: {items}.\n\n"
sf = aff_of(signatory)
sig = (f"{esc(signatory['name'])}\\\\[0.2em]\n"
       + (f"{esc(signatory['role_title'])}\\\\\n" if signatory.get("role_title") else "")
       + f"{esc(sf['department'])}, {place(sf)}.\\\\\n"
       + (f"Tel: {esc(signatory['tel'])}\\\\\n" if signatory.get("tel") else "")
       + f"E-mail: \\href{{mailto:{esc(signatory['email'])}}}{{{esc(signatory['email'])}}}\n")
refs = "\\vspace{1.4em}\n{\\bfseries References}\\par\n\\small\n" + "".join(harvard_ref(e) + "\\par\n" for e in entries)
letter_tex = fill((TPL / "Cover_Letter.tpl.tex").read_text(encoding="utf-8"), dict(
    date=esc(date), editor=esc(cfg.get("editor", "Editor")), title=esc(title),
    article_type=article_type, journal=journal,
    problem=esc(cl["problem"]), idea=esc(cl["idea"]),
    components=" ".join(esc(c) for c in cl["components"]), result=esc(cl["result"]),
    fit_topic=esc(cl["fit_topic"]), fit_citations="; ".join(harvard_inline(e) for e in entries),
    fit_gap=esc(cl["fit_gap"]), fit_close=esc(cl["fit_close"]), significance=esc(cl["significance"]),
    declarations=" ".join(decl), reviewers=reviewers, signatory_block=sig, references=refs))

# ---------------------------------------------------------------- main.tex author block (CAS syntax)
fm = PAPER / "frontmatter-authors.tex"                 # main.tex: \input{frontmatter-authors}
if anonymous:
    fm.write_text("% anonymous submission: author block intentionally empty (see submission/Title_Page.pdf)\n"
                  "% generated from submission.yaml by make_submission.py -- do not edit\n", encoding="utf-8")
else:
    tag_of = {k: str(i + 1) for i, k in enumerate(aff)}
    lines = []
    for a in authors:
        tags = ",".join(tag_of[t] for t in a["affiliations"])
        lines.append(f"\\author[{tags}]{{{esc(a['name'])}}}" + (f"[orcid={esc(a['orcid'])}]" if a.get("orcid") else ""))
        lines.append(f"\\ead{{{esc(a['email'])}}}")
        if a.get("credit"):
            lines.append(f"\\credit{{{esc(a['credit'])}}}")
        if a.get("corresponding"):
            lines.append("\\cormark[1]")
        lines.append("")
    for k, f in aff.items():
        parts = [f"organization={{{esc(f['department'])}, {esc(f['institution'])}}}"]
        if f.get("address"):
            parts.append(f"addressline={{{esc(f['address'])}}}")
        parts += [f"city={{{esc(f['city'])}}}", f"postcode={{{esc(str(f['postcode']))}}}", f"country={{{esc(f['country'])}}}"]
        lines.append(f"\\affiliation[{tag_of[k]}]{{" + ", ".join(parts) + "}")
    lines.append("\\cortext[1]{Corresponding author" + ("s" if len(corr) > 1 else "") + "}")
    fm.write_text("% generated from submission.yaml by make_submission.py -- do not edit\n" + "\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- write, compile, clean
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "Title_Page.tex").write_text(title_tex, encoding="utf-8")
(OUT / "Cover_Letter.tex").write_text(letter_tex, encoding="utf-8")
print(f"[OK] rendered {OUT / 'Title_Page.tex'} and {OUT / 'Cover_Letter.tex'}; authors -> {fm}")
if not args.no_pdf:
    for stem in ("Title_Page", "Cover_Letter"):
        for _ in range(2):
            r = subprocess.run([COMPILER, "-interaction=nonstopmode", f"{stem}.tex"], cwd=OUT, capture_output=True, text=True)
        log = (OUT / f"{stem}.log").read_text(errors="replace")
        if r.returncode != 0 or "\n!" in log:
            sys.exit(f"[FAIL] {stem}.tex did not compile; see {OUT / (stem + '.log')}")
        pages = re.search(r"Output written on .*?\((\d+) page", log)
        print(f"[OK] {stem}.pdf ({pages.group(1) if pages else '?'} pages)")
    for junk in OUT.glob("*"):
        if junk.suffix not in {".tex", ".pdf"}:
            junk.unlink()
    print(f"[OK] submission/ holds exactly: {sorted(p.name for p in OUT.iterdir())}")
