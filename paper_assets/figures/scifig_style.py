"""Shared matplotlib style contract for every data figure in this paper.

Every plot script imports `use_scifig` and `save` from here so that the whole
figure set is one visual system: same colours, same line styles, same markers
for the same method across all figures.

Design rules enforced here
  * Okabe-Ito colour-blind-safe palette.
  * Triple-redundant encoding (colour + line style + marker) so the figures
    survive greyscale printing.
  * Computer Modern text and math, which is what cas-sc sets the body in.
  * 8 pt labels (never below 7 pt).
  * Figures are sized 1:1 against the final printed width, so `\includegraphics`
    in the body carries no scaling key.

cas-sc (single column, A4) has \textwidth = 468.33 pt = 164.6 mm.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4

# Final printed widths for this document class.
FULL_MM = 164.6   # \textwidth
WIDE_MM = 164.6
HALF_MM = 80.0    # two figures side by side
MED_MM = 120.0

OI = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#6E6E6E",
    "black": "#000000",
}

# One fixed visual identity per method, used in EVERY figure of the paper.
# Our method is always blue/solid/circle; ablations warm; DRL baselines cool.
STYLE = {
    "HCMAGRL":           dict(color=OI["blue"],      ls="-",             marker="o"),
    "w/o EdgeAttn":      dict(color=OI["vermilion"], ls="--",            marker="s"),
    "w/o Hierarchy":     dict(color=OI["green"],     ls=":",             marker="^"),
    "w/o Heterogeneity": dict(color=OI["orange"],    ls="-.",            marker="D"),
    "w/o Graph":         dict(color=OI["purple"],    ls=(0, (4, 1.5)),   marker="v"),
    "DDQN":              dict(color=OI["vermilion"], ls="--",            marker="s"),
    "EDQN":              dict(color=OI["green"],     ls=":",             marker="^"),
    "SAC":               dict(color=OI["orange"],    ls="-.",            marker="D"),
    "D-DRL":             dict(color=OI["purple"],    ls=(0, (4, 1.5)),   marker="v"),
    # Metaheuristic baselines on the public benchmark.
    "SADE":              dict(color=OI["green"],     ls=":",             marker="^"),
    "MSELS":             dict(color=OI["orange"],    ls="-.",            marker="D"),
    "IGA":               dict(color=OI["purple"],    ls=(0, (4, 1.5)),   marker="v"),
    "DABC":              dict(color=OI["vermilion"], ls="--",            marker="s"),
}

HATCH = {
    "HCMAGRL": "", "w/o EdgeAttn": "///", "w/o Hierarchy": "...",
    "w/o Heterogeneity": "\\\\\\", "w/o Graph": "xxx",
    "DDQN": "///", "EDQN": "...", "SAC": "\\\\\\", "D-DRL": "xxx",
}


def style(method: str) -> dict:
    return STYLE.get(method, dict(color=OI["gray"], ls="-", marker="."))


def use_scifig(width_mm: float = FULL_MM, height_mm: float | None = None) -> None:
    if height_mm is None:
        height_mm = width_mm * 0.42
    mpl.rcParams.update({
        # cas-sc loads plain article with no font package, so the body is set
        # in Computer Modern. matplotlib ships the same family, so the figures
        # can match the running text exactly rather than approximately.
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # cmr10 has no U+2212, so negative tick labels must go through mathtext
        # or they render as missing-glyph boxes.
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        # Closed rectangular frame on every axes, as journals in this field
        # expect; all four spines are drawn.
        "axes.linewidth": 0.5,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "lines.linewidth": 0.9,
        "lines.markersize": 2.6,
        "axes.grid": True,
        "grid.color": "#BFBFBF",
        "grid.linewidth": 0.3,
        "grid.alpha": 0.7,
        "figure.figsize": (width_mm * MM, height_mm * MM),
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, stem: str) -> None:
    """Write the vector PDF the paper includes, plus a PNG for eyeballing.

    The PDF is the only product the manuscript needs, so it goes to the paper
    repository; the proof PNG stays here, next to the script that made it.
    """
    out = paper_root() / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    proofs = Path(__file__).resolve().parent / "_proofs"
    proofs.mkdir(exist_ok=True)
    fig.savefig(proofs / f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  {stem}.pdf")


def paper_root() -> Path:
    """Where the manuscript lives.

    The paper repository carries only what a compile needs -- sources, class
    files, the figure PDFs -- so this toolchain lives in the code repository
    beside the raw results and writes its products across. Override with
    $HCMAGRL_PAPER; the default is the sibling checkout.
    """
    return Path(os.environ.get(
        "HCMAGRL_PAPER",
        Path(__file__).resolve().parents[3] / "Junxin_Huang_HCMAGRL_RMS_FRT"))


def data_dir() -> Path:
    return Path(__file__).resolve().parent / "data"
