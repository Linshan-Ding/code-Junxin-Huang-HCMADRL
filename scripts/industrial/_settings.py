"""Execution settings only; experimental budgets stay in industrial_case.common.profile.

Edit these values before a run. Relative paths are resolved against the code
repository, never against the terminal's current directory. Use a NEW result
root after changing experimental code/data; do not edit an old manifest.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "result" / "industrial_motor"
DEVICE = "cpu"  # PPO updates and evaluation; rollout workers remain on CPU.
PAPER_ROOT = Path(os.environ.get("HCMAGRL_PAPER", ROOT.parent / "Junxin_Huang_HCMAGRL_RMS_FRT"))
LATEX_COMPILER = None  # Optional absolute path to tectonic.exe or latexmk[.exe].
