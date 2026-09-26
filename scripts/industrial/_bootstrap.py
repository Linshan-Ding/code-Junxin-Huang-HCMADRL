"""Small, sequential adapters over the existing industrial_case command line."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True

if __package__:
    from . import _settings
else:
    import _settings

ROOT = _settings.ROOT
STAGES = {
    "00": "Smoke: prepare, validate, train, baselines, evaluate, aggregate",
    "01": "Prepare and validate published-data instances",
    "02": "Train HCMAGRL (25 full runs)",
    "03": "Train Flat and MLP (50 full runs)",
    "04": "Run rules, TwoStage and NSGA-II",
    "05": "Evaluate final learning checkpoints",
    "06": "Audit and aggregate full results",
    "07": "Export audited full results to the paper",
    "08": "Compile the paper and check it",
}
DATA_STAGES = ("00", "01", "02", "03", "04", "05", "06")


def absolute(path):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    result_root: Path
    device: str
    paper_root: Path
    compiler: str | None = None

    @classmethod
    def current(cls):
        return cls(absolute(_settings.RESULT_ROOT), _settings.DEVICE,
                   absolute(_settings.PAPER_ROOT), _settings.LATEX_COMPILER)


def cli_commands(stage, settings):
    """Build argument vectors, never shell strings or a second experiment matrix."""
    output = ["--output", str(absolute(settings.result_root))]
    full = ["--profile", "full", *output]
    device = ["--device", settings.device]
    return {
        "00": [["prepare"], ["validate"],
               ["run", "--profile", "smoke", *output, "--resume", *device]],
        "01": [["prepare"], ["validate"]],
        "02": [["train", *full, "--resume", *device, "--methods", "HCMAGRL"]],
        "03": [["train", *full, "--resume", *device, "--methods", "Flat", "MLP"]],
        "04": [["baselines", *full, "--resume"]],
        "05": [["evaluate", *full, "--resume", *device]],
        "06": [["aggregate", *full]],
        "07": [["export-paper", *full, "--paper", str(absolute(settings.paper_root))]],
    }[stage]


class StepError(RuntimeError):
    def __init__(self, message, code=2):
        super().__init__(message)
        self.code = code if code > 0 else 1


def say(message, log):
    print(message, flush=True)
    log.write(message + "\n")
    log.flush()


def run_command(command, cwd, log):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    say("[COMMAND] " + subprocess.list2cmdline([str(x) for x in command]), log)
    with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, encoding="utf-8", errors="replace") as process:
        try:
            for line in process.stdout:
                say(line.rstrip("\r\n"), log)
            code = process.wait()
        except KeyboardInterrupt:
            # Ctrl+C also reaches console children. Give them time to exit before
            # terminating the CLI; atomic checkpoint replacement is handled there.
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
            raise
    if code:
        raise StepError(f"Command failed (exit {code}). Later stages were not started.", code)


def bundled_tectonics():
    """Discover existing installs only; never download a compiler."""
    app_path = os.environ.get("CODEX_TECTONIC_PATH")
    if app_path and Path(app_path).is_absolute() and Path(app_path).is_file():
        return [Path(app_path)]
    roots = [Path(os.environ.get("ProgramFiles", "C:/Program Files")),
             Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "Programs"]
    candidates = []
    for root in roots:
        for pattern in ["WindowsApps/OpenAI.Codex_*/app/resources/tectonic/tectonic.exe",
                        "Codex/resources/tectonic/tectonic.exe",
                        "Codex/app/resources/tectonic/tectonic.exe"]:
            try:
                candidates.extend(root.glob(pattern))
            except OSError:
                pass
    # A packaged application's parent directory can deny enumeration while the
    # current user's Appx registration exposes its exact readable install path.
    if os.name == "nt":
        powershell = shutil.which("powershell")
        if powershell:
            try:
                result = subprocess.run(
                    [powershell, "-NoProfile", "-NonInteractive", "-Command",
                     "Get-AppxPackage -Name OpenAI.Codex | Select-Object -ExpandProperty InstallLocation"],
                    capture_output=True, text=True, timeout=15, check=False)
                candidates.extend(Path(line.strip()) / "app/resources/tectonic/tectonic.exe"
                                  for line in result.stdout.splitlines() if line.strip())
            except (OSError, subprocess.TimeoutExpired):
                pass
    def version(path):
        match = re.search(r"Codex_([\d.]+)_", str(path))
        return tuple(map(int, match[1].split("."))) if match else ()
    return sorted(set(candidates), key=version, reverse=True)


def find_compiler(explicit=None):
    if explicit:
        path = absolute(explicit)
        if not path.is_file():
            raise StepError(f"Configured compiler does not exist: {path}")
        name = path.name.lower()
        if name not in {"tectonic", "tectonic.exe", "latexmk", "latexmk.exe"}:
            raise StepError("LATEX_COMPILER must name tectonic or latexmk.")
        return path, "tectonic" if name.startswith("tectonic") else "latexmk"
    located = shutil.which("tectonic")
    if located:
        return Path(located), "tectonic"
    for path in bundled_tectonics():
        if path.is_file():
            return path, "tectonic"
    located = shutil.which("latexmk")
    if located:
        return Path(located), "latexmk"
    raise StepError("No Tectonic or latexmk found. Set LATEX_COMPILER in scripts/industrial/_settings.py to an existing executable.")


def compile_commands(settings):
    paper = absolute(settings.paper_root)
    if not (paper / "main.tex").is_file():
        raise StepError(f"PAPER_ROOT must contain main.tex: {paper}")
    executable, kind = find_compiler(settings.compiler)
    args = (["-X", "compile", "--keep-logs", "--keep-intermediates", "--untrusted", "main.tex"]
            if kind == "tectonic" else ["-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
    return [(list(map(str, [executable, *args])), paper),
            ([sys.executable, "-B", "-u", str(ROOT / "paper_assets/scripts/check.py"),
              "--paper", str(paper)], ROOT)]


def required_outputs(stage, settings):
    # Import only when running a stage. This is the existing protocol's source
    # of truth; the wrappers do not redefine seeds, weights or training budgets.
    from industrial_case.common import METHODS, VARIANTS, profile, run_key
    name = "smoke" if stage == "00" else "full"
    spec = profile(name)
    out = absolute(settings.result_root) / name
    data = ROOT / "data/industrial_motor"
    files = []
    if stage in {"00", "01"}:
        files += [data / p for p in ["source/zhao_2026_motor_case.pdf", "source/manifest.json",
                                    "extracted/published.json", "extracted/cost_fit.csv", "derived.json"]]
        files += [data / "instances" / instance / filename for instance in ["MOTOR_FULL", "MOTOR_SMOKE"]
                  for filename in ["based_data.csv", "machine_data.csv", "module_data.csv", "order_data.csv", "process_data.csv"]]
    if stage in {"00", "02", "03"}:
        methods = list(VARIANTS) if stage == "00" else ["HCMAGRL"] if stage == "02" else ["Flat", "MLP"]
        files += [out / directory / (run_key(m, w, s) + extension) for m in methods
                  for w in spec["weights"] for s in spec["seeds"]
                  for directory, extension in [("checkpoints", ".pt"), ("training", ".json")]]
    if stage in {"00", "04", "05"}:
        methods = METHODS if stage == "00" else list(VARIANTS) if stage == "05" else [m for m in METHODS if m not in VARIANTS]
        files += [out / "evaluations" / (run_key(m, w, s) + ".json") for m in methods
                  for w in (spec["weights"] if m not in {"SPT", "TwoStage"} else [0.5])
                  for s in (spec["seeds"] if m in [*VARIANTS, "NSGA-II"] else [-1])]
    if stage in {"00", "04"}:
        files += [out / "nsga" / f"seed_{s}{suffix}" for s in spec["seeds"] for suffix in [".json", "_history.csv"]]
    if stage in {"00", "06", "07"}:
        files += [out / "aggregate" / filename for filename in ["detail.csv", "summary.csv", "portfolios.csv", "sensitivity.csv", "manifest.json"]]
    if stage in {"00", "02", "03", "04", "05", "06", "07"}:
        files += [out / "manifest.json", out / "references.json"]
    if stage == "07":
        paper = absolute(settings.paper_root)
        files += [paper / "tables" / f"industrial_{item}.tex" for item in
                  ["case", "cost_fit", "modules", "results", "runtime", "result_figures", "findings"]]
        files += [paper / "figures" / f"fig_industrial_{item}.pdf" for item in ["system", "performance", "gantt"]]
        files += [paper / "macros/industrial-results.tex"]
        files += [ROOT / "paper_assets/figures/data/industrial" / p.name for p in (out / "aggregate").glob("*")]
        files += [ROOT / "paper_assets/figures/_proofs/industrial" / f"fig_industrial_{item}.png"
                  for item in ["system", "performance", "gantt"]]
    if stage == "08":
        files += [absolute(settings.paper_root) / name for name in ["main.pdf", "main.log"]]
    return files


def verify_outputs(stage, settings, log):
    files = required_outputs(stage, settings)
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise StepError("Expected outputs missing:\n" + "\n".join(missing[:10]))
    say(f"[OUTPUT] {len(files)} expected files present (existence check; CLI performs the experiment audits).", log)
    for directory in sorted({p.parent for p in files}, key=str):
        say(f"[OUTPUT DIRECTORY] {directory}", log)
    if stage in {"00", "06", "07"}:
        name = "smoke" if stage == "00" else "full"
        path = absolute(settings.result_root) / name / "aggregate/manifest.json"
        info = json.loads(path.read_text(encoding="utf-8"))
        say(f"[AUDIT] {info['expected_evaluations']} evaluation records; {info['failed_evaluations']} infeasible. Manifest: {path}", log)


def execute_stage(stage, settings, log):
    started = time.perf_counter()
    say(f"[START {stage}] {STAGES[stage]}", log)
    if stage == "08":
        for command, cwd in compile_commands(settings):
            run_command(command, cwd, log)
    else:
        for args in cli_commands(stage, settings):
            run_command([sys.executable, "-B", "-u", "-m", "industrial_case", *args], ROOT, log)
    verify_outputs(stage, settings, log)
    say(f"[DONE {stage}] {time.perf_counter() - started:.2f} seconds", log)


@contextmanager
def repository_context():
    previous = Path.cwd()
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(previous)
        sys.path.remove(str(ROOT))


def entry(stage):
    if len(sys.argv) != 1:
        raise SystemExit("This entry takes no arguments. See README or edit scripts/industrial/_settings.py. No experiment was started.")
    main(stage)


def main(stage, settings=None):
    settings = settings or Settings.current()
    if stage not in {*STAGES, "all"}:
        raise ValueError(stage)
    if settings.device not in {"cpu", "cuda"}:
        raise SystemExit("DEVICE must be cpu or cuda in scripts/industrial/_settings.py")
    log_dir = absolute(settings.result_root) / "runner_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{stage}.log"
    with log_path.open("w", encoding="utf-8") as log, repository_context():
        started = time.perf_counter()
        say(f"[LOG] {log_path}", log)
        say(f"[SETTINGS] python={sys.executable}; device={settings.device}; result_root={absolute(settings.result_root)}", log)
        if stage in {"all", "02", "03"}:
            say("[BUDGET] Full protocol: 75 runs in total, 500 iterations/run, 4 CPU rollouts/iteration. Runs are sequential.", log)
        try:
            for item in DATA_STAGES if stage == "all" else [stage]:
                execute_stage(item, settings, log)
        except (StepError, OSError) as error:
            say(f"[FAILED] {error}", log)
            say("[RECOVERY] Keep existing results. Rerun to resume a matching run; for code/data/config mismatches, choose a new RESULT_ROOT in _settings.py. Do not edit manifests.", log)
            raise SystemExit(error.code if isinstance(error, StepError) else 2) from error
        except KeyboardInterrupt:
            say("[INTERRUPTED] Rerun the same entry to resume from the last complete atomic checkpoint.", log)
            raise SystemExit(130)
        say(f"[FINISHED] {time.perf_counter() - started:.2f} seconds; log={log_path}", log)
        if stage == "all":
            say("[NEXT] Data pipeline complete. Run run_07_export_paper.py, then run_08_compile_check.py separately.", log)
