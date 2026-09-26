from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "industrial_motor"
RESULTS = ROOT / "result" / "industrial_motor"
VARIANTS = {"HCMAGRL": "original", "Flat": "flat_mappo", "MLP": "mlp_encoder"}
METHODS = [*VARIANTS, "SPT", "SetupGreedy", "TwoStage", "NSGA-II"]
TIME_SCALE = 1000


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def profile(name):
    return dict(name=name, seeds=[42] if name == "smoke" else list(range(42, 47)),
                weights=[0.5] if name == "smoke" else [0, 0.25, 0.5, 0.75, 1],
                iterations=2 if name == "smoke" else 500,
                workers=1 if name == "smoke" else 4,
                nsga_evaluations=32 if name == "smoke" else 10000,
                nsga_population=8 if name == "smoke" else 100,
                variants=VARIANTS, methods=METHODS, hidden_dim=128,
                checkpoint_selection="final", schema=1)


def instance_dir(name):
    return DATA / "instances" / ("MOTOR_" + name.upper())


def source_hashes():
    paths = sorted((ROOT / "industrial_case").glob("*.py"))
    paths += [ROOT / f for f in ["agent.py", "config.py", "env.py", "rl_state.py", "graph_encoder.py",
                                "graph_state.py", "class_MO_DFRMS.py", "MO_DFRMS_instance_read.py",
                                "upper_actor.py", "lower_actor.py", "upper_critic.py", "lower_critic.py",
                                "utils.py", "Triangular_fuzzy.py", "ablation/ablation_trainer.py"]]
    paths += sorted((ROOT / "ablation" / "variants").glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def run_dir(name, output=None):
    return Path(output).resolve() / name if output else RESULTS / name


def initialize_run(name, output=None):
    out = run_dir(name, output)
    if not (DATA / "derived.json").exists():
        raise ValueError("Run prepare first.")
    data_hashes = {p.relative_to(DATA).as_posix(): digest(p)
                   for p in sorted(instance_dir(name).glob("*.csv"))}
    data_hashes["derived.json"] = digest(DATA / "derived.json")
    data_hashes["extracted/published.json"] = digest(DATA / "extracted" / "published.json")
    spec = {"profile": profile(name), "data_sha256": data_hashes, "code_sha256": source_hashes()}
    signature = object_hash(spec)
    target = out / "manifest.json"
    if target.exists():
        previous = read_json(target)
        if previous["signature"] != signature:
            raise ValueError("Existing results use different code/data/configuration. Choose a new --output directory.")
        return out, previous
    import importlib.metadata
    packages = {}
    for package in ["torch", "numpy", "scipy", "pandas", "matplotlib", "docplex", "pymoo"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    import torch
    manifest = {**spec, "signature": signature, "python": platform.python_version(),
                "platform": platform.platform(), "processor": platform.processor(),
                "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "packages": packages}
    write_json(target, manifest)
    return out, manifest


def run_key(method, weight, seed):
    return f"{method}_w{weight:g}_s{seed}"
