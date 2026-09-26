"""Completeness-gated aggregation; no repeated deterministic pseudo-samples."""
from __future__ import annotations

import math

import numpy as np

from .common import METHODS, VARIANTS, digest, initialize_run, read_json, run_key, write_csv, write_json
from .simulation import audit, replay


def nondominated(points):
    points = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
    return np.asarray([p for p in points if not any(np.all(q <= p) and np.any(q < p) for q in points)]).reshape(-1, 2)


def hypervolume(points, reference):
    points = nondominated(points)
    height, value = float(reference[1]), 0.
    for x, y in sorted(points.tolist()):
        if x < reference[0] and y < height:
            value += (reference[0] - x) * (height - y)
            height = y
    return value


def expected_records(spec):
    return [(method, weight, seed) for method in METHODS
            for weight in (spec["weights"] if method not in ["SPT", "TwoStage"] else [0.5])
            for seed in (spec["seeds"] if method in [*VARIANTS, "NSGA-II"] else [-1])]


def checked_records(name, output=None):
    out, manifest = initialize_run(name, output)
    records, missing = [], []
    hashes = {}
    for method, weight, seed in expected_records(manifest["profile"]):
        key = run_key(method, weight, seed)
        path = out / "evaluations" / (key + ".json")
        if not path.exists():
            missing.append(key)
            continue
        row = read_json(path)
        if (row["method"], row["weight"], row["seed"], row["profile"], row["signature"]) != (method, weight, seed, name, manifest["signature"]):
            raise ValueError(f"Evaluation identity mismatch: {key}")
        if row["events"]:
            independent = audit(row["events"], name, row["makespan"], row["cost"])
            if independent["feasible"] != row["feasible"]:
                raise ValueError(f"Feasibility mismatch: {key}")
            for field in ["original_matrix_cost", "reconfigurations"]:
                if abs(independent[field] - row[field]) > 1e-6:
                    raise ValueError(f"{field} mismatch: {key}")
        elif row["feasible"]:
            raise ValueError(f"Feasible result has no trace: {key}")
        if method in VARIANTS and row.get("training_rollouts") != manifest["profile"]["iterations"] * manifest["profile"]["workers"]:
            raise ValueError(f"Incomplete training budget: {key}")
        if method == "NSGA-II" and row.get("search_evaluations") != manifest["profile"]["nsga_evaluations"]:
            raise ValueError(f"Incomplete search budget: {key}")
        records.append(row)
        hashes[key] = digest(path)
    if missing:
        raise ValueError(f"Missing {len(missing)} required evaluations: {', '.join(missing[:5])}. No formal export is allowed.")
    return out, manifest, records, hashes


def mean_std(values):
    return (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.) if values else (None, None)


def aggregate(name, output=None):
    out, manifest, records, hashes = checked_records(name, output)
    refs = read_json(out / "references.json")
    scale = np.array([refs["makespan"], refs["cost"]])
    feasible_points = [[r["makespan"], r["cost"]] for r in records if r["feasible"]]
    reference = np.max(np.array(feasible_points) / scale, axis=0) * 1.1
    summary, portfolios, sensitivity, detail = [], [], [], []
    for r in records:
        detail.append({k: r.get(k) for k in ["method", "weight", "seed", "feasible", "makespan", "cost",
                                            "original_matrix_cost", "reconfigurations", "runtime_seconds",
                                            "training_seconds", "search_seconds"]})
        if r["feasible"] and r["weight"] == 0.5:
            for factor in [0.5, 1, 2]:
                replayed = replay(r["events"], name, factor)
                sensitivity.append(dict(method=r["method"], seed=r["seed"], multiplier=factor,
                                        makespan=replayed["makespan"], cost=replayed["cost"],
                                        original_matrix_cost=replayed["original_matrix_cost"]))
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        seeds = sorted({r["seed"] for r in rows})
        hvs = []
        for seed in seeds:
            subset = [r for r in rows if r["seed"] == seed and r["feasible"]]
            points = [[r["makespan"], r["cost"]] for r in subset]
            hv = hypervolume(np.array(points).reshape(-1, 2) / scale, reference)
            hvs.append(hv)
            portfolios.append(dict(method=method, seed=seed, hv=hv, feasible_weights=len(subset),
                                   distinct_nondominated=len(nondominated(points))))
        balanced = [r for r in rows if r["weight"] == 0.5]
        successful = [r for r in balanced if r["feasible"]]
        row = dict(method=method, attempted=len(balanced), feasible=len(successful),
                   feasibility_rate=len(successful)/len(balanced), independent_replicates=len(seeds))
        for key in ["makespan", "cost", "original_matrix_cost", "reconfigurations", "runtime_seconds"]:
            row[key+"_mean"], row[key+"_std"] = mean_std([r[key] for r in successful])
        row["hv_mean"], row["hv_std"] = mean_std(hvs)
        # Entire five-weight portfolio training/search cost per seed, not just inference.
        times = []
        for seed in seeds:
            group = [r for r in rows if r["seed"] == seed]
            times.append(sum(r.get("training_seconds", 0) for r in group) if method in VARIANTS else
                         max(r.get("search_seconds", 0) for r in group) if method == "NSGA-II" else
                         sum(r["runtime_seconds"] for r in group))
        row["offline_seconds_mean"], row["offline_seconds_std"] = mean_std(times)
        summary.append(row)
    directory = out / "aggregate"
    write_csv(directory / "detail.csv", detail)
    write_csv(directory / "summary.csv", summary)
    write_csv(directory / "portfolios.csv", portfolios)
    write_csv(directory / "sensitivity.csv", sensitivity)
    metadata = dict(profile=name, signature=manifest["signature"], evaluation_sha256=hashes,
                    expected_evaluations=len(records), hv_reference_normalized=reference.tolist(),
                    hv_scale=scale.tolist(), hv_rule="1.1 times coordinate maxima of pooled feasible representatives; common to all methods",
                    statistics="sample SD across independent training/search seeds; deterministic rules are single results; infeasibility reported",
                    failed_evaluations=sum(not r["feasible"] for r in records))
    write_json(directory / "manifest.json", metadata)
    print(f"Aggregated {len(records)} evaluations ({metadata['failed_evaluations']} infeasible), profile={name}")
    return out, manifest, records, summary, sensitivity
