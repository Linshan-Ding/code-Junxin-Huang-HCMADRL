"""Published observations and explicit, deterministic model reconstruction."""
from __future__ import annotations

import datetime
import urllib.request
from decimal import Decimal

import numpy as np

from .common import DATA, TIME_SCALE, digest, instance_dir, read_json, write_csv, write_json

DOI = "10.1371/journal.pone.0348884"
ARTICLE = "https://journals.plos.org/plosone/article?id=" + DOI
PDF = "https://journals.plos.org/plosone/article/file?id=" + DOI + "&type=printable"
REPOSITORY = "https://github.com/zixiangliwust/Instances_RALSP"
# Transcribed from the published article, section 5 and Tables 1--3 (p. 15).
# These are literature-reported industrial parameters, not newly collected measurements.
PUBLISHED = {
    "doi": DOI, "article": ARTICLE, "repository": REPOSITORY,
    "locator": "Section 5; Tables 1, 2, 3; printed page 15",
    "classification": "published", "time_unit": "unspecified source time-unit",
    "cost_unit": "thousand CNY", "products": ["62ZYT001", "100ZYT001", "62ZYT-SUV", "78ZYT001"],
    "product_labels": ["A", "B", "C", "D"], "demand": [5, 4, 4, 3],
    "processing": [[5, 8, 4, 6, 8], [7, 7, 9, 5, 6], [8, 5, 5, 9, 4], [6, 9, 7, 4, 7]],
    "switch_cost": [[0, 2, 4, 4], [3, 0, 5, 6], [5, 3, 0, 4], [3, 2, 7, 0]],
    "part_requirements": [[1, 1, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]],
    "excluded_source_constraints": {"startup_interval": 3, "part_frequency": [[2, 3], [3, 5], [3, 4], [1, 2]],
                                    "cyclic_return": True},
}


def download_sources():
    """Fetch the unchanged open-access article; never replace an existing source."""
    directory = DATA / "source"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "zhao_2026_motor_case.pdf"
    if not target.exists():
        request = urllib.request.Request(PDF, headers={"User-Agent": "HCMAGRL-research/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            content = response.read()
        if not content.startswith(b"%PDF"):
            raise ValueError("Source response is not a PDF.")
        target.write_bytes(content)
    record = {"url": PDF, "sha256": digest(target), "license": "CC BY (article copyright statement)",
              "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "authors": "Zhao, Huang, Liu, Li, Chen and Lu", "year": 2026, "doi": DOI,
              "note": "Unmodified published article; source tables are transcribed in extracted/published.json."}
    manifest = directory / "manifest.json"
    if manifest.exists():
        if read_json(manifest)["sha256"] != record["sha256"]:
            raise ValueError("Downloaded source checksum differs from the recorded source.")
    else:
        write_json(manifest, record)


def fit_cost(cost):
    matrix, values = [], []
    for old in range(4):
        for new in range(4):
            if old == new:
                continue
            row = np.zeros(8)
            row[old] = row[4 + new] = 1
            matrix.append(row)
            values.append(cost[old][new])
    # The equality fixes the one-dimensional additive gauge. The solution for
    # these published data is strictly positive, hence also the constrained LS solution.
    augmented = np.vstack([matrix, np.r_[np.ones(4), -np.ones(4)]])
    solution = np.linalg.lstsq(augmented, np.r_[values, 0], rcond=None)[0]
    if np.min(solution) < -1e-9 or abs(solution[:4].sum() - solution[4:].sum()) > 1e-9:
        raise ValueError("Published-data cost fit violates the declared constraints.")
    solution = np.round(solution, 12)
    residual = np.array(matrix) @ solution - values
    return solution[:4], solution[4:], {"mae_thousand_cny": float(np.abs(residual).mean()),
                                       "rmse_thousand_cny": float(np.sqrt(np.mean(residual ** 2))),
                                       "max_abs_thousand_cny": float(np.abs(residual).max())}


def ticks(value):
    scaled = Decimal(str(value)) * TIME_SCALE
    if abs(scaled - scaled.to_integral_value()) > Decimal("0.000001"):
        raise ValueError(f"Time cannot be represented exactly: {value}")
    return int(scaled.to_integral_value())


def prepare(fetch=False):
    if fetch:
        download_sources()
    published_path = DATA / "extracted" / "published.json"
    if published_path.exists() and read_json(published_path) != PUBLISHED:
        raise ValueError("Existing published values differ; refusing to overwrite them.")
    write_json(published_path, PUBLISHED)
    removal, addition, errors = fit_cost(PUBLISHED["switch_cost"])
    rows, modules = [], []
    means = np.asarray(PUBLISHED["processing"]).mean(axis=0)
    for station in range(5):
        for product in range(4):
            module = 4 * station + product
            add = round(float(0.6 * means[station]), 12)
            rem = round(float(0.4 * means[station]), 12)
            modules.append(dict(module=module, station=station, product=product,
                                mount_time=[round(f * add, 12) for f in [0.8, 1, 1.2]],
                                remove_time=[round(f * rem, 12) for f in [0.8, 1, 1.2]],
                                mount_cost_cny=round(float(addition[product] * 1000 / 5)),
                                remove_cost_cny=round(float(removal[product] * 1000 / 5))))
    for old in range(4):
        for new in range(4):
            fitted = 0.0 if old == new else float(removal[old] + addition[new])
            raw = PUBLISHED["switch_cost"][old][new]
            rows.append(dict(old="ABCD"[old], new="ABCD"[new], published=raw, fitted=fitted,
                             residual=fitted - raw, unit="thousand CNY"))
    write_csv(DATA / "extracted" / "cost_fit.csv", rows)
    derived = dict(time_scale=TIME_SCALE, cost_unit="CNY", modules=modules, cost_fit=errors,
                   cost_removal_line=removal.tolist(), cost_installation_line=addition.tolist(),
                   assumptions={"layout": "five independently reconfigurable stations; stage-specific logical modules",
                                "buffers": "unlimited", "release_time": 0, "initial_configuration": "empty",
                                "installation_share": 0.6, "removal_share": 0.4, "fuzzy_factors": [0.8, 1, 1.2],
                                "cost_station_share": 0.2, "sensitivity_time_multipliers": [0.5, 1, 2]},
                   provenance={"processing_and_demand": "published", "module_compatibility": "assumed",
                               "additive_cost": "derived least squares", "cost_station_share": "assumed",
                               "fuzzy_times": "assumed, not measured worker data"})
    write_json(DATA / "derived.json", derived)
    for name, demand in [("full", PUBLISHED["demand"]), ("smoke", [1, 1, 1, 1])]:
        directory = instance_dir(name)
        write_csv(directory / "based_data.csv", [dict(kind_count=4, machine_count=5, module_count=20, order_count=1)])
        write_csv(directory / "order_data.csv", [dict(order=0, time_arrive=0, kind_number=str(tuple(demand)))])
        write_csv(directory / "machine_data.csv", [dict(machine=j, modules=str(tuple(range(4*j, 4*j+4)))) for j in range(5)])
        write_csv(directory / "process_data.csv", [dict(kind=r, task=j, machines=str((j,)), modules=str((4*j+r,)),
                                                        times=str((ticks(PUBLISHED["processing"][r][j]),)))
                                                       for r in range(4) for j in range(5)])
        write_csv(directory / "module_data.csv", [dict(module=m["module"],
                                                       time_add=str(tuple(ticks(x) for x in m["mount_time"])),
                                                       time_rem=str(tuple(ticks(x) for x in m["remove_time"])),
                                                       cost_add=m["mount_cost_cny"], cost_rem=m["remove_cost_cny"])
                                                      for m in modules])
    print(f"Prepared {DATA}; source PDF present: {(DATA / 'source/zhao_2026_motor_case.pdf').exists()}")


def validate_data():
    from MO_DFRMS_instance_read import Data
    if read_json(DATA / "extracted/published.json") != PUBLISHED:
        raise ValueError("Published table mismatch.")
    info = read_json(DATA / "derived.json")
    for name in ["full", "smoke"]:
        directory = instance_dir(name)
        data = Data(str(directory.parent), directory.name)
        assert (data.machine_count, data.module_count, data.kind_count, data.order_count) == (5, 20, 4, 1)
        assert len(data.kind_task_tuple) == 20
        assert sum(data.count_sr_dict[0]) * 5 == (80 if name == "full" else 20)
        for r, j in data.kind_task_tuple:
            assert data.machine_module_rj_dict[(r, j)] == ((j, 4*j+r),)
            assert data.time_rja_dict[(r, j)][4*j+r] / TIME_SCALE == PUBLISHED["processing"][r][j]
        for m in info["modules"]:
            assert data.time_add_dict[m["module"]] == tuple(ticks(x) for x in m["mount_time"])
            assert data.time_rem_dict[m["module"]] == tuple(ticks(x) for x in m["remove_time"])
            assert data.cost_add_dict[m["module"]] == m["mount_cost_cny"]
            assert data.cost_rem_dict[m["module"]] == m["remove_cost_cny"]
    source = DATA / "source/manifest.json"
    if source.exists():
        assert read_json(source)["sha256"] == digest(DATA / "source/zhao_2026_motor_case.pdf")
    print("Published tables, 80/20 tasks, compatibility, units and source checksum validated.")
