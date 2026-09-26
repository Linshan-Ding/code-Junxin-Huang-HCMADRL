from __future__ import annotations

import itertools
import time

import numpy as np

from .common import DATA, TIME_SCALE, initialize_run, read_json, run_key, write_csv, write_json
from .simulation import graded, run_policy


def batch_order():
    modules = read_json(DATA / "derived.json")["modules"]
    def charge(order):
        return sum(modules[4*j+order[0]]["mount_cost_cny"] + sum(
            modules[4*j+old]["remove_cost_cny"] + modules[4*j+new]["mount_cost_cny"]
            for old, new in zip(order, order[1:])) for j in range(5))
    return min(itertools.permutations(range(4)), key=lambda order: (charge(order), order))


def rule(method, weight=0.5, references=None, keys=None):
    order = batch_order() if method == "TwoStage" else None
    ranks = {r: i for i, r in enumerate(order)} if order else None

    def score(env, m, a, lower=False):
        r, j = a % 4, m
        processing = env.time_rja_dict[(r, j)][a] / TIME_SCALE
        old = env.machine_dict[m].module
        setup_time = 0 if old == a else (graded(env.time_add_dict[a]) +
                     (graded(env.time_rem_dict[old]) if old is not None else 0)) / TIME_SCALE
        setup_cost = 0 if old == a else env.cost_add_dict[a] + (env.cost_rem_dict[old] if old is not None else 0)
        if method == "SPT":
            value = processing
        elif method == "SetupGreedy":
            value = weight * (processing + setup_time) / references["makespan"] + (1-weight) * setup_cost / references["cost"]
        elif method == "TwoStage":
            value = ranks[r]
        elif method == "NSGA-II":
            n = env.kind_task_dict[(r, j)].buffer_list[0].batch
            offset = sum(env.count_sr_dict[0][:r]) * 5 + n * 5 + j
            total = sum(env.count_sr_dict[0]) * 5
            value = keys[offset + (total if lower else 0)]
        else:
            raise ValueError(method)
        return value, r, j, m

    def choose(env):
        upper_actions = env.upper_agent_actions()
        if not upper_actions:
            return None
        upper = min(upper_actions, key=lambda ma: score(env, *ma))
        lower_actions = env.lower_agent_actions(*upper)
        if lower_actions:
            lower = min(lower_actions, key=lambda mrj: score(env, mrj[0], env.machine_dict[mrj[0]].module, True))
            return upper + lower
        return upper
    return choose


def references(name, out):
    path = out / "references.json"
    if path.exists():
        return read_json(path)
    result = run_policy(name, rule("SPT"))
    if not result["feasible"]:
        raise ValueError("Reference SPT schedule is infeasible: " + str(result["errors"]))
    refs = dict(makespan=result["makespan"], cost=result["cost"], method="SPT", source="fixed before learning/search")
    write_json(path, refs)
    return refs


def persist(out, method, weight, seed, result, signature, name):
    result.update(method=method, weight=weight, seed=seed, signature=signature, profile=name)
    write_json(out / "evaluations" / (run_key(method, weight, seed) + ".json"), result)


def run_baselines(name, output=None, resume=False):
    out, manifest = initialize_run(name, output)
    spec = manifest["profile"]
    refs = references(name, out)
    for method in ["SPT", "SetupGreedy", "TwoStage"]:
        weights = spec["weights"] if method == "SetupGreedy" else [0.5]
        for weight in weights:
            path = out / "evaluations" / (run_key(method, weight, -1) + ".json")
            if path.exists():
                if resume:
                    continue
                raise ValueError(f"Result exists: {path}; use --resume.")
            result = run_policy(name, rule(method, weight, refs))
            persist(out, method, weight, -1, result, manifest["signature"], name)
            print(method, weight, result["feasible"], result["makespan"], result["cost"], flush=True)
    for seed in spec["seeds"]:
        run_nsga(name, out, spec, refs, manifest["signature"], seed, resume)


def run_nsga(name, out, spec, refs, signature, seed, resume):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize

    archive_path = out / "nsga" / f"seed_{seed}.json"
    if archive_path.exists():
        if not resume:
            raise ValueError(f"Archive exists: {archive_path}; use --resume.")
        archive = read_json(archive_path)
    else:
        total = 80 if name == "full" else 20
        history = []

        class Problem(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=2*total, n_obj=2, n_ieq_constr=1, xl=0, xu=1)

            def _evaluate(self, x, result, *args, **kwargs):
                metrics = run_policy(name, rule("NSGA-II", keys=x))
                result["F"] = [metrics["makespan"] / refs["makespan"], metrics["cost"] / refs["cost"]]
                result["G"] = [0 if metrics["feasible"] else 1]
                history.append(dict(evaluation=len(history)+1, makespan=metrics["makespan"],
                                    cost=metrics["cost"], feasible=metrics["feasible"]))

        started = time.perf_counter()
        result = minimize(Problem(), NSGA2(pop_size=spec["nsga_population"]),
                          ("n_eval", spec["nsga_evaluations"]), seed=seed, verbose=False)
        duration = time.perf_counter() - started
        keys = np.atleast_2d(result.X).tolist() if result.X is not None else []
        write_csv(out / "nsga" / f"seed_{seed}_history.csv", history)
        archive = dict(signature=signature, seed=seed, evaluations=len(history), runtime_seconds=duration,
                       keys=keys, points=np.atleast_2d(result.F).tolist() if result.F is not None else [])
        write_json(archive_path, archive)
    if archive["signature"] != signature:
        raise ValueError("NSGA archive signature mismatch.")
    for weight in spec["weights"]:
        target = out / "evaluations" / (run_key("NSGA-II", weight, seed) + ".json")
        if target.exists():
            if resume and read_json(target)["signature"] == signature:
                continue
            raise ValueError(f"Evaluation already exists or is incompatible: {target}")
        if not archive["keys"]:
            result = dict(feasible=False, errors=["no feasible NSGA-II solution"], events=[], makespan=None,
                          cost=None, original_matrix_cost=None, reconfigurations=None, runtime_seconds=0)
        else:
            best = min(range(len(archive["keys"])), key=lambda i: weight*archive["points"][i][0] + (1-weight)*archive["points"][i][1])
            result = run_policy(name, rule("NSGA-II", keys=archive["keys"][best]))
        result["search_seconds"] = archive["runtime_seconds"]
        result["search_evaluations"] = archive["evaluations"]
        persist(out, "NSGA-II", weight, seed, result, signature, name)
    print(f"NSGA-II seed {seed}: {archive['evaluations']} evaluations", flush=True)
