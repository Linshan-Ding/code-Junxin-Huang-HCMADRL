"""Shared environment, event recording, and an independent schedule audit."""
from __future__ import annotations

import math
import time

from env import MO_DFRMS_Environment
from .common import DATA, TIME_SCALE, instance_dir, read_json
from .data import PUBLISHED


def graded(triple):
    return (triple[0] + 4 * triple[1] + triple[2]) / 6


class RecordedEnvironment(MO_DFRMS_Environment):
    def __init__(self, name):
        directory = instance_dir(name)
        self.events = []
        self.switches = 0
        self.profile_name = name
        super().__init__(use_instance=False, path=str(directory.parent), file_name=directory.name)

    def reset(self):
        super().reset()
        self.events = []
        self.switches = 0

    def event(self, kind, machine, module, start, duration, cost=0, product=None, stage=None, job=None, group=None, ready_job=None):
        self.events.append(dict(index=len(self.events), kind=kind, machine=int(machine), module=int(module),
                                start=start / TIME_SCALE, end=(start + duration) / TIME_SCALE,
                                cost=float(cost), product=product, stage=stage, job=job, group=group, ready_job=ready_job))

    def step(self, action):
        upper = tuple(action[:2])
        if upper not in self.upper_agent_actions():
            raise ValueError(f"Illegal upper action: {action}")
        if len(action) == 5 and tuple(action[2:]) not in self.lower_agent_actions(*upper):
            raise ValueError(f"Illegal lower action: {action}")
        machine, new = upper
        old = self.machine_dict[machine].module
        start = self.step_time
        if old != new:
            group = self.switches
            self.switches += 1
            r, j = new % 4, machine
            ready_job = [r, j, int(self.kind_task_dict[(r, j)].buffer_list[0].batch)]
            if old is not None:
                duration = graded(self.time_rem_dict[old])
                self.event("remove", machine, old, start, duration, self.cost_rem_dict[old], group=group, ready_job=ready_job)
                start += duration
            self.event("install", machine, new, start, graded(self.time_add_dict[new]), self.cost_add_dict[new], group=group, ready_job=ready_job)
        if len(action) == 5:
            m, r, j = action[2:]
            module = self.machine_dict[m].module
            job = int(self.kind_task_dict[(r, j)].buffer_list[0].batch)
            self.event("process", m, module, self.step_time, self.time_rja_dict[(r, j)][module],
                       product=int(r), stage=int(j), job=job)
        return super().step(action)


def audit(events, name, expected_makespan=None, expected_cost=None, time_multiplier=1):
    """Rebuild feasibility and objectives without invoking the simulator."""
    info = read_json(DATA / "derived.json")
    modules = info["modules"]
    demand = PUBLISHED["demand"] if name == "full" else [1] * 4
    installed, ends, processed = [None] * 5, [0.] * 5, {}
    errors, cost, original_cost, setups = [], 0., 0., 0
    previous_product = [None] * 5
    for event in sorted(events, key=lambda e: (e["start"], e["index"])):
        m, a, kind = event["machine"], event["module"], event["kind"]
        module = modules[a]
        if m != module["station"] or event["start"] + 1e-7 < ends[m]:
            errors.append("station incompatibility or overlap")
        if not math.isfinite(event["end"]) or event["start"] < 0 or event["end"] < event["start"]:
            errors.append("invalid event time")
        if kind != "process":
            ready = event.get("ready_job")
            if not ready:
                errors.append("missing reconfiguration readiness witness")
            else:
                r, j, n = ready
                if not (0 <= r < 4 and j == m and 0 <= n < demand[r]) or tuple(ready) in processed:
                    errors.append("invalid reconfiguration readiness witness")
                if j and processed.get((r, j-1, n), math.inf) > event["start"] + 1e-7:
                    errors.append("reconfiguration precedes enabling operation")
                if kind == "install" and r != module["product"]:
                    errors.append("readiness witness has wrong product")
        if kind == "remove":
            if installed[m] != a:
                errors.append("removed module was not installed")
            duration = graded(module["remove_time"]) * time_multiplier
            charge = module["remove_cost_cny"]
            installed[m] = None
        elif kind == "install":
            if installed[m] is not None:
                errors.append("installation without removal")
            duration = graded(module["mount_time"]) * time_multiplier
            charge = module["mount_cost_cny"]
            old_product = previous_product[m]
            original_cost += charge if old_product is None else PUBLISHED["switch_cost"][old_product][module["product"]] * 1000 / 5
            previous_product[m] = module["product"]
            installed[m] = a
            setups += 1
        elif kind == "process":
            r, j, n = event["product"], event["stage"], event["job"]
            if not (0 <= r < 4 and 0 <= n < demand[r] and j == m and a == 4*j+r and installed[m] == a):
                errors.append("operation, job or module mismatch")
            key = (r, j, n)
            if key in processed:
                errors.append("duplicate operation")
            if j and (processed.get((r, j-1, n), math.inf) > event["start"] + 1e-7):
                errors.append("precedence violation")
            processed[key] = event["end"]
            duration = PUBLISHED["processing"][r][j]
            charge = 0
        else:
            raise ValueError(f"Unknown event kind {kind}")
        if abs(event["end"] - event["start"] - duration) > 1e-6 or abs(event["cost"] - charge) > 1e-6:
            errors.append("duration or event charge mismatch")
        cost += charge
        ends[m] = event["end"]
    expected = {(r, j, n) for r in range(4) for j in range(5) for n in range(demand[r])}
    complete = set(processed) == expected
    if not complete:
        errors.append("unfinished production")
    makespan = max(ends)
    if expected_makespan is not None and abs(makespan - expected_makespan) > 1e-6:
        errors.append("simulator makespan mismatch")
    if expected_cost is not None and abs(cost - expected_cost) > 1e-6:
        errors.append("simulator cost mismatch")
    return dict(feasible=not errors, complete=complete, errors=sorted(set(errors)), makespan=makespan,
                cost=cost, original_matrix_cost=original_cost, reconfigurations=max(0, setups - 5),
                initial_installations=min(5, sum(1 for e in events if e["kind"] == "install" and
                                               not any(x["machine"] == e["machine"] and x["index"] < e["index"] for x in events))),
                operations=len(processed))


def replay(events, name, multiplier):
    """Earliest-start DAG replay; preserves machine order, not policy decisions."""
    end_machine, end_job = [0.] * 5, {}
    result = []
    # Original chronological order is topological for machine/precedence edges.
    for e in sorted(events, key=lambda x: (x["start"], x["index"])):
        e = dict(e)
        duration = e["end"] - e["start"]
        if e["kind"] != "process":
            duration *= multiplier
        start = end_machine[e["machine"]]
        if e["kind"] == "process" and e["stage"]:
            start = max(start, end_job[(e["product"], e["stage"]-1, e["job"])])
        elif e["kind"] != "process" and e["ready_job"][1]:
            r, j, n = e["ready_job"]
            start = max(start, end_job[(r, j-1, n)])
        e.update(start=start, end=start + duration, index=len(result))
        end_machine[e["machine"]] = e["end"]
        if e["kind"] == "process":
            end_job[(e["product"], e["stage"], e["job"])] = e["end"]
        result.append(e)
    report = audit(result, name, time_multiplier=multiplier)
    if not report["feasible"]:
        raise ValueError(f"Invalid replay: {report['errors']}")
    return report


def run_policy(name, policy):
    env = RecordedEnvironment(name)
    env.reset()
    start = time.perf_counter()
    limit = 20 * sum(env.count_sr_dict[0]) * 5
    steps = 0
    while not env.done and steps < limit:
        action = policy(env)
        if action is None:
            break
        env.step(action)
        steps += 1
    report = audit(env.events, name, env.step_time / TIME_SCALE,
                   sum(m.reconfig_cost_sum for m in env.machine_dict.values()))
    return dict(**report, runtime_seconds=time.perf_counter()-start, steps=steps,
                termination="complete" if env.done else "step_limit_or_no_action", events=env.events)
