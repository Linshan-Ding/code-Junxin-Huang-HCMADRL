"""PPO orchestration using the repository's agents, with resumable final states."""
from __future__ import annotations

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch

from ablation.ablation_trainer import create_agent, infer_dims
from agent import RolloutBuffer
from config import config
from utils import set_seed
from .baselines import persist, references
from .common import TIME_SCALE, VARIANTS, initialize_run, read_json, run_key, write_json
from .simulation import RecordedEnvironment, audit


def configure(weight, refs, device="cpu"):
    torch.set_num_threads(1)
    config.DEVICE = torch.device(device)
    config.alpha_t, config.alpha_c = weight, 1-weight
    config.REWARD_NORMALIZE = True
    config.MAKESPAN_REF = refs["makespan"] * TIME_SCALE
    config.COST_REF = refs["cost"]
    config.UPDATE_BATCH_SIZE = 64


def make_agent(name, method, device):
    dims = infer_dims(RecordedEnvironment(name), torch.device(device))
    return create_agent(VARIANTS[method], *dims, hidden_dim=128, device=torch.device(device))


def episode(name, agent, method, deterministic=False, collect=False):
    env = RecordedEnvironment(name)
    env.reset()
    buffer = RolloutBuffer()
    limit = 20 * sum(env.count_sr_dict[0]) * 5
    started = time.perf_counter()
    steps = 0
    agent.eval()
    while not env.done and steps < limit:
        if method == "Flat":
            action, upper_transition = agent.select_action(env, deterministic=deterministic)
            lower_transition = None
        else:
            upper, upper_transition = agent.select_upper_action(env, deterministic=deterministic)
            if upper is None:
                break
            lower, lower_transition = agent.select_lower_action(env, upper, deterministic=deterministic)
            action = tuple(upper + lower) if lower is not None else tuple(upper)
        if action is None:
            break
        _, upper_reward, lower_reward, done = env.step(action)
        if collect:
            for transition, reward, destination in [(upper_transition, upper_reward, buffer.upper),
                                                     (lower_transition, lower_reward, buffer.lower)]:
                if transition is not None:
                    transition.reward, transition.done = float(reward), done
                    destination.append(transition)
        steps += 1
    # Match the existing rollout truncation convention; incomplete episodes are
    # explicitly counted rather than exported as successful schedules.
    for transitions in [buffer.upper, buffer.lower]:
        if transitions:
            transitions[-1].done = True
    report = audit(env.events, name, env.step_time / TIME_SCALE,
                   sum(m.reconfig_cost_sum for m in env.machine_dict.values()))
    report.update(runtime_seconds=time.perf_counter()-started, steps=steps,
                  reward=float(env.upper_reward_sum), events=env.events,
                  termination="complete" if env.done else "step_limit_or_no_action")
    return buffer, report


def worker(name, method, weight, refs, state, seed):
    configure(weight, refs)
    set_seed(seed)
    agent = make_agent(name, method, "cpu")
    agent.load_state_dict(state)
    buffer, report = episode(name, agent, method, collect=True)
    report.pop("events")
    return buffer, report


def optimizers(agent):
    return {name: value for name, value in vars(agent).items() if isinstance(value, torch.optim.Optimizer)}


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def train(name, output=None, resume=False, stop_after=None, device="cpu", methods=None):
    out, manifest = initialize_run(name, output)
    spec, refs = manifest["profile"], references(name, out)
    selected = methods or list(VARIANTS)
    for method in selected:
        for weight in spec["weights"]:
            for seed in spec["seeds"]:
                train_one(name, method, weight, seed, out, manifest, refs, resume, stop_after, device)


def train_one(name, method, weight, seed, out, manifest, refs, resume=False, stop_after=None, device="cpu"):
    spec = manifest["profile"]
    path = out / "checkpoints" / (run_key(method, weight, seed) + ".pt")
    configure(weight, refs, device)
    set_seed(seed)
    agent = make_agent(name, method, device)
    history, start_iteration = [], 0
    if path.exists():
        if not resume:
            raise ValueError(f"Checkpoint exists: {path}; use --resume.")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["signature"] != manifest["signature"] or saved["device"] != device:
            raise ValueError("Checkpoint signature/device mismatch.")
        agent.load_state_dict(saved["model_state_dict"])
        for key, optimizer in optimizers(agent).items():
            optimizer.load_state_dict(saved["optimizers"][key])
        restore_rng(saved["rng"])
        history, start_iteration = saved["history"], saved["iteration"]
    final_iteration = min(spec["iterations"], stop_after) if stop_after is not None else spec["iterations"]
    if start_iteration >= final_iteration:
        return
    pool = ProcessPoolExecutor(spec["workers"], mp_context=get_context("spawn")) if spec["workers"] > 1 else None
    try:
        for iteration in range(start_iteration + 1, final_iteration + 1):
            started = time.perf_counter()
            state = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
            tasks = [(name, method, weight, refs, state, seed + iteration*10007 + w*97) for w in range(spec["workers"])]
            if pool:
                futures = [pool.submit(worker, *task) for task in tasks]
                outputs = [future.result() for future in futures]
            else:
                outputs = [worker(*task) for task in tasks]
                configure(weight, refs, device)
            merged = RolloutBuffer()
            metrics = []
            for buffer, report in outputs:
                merged.extend(buffer)
                metrics.append(report)
            agent.train()
            agent.update(merged)
            entry = dict(iteration=iteration, seconds=time.perf_counter()-started, workers=metrics,
                         feasible_rollouts=sum(r["feasible"] for r in metrics),
                         mean_makespan=float(np.mean([r["makespan"] for r in metrics])),
                         mean_cost=float(np.mean([r["cost"] for r in metrics])))
            history.append(entry)
            checkpoint = dict(signature=manifest["signature"], profile=name, method=method, weight=weight, seed=seed,
                              iteration=iteration, completed=iteration == spec["iterations"], device=device,
                              model_state_dict={k: v.detach().cpu().clone() for k, v in agent.state_dict().items()},
                              optimizers={k: v.state_dict() for k, v in optimizers(agent).items()},
                              rng=rng_state(), history=history, references=refs)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
            write_json(out / "training" / (path.stem + ".json"), history)
            print(f"{path.stem}: {iteration}/{spec['iterations']} | feasible {entry['feasible_rollouts']}/{spec['workers']} | {entry['seconds']:.2f}s", flush=True)
    finally:
        if pool:
            pool.shutdown()


def evaluate(name, output=None, resume=False, device="cpu"):
    out, manifest = initialize_run(name, output)
    spec, refs = manifest["profile"], references(name, out)
    for method in VARIANTS:
        for weight in spec["weights"]:
            for seed in spec["seeds"]:
                key = run_key(method, weight, seed)
                target = out / "evaluations" / (key + ".json")
                if target.exists():
                    if resume:
                        continue
                    raise ValueError(f"Evaluation exists: {target}; use --resume.")
                path = out / "checkpoints" / (key + ".pt")
                if not path.exists():
                    raise ValueError(f"Missing checkpoint: {path}")
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if not saved["completed"] or saved["signature"] != manifest["signature"]:
                    raise ValueError(f"Incomplete or incompatible checkpoint: {path}")
                configure(weight, refs, device)
                set_seed(seed)
                agent = make_agent(name, method, device)
                agent.load_state_dict(saved["model_state_dict"])
                _, result = episode(name, agent, method, deterministic=True)
                result["training_seconds"] = sum(row["seconds"] for row in saved["history"])
                result["training_rollouts"] = spec["iterations"] * spec["workers"]
                persist(out, method, weight, seed, result, manifest["signature"], name)
                print(f"Evaluate {key}: feasible={result['feasible']} {result['errors']}", flush=True)
