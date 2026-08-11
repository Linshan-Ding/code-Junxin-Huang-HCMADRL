# ============================================================
# 文件名：ablation/ablation_trainer.py
# 消融实验轮训引擎（多 worker 并行版）
#
# 与原始 train_visdom.py 保持一致：
#   - 多 worker 并行采样 → 合并 buffer → 主进程 PPO 更新
#   - Visdom 画多 worker 均值曲线
#   - pt 文件格式与原始完全一致
# ============================================================

import os
import sys
import time
import random
import argparse
from typing import Dict, Tuple, Optional, List
from statistics import mean

import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from config import config as global_config
from agent import HiFAMRSyncMAGRL, RolloutBuffer, Transition
from env import MO_DFRMS_Environment
from rl_state import graph_tensor_from_env
from utils import ensure_dir, set_seed

import Triangular_fuzzy as tf


# ============================================================
# 工厂函数
# ============================================================
def create_env(file_name: str, path: str) -> MO_DFRMS_Environment:
    """所有变体使用相同的环境。"""
    return MO_DFRMS_Environment(use_instance=False, file_name=file_name, path=path)


# 四种论文算法的标识集合
FLAT_VARIANTS = {"flat_mappo", "ddqn", "edqn", "sac", "td3"}


def create_agent(
    variant: str,
    op_input_dim: int,
    machine_input_dim: int,
    module_input_dim: int,
    hidden_dim: int = 128,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """根据 variant 创建 Agent。"""
    if variant == "flat_mappo":
        from ablation.variants.variant_flat_mappo import FlatMAPPOAgent
        return FlatMAPPOAgent(
            op_input_dim=op_input_dim, machine_input_dim=machine_input_dim,
            module_input_dim=module_input_dim, hidden_dim=hidden_dim, device=device,
        )
    else:
        agent = HiFAMRSyncMAGRL(
            op_input_dim=op_input_dim, machine_input_dim=machine_input_dim,
            module_input_dim=module_input_dim, hidden_dim=hidden_dim, device=device,
        )
        # 替换 encoder 后必须重建 optimizer，否则 encoder 权重永远不更新
        encoder_replaced = False
        if variant == "mlp_encoder":
            from ablation.variants.variant_mlp_encoder import MLPEncoder
            agent.encoder = MLPEncoder(
                op_input_dim=op_input_dim, machine_input_dim=machine_input_dim,
                module_input_dim=module_input_dim, hidden_dim=hidden_dim,
                dropout=global_config.DROPOUT,
            )
            encoder_replaced = True
        elif variant == "homo_gnn":
            from ablation.variants.variant_homo_gnn import HomoGNNEncoder
            agent.encoder = HomoGNNEncoder(
                op_input_dim=op_input_dim, machine_input_dim=machine_input_dim,
                module_input_dim=module_input_dim, hidden_dim=hidden_dim,
                dropout=global_config.DROPOUT,
            )
            encoder_replaced = True
        elif variant == "no_attn":
            from ablation.variants.variant_no_attn import NoAttnEncoder
            agent.encoder = NoAttnEncoder(
                op_input_dim=op_input_dim, machine_input_dim=machine_input_dim,
                module_input_dim=module_input_dim, hidden_dim=hidden_dim,
                dropout=global_config.DROPOUT,
            )
            encoder_replaced = True

        if encoder_replaced:
            agent.to(device)
            agent.upper_optimizer = torch.optim.Adam(
                list(agent.encoder.parameters())
                + list(agent.upper_actor.parameters())
                + list(agent.upper_critic.parameters()),
                lr=global_config.UPPER_LR, weight_decay=global_config.WEIGHT_DECAY,
            )
            agent.lower_optimizer = torch.optim.Adam(
                list(agent.encoder.parameters())
                + list(agent.lower_actor.parameters())
                + list(agent.lower_critic.parameters()),
                lr=global_config.LOWER_LR, weight_decay=global_config.WEIGHT_DECAY,
            )
        return agent


def infer_dims(env: MO_DFRMS_Environment, device: torch.device) -> Tuple[int, int, int]:
    env.reset()
    graph = graph_tensor_from_env(env, device)
    return graph.op_x.shape[-1], graph.machine_x.shape[-1], graph.module_x.shape[-1]


# ============================================================
# Episode 执行（消融变体）
# ============================================================
def run_episode_hierarchical(env, agent, deterministic=False, train_mode=True):
    buffer = RolloutBuffer()
    env.reset()
    step = 0
    while not env.done and step < global_config.MAX_EPISODE_STEPS:
        upper_action, upper_transition = agent.select_upper_action(env=env, deterministic=deterministic)
        if upper_action is None:
            break
        lower_action, lower_transition = agent.select_lower_action(
            env=env, upper_action=upper_action, deterministic=deterministic)
        full_action = tuple(upper_action + lower_action) if lower_action is not None else tuple(upper_action)
        _, upper_reward, lower_reward, done = env.step(full_action)
        if upper_transition is not None:
            upper_transition.reward = float(upper_reward)
            upper_transition.done = bool(done)
            if train_mode:
                buffer.upper.append(upper_transition)
        if lower_transition is not None:
            lower_transition.reward = float(lower_reward)
            lower_transition.done = bool(done)
            if train_mode:
                buffer.lower.append(lower_transition)
        step += 1
    if train_mode:
        if len(buffer.upper) > 0:
            buffer.upper[-1].done = True
        if len(buffer.lower) > 0:
            buffer.lower[-1].done = True
    total_cost = sum(env.machine_dict[m].reconfig_cost_sum for m in env.machine_tuple)
    metrics = {
        "makespan": float(env.step_time), "cost": float(total_cost), "steps": int(step),
        "upper_reward_sum": float(env.upper_reward_sum),
        "lower_reward_sum": float(env.lower_reward_sum),
        "upper_lower_reward_sum": float(env.upper_reward_sum),  # 双层各自同值，取一个
        "done": bool(env.done),
    }
    return buffer, metrics


def run_episode_flat(env, agent, deterministic=False, train_mode=True):
    buffer = RolloutBuffer()
    env.reset()
    step = 0
    while not env.done and step < global_config.MAX_EPISODE_STEPS:
        action, transition = agent.select_action(env=env, deterministic=deterministic)
        if action is None:
            break
        if len(action) == 5:
            full_action = (action[0], action[1], action[2], action[3], action[4])
        else:
            full_action = tuple(action)
        _, upper_reward, lower_reward, done = env.step(full_action)
        if transition is not None:
            # upper_reward == lower_reward（env 中两者相同），单层智能体只需取一个
            transition.reward = float(upper_reward)
            transition.done = bool(done)
            if train_mode:
                buffer.upper.append(transition)
        step += 1
    if train_mode and len(buffer.upper) > 0:
        buffer.upper[-1].done = True
    total_cost = sum(env.machine_dict[m].reconfig_cost_sum for m in env.machine_tuple)
    metrics = {
        "makespan": float(env.step_time), "cost": float(total_cost), "steps": int(step),
        "upper_reward_sum": float(env.upper_reward_sum),
        "lower_reward_sum": float(env.lower_reward_sum),
        "upper_lower_reward_sum": float(env.upper_reward_sum),  # 单层：取一个
        "done": bool(env.done),
    }
    return buffer, metrics


# ============================================================
# Episode 执行（离线策略算法：DDQN/EDQN/SAC/TD3）
# ============================================================
def run_episode_offpolicy(env, agent, deterministic=False, train_mode=True):
    """
    离线策略算法 episode：store_transition 模式。
    每步调用 agent.cache_state → select_action → env.step → agent.store_transition
    """
    env.reset()
    step = 0
    total_reward = 0.0
    while not env.done and step < global_config.MAX_EPISODE_STEPS:
        # 缓存当前状态（供后续 store_transition 使用）
        if train_mode and hasattr(agent, 'cache_state'):
            agent.cache_state(env)

        action, transition = agent.select_action(env=env, deterministic=deterministic)
        if action is None:
            break

        if len(action) == 5:
            full_action = (action[0], action[1], action[2], action[3], action[4])
        else:
            full_action = tuple(action)

        _, upper_reward, lower_reward, done = env.step(full_action)

        # upper_reward == lower_reward（env 中两者相同），单智能体只需取一个
        step_reward = float(upper_reward)
        total_reward += step_reward

        if transition is not None and train_mode and hasattr(agent, 'store_transition'):
            agent.store_transition(env, transition, step_reward, bool(done))

        step += 1

    total_cost = sum(env.machine_dict[m].reconfig_cost_sum for m in env.machine_tuple)
    metrics = {
        "makespan": float(env.step_time), "cost": float(total_cost), "steps": int(step),
        "upper_reward_sum": float(env.upper_reward_sum),
        "lower_reward_sum": float(env.lower_reward_sum),
        "upper_lower_reward_sum": float(env.upper_reward_sum),  # 单层：取一个
        "done": bool(env.done),
    }
    # 返回空 buffer（离线策略不通过 buffer 更新）
    return RolloutBuffer(), metrics


# ============================================================
# 多 worker 并行采样
# ============================================================
def state_dict_to_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _seed_worker(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)


def _rollout_worker(worker_id, state_dict, dims, variant, file_name, path,
                    hidden_dim, deterministic, seed, rollout_device):
    torch.set_num_threads(1)
    _seed_worker(seed)
    device = torch.device(rollout_device)
    env = create_env(file_name=file_name, path=path)
    op_dim, machine_dim, module_dim = dims
    is_flat = (variant in FLAT_VARIANTS)

    agent = create_agent(variant=variant, op_input_dim=op_dim,
                         machine_input_dim=machine_dim, module_input_dim=module_dim,
                         hidden_dim=hidden_dim, device=device)
    agent.load_state_dict(state_dict)
    agent.eval()

    with torch.no_grad():
        if is_flat:
            buffer, metrics = run_episode_flat(env, agent, deterministic=deterministic, train_mode=True)
        else:
            buffer, metrics = run_episode_hierarchical(env, agent, deterministic=deterministic, train_mode=True)

    metrics["worker_id"] = int(worker_id)
    metrics["upper_transition_count"] = len(buffer.upper)
    metrics["lower_transition_count"] = len(buffer.lower)
    return buffer, metrics


def collect_parallel_rollouts(agent, variant, dims, file_name, path, num_workers,
                              iteration, deterministic, base_seed, mp_start_method,
                              rollout_device, executor=None):
    is_flat = (variant in FLAT_VARIANTS)

    # 离线策略算法不支持多 worker 并行（replay buffer 不共享），强制单 worker
    actual_workers =  num_workers

    if actual_workers <= 1:
        if base_seed >= 0:
            _seed_worker(base_seed + iteration * 10007)
        env = create_env(file_name, path)
        if is_flat:
            buffer, metrics = run_episode_flat(env, agent, deterministic=deterministic, train_mode=True)
        else:
            buffer, metrics = run_episode_hierarchical(env, agent, deterministic=deterministic, train_mode=True)
        metrics["worker_id"] = 0
        metrics["upper_transition_count"] = len(buffer.upper)
        metrics["lower_transition_count"] = len(buffer.lower)
        return _aggregate([(buffer, metrics)])

    state_dict = state_dict_to_cpu(agent)
    if base_seed < 0:
        base_seed = int(time.time_ns() % (2 ** 31 - 1))

    def _submit_jobs(exec):
        jobs = []
        for worker_id in range(actual_workers):
            worker_seed = int(base_seed + iteration * 10007 + worker_id * 97)
            jobs.append(exec.submit(
                _rollout_worker, worker_id, state_dict, dims, variant, file_name, path,
                global_config.HIDDEN_DIM, deterministic, worker_seed, rollout_device,
            ))
        return [job.result() for job in jobs]

    if executor is not None:
        worker_outputs = _submit_jobs(executor)
    else:
        ctx = get_context(mp_start_method)
        with ProcessPoolExecutor(max_workers=actual_workers, mp_context=ctx) as exec:
            worker_outputs = _submit_jobs(exec)

    return _aggregate(worker_outputs)


def _aggregate(worker_outputs):
    merged_buffer = RolloutBuffer()
    worker_metrics = []
    for buffer, metrics in worker_outputs:
        merged_buffer.extend(buffer)
        worker_metrics.append(metrics)

    numeric_keys = ["makespan", "cost", "steps", "upper_reward_sum", "lower_reward_sum",
                    "upper_lower_reward_sum", "upper_transition_count", "lower_transition_count"]
    avg_metrics = {}
    for key in numeric_keys:
        values = [float(m[key]) for m in worker_metrics if key in m]
        avg_metrics[key] = mean(values) if values else 0.0
    done_values = [bool(m.get("done", False)) for m in worker_metrics]
    avg_metrics["done_rate"] = sum(done_values) / max(1, len(done_values))
    avg_metrics["done"] = all(done_values) if done_values else False
    avg_metrics["worker_count"] = len(worker_metrics)
    return merged_buffer, avg_metrics, worker_metrics


# ============================================================
# 保存 checkpoint（与原始 train_visdom.py 格式一致）
# ============================================================
def save_checkpoint(path, best_state, all_makespan, all_cost, all_episode_runtime,
                    all_episode_steps, all_upper_reward_sum, all_lower_reward_sum,
                    all_upper_lower_reward_sum, all_done_rate, all_worker_metrics):
    ensure_dir(os.path.dirname(path))
    torch.save({
        "model_state_dict": best_state,
        "all_makespan": all_makespan,
        "all_cost": all_cost,
        "all_episode_runtime": all_episode_runtime,
        "all_episode_steps": all_episode_steps,
        "all_upper_reward_sum": all_upper_reward_sum,
        "all_lower_reward_sum": all_lower_reward_sum,
        "all_upper_lower_reward_sum": all_upper_lower_reward_sum,
        "all_done_rate": all_done_rate,
        "all_worker_metrics": all_worker_metrics,
    }, path)


# ============================================================
# Visdom 辅助
# ============================================================
def _append_visdom_line(viz, win, x_value, y_value, title, xlabel, ylabel):
    if win is None:
        return viz.line(Y=[y_value], X=[x_value],
                        opts=dict(title=title, xlabel=xlabel, ylabel=ylabel))
    viz.line(Y=[y_value], X=[x_value], win=win, update="append")
    return win


def format_weight(w: float) -> str:
    w = float(w)
    if abs(w - round(w)) < 1e-9:
        return str(int(round(w)))
    return f"{w:.2f}".rstrip("0").rstrip(".")


# ============================================================
# 单轮训练（多 worker 并行版）
# ============================================================
def train_one_round(
    variant: str,
    alpha_t: float,
    alpha_c: float,
    file_name: str,
    episodes: int,
    reward_normalize: bool,
    makespan_ref: float,
    cost_ref: float,
    num_workers: int,
    update_batch_size: int,
    mp_start_method: str,
    rollout_device: str,
    data_path: str,
    save_dir: str,
    description: str,
    visdom_env: str,
    seed: int = -1,
) -> Dict:
    global_config.alpha_t = alpha_t
    global_config.alpha_c = alpha_c
    global_config.REWARD_NORMALIZE = reward_normalize
    global_config.MAKESPAN_REF = makespan_ref
    global_config.COST_REF = cost_ref
    global_config.UPDATE_BATCH_SIZE = update_batch_size

    # ★ 同步到环境变量：多 worker 进程通过 os.environ 读取 config 默认值
    os.environ["ALPHA_T"] = str(alpha_t)
    os.environ["ALPHA_C"] = str(alpha_c)
    os.environ["REWARD_NORMALIZE"] = "1" if reward_normalize else "0"
    os.environ["MAKESPAN_REF"] = str(makespan_ref)
    os.environ["COST_REF"] = str(cost_ref)

    env = create_env(file_name, data_path)
    op_dim, machine_dim, module_dim = infer_dims(env, global_config.DEVICE)
    dims = (op_dim, machine_dim, module_dim)

    # ── 创建 Agent ──
    agent = create_agent(
        variant=variant, op_input_dim=op_dim, machine_input_dim=machine_dim,
        module_input_dim=module_dim, hidden_dim=global_config.HIDDEN_DIM,
        device=global_config.DEVICE,
    )

    save_path = os.path.join(
        save_dir,
        f"{variant}_{format_weight(alpha_t)}makespan_{format_weight(alpha_c)}cost_{file_name}.pt",
    )
    ensure_dir(os.path.dirname(save_path))

    # Visdom
    viz = None
    try:
        from visdom import Visdom
        viz = Visdom(env=visdom_env)
    except Exception:
        pass

    best_reward_sum = -float("inf")
    best_state = None
    best_makespan = float("inf")
    best_cost = float("inf")

    all_makespan = []
    all_cost = []
    all_episode_runtime = []
    all_episode_steps = []
    all_upper_reward_sum = []
    all_lower_reward_sum = []
    all_upper_lower_reward_sum = []
    all_done_rate = []
    all_worker_metrics = []

    cost_win = None
    makespan_win = None


    # 创建持久进程池（仅 PPO 变体需要）
    persistent_executor = None
    if num_workers > 1 :
        ctx = get_context(mp_start_method)
        persistent_executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx)
        print(f"  [INFO] 持久进程池已创建：{num_workers} workers", flush=True)

    for ep in range(1, episodes + 1):
        iteration_start = time.perf_counter()

        # ── 采样 ──
        buffer, avg_metrics, worker_metrics = collect_parallel_rollouts(
            agent=agent, variant=variant, dims=dims, file_name=file_name, path=data_path,
            num_workers=num_workers, iteration=ep, deterministic=False,
            base_seed=seed, mp_start_method=mp_start_method, rollout_device=rollout_device,
            executor=persistent_executor,
        )

        agent.update(buffer)

        iteration_runtime = time.perf_counter() - iteration_start

        # 记录（与原始格式一致）
        all_makespan.append(float(avg_metrics["makespan"]))
        all_cost.append(float(avg_metrics["cost"]))
        all_episode_runtime.append(float(iteration_runtime))
        all_episode_steps.append(float(avg_metrics["steps"]))
        all_upper_reward_sum.append(float(avg_metrics["upper_reward_sum"]))
        all_lower_reward_sum.append(float(avg_metrics["lower_reward_sum"]))
        all_upper_lower_reward_sum.append(float(avg_metrics["upper_lower_reward_sum"]))
        all_done_rate.append(float(avg_metrics["done_rate"]))
        all_worker_metrics.append(worker_metrics)

        # 最佳模型
        current_reward_sum = avg_metrics["upper_lower_reward_sum"]
        if current_reward_sum > best_reward_sum:
            best_reward_sum = current_reward_sum
            best_makespan = avg_metrics["makespan"]
            best_cost = avg_metrics["cost"]
            best_state = state_dict_to_cpu(agent)

        # Visdom
        worker_label = avg_metrics.get("worker_count", num_workers)
        if viz is not None:
            try:
                cost_win = _append_visdom_line(
                    viz, cost_win, ep, avg_metrics["cost"],
                    f"{file_name}_{format_weight(alpha_c)} Average Cost per Iteration ({worker_label} workers)",
                    "Episode / Iteration", "Average Cost",
                )
                makespan_win = _append_visdom_line(
                    viz, makespan_win, ep, avg_metrics["makespan"],
                    f"{file_name}_{format_weight(alpha_t)} Average Makespan per Iteration ({worker_label} workers)",
                    "Episode / Iteration", "Average Makespan",
                )
            except Exception:
                pass

        # 日志
        if ep % 1 == 0 or ep == 1:
            print(
                f"  [{description}] ep={ep}/{episodes} "
                f"avg_makespan={avg_metrics['makespan']:.1f} avg_cost={avg_metrics['cost']:.1f} "
                f"reward={avg_metrics['upper_lower_reward_sum']:.1f} "
                f"done_rate={avg_metrics['done_rate']:.2f} "
                f"workers={worker_label} runtime={iteration_runtime:.2f}s"
            )

    # 关闭持久进程池
    if persistent_executor is not None:
        persistent_executor.shutdown(wait=True)
        print("  [INFO] 持久进程池已关闭", flush=True)

    if best_state is None:
        best_state = state_dict_to_cpu(agent)

    # 保存（与原始 train_visdom.py 格式完全一致）
    save_checkpoint(
        path=save_path, best_state=best_state,
        all_makespan=all_makespan, all_cost=all_cost,
        all_episode_runtime=all_episode_runtime, all_episode_steps=all_episode_steps,
        all_upper_reward_sum=all_upper_reward_sum, all_lower_reward_sum=all_lower_reward_sum,
        all_upper_lower_reward_sum=all_upper_lower_reward_sum,
        all_done_rate=all_done_rate, all_worker_metrics=all_worker_metrics,
    )

    result = {
        "variant": variant, "alpha_t": alpha_t, "alpha_c": alpha_c,
        "file_name": file_name, "best_makespan": best_makespan, "best_cost": best_cost,
        "episodes": episodes, "checkpoint": save_path, "description": description,
    }
    print(f"  [OK] best_avg_makespan={best_makespan:.1f} best_avg_cost={best_cost:.1f} -> {save_path}")
    return result


# ============================================================
# 参考值获取
# ============================================================
def get_reference_values(variant, file_names, data_path, save_dir, episodes,
                         num_workers, update_batch_size, mp_start_method,
                         rollout_device, visdom_env):
    ref_makespan = 1.0
    ref_cost = 1.0
    try:
        print("\n--- 获取 MAKESPAN_REF（纯 Makespan）---")
        r = train_one_round(variant=variant, alpha_t=1.0, alpha_c=0.0,
                            file_name=file_names[0], episodes=min(300, episodes),
                            reward_normalize=False, makespan_ref=1.0, cost_ref=1.0,
                            num_workers=num_workers, update_batch_size=update_batch_size,
                            mp_start_method=mp_start_method, rollout_device=rollout_device,
                            data_path=data_path, save_dir=save_dir,
                            description="REF-Makespan", visdom_env=visdom_env)
        ref_makespan = max(r["best_makespan"], 1.0)

        print("\n--- 获取 COST_REF（纯 Cost）---")
        r = train_one_round(variant=variant, alpha_t=0.0, alpha_c=1.0,
                            file_name=file_names[0], episodes=min(300, episodes),
                            reward_normalize=False, makespan_ref=1.0, cost_ref=1.0,
                            num_workers=num_workers, update_batch_size=update_batch_size,
                            mp_start_method=mp_start_method, rollout_device=rollout_device,
                            data_path=data_path, save_dir=save_dir,
                            description="REF-Cost", visdom_env=visdom_env)
        ref_cost = max(r["best_cost"], 1.0)
    except Exception as e:
        print(f"  [WARN] ref values failed: {e}, using default 1.0")
    print(f"  MAKESPAN_REF={ref_makespan:.2f}, COST_REF={ref_cost:.2f}")
    return ref_makespan, ref_cost


def is_port_open(host="127.0.0.1", port=8097, timeout=1.0):
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
