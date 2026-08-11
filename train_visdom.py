# =========================
# 文件名：train_visdom.py
# 多 worker / actor 并行采样 + 主进程统一 mini-batch PPO 更新 + Visdom 均值曲线
# =========================
import argparse
import os
import time
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import torch
from visdom import Visdom

import Triangular_fuzzy as tf
from agent import HiFAMRSyncMAGRL, RolloutBuffer
from config import config
from env import MO_DFRMS_Environment
from rl_state import graph_tensor_from_env
from utils import ensure_dir, set_seed


def make_env(file_name: str, path: str) -> MO_DFRMS_Environment:
    # 训练数据已经生成好，因此这里保持 use_instance=False，只读取已有 csv 实例。
    return MO_DFRMS_Environment(
        use_instance=False,
        file_name=file_name,
        path=path,
    )


def infer_dims(env: MO_DFRMS_Environment) -> Tuple[int, int, int]:
    env.reset()
    graph = graph_tensor_from_env(env, config.DEVICE)
    return graph.op_x.shape[-1], graph.machine_x.shape[-1], graph.module_x.shape[-1]


def build_empty_trace(env: MO_DFRMS_Environment):
    schedule_dict = {m: [] for m in env.machine_tuple}
    module_change_dict = {m: [] for m in env.machine_tuple}
    return schedule_dict, module_change_dict


def get_batch_id_from_env(env: MO_DFRMS_Environment, r: int, j: int) -> int:
    """
    从环境当前工序缓冲区中读取即将加工的批量号。
    kind_task_dict[(r,j)].buffer_list 中存放 Job 对象。
    Job 对象中批量字段为 batch。
    """
    kind_task_object = env.kind_task_dict[(r, j)]

    if not hasattr(kind_task_object, "buffer_list"):
        return -1

    if len(kind_task_object.buffer_list) == 0:
        return -1

    job_object = kind_task_object.buffer_list[0]

    if hasattr(job_object, "batch"):
        return int(job_object.batch)

    if hasattr(job_object, "n"):
        return int(job_object.n)

    return -1


def record_action_trace(
    env: MO_DFRMS_Environment,
    action: Tuple,
    schedule_dict: Dict[int, List[Tuple]],
    module_change_dict: Dict[int, List[Tuple]],
):
    current_time = float(env.step_time)

    # =========================
    # 记录上层模块切换
    # =========================
    m_u = action[0]
    a_u = action[1]
    machine_object = env.machine_dict[m_u]
    old_module = machine_object.module

    if old_module != a_u:
        if old_module is None:
            reconfig_time = tf.Triangular_fuzzy_time(
                (0, 0, 0),
                env.time_add_dict[a_u],
            )
        else:
            reconfig_time = tf.Triangular_fuzzy_time(
                env.time_rem_dict[old_module],
                env.time_add_dict[a_u],
            )

        start_time = current_time
        end_time = current_time + float(reconfig_time)

        module_change_dict[m_u].append(
            (
                int(a_u),
                float(start_time),
                float(end_time),
                None if old_module is None else int(old_module),
            )
        )

    if len(action) == 5:
        m_l = action[2]
        r = action[3]
        j = action[4]

        lower_machine_object = env.machine_dict[m_l]
        current_module = lower_machine_object.module

        if current_module is not None:
            batch_id = get_batch_id_from_env(env, r, j)

            process_time = env.time_arj_dict[current_module][(r, j)]
            start_time = current_time
            end_time = current_time + float(process_time)

            schedule_dict[m_l].append(
                (
                    int(r),
                    int(j),
                    int(batch_id),
                    float(start_time),
                    float(end_time),
                    int(current_module),
                )
            )


def run_episode_with_trace(
    env: MO_DFRMS_Environment,
    agent: HiFAMRSyncMAGRL,
    deterministic: bool = False,
    train_mode: bool = True,
):
    buffer = RolloutBuffer()
    env.reset()

    schedule_dict, module_change_dict = build_empty_trace(env)

    step = 0

    while not env.done and step < config.MAX_EPISODE_STEPS:
        upper_action, upper_transition = agent.select_upper_action(
            env=env,
            deterministic=deterministic,
        )

        if upper_action is None:
            break

        lower_action, lower_transition = agent.select_lower_action(
            env=env,
            upper_action=upper_action,
            deterministic=deterministic,
        )

        if lower_action is not None:
            full_action = tuple(upper_action + lower_action)
        else:
            full_action = tuple(upper_action)

        # 必须在 env.step(full_action) 前记录，
        # 因为 env.step 内部会把对应 Job 从 buffer_list 中取走。
        record_action_trace(
            env=env,
            action=full_action,
            schedule_dict=schedule_dict,
            module_change_dict=module_change_dict,
        )

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

    # 一个 worker 对应一条 episode trajectory。即使因为达到 MAX_EPISODE_STEPS
    # 或无可行动作提前退出，也要把该 worker 的最后一条 transition 标成 done，
    # 防止多个 worker 的轨迹在合并后计算 GAE 时互相 bootstrap。
    if train_mode:
        if len(buffer.upper) > 0:
            buffer.upper[-1].done = True
        if len(buffer.lower) > 0:
            buffer.lower[-1].done = True

    total_cost = sum(env.machine_dict[m].reconfig_cost_sum for m in env.machine_tuple)

    upper_reward_sum = float(env.upper_reward_sum)
    lower_reward_sum = float(env.lower_reward_sum)
    reward_sum = upper_reward_sum + lower_reward_sum

    metrics = {
        "makespan": float(env.step_time),
        "cost": float(total_cost),
        "steps": int(step),
        "upper_reward_sum": upper_reward_sum,
        "lower_reward_sum": lower_reward_sum,
        "upper_lower_reward_sum": float(reward_sum),
        "done": bool(env.done),
    }

    return buffer, metrics, schedule_dict, module_change_dict


def state_dict_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _seed_worker(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32 - 1))
    torch.manual_seed(seed)


def _rollout_worker(
    worker_id: int,
    state_dict: Dict[str, torch.Tensor],
    dims: Tuple[int, int, int],
    file_name: str,
    path: str,
    hidden_dim: int,
    deterministic: bool,
    seed: int,
    rollout_device: str,
):
    """单个 worker/actor：只负责采样，不更新网络参数。"""
    t_start = time.perf_counter()
    torch.set_num_threads(1)
    _seed_worker(seed)

    t0 = time.perf_counter()
    device = torch.device(rollout_device)
    env = make_env(file_name=file_name, path=path)
    t_env = time.perf_counter() - t0

    t0 = time.perf_counter()
    op_dim, machine_dim, module_dim = dims
    agent = HiFAMRSyncMAGRL(
        op_input_dim=op_dim,
        machine_input_dim=machine_dim,
        module_input_dim=module_dim,
        hidden_dim=hidden_dim,
        device=device,
    )
    agent.load_state_dict(state_dict)
    agent.eval()
    t_model = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        buffer, metrics, schedule_dict, module_change_dict = run_episode_with_trace(
            env=env,
            agent=agent,
            deterministic=deterministic,
            train_mode=True,
        )
    t_rollout = time.perf_counter() - t0

    t_total = time.perf_counter() - t_start
    # print(f"[WORKER {worker_id}] start={t_start:.3f} | env={t_env:.2f}s | model={t_model:.2f}s | rollout={t_rollout:.2f}s | total={t_total:.2f}s", flush=True)

    metrics["worker_id"] = int(worker_id)
    metrics["upper_transition_count"] = len(buffer.upper)
    metrics["lower_transition_count"] = len(buffer.lower)
    return buffer, metrics, schedule_dict, module_change_dict


def aggregate_worker_outputs(worker_outputs):
    merged_buffer = RolloutBuffer()
    worker_metrics = []
    schedule_dicts = []
    module_change_dicts = []

    for buffer, metrics, schedule_dict, module_change_dict in worker_outputs:
        merged_buffer.extend(buffer)
        worker_metrics.append(metrics)
        schedule_dicts.append(schedule_dict)
        module_change_dicts.append(module_change_dict)

    numeric_keys = [
        "makespan",
        "cost",
        "steps",
        "upper_reward_sum",
        "lower_reward_sum",
        "upper_lower_reward_sum",
        "upper_transition_count",
        "lower_transition_count",
    ]
    avg_metrics = {}
    for key in numeric_keys:
        values = [float(m[key]) for m in worker_metrics if key in m]
        avg_metrics[key] = mean(values) if values else 0.0

    done_values = [bool(m.get("done", False)) for m in worker_metrics]
    avg_metrics["done_rate"] = sum(done_values) / max(1, len(done_values))
    avg_metrics["done"] = all(done_values) if done_values else False
    avg_metrics["worker_count"] = len(worker_metrics)

    return merged_buffer, avg_metrics, worker_metrics, schedule_dicts, module_change_dicts


def collect_parallel_rollouts(
    agent: HiFAMRSyncMAGRL,
    env: MO_DFRMS_Environment,
    dims: Tuple[int, int, int],
    file_name: str,
    path: str,
    num_workers: int,
    iteration: int,
    deterministic: bool,
    base_seed: int,
    mp_start_method: str,
    rollout_device: str,
    executor: ProcessPoolExecutor = None,
):
    """
    并行采样流程：
    多个 worker/actor 并行采样 -> 收集 trajectory/transition -> 合并成 RolloutBuffer。

    如果传入 executor，则复用已有进程池（避免每 episode 重复创建进程）；
    否则内部创建临时进程池（向后兼容）。
    """
    if num_workers <= 1:
        # 单 worker 时避免进程开销，便于调试。
        if base_seed >= 0:
            _seed_worker(base_seed + iteration * 10007)
        buffer, metrics, schedule_dict, module_change_dict = run_episode_with_trace(
            env=env,
            agent=agent,
            deterministic=deterministic,
            train_mode=True,
        )
        metrics["worker_id"] = 0
        metrics["upper_transition_count"] = len(buffer.upper)
        metrics["lower_transition_count"] = len(buffer.lower)
        return aggregate_worker_outputs([(buffer, metrics, schedule_dict, module_change_dict)])

    state_dict = state_dict_to_cpu(agent)
    if base_seed < 0:
        base_seed = time.time_ns() % (2 ** 31 - 1)

    def _submit_jobs(exec):
        jobs = []
        for worker_id in range(num_workers):
            worker_seed = int(base_seed + iteration * 10007 + worker_id * 97)
            jobs.append(
                exec.submit(
                    _rollout_worker,
                    worker_id,
                    state_dict,
                    dims,
                    file_name,
                    path,
                    config.HIDDEN_DIM,
                    deterministic,
                    worker_seed,
                    rollout_device,
                )
            )
        return [job.result() for job in jobs]

    if executor is not None:
        # 复用外部传入的持久进程池
        worker_outputs = _submit_jobs(executor)
    else:
        # 向后兼容：内部临时创建进程池
        ctx = get_context(mp_start_method)
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as exec:
            worker_outputs = _submit_jobs(exec)

    return aggregate_worker_outputs(worker_outputs)


def save_checkpoint(
    path,
    best_state,
    schedule_dict,
    module_change_dict,
    all_makespan,
    all_cost,
    all_episode_runtime,
    all_episode_steps,
    all_upper_reward_sum,
    all_lower_reward_sum,
    all_upper_lower_reward_sum,
    all_done_rate,
    all_worker_metrics,
):
    ensure_dir(os.path.dirname(path))

    torch.save(
        {
            "model_state_dict": best_state,

            "schedule_dict": schedule_dict,
            "module_change_dict": module_change_dict,

            "all_makespan": all_makespan,
            "all_cost": all_cost,

            # 每个训练周期的总运行时间，单位：秒
            "all_episode_runtime": all_episode_runtime,

            "all_episode_steps": all_episode_steps,
            "all_upper_reward_sum": all_upper_reward_sum,
            "all_lower_reward_sum": all_lower_reward_sum,
            "all_upper_lower_reward_sum": all_upper_lower_reward_sum,
            "all_done_rate": all_done_rate,

            "all_worker_metrics": all_worker_metrics,
        },
        path,
    )


def _append_visdom_line(viz, win, x_value, y_value, title, xlabel, ylabel):
    if win is None:
        return viz.line(
            Y=[y_value],
            X=[x_value],
            opts=dict(title=title, xlabel=xlabel, ylabel=ylabel),
        )
    viz.line(Y=[y_value], X=[x_value], win=win, update="append")
    return win

def format_weight(w: float) -> str:
    w = float(w)
    if abs(w - round(w)) < 1e-9:
        return str(int(round(w)))
    return f"{w:.2f}".rstrip("0").rstrip(".")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=config.DATA_PATH)
    parser.add_argument("--file_name", type=str, default="M4_A3_R4_J2")
    parser.add_argument("--episodes", type=int, default=config.TRAIN_EPISODES)
    parser.add_argument("--save_dir", type=str, default=config.SAVE_DIR)
    parser.add_argument("--num_workers", type=int, default=getattr(config, "NUM_WORKERS", 4))
    parser.add_argument("--update_batch_size", type=int, default=getattr(config, "UPDATE_BATCH_SIZE", 64))
    parser.add_argument("--mp_start_method", type=str, default=getattr(config, "MP_START_METHOD", "spawn"))
    parser.add_argument("--rollout_device", type=str, default=getattr(config, "ROLLOUT_DEVICE", "cpu"))
    parser.add_argument("--seed", type=int, default=-1, help="-1 表示不固定随机种子")
    args = parser.parse_args()


    # 命令行参数覆盖 config，便于不改代码直接实验。
    config.UPDATE_BATCH_SIZE = int(args.update_batch_size)

    ensure_dir(args.save_dir)
    if args.seed >= 0:
        set_seed(args.seed)

    viz = Visdom(env="MOFRMS_Training")
    cost_win = None
    makespan_win = None

    env = make_env(file_name=args.file_name, path=args.path)
    op_dim, machine_dim, module_dim = infer_dims(env)
    dims = (op_dim, machine_dim, module_dim)

    agent = HiFAMRSyncMAGRL(
        op_input_dim=op_dim,
        machine_input_dim=machine_dim,
        module_input_dim=module_dim,
        hidden_dim=config.HIDDEN_DIM,
        device=config.DEVICE,
    )

    best_reward_sum = -float("inf")
    best_path = os.path.join(
        config.SAVE_DIR,
        f"{format_weight(config.alpha_t)}makespan_"
        f"{format_weight(config.alpha_c)}cost_"
        f"{args.file_name}.pt",
    )

    # =========================
    # 记录每个 iteration 的 worker 均值数据
    # =========================
    all_makespan = []
    all_cost = []
    all_episode_runtime = []
    all_episode_steps = []
    all_upper_reward_sum = []
    all_lower_reward_sum = []
    all_upper_lower_reward_sum = []
    all_done_rate = []
    all_worker_metrics = []

    final_schedule_dict = None
    final_module_change_dict = None
    best_state = None

    # =========================
    # 创建持久进程池（复用跨 episode，避免每轮 Windows spawn 开销）
    # =========================
    if args.num_workers > 1:
        ctx = get_context(args.mp_start_method)
        persistent_executor = ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx)

    else:
        persistent_executor = None

    for episode in range(1, args.episodes + 1):
        iteration_start_time = time.perf_counter()

        (
            buffer,
            avg_metrics,
            worker_metrics,
            schedule_dicts,
            module_change_dicts,
        ) = collect_parallel_rollouts(
            agent=agent,
            env=env,
            dims=dims,
            file_name=args.file_name,
            path=args.path,
            num_workers=args.num_workers,
            iteration=episode,
            deterministic=False,
            base_seed=args.seed,
            mp_start_method=args.mp_start_method,
            rollout_device=args.rollout_device,
            executor=persistent_executor,
        )

        update_logs = agent.update(buffer)
        update_time = float(update_logs.get("update_time_sec", 0.0))

        # 一个训练周期的总运行时间：并行采样 + batch 更新 + 基本统计
        iteration_runtime = time.perf_counter() - iteration_start_time


        # =========================
        # 保存每个周期的 worker 均值数据
        # =========================
        all_makespan.append(float(avg_metrics["makespan"]))
        all_cost.append(float(avg_metrics["cost"]))
        all_episode_runtime.append(float(iteration_runtime))
        all_episode_steps.append(float(avg_metrics["steps"]))
        all_upper_reward_sum.append(float(avg_metrics["upper_reward_sum"]))
        all_lower_reward_sum.append(float(avg_metrics["lower_reward_sum"]))
        all_upper_lower_reward_sum.append(float(avg_metrics["upper_lower_reward_sum"]))
        all_done_rate.append(float(avg_metrics["done_rate"]))
        all_worker_metrics.append(worker_metrics)

        print(
            f"episode={episode} "
            f"workers={args.num_workers} "
            f"done_rate={avg_metrics['done_rate']:.4f} "
            f"avg_steps={avg_metrics['steps']:.2f} "
            f"runtime={iteration_runtime:.4f}s "
            f"avg_makespan={avg_metrics['makespan']:.4f} "
            f"avg_cost={avg_metrics['cost']:.4f} "
        )

        # =========================
        # Visdom：每个 episode/iteration 画多个 worker 的均值
        # =========================
        cost_win = _append_visdom_line(
            viz=viz,
            win=cost_win,
            x_value=episode,
            y_value=avg_metrics["cost"],
            title=f"{args.file_name}_{config.alpha_c} Average Cost per Iteration ({args.num_workers} workers)",
            xlabel="Episode / Iteration",
            ylabel="Average Cost",
        )

        makespan_win = _append_visdom_line(
            viz=viz,
            win=makespan_win,
            x_value=episode,
            y_value=avg_metrics["makespan"],
            title=f"{args.file_name}_{config.alpha_t} Average Makespan per Iteration ({args.num_workers} workers)",
            xlabel="Episode / Iteration",
            ylabel="Average Makespan",
        )

        # =========================
        # 保存最优模型与甘特图
        # 依据：多个 worker 的 average upper_reward_sum + lower_reward_sum 最大。
        # 甘特图保存该 iteration 中回报最高且完成的 worker；若没有完成，则保存回报最高 worker。
        # =========================
        current_reward_sum = avg_metrics["upper_lower_reward_sum"]
        if current_reward_sum > best_reward_sum:
            best_reward_sum = current_reward_sum
            best_worker_idx = max(
                range(len(worker_metrics)),
                key=lambda i: (
                    bool(worker_metrics[i].get("done", False)),
                    float(worker_metrics[i].get("upper_lower_reward_sum", -float("inf"))),
                ),
            )
            final_schedule_dict = schedule_dicts[best_worker_idx]
            final_module_change_dict = module_change_dicts[best_worker_idx]
            best_state = state_dict_to_cpu(agent)

    if best_state is None:
        best_state = state_dict_to_cpu(agent)

    # 训练结束，关闭持久进程池
    if persistent_executor is not None:
        persistent_executor.shutdown(wait=True)
        print("[INFO] 持久进程池已关闭", flush=True)

    save_checkpoint(
        path=best_path,
        best_state=best_state,
        schedule_dict=final_schedule_dict,
        module_change_dict=final_module_change_dict,
        all_makespan=all_makespan,
        all_cost=all_cost,
        all_episode_runtime=all_episode_runtime,
        all_episode_steps=all_episode_steps,
        all_upper_reward_sum=all_upper_reward_sum,
        all_lower_reward_sum=all_lower_reward_sum,
        all_upper_lower_reward_sum=all_upper_lower_reward_sum,
        all_done_rate=all_done_rate,
        all_worker_metrics=all_worker_metrics,
    )

    print(
        f"best_avg_makespan={min(all_makespan):.4f} "
        f"best_avg_cost={min(all_cost):.4f} "
        f"checkpoint={best_path}"
    )


if __name__ == "__main__":
    # 启动 Visdom:
    # python -m visdom.server
    main()
