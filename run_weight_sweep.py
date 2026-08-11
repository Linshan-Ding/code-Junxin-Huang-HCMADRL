from pathlib import Path
import os
import torch

import socket
import time
import subprocess
import sys

PROJECT_DIR = Path(r"F:\Flexible Reconfigurable Manufacturing System\HiFAMR-BMAPPO")
TRAIN_SCRIPT = PROJECT_DIR / "train_visdom.py"

DATA_PATH = PROJECT_DIR / "data"
SAVE_DIR = PROJECT_DIR / "checkpoints"

FILE_NAME = "M4_A4_R4_J2"

EPISODES = 500
NUM_WORKERS = 4
UPDATE_BATCH_SIZE = 128
MP_START_METHOD = "spawn"

def is_port_open(host="127.0.0.1", port=8097, timeout=1.0):
    """检查 Visdom 端口是否已经启动。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_visdom_server(port=8097):
    """
    如果 Visdom server 没启动，则自动启动：
        python -m visdom.server -port 8097
    """
    if is_port_open(port=port):
        print(f"Visdom server 已经在运行：http://127.0.0.1:{port}")
        return None

    print(f"正在启动 Visdom server：http://127.0.0.1:{port}")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "visdom.server",
            "-port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 等待 Visdom 启动
    for _ in range(30):
        if is_port_open(port=port):
            print(f"Visdom server 启动成功：http://127.0.0.1:{port}")
            return process
        time.sleep(1)

    print("警告：Visdom server 可能没有正常启动，但训练会继续运行。")
    return process

def format_weight(w: float) -> str:
    w = float(w)
    if abs(w - round(w)) < 1e-9:
        return str(int(round(w)))
    return f"{w:.2f}".rstrip("0").rstrip(".")


def checkpoint_path(alpha_t: float, alpha_c: float) -> Path:
    return SAVE_DIR / (
        f"{format_weight(alpha_t)}makespan_"
        f"{format_weight(alpha_c)}cost_"
        f"{FILE_NAME}.pt"
    )


def run_train(
    alpha_t: float,
    alpha_c: float,
    reward_normalize: bool,
    makespan_ref: float = 1.0,
    cost_ref: float = 1.0,
):
    env = os.environ.copy()

    env["ALPHA_T"] = str(alpha_t)
    env["ALPHA_C"] = str(alpha_c)
    env["REWARD_NORMALIZE"] = "1" if reward_normalize else "0"
    env["MAKESPAN_REF"] = str(makespan_ref)
    env["COST_REF"] = str(cost_ref)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--path", str(DATA_PATH),
        "--file_name", FILE_NAME,
        "--episodes", str(EPISODES),
        "--num_workers", str(NUM_WORKERS),
        "--update_batch_size", str(UPDATE_BATCH_SIZE),
        "--mp_start_method", MP_START_METHOD,
    ]

    print("\n" + "=" * 100)
    print(
        f"开始训练：alpha_t={alpha_t}, alpha_c={alpha_c}, "
        f"reward_normalize={reward_normalize}, "
        f"MAKESPAN_REF={makespan_ref}, COST_REF={cost_ref}"
    )
    print("=" * 100)

    subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        env=env,
        check=True,
    )

    pt_path = checkpoint_path(alpha_t, alpha_c)

    if not pt_path.exists():
        raise FileNotFoundError(f"训练结束，但没有找到 checkpoint：{pt_path}")

    return pt_path


def extract_best_metric(pt_path: Path, metric_name: str) -> float:
    data = torch.load(pt_path, map_location="cpu")

    values = []

    # 优先从所有 worker 的原始数据中取最优
    for episode_metrics in data.get("all_worker_metrics", []):
        for m in episode_metrics:
            if not m.get("done", True):
                continue

            if metric_name in m:
                values.append(float(m[metric_name]))

    # 如果没有 all_worker_metrics，就从平均曲线里取
    if not values:
        if metric_name == "makespan":
            values = [float(v) for v in data["all_makespan"]]
        elif metric_name == "cost":
            values = [float(v) for v in data["all_cost"]]
        else:
            raise KeyError(f"不支持的 metric_name：{metric_name}")

    return min(values)


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    # 训练开始前自动启动 Visdom server
    visdom_process = start_visdom_server(port=8097)
    # =====================================================
    # 阶段 1：先跑两个单目标权重
    # =====================================================
    pt_makespan = run_train(
        alpha_t=1.0,
        alpha_c=0.0,
        reward_normalize=False,
    )

    pt_cost = run_train(
        alpha_t=0.0,
        alpha_c=1.0,
        reward_normalize=False,
    )

    makespan_ref = extract_best_metric(pt_makespan, "makespan")
    cost_ref = extract_best_metric(pt_cost, "cost")

    print("\n" + "=" * 100)
    print("标准化参考值已确定：")
    print(f"MAKESPAN_REF = {makespan_ref:.6f}，来自 {pt_makespan.name}")
    print(f"COST_REF     = {cost_ref:.6f}，来自 {pt_cost.name}")
    print("=" * 100)

    # =====================================================
    # 阶段 2：再跑三组混合权重
    # =====================================================
    mixed_weights = [
        (0.25, 0.75),
        (0.5, 0.5),
        (0.75, 0.25),
    ]

    for alpha_t, alpha_c in mixed_weights:
        run_train(
            alpha_t=alpha_t,
            alpha_c=alpha_c,
            reward_normalize=True,
            makespan_ref=makespan_ref,
            cost_ref=cost_ref,
        )

    print("\n5 组权重全部训练完成。")
    print("\n5 组权重全部训练完成，开始绘制 Pareto 前沿图...")

    # =========================
    # 阶段 3：调用独立 Pareto 绘图脚本
    # =========================
    PLOT_SCRIPT = PROJECT_DIR / "plot_pareto_front.py"

    subprocess.run(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--checkpoint_dir", str(SAVE_DIR),
            "--file_name", FILE_NAME,
            "--last_n", "10",
        ],
        cwd=str(PROJECT_DIR),
        check=True,
    )

    print("\n全部完成。")

if __name__ == "__main__":
    main()