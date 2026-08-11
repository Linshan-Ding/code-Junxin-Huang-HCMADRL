# =========================
# 文件名：evaluate.py
# =========================
import argparse
import os
from statistics import mean, pstdev

import torch

from agent import HiFAMRSyncMAGRL
from config import config
from env import MO_DFRMS_Environment
from rl_state import graph_tensor_from_env
from train import infer_dims, run_episode
from utils import set_seed


def make_env(file_name: str, path: str) -> MO_DFRMS_Environment:
    return MO_DFRMS_Environment(
        use_instance=False,
        file_name=file_name,
        path=path,
    )


def load_agent(checkpoint_path: str, env: MO_DFRMS_Environment) -> HiFAMRSyncMAGRL:
    op_dim, machine_dim, module_dim = infer_dims(env)
    agent = HiFAMRSyncMAGRL(
        op_input_dim=op_dim,
        machine_input_dim=machine_dim,
        module_input_dim=module_dim,
        hidden_dim=config.HIDDEN_DIM,
        device=config.DEVICE,
    )

    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=config.DATA_PATH)
    parser.add_argument("--file_name", type=str, default="M8_A6_S1")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(config.SAVE_DIR, config.MODEL_NAME),
    )
    parser.add_argument("--episodes", type=int, default=config.EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    env = make_env(args.file_name, args.path)
    agent = load_agent(args.checkpoint, env)

    makespans = []
    costs = []
    steps = []
    dones = []

    for episode in range(1, args.episodes + 1):
        _, metrics = run_episode(
            env=env,
            agent=agent,
            deterministic=config.DETERMINISTIC_EVAL,
            train_mode=False,
        )
        makespans.append(metrics["makespan"])
        costs.append(metrics["cost"])
        steps.append(metrics["steps"])
        dones.append(metrics["done"])

        print(
            f"eval_episode={episode} "
            f"done={metrics['done']} "
            f"steps={metrics['steps']} "
            f"makespan={metrics['makespan']:.4f} "
            f"cost={metrics['cost']:.4f}"
        )

    print(
        f"summary "
        f"done_rate={sum(dones) / max(1, len(dones)):.4f} "
        f"makespan_mean={mean(makespans):.4f} "
        f"makespan_std={pstdev(makespans):.4f} "
        f"cost_mean={mean(costs):.4f} "
        f"cost_std={pstdev(costs):.4f} "
        f"steps_mean={mean(steps):.4f}"
    )


if __name__ == "__main__":
    main()