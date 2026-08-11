# =========================
# 文件名：config.py
# =========================
import os
import torch


class _DefaultCostDict(dict):
    def __missing__(self, key):
        return config.COST_REF_DEFAULT


class config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alpha_t = float(os.environ.get("ALPHA_T", "0.5"))
    alpha_c = float(os.environ.get("ALPHA_C", "0.5"))

    # 是否对 reward 中的 makespan / cost 项归一化
    REWARD_NORMALIZE = os.environ.get("REWARD_NORMALIZE", "0") == "1"

    # 前两组单目标训练得到的参考值
    MAKESPAN_REF = float(os.environ.get("MAKESPAN_REF", "1.0"))
    COST_REF = float(os.environ.get("COST_REF", "1.0"))

    new_file_names = [
        'M4_A3_R4_J2', 'M4_A3_R6_J3', 'M4_A3_R8_J4', 'M4_A4_R4_J2', 'M4_A4_R6_J3', 'M4_A4_R8_J4',
        'M4_A5_R4_J2', 'M4_A5_R6_J3', 'M4_A5_R8_J4', 'M8_A6_R4_J2', 'M8_A6_R6_J3', 'M8_A6_R8_J4',
        'M8_A8_R4_J2', 'M8_A8_R6_J3', 'M8_A8_R8_J4', 'M8_A10_R4_J2', 'M8_A10_R6_J3', 'M8_A10_R8_J4',
        'M12_A9_R4_J2', 'M12_A9_R6_J3', 'M12_A9_R8_J4', 'M12_A12_R4_J2', 'M12_A12_R6_J3', 'M12_A12_R8_J4',
        'M12_A15_R4_J2', 'M12_A15_R6_J3', 'M12_A15_R8_J4'
    ]

    cost = _DefaultCostDict({
        'M4_A3_R4_J2': 0, 'M4_A3_R6_J3': 0, 'M4_A3_R8_J4': 0,
        'M4_A4_R4_J2': 0, 'M4_A4_R6_J3': 0, 'M4_A4_R8_J4': 0,
        'M4_A5_R4_J2': 0, 'M4_A5_R6_J3': 0, 'M4_A5_R8_J4': 0,
        'M8_A6_R4_J2': 0, 'M8_A6_R6_J3': 0, 'M8_A6_R8_J4': 0,
        'M8_A8_R4_J2': 0, 'M8_A8_R6_J3': 0, 'M8_A8_R8_J4': 0,
        'M8_A10_R4_J2': 0, 'M8_A10_R6_J3': 0, 'M8_A10_R8_J4': 0,
        'M12_A9_R4_J2': 0, 'M12_A9_R6_J3': 0, 'M12_A9_R8_J4': 0,
        'M12_A12_R4_J2': 0, 'M12_A12_R6_J3': 0, 'M12_A12_R8_J4': 0,
        'M12_A15_R4_J2': 0, 'M12_A15_R6_J3': 0, 'M12_A15_R8_J4': 0
    })

    makespan = _DefaultCostDict({
        'M4_A3_R4_J2': 0, 'M4_A3_R6_J3': 0, 'M4_A3_R8_J4': 0,
        'M4_A4_R4_J2': 0, 'M4_A4_R6_J3': 0, 'M4_A4_R8_J4': 0,
        'M4_A5_R4_J2': 0, 'M4_A5_R6_J3': 0, 'M4_A5_R8_J4': 0,
        'M8_A6_R4_J2': 0, 'M8_A6_R6_J3': 0, 'M8_A6_R8_J4': 0,
        'M8_A8_R4_J2': 0, 'M8_A8_R6_J3': 0, 'M8_A8_R8_J4': 0,
        'M8_A10_R4_J2': 0, 'M8_A10_R6_J3': 0, 'M8_A10_R8_J4': 0,
        'M12_A9_R4_J2': 0, 'M12_A9_R6_J3': 0, 'M12_A9_R8_J4': 0,
        'M12_A12_R4_J2': 0, 'M12_A12_R6_J3': 0, 'M12_A12_R8_J4': 0,
        'M12_A15_R4_J2': 0, 'M12_A15_R6_J3': 0, 'M12_A15_R8_J4': 0
    })
    DATA_PATH = os.environ.get("HIFAMR_DATA_PATH", os.path.join(os.getcwd(), "data"))




    HIDDEN_DIM = 128
    GRAPH_LAYERS = 1
    DROPOUT = 0.0

    UPPER_LR = 3e-4
    LOWER_LR = 3e-4
    WEIGHT_DECAY = 0.0

    GAMMA_U = 0.99
    GAMMA_L = 0.99
    GAE_LAMBDA_U = 0.95
    GAE_LAMBDA_L = 0.95

    PPO_CLIP = 0.2
    PPO_EPOCHS = 4
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5

    TRAIN_EPISODES = 500
    MAX_EPISODE_STEPS = 5000000
    SAVE_DIR = r"D:\HCMADRL\checkpoints"

    EVAL_EPISODES = 500
    DETERMINISTIC_EVAL = True
# =========================
# 并行 rollout + mini-batch PPO 更新配置
# =========================
# 每个训练 iteration 同时启动多少个 worker/actor 采样。
# 外层 episode/iteration 的 Visdom 曲线会记录这些 worker 的均值。
config.NUM_WORKERS = int(os.environ.get("HIFAMR_NUM_WORKERS", "4"))

# PPO update 阶段的 mini-batch 大小：先合并所有 worker 的 transition，再打乱切 batch。
config.UPDATE_BATCH_SIZE = int(os.environ.get("HIFAMR_UPDATE_BATCH_SIZE", "64"))
config.UPDATE_SHUFFLE = os.environ.get("HIFAMR_UPDATE_SHUFFLE", "1") != "0"

# 多进程启动方式：Windows 建议 spawn；Linux 可用 fork 或 spawn。
config.MP_START_METHOD = os.environ.get("HIFAMR_MP_START_METHOD", "spawn")

# rollout worker 默认用 CPU，主进程仍按 config.DEVICE 在 GPU/CPU 上统一更新网络。
config.ROLLOUT_DEVICE = os.environ.get("HIFAMR_ROLLOUT_DEVICE", "cpu")


