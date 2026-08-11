# ============================================================
# 文件名：ablation/ablation_config.py
# 消融实验调度配置 —— 用户在此文件中指定每轮训练的变体、权重和算例
# ============================================================
#
# 训练模式：轮训 (Round-Robin)
#   - 按 SCHEDULE 列表顺序，依次执行每个 ScheduleItem
#   - 每个 ScheduleItem 指定：使用哪个消融变体、哪个权重、哪些算例
#   - 训练完一轮后，自动进行下一轮
#   - 支持多轮循环（n_rounds > 1 时重复整个 SCHEDULE）
#
# 使用方法：
#   1. 修改本文件中的 SCHEDULE 列表
#   2. 修改 NUM_ROUNDS 指定循环轮数
#   3. 运行：python ablation/run_ablation.py
# ============================================================

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ============================================================
# 变体名称常量（方便引用，避免拼写错误）
# ============================================================
VARIANT_ORIGINAL    = "original"       # 原始 HiFAMR-BMAPPO（不做任何修改）
VARIANT_FLAT_MAPPO  = "flat_mappo"     # 变体1：扁平化单层级结构
VARIANT_MLP_ENCODER = "mlp_encoder"    # 变体2：MLP 替代异构图神经网络
VARIANT_HOMO_GNN    = "homo_gnn"       # 变体3：同质图替代异构图
VARIANT_NO_ATTN     = "no_attn"        # 变体4：去除边注意力机制（均值聚合）



# 所有消融变体列表
ABLATION_VARIANTS = [
    VARIANT_ORIGINAL,
    VARIANT_FLAT_MAPPO,
    VARIANT_MLP_ENCODER,
    VARIANT_HOMO_GNN,
    VARIANT_NO_ATTN,
]




# ============================================================
# 可用算例列表（来自 ablation_config_v1.py 的 27 个实例）
# ============================================================
ALL_INSTANCES = [
    # M=4 小型实例 (9个)
    'M4_A3_R4_J2',  'M4_A3_R6_J3',  'M4_A3_R8_J4',
    'M4_A4_R4_J2',  'M4_A4_R6_J3',  'M4_A4_R8_J4',
    'M4_A5_R4_J2',  'M4_A5_R6_J3',  'M4_A5_R8_J4',
    # M=8 中型实例 (9个)
    'M8_A6_R4_J2',  'M8_A6_R6_J3',  'M8_A6_R8_J4',
    'M8_A8_R4_J2',  'M8_A8_R6_J3',  'M8_A8_R8_J4',
    'M8_A10_R4_J2', 'M8_A10_R6_J3', 'M8_A10_R8_J4',
    # M=12 大型实例 (9个)
    'M12_A9_R4_J2',  'M12_A9_R6_J3',  'M12_A9_R8_J4',
    'M12_A12_R4_J2', 'M12_A12_R6_J3', 'M12_A12_R8_J4',
    'M12_A15_R4_J2', 'M12_A15_R6_J3', 'M12_A15_R8_J4',
]


# ============================================================
# ScheduleItem 定义
# ============================================================
@dataclass
class ScheduleItem:
    """
    一轮训练的完整配置。

    Parameters
    ----------
    variant : str
        消融变体名称，使用 VARIANT_* 常量
    alpha_t : float
        Makespan 权重 (0.0 ~ 1.0)
    alpha_c : float
        Cost 权重 (0.0 ~ 1.0)，注意 alpha_t + alpha_c 通常 = 1.0
    file_names : List[str]
        本轮要训练的算例名称列表
    reward_normalize : bool
        是否对奖励进行归一化（混合权重建议开启）
    episodes : int
        每个算例训练多少个 episode
    description : str
        本轮训练的描述标签（用于日志）
    """
    variant: str
    alpha_t: float
    alpha_c: float
    file_names: List[str]
    reward_normalize: bool = False
    episodes: int = 1000
    description: str = ""


# ============================================================
# ★ 用户在此指定训练调度 ★
# ============================================================
# 修改下面的 SCHEDULE 列表来控制每轮训练的内容
# 每个 ScheduleItem 代表一轮训练

# SCHEDULE 移到文件末尾（函数定义之后），见下方
# ============================================================
# 全局训练参数
# ============================================================

# 训练轮数：整个 SCHEDULE 重复执行的次数
# 设为 1 表示每个 ScheduleItem 只执行一次
# 设为 3 表示整个 SCHEDULE 循环执行 3 遍
NUM_ROUNDS: int = 1

# 并行 worker 数（每个算例训练时启动多少个采样进程）
NUM_WORKERS: int = 4



# PPO mini-batch 大小
UPDATE_BATCH_SIZE: int = 64

# 多进程启动方式（Windows 用 spawn）
MP_START_METHOD: str = "spawn"

# rollout worker 使用的设备
ROLLOUT_DEVICE: str = "cpu"

# 是否每轮训练前自动启动 Visdom
AUTO_START_VISDOM: bool = True

# Visdom 端口
VISDOM_PORT: int = 8097

# Visdom 环境名
VISDOM_ENV: str = "MOFRMS_Training"


# ============================================================
# 路径配置
# ============================================================
# 项目根目录
PROJECT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_PATH: str = os.path.join(PROJECT_DIR, "data")

# 消融实验模型保存目录
ABLATION_SAVE_DIR: str = os.path.join(PROJECT_DIR, "checkpoints", "ablation")

# 原始模型保存目录
ORIGINAL_SAVE_DIR: str = os.path.join(PROJECT_DIR, "checkpoints")


# ============================================================
# 辅助函数：快速构建自定义调度
# ============================================================
def make_weight_sweep_schedule(
    variant: str,
    file_names: List[str],
    episodes: int = 1000,
    description_prefix: str = "",
) -> List[ScheduleItem]:
    """
    为一组权重生成完整 Pareto 前沿调度。

    生成 5 组权重：纯Makespan → 混合 → 纯Cost
    """
    # 顺序：先训练两个纯目标（提供标准化参考值），再训练三个混合权重
    weights = [
        (1.0, 0.0, False, "纯Makespan"),
        (0.0, 1.0, False, "纯Cost"),
        (0.75, 0.25, True, "Makespan偏重"),
        (0.5, 0.5, True, "均衡"),
        (0.25, 0.75, True, "Cost偏重"),
    ]
    return [
        ScheduleItem(
            variant=variant,
            alpha_t=at,
            alpha_c=ac,
            file_names=file_names,
            reward_normalize=rn,
            episodes=episodes,
            description=f"{description_prefix}-{label}",
        )
        for at, ac, rn, label in weights
    ]


def make_ablation_comparison_schedule(
    file_names: List[str],
    episodes: int = 1000,
) -> List[ScheduleItem]:
    """
    生成完整的消融实验对比调度。

    包含：原始算法 + 4个变体，每个运行 5 组权重（完整 Pareto 前沿）。
    总计 5×5 = 25 轮。
    """
    all_variants = [
        (VARIANT_ORIGINAL,    "原始算法"),
        (VARIANT_FLAT_MAPPO,  "变体1-FlatMAPPO"),
        (VARIANT_MLP_ENCODER, "变体2-MLP"),
        (VARIANT_HOMO_GNN,    "变体3-HomoGNN"),
        (VARIANT_NO_ATTN,     "变体4-NoAttn"),
    ]
    schedule = []
    for variant, desc_prefix in all_variants:
        schedule.extend(
            make_weight_sweep_schedule(variant, file_names, episodes, desc_prefix)
        )
    return schedule








# ============================================================
# ★ 当前生效的调度（在函数定义之后，可正常调用）★
# ============================================================
SCHEDULE = make_weight_sweep_schedule(
    variant=VARIANT_ORIGINAL,
    file_names=['M12_A15_R4_J2'],
    episodes=500,
)
