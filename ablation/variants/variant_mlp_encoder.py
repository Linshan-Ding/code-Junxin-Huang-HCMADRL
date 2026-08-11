# ============================================================
# 文件名：ablation/variants/variant_mlp_encoder.py
# 变体2：MLP-Encoder（替换异构图神经网络为多层感知机）
#
# 消融目标：验证异构关系图神经网络（FMCO_RHGN）相对于
#           传统向量编码器的性能优势。
#
# 修改内容：
#   1. 将所有 Operation、Machine、Module 节点特征展平并拼接
#   2. 通过 3 层 MLP 得到全局状态表示
#   3. 图结构信息完全丢弃，仅依赖特征值
# ============================================================

from typing import Dict, Union

import torch
import torch.nn as nn

from rl_state import GraphTensor, BatchedGraphTensor


class MLPEncoder(nn.Module):
    """
    MLP 编码器：将图状态展平为固定向量，通过 MLP 编码。

    与 FMCO_RHGN 的区别：
    - FMCO_RHGN：通过异构图注意力网络建模节点间关系
    - MLPEncoder：简单展平所有节点特征，忽略图拓扑结构

    输出格式与 FMCO_RHGN 完全兼容：
      {"op": [B,O,H], "machine": [B,M,H], "module": [B,A,H], "global": [B,H]}
    """

    def __init__(
        self,
        op_input_dim: int,
        machine_input_dim: int,
        module_input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 节点级投影（与原始保持一致）
        self.op_proj = nn.Sequential(
            nn.Linear(op_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.machine_proj = nn.Sequential(
            nn.Linear(machine_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.module_proj = nn.Sequential(
            nn.Linear(module_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 全局 MLP：将展平的所有节点特征拼接后编码
        # 注意：这里不固定输入维度，而是对每种节点分别池化后拼接
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, graph: Union[GraphTensor, BatchedGraphTensor]) -> Dict[str, torch.Tensor]:
        """
        MLP 编码器前向传播。

        与 FMCO_RHGN 的关键区别：
        - 不执行任何图消息传递（无边注意力、无聚合）
        - 节点表示仅通过各自的 MLP 投影获得
        - 全局表示 = 各节点池化后的拼接 → MLP
        """
        op_h = self.op_proj(graph.op_x)
        machine_h = self.machine_proj(graph.machine_x)
        module_h = self.module_proj(graph.module_x)

        # 池化
        if op_h.dim() == 3:
            op_pool = op_h.mean(dim=1)
            machine_pool = machine_h.mean(dim=1)
            module_pool = module_h.mean(dim=1)
        else:
            op_pool = op_h.mean(dim=0)
            machine_pool = machine_h.mean(dim=0)
            module_pool = module_h.mean(dim=0)

        global_h = self.global_proj(torch.cat([op_pool, machine_pool, module_pool], dim=-1))

        return {
            "op": op_h,
            "machine": machine_h,
            "module": module_h,
            "global": global_h,
        }
