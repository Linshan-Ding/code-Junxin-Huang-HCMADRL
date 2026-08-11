# ============================================================
# 文件名：ablation/variants/variant_no_attn.py
# 变体4：No-Attn（去除边注意力机制）
#
# 消融目标：验证图神经网络中边注意力（Edge Attention）加权机制
#           相对于简单均值聚合的性能贡献。
#
# 修改内容：
#   1. 移除 EdgeAttentionLayer 中的 score 子模块（注意力打分）
#   2. 移除 _segment_softmax / _segment_softmax_batched（注意力归一化）
#   3. 消息传递改为：msg = MLP([src_feat, edge_h]) / num_neighbors
#   4. 保留边投影（edge_proj）、消息MLP（message）、残差连接、LayerNorm
#   5. 异构节点类型和四种边类型结构完全保留
# ============================================================

from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_state import GraphTensor, BatchedGraphTensor


# ============================================================
# MeanAggregationLayer：均值聚合层（替代 EdgeAttentionLayer）
# ============================================================
class MeanAggregationLayer(nn.Module):
    """
    均值聚合层：用 1/num_neighbors 均匀权重替代注意力加权。

    与 EdgeAttentionLayer 的区别：
    - EdgeAttentionLayer：attention = softmax(score([dst, src, edge]))，动态学习权重
    - MeanAggregationLayer：weight = 1 / num_neighbors，所有邻居等权重
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        # 保留边投影（不做注意力打分）
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 保留消息 MLP
        self.message = nn.Linear(hidden_dim * 2, hidden_dim)
        # 保留 LayerNorm 和残差
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        src_h: torch.Tensor,
        dst_h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_dst: int,
    ) -> torch.Tensor:
        """
        均值聚合前向传播。

        输入格式与 EdgeAttentionLayer 完全兼容：
        - 单图：src_h [Nsrc,H], dst_h [Ndst,H], edge_attr [E,F]
        - 批量：src_h [B,Nsrc,H], dst_h [B,Ndst,H], edge_attr [B,E,F]
        """
        if edge_index.numel() == 0:
            return dst_h

        single_graph = False
        if src_h.dim() == 2:
            single_graph = True
            src_h = src_h.unsqueeze(0)
            dst_h = dst_h.unsqueeze(0)
            edge_attr = edge_attr.unsqueeze(0)

        src = edge_index[0].long()
        dst = edge_index[1].long()

        # 边投影（与原始相同）
        edge_h = self.edge_proj(edge_attr)       # [B, E, H]
        src_feat = src_h[:, src, :]              # [B, E, H]

        # ★ 关键区别：无注意力分数计算，无 softmax
        # 直接构建消息
        msg_input = torch.cat([src_feat, edge_h], dim=-1)
        msg = self.message(msg_input)            # [B, E, H]

        # ★ 均值聚合：msg / num_neighbors，使用批量 index_add_（一条 GPU 指令）
        agg = torch.zeros_like(dst_h)                   # [B, Ndst, H]
        agg.index_add_(1, dst, msg)                     # 批量聚合，与原始一致

        # 除以每个目标节点的入度（均值归一化）
        degree = torch.zeros(num_dst, device=dst_h.device)
        degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        degree = degree.clamp(min=1.0)

        agg = agg / degree.unsqueeze(0).unsqueeze(-1)   # 统一 batch 除法

        # 残差 + LayerNorm
        out = self.norm(dst_h + agg)

        if single_graph:
            out = out.squeeze(0)
        return out


# ============================================================
# NoAttnEncoder：无注意力的异构关系图编码器
# ============================================================
class NoAttnEncoder(nn.Module):
    """
    无注意力版本的异构关系图编码器。

    与 FMCO_RHGN 的区别：
    - FMCO_RHGN：4 种边类型各使用 EdgeAttentionLayer（带注意力打分+softmax）
    - NoAttnEncoder：4 种边类型各使用 MeanAggregationLayer（1/N 均匀聚合）

    节点投影、异构边类型区分、全局池化等结构完全保留。
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

        # 节点投影（与原始 FMCO_RHGN 完全一致）
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

        # ★ 四种边类型仍然各自有独立的 MeanAggregationLayer（保留异构性）
        # 但内部不使用注意力，而是均值聚合
        self.o2o_layer = MeanAggregationLayer(hidden_dim=hidden_dim, edge_dim=4)
        self.m2a_to_machine_layer = MeanAggregationLayer(hidden_dim=hidden_dim, edge_dim=8)
        self.o2a_to_operation_layer = MeanAggregationLayer(hidden_dim=hidden_dim, edge_dim=4)
        self.o2m_to_machine_layer = MeanAggregationLayer(hidden_dim=hidden_dim, edge_dim=5)

        # 全局投影（与原始一致）
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, graph: Union[GraphTensor, BatchedGraphTensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播。消息传递流程与 FMCO_RHGN 完全一致，
        但每个 EdgeAttentionLayer 替换为 MeanAggregationLayer。
        """
        op_h0 = self.op_proj(graph.op_x)
        machine_h0 = self.machine_proj(graph.machine_x)
        module_h0 = self.module_proj(graph.module_x)

        # o2o: 工序→工序
        op_h1 = self.o2o_layer(
            src_h=op_h0,
            dst_h=op_h0,
            edge_index=graph.o2o_edge_index,
            edge_attr=graph.o2o_edge_attr,
            num_dst=graph.op_count,
        )

        # m2a→machine: 机器→模块 反向（模块→机器）
        m2a_reverse = (
            torch.stack([graph.m2a_edge_index[1], graph.m2a_edge_index[0]], dim=0)
            if graph.m2a_edge_index.numel() > 0
            else graph.m2a_edge_index
        )
        machine_h1 = self.m2a_to_machine_layer(
            src_h=module_h0,
            dst_h=machine_h0,
            edge_index=m2a_reverse,
            edge_attr=graph.m2a_edge_attr,
            num_dst=graph.machine_count,
        )

        # o2a→operation: 工序→模块 反向（模块→工序）
        o2a_reverse = (
            torch.stack([graph.o2a_edge_index[1], graph.o2a_edge_index[0]], dim=0)
            if graph.o2a_edge_index.numel() > 0
            else graph.o2a_edge_index
        )
        op_h2 = self.o2a_to_operation_layer(
            src_h=module_h0,
            dst_h=op_h1,
            edge_index=o2a_reverse,
            edge_attr=graph.o2a_edge_attr,
            num_dst=graph.op_count,
        )

        # o2m→machine: 工序→机器
        machine_h2 = self.o2m_to_machine_layer(
            src_h=op_h2,
            dst_h=machine_h1,
            edge_index=graph.o2m_edge_index,
            edge_attr=graph.o2m_edge_attr,
            num_dst=graph.machine_count,
        )

        # 全局池化（与原始一致）
        if op_h2.dim() == 3:
            op_pool = op_h2.mean(dim=1)
            machine_pool = machine_h2.mean(dim=1)
            module_pool = module_h0.mean(dim=1)
        else:
            op_pool = op_h2.mean(dim=0)
            machine_pool = machine_h2.mean(dim=0)
            module_pool = module_h0.mean(dim=0)

        global_h = self.global_proj(
            torch.cat([op_pool, machine_pool, module_pool], dim=-1)
        )

        return {
            "op": op_h2,
            "machine": machine_h2,
            "module": module_h0,
            "global": global_h,
        }
