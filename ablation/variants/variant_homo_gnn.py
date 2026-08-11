# ============================================================
# 文件名：ablation/variants/variant_homo_gnn.py
# 变体3：HomoGNN（同质图神经网络替代异构图）
#
# 消融目标：验证异构关系图网络中区分实体类型和关系类型的必要性。
#
# 修改内容：
#   1. 所有三类节点（Operation, Machine, Module）统一投影到同一特征空间
#   2. 所有四种边类型合并为统一的边类型
#   3. 使用标准 GAT 层进行消息传递，共享同一组注意力参数
#   4. 移除分类型消息传递和分类型注意力
# ============================================================

from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_state import GraphTensor, BatchedGraphTensor


class HomoGATLayer(nn.Module):
    """
    标准同质图注意力层（GAT）。

    所有节点共享同一组参数，不区分节点/边类型。
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn_src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_edge = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.attn_out = nn.Linear(hidden_dim, 1)

        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        同质 GAT 层前向传播。

        node_h: [B,N,H] 或 [N,H]
        edge_index: [2,E]
        edge_attr: [B,E,F_edge] 或 [E,F_edge]
        """
        if edge_index.numel() == 0:
            return node_h

        single_graph = False
        if node_h.dim() == 2:
            single_graph = True
            node_h = node_h.unsqueeze(0)
            edge_attr = edge_attr.unsqueeze(0)

        src = edge_index[0].long()
        dst = edge_index[1].long()

        B, N, H = node_h.shape
        src_feat = node_h[:, src, :]   # [B, E, H]
        dst_feat = node_h[:, dst, :]   # [B, E, H]

        # 统一边特征投影
        edge_h = F.linear(edge_attr, self.attn_edge.weight, self.attn_edge.bias)

        # 注意力分数
        attn_input = self.attn_src(src_feat) + self.attn_dst(dst_feat) + edge_h
        raw_scores = self.attn_out(self.leaky_relu(attn_input)).squeeze(-1)  # [B, E]

        # 分段 softmax
        if raw_scores.numel() > 0:
            output = torch.zeros_like(raw_scores)
            unique_dst = torch.unique(dst)
            for d in unique_dst:
                mask = dst == d
                output[:, mask] = F.softmax(raw_scores[:, mask], dim=1)
            attn = output
        else:
            attn = raw_scores

        # 消息
        msg = self.msg_mlp(torch.cat([src_feat, edge_h], dim=-1)) * attn.unsqueeze(-1)

        # 聚合
        agg = torch.zeros_like(node_h)  # [B, N, H]
        agg.scatter_add_(1, dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, H), msg)

        out = self.norm(node_h + agg)

        if single_graph:
            out = out.squeeze(0)
        return out


class HomoGNNEncoder(nn.Module):
    """
    同质图神经网络编码器。

    与 FMCO_RHGN 的区别：
    - FMCO_RHGN：4 种边类型各有独立的 EdgeAttentionLayer，3 类节点各有独立的投影
    - HomoGNNEncoder：所有节点统一投影，所有边合并为单一边类型，共享 GAT 参数

    输出格式与 FMCO_RHGN 完全兼容。
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

        # 关键区别：三类节点先各自投影到 hidden_dim，再统一编码
        self.op_proj = nn.Linear(op_input_dim, hidden_dim)
        self.machine_proj = nn.Linear(machine_input_dim, hidden_dim)
        self.module_proj = nn.Linear(module_input_dim, hidden_dim)

        # 统一边投影（边特征维度不同，各自投影到 hidden_dim）
        self.o2o_edge_proj = nn.Linear(4, hidden_dim)
        self.o2m_edge_proj = nn.Linear(5, hidden_dim)
        self.o2a_edge_proj = nn.Linear(4, hidden_dim)
        self.m2a_edge_proj = nn.Linear(8, hidden_dim)

        # 统一 GAT 层
        self.gat1 = HomoGATLayer(hidden_dim, dropout)
        self.gat2 = HomoGATLayer(hidden_dim, dropout)

        self.dropout = nn.Dropout(dropout)

        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _build_unified_graph(self, graph: Union[GraphTensor, BatchedGraphTensor]):
        """
        将异构图的 3 类节点合并为统一节点序列，4 类边合并为统一边序列。

        返回: (unified_nodes, unified_edge_index, unified_edge_attr, op_count, machine_count)
        """
        single = graph.op_x.dim() == 2

        if single:
            op_h = self.op_proj(graph.op_x)
            machine_h = self.machine_proj(graph.machine_x)
            module_h = self.module_proj(graph.module_x)
        else:
            op_h = self.op_proj(graph.op_x)          # [B,O,H]
            machine_h = self.machine_proj(graph.machine_x)  # [B,M,H]
            module_h = self.module_proj(graph.module_x)     # [B,A,H]

        O = graph.op_count
        M = graph.machine_count
        A = graph.module_count

        if single:
            unified_nodes = torch.cat([op_h, machine_h, module_h], dim=0)  # [O+M+A, H]
        else:
            unified_nodes = torch.cat([op_h, machine_h, module_h], dim=1)  # [B, O+M+A, H]

        # 合并边索引（需要偏移 machine 和 module 的索引）
        all_edges_src = []
        all_edges_dst = []
        all_edges_attr = []

        # o2o: 工序→工序（不需要偏移）
        if graph.o2o_edge_index.numel() > 0:
            all_edges_src.append(graph.o2o_edge_index[0])
            all_edges_dst.append(graph.o2o_edge_index[1])
            if single:
                all_edges_attr.append(self.o2o_edge_proj(graph.o2o_edge_attr))
            else:
                all_edges_attr.append(self.o2o_edge_proj(graph.o2o_edge_attr))

        # o2m: 工序→机器（机器索引 + O）
        if graph.o2m_edge_index.numel() > 0:
            all_edges_src.append(graph.o2m_edge_index[0])
            all_edges_dst.append(graph.o2m_edge_index[1] + O)
            if single:
                all_edges_attr.append(self.o2m_edge_proj(graph.o2m_edge_attr))
            else:
                all_edges_attr.append(self.o2m_edge_proj(graph.o2m_edge_attr))

        # o2a: 工序→模块（模块索引 + O + M）
        if graph.o2a_edge_index.numel() > 0:
            all_edges_src.append(graph.o2a_edge_index[0])
            all_edges_dst.append(graph.o2a_edge_index[1] + O + M)
            if single:
                all_edges_attr.append(self.o2a_edge_proj(graph.o2a_edge_attr))
            else:
                all_edges_attr.append(self.o2a_edge_proj(graph.o2a_edge_attr))

        # m2a: 机器→模块（机器索引 + O，模块索引 + O + M）
        if graph.m2a_edge_index.numel() > 0:
            all_edges_src.append(graph.m2a_edge_index[0] + O)
            all_edges_dst.append(graph.m2a_edge_index[1] + O + M)
            if single:
                all_edges_attr.append(self.m2a_edge_proj(graph.m2a_edge_attr))
            else:
                all_edges_attr.append(self.m2a_edge_proj(graph.m2a_edge_attr))

        if all_edges_src:
            edge_index = torch.stack([
                torch.cat(all_edges_src, dim=0),
                torch.cat(all_edges_dst, dim=0),
            ], dim=0)
            edge_attr = torch.cat(all_edges_attr, dim=-2) if single else torch.cat(all_edges_attr, dim=-2)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            if single:
                edge_attr = torch.zeros((0, hidden_dim))
            else:
                B = op_h.shape[0]
                edge_attr = torch.zeros((B, 0, hidden_dim))

        return unified_nodes, edge_index, edge_attr, O, M

    def forward(self, graph: Union[GraphTensor, BatchedGraphTensor]) -> Dict[str, torch.Tensor]:
        single = graph.op_x.dim() == 2
        unified_nodes, edge_index, edge_attr, O, M = self._build_unified_graph(graph)

        # GAT 消息传递
        h1 = self.gat1(unified_nodes, edge_index, edge_attr)
        h1 = self.dropout(h1)
        h2 = self.gat2(h1, edge_index, edge_attr)

        # 拆分回三类节点
        if single:
            op_h = h2[:O]
            machine_h = h2[O:O + M]
            module_h = h2[O + M:]
        else:
            op_h = h2[:, :O, :]
            machine_h = h2[:, O:O + M, :]
            module_h = h2[:, O + M:, :]

        # 池化 + 全局嵌入
        if single:
            op_pool = op_h.mean(dim=0)
            machine_pool = machine_h.mean(dim=0)
            module_pool = module_h.mean(dim=0)
        else:
            op_pool = op_h.mean(dim=1)
            machine_pool = machine_h.mean(dim=1)
            module_pool = module_h.mean(dim=1)

        global_h = self.global_proj(torch.cat([op_pool, machine_pool, module_pool], dim=-1))

        return {
            "op": op_h,
            "machine": machine_h,
            "module": module_h,
            "global": global_h,
        }
