# =========================
# 文件名：graph_encoder.py
# 支持单图 forward 与 [B, ...] 批量 forward 的 HGNN Encoder
# =========================
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_state import GraphTensor, BatchedGraphTensor


def _segment_softmax(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """单图分段 softmax。scores: [E]。"""
    if scores.numel() == 0:
        return scores
    output = torch.zeros_like(scores)
    unique_index = torch.unique(index)
    for idx in unique_index:
        mask = index == idx
        output[mask] = F.softmax(scores[mask], dim=0)
    return output


def _segment_softmax_batched(scores: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """批量分段 softmax。scores: [B, E]，同一目标节点的入边在 E 维上归一化。"""
    if scores.numel() == 0:
        return scores
    output = torch.zeros_like(scores)
    unique_index = torch.unique(index)
    for idx in unique_index:
        mask = index == idx
        output[:, mask] = F.softmax(scores[:, mask], dim=1)
    return output


class EdgeAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.message = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        src_h: torch.Tensor,
        dst_h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_dst: int,
    ) -> torch.Tensor:
        """支持两种输入：
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

        edge_h = self.edge_proj(edge_attr)       # [B, E, H]
        src_feat = src_h[:, src, :]              # [B, E, H]
        dst_feat = dst_h[:, dst, :]              # [B, E, H]

        score_input = torch.cat([dst_feat, src_feat, edge_h], dim=-1)
        raw_scores = self.score(score_input).squeeze(-1)  # [B, E]
        attn = _segment_softmax_batched(raw_scores, dst, num_dst)

        msg_input = torch.cat([src_feat, edge_h], dim=-1)
        msg = self.message(msg_input) * attn.unsqueeze(-1)  # [B, E, H]

        agg = torch.zeros_like(dst_h)
        agg.index_add_(1, dst, msg)
        out = self.norm(dst_h + agg)

        if single_graph:
            out = out.squeeze(0)
        return out


class FMCO_RHGN(nn.Module):
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

        self.o2o_layer = EdgeAttentionLayer(hidden_dim=hidden_dim, edge_dim=4)
        self.m2a_to_machine_layer = EdgeAttentionLayer(hidden_dim=hidden_dim, edge_dim=8)
        self.o2a_to_operation_layer = EdgeAttentionLayer(hidden_dim=hidden_dim, edge_dim=4)
        self.o2m_to_machine_layer = EdgeAttentionLayer(hidden_dim=hidden_dim, edge_dim=5)

        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, graph: Union[GraphTensor, BatchedGraphTensor]) -> Dict[str, torch.Tensor]:
        op_h0 = self.op_proj(graph.op_x)
        machine_h0 = self.machine_proj(graph.machine_x)
        module_h0 = self.module_proj(graph.module_x)

        op_h1 = self.o2o_layer(
            src_h=op_h0,
            dst_h=op_h0,
            edge_index=graph.o2o_edge_index,
            edge_attr=graph.o2o_edge_attr,
            num_dst=graph.op_count,
        )

        m2a_reverse = torch.stack(
            [graph.m2a_edge_index[1], graph.m2a_edge_index[0]],
            dim=0,
        ) if graph.m2a_edge_index.numel() > 0 else graph.m2a_edge_index

        machine_h1 = self.m2a_to_machine_layer(
            src_h=module_h0,
            dst_h=machine_h0,
            edge_index=m2a_reverse,
            edge_attr=graph.m2a_edge_attr,
            num_dst=graph.machine_count,
        )

        o2a_reverse = torch.stack(
            [graph.o2a_edge_index[1], graph.o2a_edge_index[0]],
            dim=0,
        ) if graph.o2a_edge_index.numel() > 0 else graph.o2a_edge_index

        op_h2 = self.o2a_to_operation_layer(
            src_h=module_h0,
            dst_h=op_h1,
            edge_index=o2a_reverse,
            edge_attr=graph.o2a_edge_attr,
            num_dst=graph.op_count,
        )

        machine_h2 = self.o2m_to_machine_layer(
            src_h=op_h2,
            dst_h=machine_h1,
            edge_index=graph.o2m_edge_index,
            edge_attr=graph.o2m_edge_attr,
            num_dst=graph.machine_count,
        )

        if op_h2.dim() == 3:
            op_pool = op_h2.mean(dim=1)          # [B,H]
            machine_pool = machine_h2.mean(dim=1)
            module_pool = module_h0.mean(dim=1)
        else:
            op_pool = op_h2.mean(dim=0)          # [H]
            machine_pool = machine_h2.mean(dim=0)
            module_pool = module_h0.mean(dim=0)

        global_h = self.global_proj(torch.cat([op_pool, machine_pool, module_pool], dim=-1))

        return {
            "op": op_h2,
            "machine": machine_h2,
            "module": module_h0,
            "global": global_h,
        }
