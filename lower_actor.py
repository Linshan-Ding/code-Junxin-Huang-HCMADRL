# =========================
# 文件名：lower_actor.py
# 支持单步动作打分与批量动作打分
# =========================
from typing import List, Tuple

import torch
import torch.nn as nn

from upper_actor import batch_gather_nodes


class LowerActor(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.action_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_batch(
        self,
        op_h: torch.Tensor,
        machine_h: torch.Tensor,
        global_h: torch.Tensor,
        action_ids: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """批量下层动作打分。

        op_h: [B,O,H]
        machine_h: [B,M,H]
        global_h: [B,H]
        action_ids: [B,Amax,2]，最后一维为 (m, op_idx)
        action_features: [B,Amax,4]
        action_mask: [B,Amax]
        """
        if op_h.dim() == 2:
            op_h = op_h.unsqueeze(0)
            machine_h = machine_h.unsqueeze(0)
            global_h = global_h.unsqueeze(0)
        if action_ids.dim() == 2:
            action_ids = action_ids.unsqueeze(0)
            action_features = action_features.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        m_idx = action_ids[..., 0]
        op_idx = action_ids[..., 1]
        m_h = batch_gather_nodes(machine_h, m_idx)
        o_h = batch_gather_nodes(op_h, op_idx)
        g_h = global_h.unsqueeze(1).expand(-1, action_ids.shape[1], -1)

        x = torch.cat([m_h, o_h, g_h, action_features.to(machine_h.dtype)], dim=-1)
        logits = self.action_mlp(x).squeeze(-1)
        logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits

    def forward(
        self,
        op_h: torch.Tensor,
        machine_h: torch.Tensor,
        global_h: torch.Tensor,
        actions: List[Tuple[int, int, int]],
        env,
    ) -> torch.Tensor:
        """兼容旧接口：单状态、变长 action list。"""
        if len(actions) == 0:
            raise ValueError("LowerActor received empty action list")

        ids = []
        feats = []
        for m, r, j in actions:
            op_idx = env.kind_task_tuple.index((r, j))
            machine_obj = env.machine_dict[m]
            op_obj = env.kind_task_dict[(r, j)]
            current_module = machine_obj.module
            feats.append([
                1.0 if current_module is not None and (r, j) in env.kind_task_a_dict[current_module] else 0.0,
                1.0 if op_obj.buffer > 0 else 0.0,
                1.0 if machine_obj.proc_state == 0 and machine_obj.reconfig_state == 0 else 0.0,
                float(op_obj.unprocessed_count) / max(1.0, float(op_obj.unprocessed_count + op_obj.processed_count)),
            ])
            ids.append([m, op_idx])

        action_ids = torch.tensor(ids, dtype=torch.long, device=machine_h.device)
        action_features = torch.tensor(feats, dtype=machine_h.dtype, device=machine_h.device)
        action_mask = torch.ones(len(actions), dtype=torch.bool, device=machine_h.device)
        return self.forward_batch(op_h, machine_h, global_h, action_ids, action_features, action_mask).squeeze(0)
