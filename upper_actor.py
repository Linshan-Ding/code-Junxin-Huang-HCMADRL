# =========================
# 文件名：upper_actor.py
# 支持单步动作打分与批量动作打分
# =========================
from typing import List, Tuple

import torch
import torch.nn as nn


def batch_gather_nodes(node_h: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """从 [B,N,H] 中按 [B,A] 批量 gather，返回 [B,A,H]。"""
    if node_h.dim() != 3:
        raise ValueError(f"node_h must be [B,N,H], got {tuple(node_h.shape)}")
    if index.dim() != 2:
        raise ValueError(f"index must be [B,A], got {tuple(index.shape)}")
    B, A = index.shape
    H = node_h.shape[-1]
    idx = index.clamp(min=0).unsqueeze(-1).expand(B, A, H)
    return torch.gather(node_h, dim=1, index=idx)


class UpperActor(nn.Module):
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
        machine_h: torch.Tensor,
        module_h: torch.Tensor,
        global_h: torch.Tensor,
        action_ids: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """批量上层动作打分。

        machine_h: [B,M,H]
        module_h: [B,A_module,H]
        global_h: [B,H]
        action_ids: [B,Amax,2]，最后一维为 (m, a)
        action_features: [B,Amax,4]
        action_mask: [B,Amax]
        """
        if machine_h.dim() == 2:
            machine_h = machine_h.unsqueeze(0)
            module_h = module_h.unsqueeze(0)
            global_h = global_h.unsqueeze(0)
        if action_ids.dim() == 2:
            action_ids = action_ids.unsqueeze(0)
            action_features = action_features.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        m_idx = action_ids[..., 0]
        a_idx = action_ids[..., 1]
        m_h = batch_gather_nodes(machine_h, m_idx)
        a_h = batch_gather_nodes(module_h, a_idx)
        g_h = global_h.unsqueeze(1).expand(-1, action_ids.shape[1], -1)

        x = torch.cat([m_h, a_h, g_h, action_features.to(machine_h.dtype)], dim=-1)
        logits = self.action_mlp(x).squeeze(-1)
        logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits

    def forward(
        self,
        machine_h: torch.Tensor,
        module_h: torch.Tensor,
        global_h: torch.Tensor,
        actions: List[Tuple[int, int]],
        env,
    ) -> torch.Tensor:
        """兼容旧接口：单状态、变长 action list。"""
        if len(actions) == 0:
            raise ValueError("UpperActor received empty action list")

        ids = []
        feats = []
        for m, a in actions:
            machine_obj = env.machine_dict[m]
            current_module = machine_obj.module
            feats.append([
                1.0 if current_module == a else 0.0,
                1.0 if machine_obj.proc_state == 1 or machine_obj.reconfig_state == 1 else 0.0,
                1.0 if a in env.machine_module_dict[m] else 0.0,
                1.0 if current_module is None else 0.0,
            ])
            ids.append([m, a])

        action_ids = torch.tensor(ids, dtype=torch.long, device=machine_h.device)
        action_features = torch.tensor(feats, dtype=machine_h.dtype, device=machine_h.device)
        action_mask = torch.ones(len(actions), dtype=torch.bool, device=machine_h.device)
        return self.forward_batch(machine_h, module_h, global_h, action_ids, action_features, action_mask).squeeze(0)
