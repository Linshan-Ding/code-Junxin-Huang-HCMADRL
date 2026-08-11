# =========================
# 文件名：lower_critic.py
# 支持单图/批量价值估计
# =========================
import torch
import torch.nn as nn


class LowerCritic(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        op_h: torch.Tensor,
        machine_h: torch.Tensor,
        global_h: torch.Tensor,
    ) -> torch.Tensor:
        if op_h.dim() == 3:
            op_pool = op_h.mean(dim=1)             # [B,H]
            machine_pool = machine_h.mean(dim=1)
            x = torch.cat([global_h, machine_pool, op_pool], dim=-1)
            return self.value_mlp(x).squeeze(-1)   # [B]
        op_pool = op_h.mean(dim=0)
        machine_pool = machine_h.mean(dim=0)
        x = torch.cat([global_h, machine_pool, op_pool], dim=-1)
        return self.value_mlp(x).squeeze(-1)
