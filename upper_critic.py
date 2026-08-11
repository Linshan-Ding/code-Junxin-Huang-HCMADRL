# =========================
# 文件名：upper_critic.py
# 支持单图/批量价值估计
# =========================
import torch
import torch.nn as nn


class UpperCritic(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, machine_h: torch.Tensor, global_h: torch.Tensor) -> torch.Tensor:
        if machine_h.dim() == 3:
            machine_pool = machine_h.mean(dim=1)   # [B,H]
            x = torch.cat([global_h, machine_pool], dim=-1)
            return self.value_mlp(x).squeeze(-1)   # [B]
        machine_pool = machine_h.mean(dim=0)
        x = torch.cat([global_h, machine_pool], dim=-1)
        return self.value_mlp(x).squeeze(-1)
