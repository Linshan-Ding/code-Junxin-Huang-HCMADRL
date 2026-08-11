# =========================
# 文件名：utils.py
# =========================
import os
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fuzzy_time_to_stats(fuzzy_time: Tuple[float, float, float]) -> Tuple[float, float]:
    l_value, m_value, u_value = fuzzy_time
    centroid = (l_value + 4.0 * m_value + u_value) / 6.0
    spread = u_value - l_value
    return float(centroid), float(spread)


def realize_fuzzy_time(
    fuzzy_time: Tuple[float, float, float],
    mode: str = "deterministic",
) -> float:
    l_value, m_value, u_value = fuzzy_time
    if mode == "deterministic":
        return fuzzy_time_to_stats(fuzzy_time)[0]
    if mode == "sample":
        return float(random.triangular(l_value, u_value, m_value))
    raise ValueError(f"Unsupported fuzzy time mode: {mode}")


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return float(numerator) / float(denominator)


def to_float_tensor(data: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.float32, device=device)


def to_long_tensor(data: Any, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(data, dtype=torch.long, device=device)


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1-D, got shape {tuple(logits.shape)}")
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1-D, got shape {tuple(mask.shape)}")
    if logits.shape[0] != mask.shape[0]:
        raise ValueError(
            f"logits and mask length mismatch: {logits.shape[0]} vs {mask.shape[0]}"
        )
    return logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)


def categorical_sample_from_logits(
    logits: torch.Tensor,
    deterministic: bool = False,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = torch.distributions.Categorical(logits=logits)
    if deterministic:
        action = int(torch.argmax(logits).item())
        action_tensor = torch.tensor(action, dtype=torch.long, device=logits.device)
    else:
        action_tensor = dist.sample()
        action = int(action_tensor.item())
    log_prob = dist.log_prob(action_tensor)
    entropy = dist.entropy()
    probs = dist.probs
    return action, log_prob, entropy, probs


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(rewards) != len(values) or len(rewards) != len(dones):
        raise ValueError("rewards, values and dones must have the same length")

    advantages = []
    gae = 0.0
    next_value = 0.0

    for step in reversed(range(len(rewards))):
        non_terminal = 0.0 if dones[step] else 1.0
        delta = rewards[step] + gamma * next_value * non_terminal - values[step]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages.insert(0, gae)
        next_value = values[step]

    adv_tensor = torch.tensor(advantages, dtype=torch.float32)
    value_tensor = torch.tensor(values, dtype=torch.float32)
    returns = adv_tensor + value_tensor

    if adv_tensor.numel() > 1:
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std(unbiased=False) + 1e-8)

    return adv_tensor, returns


def flatten_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def as_tuple(value: Any) -> Tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)