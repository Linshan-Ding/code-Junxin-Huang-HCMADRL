# =========================
# 文件名：agent.py
# 多 worker rollout + 真正 GPU mini-batch PPO 更新
# =========================
from dataclasses import dataclass
import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from graph_encoder import FMCO_RHGN
from lower_actor import LowerActor
from lower_critic import LowerCritic
from rl_state import (
    GraphTensor,
    BatchedGraphTensor,
    batch_graph_index_select,
    batch_graph_tensors,
    graph_tensor_from_env,
)
from upper_actor import UpperActor
from upper_critic import UpperCritic
from utils import categorical_sample_from_logits, compute_gae


@dataclass
class Transition:
    """一条 PPO transition。

    为了实现 PDF 式 batch update，这里在采样时就固定动作元数据：
    - upper: action_ids 每行为 (m, a)
    - lower: action_ids 每行为 (m, op_idx)
    - action_features 为 actor 原来依赖 env 计算的 4 维动作附加特征

    更新阶段不再依赖 env，也不再逐 transition 调 actor/critic。
    """
    graph: GraphTensor
    actions: List[Tuple]
    action_ids: torch.Tensor          # [A,2]
    action_features: torch.Tensor     # [A,4]
    action_index: int
    old_log_prob: float
    value: float
    reward: float
    done: bool


@dataclass
class AgentBatch:
    graph: BatchedGraphTensor
    action_ids: torch.Tensor          # [N,Amax,2]
    action_features: torch.Tensor     # [N,Amax,4]
    action_mask: torch.Tensor         # [N,Amax]
    action_index: torch.Tensor        # [N]
    old_log_probs: torch.Tensor       # [N]
    advantages: torch.Tensor          # [N]
    returns: torch.Tensor             # [N]

    def to(self, device: torch.device) -> "AgentBatch":
        return AgentBatch(
            graph=self.graph.to(device),
            action_ids=self.action_ids.to(device),
            action_features=self.action_features.to(device),
            action_mask=self.action_mask.to(device),
            action_index=self.action_index.to(device),
            old_log_probs=self.old_log_probs.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
        )

    def index_select(self, idx: torch.Tensor) -> "AgentBatch":
        idx_graph = idx.to(self.graph.op_x.device)
        idx_tensor = idx.to(self.action_ids.device)
        return AgentBatch(
            graph=batch_graph_index_select(self.graph, idx_graph),
            action_ids=self.action_ids.index_select(0, idx_tensor),
            action_features=self.action_features.index_select(0, idx_tensor),
            action_mask=self.action_mask.index_select(0, idx_tensor),
            action_index=self.action_index.index_select(0, idx_tensor),
            old_log_probs=self.old_log_probs.index_select(0, idx_tensor),
            advantages=self.advantages.index_select(0, idx_tensor),
            returns=self.returns.index_select(0, idx_tensor),
        )


class RolloutBuffer:
    def __init__(self):
        self.upper: List[Transition] = []
        self.lower: List[Transition] = []

    def clear(self) -> None:
        self.upper.clear()
        self.lower.clear()

    def extend(self, other: "RolloutBuffer") -> None:
        self.upper.extend(other.upper)
        self.lower.extend(other.lower)

    @property
    def transition_count(self) -> int:
        return len(self.upper) + len(self.lower)


def _cfg(name: str, default):
    return getattr(config, name, default)


def build_agent_batch(
    transitions: List[Transition],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    device: torch.device,
) -> AgentBatch:
    """vectorize_buffer：把 transition list 整理为统一 batch tensor。"""
    if len(transitions) == 0:
        raise ValueError("build_agent_batch received empty transitions")

    graph_batch = batch_graph_tensors([t.graph for t in transitions])

    max_actions = max(int(t.action_ids.shape[0]) for t in transitions)
    id_dim = int(transitions[0].action_ids.shape[1])
    feat_dim = int(transitions[0].action_features.shape[1])
    n = len(transitions)

    action_ids = torch.full((n, max_actions, id_dim), -1, dtype=torch.long)
    action_features = torch.zeros((n, max_actions, feat_dim), dtype=torch.float32)
    action_mask = torch.zeros((n, max_actions), dtype=torch.bool)

    for i, t in enumerate(transitions):
        a_count = int(t.action_ids.shape[0])
        action_ids[i, :a_count] = t.action_ids.long()
        action_features[i, :a_count] = t.action_features.float()
        action_mask[i, :a_count] = True

    batch = AgentBatch(
        graph=graph_batch,
        action_ids=action_ids,
        action_features=action_features,
        action_mask=action_mask,
        action_index=torch.tensor([t.action_index for t in transitions], dtype=torch.long),
        old_log_probs=torch.tensor([t.old_log_prob for t in transitions], dtype=torch.float32),
        advantages=advantages.float(),
        returns=returns.float(),
    )
    return batch.to(device)


class HiFAMRSyncMAGRL(nn.Module):
    def __init__(
        self,
        op_input_dim: int,
        machine_input_dim: int,
        module_input_dim: int,
        hidden_dim: int = 128,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device

        self.encoder = FMCO_RHGN(
            op_input_dim=op_input_dim,
            machine_input_dim=machine_input_dim,
            module_input_dim=module_input_dim,
            hidden_dim=hidden_dim,
            dropout=config.DROPOUT,
        )

        self.upper_actor = UpperActor(hidden_dim=hidden_dim)
        self.upper_critic = UpperCritic(hidden_dim=hidden_dim)

        self.lower_actor = LowerActor(hidden_dim=hidden_dim)
        self.lower_critic = LowerCritic(hidden_dim=hidden_dim)

        self.to(device)

        self.upper_optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.upper_actor.parameters())
            + list(self.upper_critic.parameters()),
            lr=config.UPPER_LR,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.lower_optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.lower_actor.parameters())
            + list(self.lower_critic.parameters()),
            lr=config.LOWER_LR,
            weight_decay=config.WEIGHT_DECAY,
        )

    def encode(self, graph: Union[GraphTensor, BatchedGraphTensor]):
        return self.encoder(graph.to(self.device))

    @staticmethod
    def _upper_action_tensors(env, actions: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = []
        features = []
        for m, a in actions:
            machine_obj = env.machine_dict[m]
            current_module = machine_obj.module

            same_module = 1.0 if current_module == a else 0.0
            busy = 1.0 if machine_obj.proc_state == 1 or machine_obj.reconfig_state == 1 else 0.0
            selectable_rate = 1.0 if a in env.machine_module_dict[m] else 0.0
            current_empty = 1.0 if current_module is None else 0.0

            ids.append([m, a])
            features.append([same_module, busy, selectable_rate, current_empty])

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(features, dtype=torch.float32),
        )

    @staticmethod
    def _lower_action_tensors(env, actions: List[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = []
        features = []
        for m, r, j in actions:
            op_idx = env.kind_task_tuple.index((r, j))
            machine_obj = env.machine_dict[m]
            op_obj = env.kind_task_dict[(r, j)]
            current_module = machine_obj.module

            current_feasible = (
                1.0
                if current_module is not None and (r, j) in env.kind_task_a_dict[current_module]
                else 0.0
            )
            ready = 1.0 if op_obj.buffer > 0 else 0.0
            machine_idle = (
                1.0
                if machine_obj.proc_state == 0 and machine_obj.reconfig_state == 0
                else 0.0
            )
            remaining_rate = (
                float(op_obj.unprocessed_count)
                / max(1.0, float(op_obj.unprocessed_count + op_obj.processed_count))
            )

            ids.append([m, op_idx])
            features.append([current_feasible, ready, machine_idle, remaining_rate])

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(features, dtype=torch.float32),
        )

    def select_upper_action(
        self,
        env,
        deterministic: bool = False,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Transition]]:
        graph = graph_tensor_from_env(env, self.device)
        actions = list(env.upper_agent_actions())

        if len(actions) == 0:
            return None, None

        action_ids_cpu, action_features_cpu = self._upper_action_tensors(env, actions)
        action_ids = action_ids_cpu.to(self.device)
        action_features = action_features_cpu.to(self.device)
        action_mask = torch.ones(action_ids.shape[0], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            emb = self.encode(graph)
            logits = self.upper_actor.forward_batch(
                machine_h=emb["machine"],
                module_h=emb["module"],
                global_h=emb["global"],
                action_ids=action_ids,
                action_features=action_features,
                action_mask=action_mask,
            ).squeeze(0)
            action_index, log_prob, _, _ = categorical_sample_from_logits(
                logits,
                deterministic=deterministic,
            )
            value = self.upper_critic(
                machine_h=emb["machine"],
                global_h=emb["global"],
            )

        transition = Transition(
            graph=graph.detach_cpu(),
            actions=actions,
            action_ids=action_ids_cpu.cpu(),
            action_features=action_features_cpu.cpu(),
            action_index=action_index,
            old_log_prob=float(log_prob.item()),
            value=float(value.item()),
            reward=0.0,
            done=False,
        )

        return actions[action_index], transition

    def select_lower_action(
        self,
        env,
        upper_action: Tuple[int, int],
        deterministic: bool = False,
    ) -> Tuple[Optional[Tuple[int, int, int]], Optional[Transition]]:
        graph = graph_tensor_from_env(env, self.device)
        actions = list(env.lower_agent_actions(upper_action[0], upper_action[1]))

        if len(actions) == 0:
            return None, None

        action_ids_cpu, action_features_cpu = self._lower_action_tensors(env, actions)
        action_ids = action_ids_cpu.to(self.device)
        action_features = action_features_cpu.to(self.device)
        action_mask = torch.ones(action_ids.shape[0], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            emb = self.encode(graph)
            logits = self.lower_actor.forward_batch(
                op_h=emb["op"],
                machine_h=emb["machine"],
                global_h=emb["global"],
                action_ids=action_ids,
                action_features=action_features,
                action_mask=action_mask,
            ).squeeze(0)
            action_index, log_prob, _, _ = categorical_sample_from_logits(
                logits,
                deterministic=deterministic,
            )
            value = self.lower_critic(
                op_h=emb["op"],
                machine_h=emb["machine"],
                global_h=emb["global"],
            )

        transition = Transition(
            graph=graph.detach_cpu(),
            actions=actions,
            action_ids=action_ids_cpu.cpu(),
            action_features=action_features_cpu.cpu(),
            action_index=action_index,
            old_log_prob=float(log_prob.item()),
            value=float(value.item()),
            reward=0.0,
            done=False,
        )

        return actions[action_index], transition

    def _upper_batch_loss(self, mb: AgentBatch) -> torch.Tensor:
        emb = self.encoder(mb.graph)
        logits = self.upper_actor.forward_batch(
            machine_h=emb["machine"],
            module_h=emb["module"],
            global_h=emb["global"],
            action_ids=mb.action_ids,
            action_features=mb.action_features,
            action_mask=mb.action_mask,
        )

        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(mb.action_index)
        entropy = dist.entropy()

        ratio = torch.exp(log_prob - mb.old_log_probs)
        surr1 = ratio * mb.advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - config.PPO_CLIP,
            1.0 + config.PPO_CLIP,
        ) * mb.advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value = self.upper_critic(machine_h=emb["machine"], global_h=emb["global"])
        value_loss = F.mse_loss(value, mb.returns)
        entropy_loss = -entropy.mean()

        return policy_loss + config.VALUE_COEF * value_loss + config.ENTROPY_COEF * entropy_loss

    def _lower_batch_loss(self, mb: AgentBatch) -> torch.Tensor:
        emb = self.encoder(mb.graph)
        logits = self.lower_actor.forward_batch(
            op_h=emb["op"],
            machine_h=emb["machine"],
            global_h=emb["global"],
            action_ids=mb.action_ids,
            action_features=mb.action_features,
            action_mask=mb.action_mask,
        )

        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(mb.action_index)
        entropy = dist.entropy()

        ratio = torch.exp(log_prob - mb.old_log_probs)
        surr1 = ratio * mb.advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - config.PPO_CLIP,
            1.0 + config.PPO_CLIP,
        ) * mb.advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value = self.lower_critic(op_h=emb["op"], machine_h=emb["machine"], global_h=emb["global"])
        value_loss = F.mse_loss(value, mb.returns)
        entropy_loss = -entropy.mean()

        return policy_loss + config.VALUE_COEF * value_loss + config.ENTROPY_COEF * entropy_loss

    @staticmethod
    def _mini_batch_indices(n: int, batch_size: int, shuffle: bool, device: torch.device):
        if n <= 0:
            return
        if batch_size <= 0:
            batch_size = n
        order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
        for start in range(0, n, batch_size):
            yield order[start:start + batch_size]

    def update(self, buffer: RolloutBuffer, env=None) -> dict:
        """PDF 式 update：
        1. 合并多个 worker 的 rollout buffer；
        2. 一次性 vectorize 成 batch tensor 并搬到 device；
        3. PPO epoch 内随机打乱；
        4. 每个 mini-batch 一次 HGNN/Actor/Critic 批量前向与一次 backward。
        """
        logs = {
            "upper_loss": 0.0,
            "lower_loss": 0.0,
            "upper_transition_count": len(buffer.upper),
            "lower_transition_count": len(buffer.lower),
            "upper_minibatches": 0,
            "lower_minibatches": 0,
            "batch_prepare_time_sec": 0.0,
            "upper_update_time_sec": 0.0,
            "lower_update_time_sec": 0.0,
            "update_time_sec": 0.0,
        }

        update_batch_size = int(_cfg("UPDATE_BATCH_SIZE", 64))
        update_shuffle = bool(_cfg("UPDATE_SHUFFLE", True))
        t_update0 = time.perf_counter()

        if len(buffer.upper) > 0:
            rewards = [t.reward for t in buffer.upper]
            values = [t.value for t in buffer.upper]
            dones = [t.done for t in buffer.upper]
            advantages, returns = compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                gamma=config.GAMMA_U,
                gae_lambda=config.GAE_LAMBDA_U,
            )

            t0 = time.perf_counter()
            upper_batch = build_agent_batch(buffer.upper, advantages, returns, self.device)
            logs["batch_prepare_time_sec"] += time.perf_counter() - t0
            n_upper = len(buffer.upper)

            for _ in range(config.PPO_EPOCHS):
                for idx in self._mini_batch_indices(n_upper, update_batch_size, update_shuffle, self.device):
                    mb = upper_batch.index_select(idx)
                    t_mb = time.perf_counter()
                    self.upper_optimizer.zero_grad()
                    upper_loss = self._upper_batch_loss(mb)
                    upper_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters())
                        + list(self.upper_actor.parameters())
                        + list(self.upper_critic.parameters()),
                        config.MAX_GRAD_NORM,
                    )
                    self.upper_optimizer.step()
                    logs["upper_update_time_sec"] += time.perf_counter() - t_mb
                    logs["upper_loss"] = float(upper_loss.item())
                    logs["upper_minibatches"] += 1

        if len(buffer.lower) > 0:
            rewards = [t.reward for t in buffer.lower]
            values = [t.value for t in buffer.lower]
            dones = [t.done for t in buffer.lower]
            advantages, returns = compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                gamma=config.GAMMA_L,
                gae_lambda=config.GAE_LAMBDA_L,
            )

            t0 = time.perf_counter()
            lower_batch = build_agent_batch(buffer.lower, advantages, returns, self.device)
            logs["batch_prepare_time_sec"] += time.perf_counter() - t0
            n_lower = len(buffer.lower)

            for _ in range(config.PPO_EPOCHS):
                for idx in self._mini_batch_indices(n_lower, update_batch_size, update_shuffle, self.device):
                    mb = lower_batch.index_select(idx)
                    t_mb = time.perf_counter()
                    self.lower_optimizer.zero_grad()
                    lower_loss = self._lower_batch_loss(mb)
                    lower_loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters())
                        + list(self.lower_actor.parameters())
                        + list(self.lower_critic.parameters()),
                        config.MAX_GRAD_NORM,
                    )
                    self.lower_optimizer.step()
                    logs["lower_update_time_sec"] += time.perf_counter() - t_mb
                    logs["lower_loss"] = float(lower_loss.item())
                    logs["lower_minibatches"] += 1

        logs["update_time_sec"] = time.perf_counter() - t_update0
        return logs
