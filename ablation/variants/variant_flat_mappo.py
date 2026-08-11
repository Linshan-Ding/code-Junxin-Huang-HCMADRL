# ============================================================
# 文件名：ablation/variants/variant_flat_mappo.py
# 变体1：Flat-MAPPO（扁平化单层级结构）
#
# 消融目标：验证层次化双层决策架构相对于扁平化单层级决策的性能增益。
#
# 修改内容：
#   1. 合并 UpperActor + LowerActor → FlatActor（联合输出 (m, a, r, j) 或 (m, a)）
#   2. 合并 UpperCritic + LowerCritic → FlatCritic（联合价值估计）
#   3. FlatMAPPOAgent：单层 Agent，每步直接输出联合动作
#   4. 移除上下层分离的 transition / buffer / update 逻辑
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Union

from config import config
from graph_encoder import FMCO_RHGN
from rl_state import (
    GraphTensor,
    BatchedGraphTensor,
    batch_graph_index_select,
    batch_graph_tensors,
    graph_tensor_from_env,
)
from agent import (
    Transition,
    AgentBatch,
    RolloutBuffer,
    build_agent_batch,
    _cfg,
)
from utils import categorical_sample_from_logits, compute_gae
from upper_actor import batch_gather_nodes


# ============================================================
# FlatActor：扁平化联合动作打分
# ============================================================
class FlatActor(nn.Module):
    """
    扁平化 Actor，同时输出 (m, a, r, j) 或孤立 (m, a) 的动作概率。

    动作空间 = 上层动作空间 ∪ 下层动作空间
    - 纯重构动作: (m, a)，不跟任何工序
    - 联合动作:   (m, a, r, j)，重构+加工
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.action_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 6, hidden_dim),  # machine + module + op + global + 6feat
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_batch(
        self,
        op_h: torch.Tensor,
        machine_h: torch.Tensor,
        module_h: torch.Tensor,
        global_h: torch.Tensor,
        action_ids: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        批量联合动作打分。action_ids: [B, Amax, 4]，最后一维为 (m, a, r, j)。
        对于纯重构动作，(r, j) 位置填充 -1。
        """
        if machine_h.dim() == 2:
            machine_h = machine_h.unsqueeze(0)
            module_h = module_h.unsqueeze(0)
            op_h = op_h.unsqueeze(0)
            global_h = global_h.unsqueeze(0)
        if action_ids.dim() == 2:
            action_ids = action_ids.unsqueeze(0)
            action_features = action_features.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        m_idx = action_ids[..., 0].clamp(min=0)
        a_idx = action_ids[..., 1].clamp(min=0)
        op_idx = action_ids[..., 2].clamp(min=0)

        m_h = batch_gather_nodes(machine_h, m_idx)
        a_h = batch_gather_nodes(module_h, a_idx)
        o_h = batch_gather_nodes(op_h, op_idx)
        g_h = global_h.unsqueeze(1).expand(-1, action_ids.shape[1], -1)

        x = torch.cat([m_h, a_h, o_h, g_h, action_features.to(machine_h.dtype)], dim=-1)
        logits = self.action_mlp(x).squeeze(-1)
        logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits


# ============================================================
# FlatCritic：扁平化状态价值估计
# ============================================================
class FlatCritic(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # op + machine + global
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, op_h, machine_h, global_h):
        if op_h.dim() == 3:
            op_pool = op_h.mean(dim=1)
            machine_pool = machine_h.mean(dim=1)
            x = torch.cat([global_h, machine_pool, op_pool], dim=-1)
            return self.value_mlp(x).squeeze(-1)
        op_pool = op_h.mean(dim=0)
        machine_pool = machine_h.mean(dim=0)
        x = torch.cat([global_h, machine_pool, op_pool], dim=-1)
        return self.value_mlp(x).squeeze(-1)


# ============================================================
# FlatMAPPOAgent：扁平化单层 Agent
# ============================================================
class FlatMAPPOAgent(nn.Module):
    """
    扁平化单层多智能体 Agent。

    与原始 HiFAMRSyncMAGRL 的区别：
    - 原始：上层选 (m,a) → 下层选 (m,r,j)，两层分离
    - Flat：一次选联合动作，可能是纯重构 (m,a) 或重构+加工 (m,a,r,j)
    """

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

        self.actor = FlatActor(hidden_dim=hidden_dim)
        self.critic = FlatCritic(hidden_dim=hidden_dim)

        self.to(device)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters()),
            lr=config.UPPER_LR,
            weight_decay=config.WEIGHT_DECAY,
        )

    def encode(self, graph: Union[GraphTensor, BatchedGraphTensor]):
        return self.encoder(graph.to(self.device))

    @staticmethod
    def _flat_action_tensors(env, actions: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建扁平化动作张量。

        actions 中的每项可能是：
        - (m, a, r, j)：重构+加工联合动作
        - (m, a)：纯重构动作

        action_ids: [A, 4]，每行为 (m, a, r_idx, j_idx)
          对于纯重构动作，(r_idx, j_idx) 填 0
        action_features: [A, 6]，每行为 6 维特征
          [same_module, busy, selectable, empty, has_lower, current_feasible]
        """
        ids = []
        features = []
        for action in actions:
            m = action[0]
            a = action[1]
            machine_obj = env.machine_dict[m]

            same_module = 1.0 if machine_obj.module == a else 0.0
            busy = 1.0 if machine_obj.proc_state == 1 or machine_obj.reconfig_state == 1 else 0.0
            selectable = 1.0 if a in env.machine_module_dict[m] else 0.0
            current_empty = 1.0 if machine_obj.module is None else 0.0

            if len(action) == 5:
                r, j = action[3], action[4]
                r_idx = r
                j_idx = j
                has_lower = 1.0
                current_feasible = (
                    1.0 if (r, j) in env.kind_task_a_dict.get(a, ()) else 0.0
                )
            else:
                r_idx = 0
                j_idx = 0
                has_lower = 0.0
                current_feasible = 0.0

            ids.append([m, a, r_idx, j_idx])
            features.append([same_module, busy, selectable, current_empty, has_lower, current_feasible])

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(features, dtype=torch.float32),
        )

    @staticmethod
    def _build_flat_actions(env) -> List[Tuple]:
        """
        构建扁平化动作空间。

        全局屏蔽：纯重构 (m,a) 仅在全局无任何派发动作时作为兜底，
        只要系统中有任何机器可派发工序，纯重构就不出现在候选列表中。

        与原始层次化版本等价：
        - (m, a, ml, r, j)：重构 m→a + 机器 ml 加工工序 (r,j)
        - (m, a)：纯重构兜底（仅全局无派发时）
        """
        actions = []
        pure_reconfigs = []
        upper_actions = list(env.upper_agent_actions())

        if len(upper_actions) == 0:
            return actions

        for m, a in upper_actions:
            has_reconfig = (env.machine_dict[m].module != a)
            lower_actions = env.lower_agent_actions(m, a)

            if has_reconfig:
                for ml, r, j in lower_actions:
                    actions.append((m, a, ml, r, j))
                if not lower_actions:
                    pure_reconfigs.append((m, a))
            else:
                for ml, r, j in lower_actions:
                    actions.append((m, a, ml, r, j))

        if len(actions) == 0:
            actions.extend(pure_reconfigs)

        if len(actions) == 0:
            actions.extend(upper_actions)

        return actions

    def select_action(
        self,
        env,
        deterministic: bool = False,
    ) -> Tuple[Optional[Tuple], Optional[Transition]]:
        graph = graph_tensor_from_env(env, self.device)
        actions = self._build_flat_actions(env)

        if len(actions) == 0:
            return None, None

        action_ids_cpu, action_features_cpu = self._flat_action_tensors(env, actions)
        action_ids = action_ids_cpu.to(self.device)
        action_features = action_features_cpu.to(self.device)
        action_mask = torch.ones(action_ids.shape[0], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            emb = self.encode(graph)
            logits = self.actor.forward_batch(
                op_h=emb["op"],
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
            value = self.critic(
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

    def _batch_loss(self, mb: AgentBatch) -> torch.Tensor:
        emb = self.encoder(mb.graph)
        logits = self.actor.forward_batch(
            op_h=emb["op"],
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
        surr2 = torch.clamp(ratio, 1.0 - config.PPO_CLIP, 1.0 + config.PPO_CLIP) * mb.advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value = self.critic(
            op_h=emb["op"],
            machine_h=emb["machine"],
            global_h=emb["global"],
        )
        value_loss = F.mse_loss(value, mb.returns)
        entropy_loss = -entropy.mean()

        return policy_loss + config.VALUE_COEF * value_loss + config.ENTROPY_COEF * entropy_loss

    @staticmethod
    def _mini_batch_indices(n, batch_size, shuffle, device):
        if n <= 0:
            return
        if batch_size <= 0:
            batch_size = n
        order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
        for start in range(0, n, batch_size):
            yield order[start:start + batch_size]

    def update(self, buffer: RolloutBuffer, env=None) -> dict:
        """
        扁平化 PPO 更新：合并所有 transition 进行一次 update。
        """
        logs = {
            "loss": 0.0,
            "transition_count": buffer.transition_count,
            "minibatches": 0,
            "update_time_sec": 0.0,
        }

        import time
        t_start = time.perf_counter()

        # 合并上层和下层 transition
        all_transitions = buffer.upper + buffer.lower
        if len(all_transitions) == 0:
            return logs

        rewards = [t.reward for t in all_transitions]
        values = [t.value for t in all_transitions]
        dones = [t.done for t in all_transitions]

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=config.GAMMA_U,
            gae_lambda=config.GAE_LAMBDA_U,
        )

        batch = build_agent_batch(all_transitions, advantages, returns, self.device)

        update_batch_size = int(_cfg("UPDATE_BATCH_SIZE", 64))
        update_shuffle = bool(_cfg("UPDATE_SHUFFLE", True))
        n = len(all_transitions)

        for _ in range(config.PPO_EPOCHS):
            for idx in self._mini_batch_indices(n, update_batch_size, update_shuffle, self.device):
                mb = batch.index_select(idx)
                self.optimizer.zero_grad()
                loss = self._batch_loss(mb)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters())
                    + list(self.actor.parameters())
                    + list(self.critic.parameters()),
                    config.MAX_GRAD_NORM,
                )
                self.optimizer.step()
                logs["loss"] = float(loss.item())
                logs["minibatches"] += 1

        logs["update_time_sec"] = time.perf_counter() - t_start
        return logs
