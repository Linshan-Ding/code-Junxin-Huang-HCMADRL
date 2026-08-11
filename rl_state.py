# =========================
# 文件名：rl_state.py
# =========================
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from utils import safe_div, to_float_tensor, to_long_tensor
import Triangular_fuzzy as tf


@dataclass
class GraphTensor:
    op_x: torch.Tensor
    machine_x: torch.Tensor
    module_x: torch.Tensor

    o2o_edge_index: torch.Tensor
    o2m_edge_index: torch.Tensor
    o2a_edge_index: torch.Tensor
    m2a_edge_index: torch.Tensor

    o2o_edge_attr: torch.Tensor
    o2m_edge_attr: torch.Tensor
    o2a_edge_attr: torch.Tensor
    m2a_edge_attr: torch.Tensor

    op_count: int
    machine_count: int
    module_count: int

    def to(self, device: torch.device) -> "GraphTensor":
        return GraphTensor(
            op_x=self.op_x.to(device),
            machine_x=self.machine_x.to(device),
            module_x=self.module_x.to(device),
            o2o_edge_index=self.o2o_edge_index.to(device),
            o2m_edge_index=self.o2m_edge_index.to(device),
            o2a_edge_index=self.o2a_edge_index.to(device),
            m2a_edge_index=self.m2a_edge_index.to(device),
            o2o_edge_attr=self.o2o_edge_attr.to(device),
            o2m_edge_attr=self.o2m_edge_attr.to(device),
            o2a_edge_attr=self.o2a_edge_attr.to(device),
            m2a_edge_attr=self.m2a_edge_attr.to(device),
            op_count=self.op_count,
            machine_count=self.machine_count,
            module_count=self.module_count,
        )

    def detach_cpu(self) -> "GraphTensor":
        return GraphTensor(
            op_x=self.op_x.detach().cpu(),
            machine_x=self.machine_x.detach().cpu(),
            module_x=self.module_x.detach().cpu(),
            o2o_edge_index=self.o2o_edge_index.detach().cpu(),
            o2m_edge_index=self.o2m_edge_index.detach().cpu(),
            o2a_edge_index=self.o2a_edge_index.detach().cpu(),
            m2a_edge_index=self.m2a_edge_index.detach().cpu(),
            o2o_edge_attr=self.o2o_edge_attr.detach().cpu(),
            o2m_edge_attr=self.o2m_edge_attr.detach().cpu(),
            o2a_edge_attr=self.o2a_edge_attr.detach().cpu(),
            m2a_edge_attr=self.m2a_edge_attr.detach().cpu(),
            op_count=self.op_count,
            machine_count=self.machine_count,
            module_count=self.module_count,
        )


@dataclass
class BatchedGraphTensor:
    """多个 GraphTensor 堆叠后的 batch 版本。

    节点和边属性带有 batch 维度：
    - op_x: [B, O, F_op]
    - machine_x: [B, M, F_m]
    - module_x: [B, A, F_a]
    - *_edge_attr: [B, E, F_e]

    edge_index 在同一个算例内拓扑一致，因此共享一份 [2, E]。
    """
    op_x: torch.Tensor
    machine_x: torch.Tensor
    module_x: torch.Tensor

    o2o_edge_index: torch.Tensor
    o2m_edge_index: torch.Tensor
    o2a_edge_index: torch.Tensor
    m2a_edge_index: torch.Tensor

    o2o_edge_attr: torch.Tensor
    o2m_edge_attr: torch.Tensor
    o2a_edge_attr: torch.Tensor
    m2a_edge_attr: torch.Tensor

    op_count: int
    machine_count: int
    module_count: int

    def to(self, device: torch.device) -> "BatchedGraphTensor":
        return BatchedGraphTensor(
            op_x=self.op_x.to(device),
            machine_x=self.machine_x.to(device),
            module_x=self.module_x.to(device),
            o2o_edge_index=self.o2o_edge_index.to(device),
            o2m_edge_index=self.o2m_edge_index.to(device),
            o2a_edge_index=self.o2a_edge_index.to(device),
            m2a_edge_index=self.m2a_edge_index.to(device),
            o2o_edge_attr=self.o2o_edge_attr.to(device),
            o2m_edge_attr=self.o2m_edge_attr.to(device),
            o2a_edge_attr=self.o2a_edge_attr.to(device),
            m2a_edge_attr=self.m2a_edge_attr.to(device),
            op_count=self.op_count,
            machine_count=self.machine_count,
            module_count=self.module_count,
        )


def _assert_same_graph_shape(graphs: List[GraphTensor]) -> None:
    if not graphs:
        raise ValueError("batch_graph_tensors received empty graph list")
    g0 = graphs[0]
    ref = (
        tuple(g0.op_x.shape),
        tuple(g0.machine_x.shape),
        tuple(g0.module_x.shape),
        tuple(g0.o2o_edge_index.shape),
        tuple(g0.o2m_edge_index.shape),
        tuple(g0.o2a_edge_index.shape),
        tuple(g0.m2a_edge_index.shape),
        tuple(g0.o2o_edge_attr.shape),
        tuple(g0.o2m_edge_attr.shape),
        tuple(g0.o2a_edge_attr.shape),
        tuple(g0.m2a_edge_attr.shape),
    )
    for i, g in enumerate(graphs[1:], start=1):
        cur = (
            tuple(g.op_x.shape),
            tuple(g.machine_x.shape),
            tuple(g.module_x.shape),
            tuple(g.o2o_edge_index.shape),
            tuple(g.o2m_edge_index.shape),
            tuple(g.o2a_edge_index.shape),
            tuple(g.m2a_edge_index.shape),
            tuple(g.o2o_edge_attr.shape),
            tuple(g.o2m_edge_attr.shape),
            tuple(g.o2a_edge_attr.shape),
            tuple(g.m2a_edge_attr.shape),
        )
        if cur != ref:
            raise ValueError(
                "当前批量化更新要求同一 mini-batch 内图规模一致。"
                f"第 0 个图 shape={ref}，第 {i} 个图 shape={cur}。"
                "如果要混合不同规模算例，需要先按算例规模分组，或进一步实现节点 padding。"
            )


def batch_graph_tensors(graphs: List[GraphTensor]) -> BatchedGraphTensor:
    """把同一算例下的多个 GraphTensor 堆叠成 BatchedGraphTensor。"""
    _assert_same_graph_shape(graphs)
    g0 = graphs[0]
    return BatchedGraphTensor(
        op_x=torch.stack([g.op_x for g in graphs], dim=0),
        machine_x=torch.stack([g.machine_x for g in graphs], dim=0),
        module_x=torch.stack([g.module_x for g in graphs], dim=0),
        o2o_edge_index=g0.o2o_edge_index,
        o2m_edge_index=g0.o2m_edge_index,
        o2a_edge_index=g0.o2a_edge_index,
        m2a_edge_index=g0.m2a_edge_index,
        o2o_edge_attr=torch.stack([g.o2o_edge_attr for g in graphs], dim=0),
        o2m_edge_attr=torch.stack([g.o2m_edge_attr for g in graphs], dim=0),
        o2a_edge_attr=torch.stack([g.o2a_edge_attr for g in graphs], dim=0),
        m2a_edge_attr=torch.stack([g.m2a_edge_attr for g in graphs], dim=0),
        op_count=g0.op_count,
        machine_count=g0.machine_count,
        module_count=g0.module_count,
    )


def batch_graph_index_select(graph: BatchedGraphTensor, idx: torch.Tensor) -> BatchedGraphTensor:
    """在 batch 维度上选择 mini-batch。idx 可以在 CPU 或 GPU。"""
    idx = idx.to(graph.op_x.device)
    return BatchedGraphTensor(
        op_x=graph.op_x.index_select(0, idx),
        machine_x=graph.machine_x.index_select(0, idx),
        module_x=graph.module_x.index_select(0, idx),
        o2o_edge_index=graph.o2o_edge_index,
        o2m_edge_index=graph.o2m_edge_index,
        o2a_edge_index=graph.o2a_edge_index,
        m2a_edge_index=graph.m2a_edge_index,
        o2o_edge_attr=graph.o2o_edge_attr.index_select(0, idx),
        o2m_edge_attr=graph.o2m_edge_attr.index_select(0, idx),
        o2a_edge_attr=graph.o2a_edge_attr.index_select(0, idx),
        m2a_edge_attr=graph.m2a_edge_attr.index_select(0, idx),
        op_count=graph.op_count,
        machine_count=graph.machine_count,
        module_count=graph.module_count,
    )


def _max_positive(values: List[float], default: float = 1.0) -> float:
    positives = [float(v) for v in values if float(v) > 0]
    return max(positives) if positives else default


def _operation_feature(env, r: int, j: int) -> List[float]:
    obj = env.kind_task_dict[(r, j)]

    max_buffer = _max_positive(
        [env.kind_task_dict[o].buffer for o in env.kind_task_tuple],
        default=1.0,
    )
    total_initial = max(1.0, float(env.sum_kind_task_count[r]))
    task_count = max(env.task_r_dict[r]) + 1

    job_now_num = safe_div(obj.buffer, max_buffer)
    processed_rate = safe_div(obj.processed_count, obj.processed_count + obj.unprocessed_count)
    pre_rj_num = safe_div(j, task_count)
    remain_rj_num = safe_div(max(env.task_r_dict[r]) - j, task_count)

    machine_num = safe_div(len(env.machine_rj_dict[(r, j)]), env.machine_count)

    feasible_times = []
    feasible_machine_count = 0
    for m in env.machine_tuple:
        machine_obj = env.machine_dict[m]
        current_module = machine_obj.module
        if current_module is not None and (r, j) in env.kind_task_a_dict[current_module]:
            feasible_machine_count += 1
            feasible_times.append(float(env.time_rja_dict[(r, j)][current_module]))

    machine_module_num = safe_div(feasible_machine_count, env.machine_count)
    max_time = _max_positive(feasible_times, default=1.0)
    proc_min_time = safe_div(min(feasible_times), max_time) if feasible_times else 0.0
    proc_avg_time = safe_div(sum(feasible_times) / len(feasible_times), max_time) if feasible_times else 0.0

    ready_flag = 1.0 if obj.buffer > 0 else 0.0
    completed_rate = safe_div(obj.processed_count, total_initial)
    unprocessed_rate = safe_div(obj.unprocessed_count, total_initial)

    return [
        job_now_num,
        processed_rate,
        pre_rj_num,
        remain_rj_num,
        machine_num,
        machine_module_num,
        proc_min_time,
        proc_avg_time,
        ready_flag,
        completed_rate,
        unprocessed_rate,
    ]


def _machine_feature(env, m: int) -> List[float]:
    machine_obj = env.machine_dict[m]
    current_module = machine_obj.module

    is_busy = 1.0 if machine_obj.proc_state == 1 or machine_obj.reconfig_state == 1 else 0.0
    is_processing = float(machine_obj.proc_state)
    is_reconfigure = float(machine_obj.reconfig_state)

    remain_proc_den = max(1.0, float(machine_obj.proc_time))
    remain_reconfig_den = max(1.0, float(machine_obj.reconfig_time))

    remain_proc_time = safe_div(machine_obj.remain_proc_time, remain_proc_den)
    remain_reconfig_time = safe_div(machine_obj.remain_reconfig_time, remain_reconfig_den)

    module_onehot = [0.0] * env.module_count
    if current_module is not None:
        module_onehot[current_module] = 1.0

    installed_module_rate = 1.0 if current_module is not None else 0.0

    if current_module is not None:
        opt_tuple = tuple(
            o for o in env.kind_task_a_dict[current_module]
            if env.kind_task_dict[o].unprocessed_count > 0
        )
        remain_operation_num = safe_div(len(opt_tuple), len(env.kind_task_tuple))
        proc_times = [float(env.time_rja_dict[o][current_module]) for o in opt_tuple]
        max_proc = _max_positive(proc_times, default=1.0)
        proc_avg_time = safe_div(sum(proc_times) / len(proc_times), max_proc) if proc_times else 0.0
        task_machine_mismatch = safe_div(len(env.kind_task_a_dict[current_module]), len(env.kind_task_tuple))
    else:
        remain_operation_num = 0.0
        proc_avg_time = 0.0
        task_machine_mismatch = 0.0

    selectable_modules = env.machine_module_dict[m]
    reconfig_times = []
    reconfig_spreads = []
    reconfig_costs = []

    if current_module is None:
        for a in selectable_modules:
            reconfig_times.append(tf.Triangular_fuzzy_time(env.time_add_dict[a], (0, 0, 0)))
            reconfig_spreads.append(tf.delta_fuzzy_time(env.time_add_dict[a], (0, 0, 0)))
            reconfig_costs.append(float(env.cost_add_dict[a]))
    else:
        for a in selectable_modules:
            if a == current_module:
                continue
            reconfig_times.append(tf.Triangular_fuzzy_time(env.time_add_dict[a], env.time_rem_dict[current_module]))
            reconfig_spreads.append(tf.delta_fuzzy_time(env.time_add_dict[a], env.time_rem_dict[current_module]))
            reconfig_costs.append(float(env.cost_add_dict[a] + env.cost_rem_dict[current_module]))

    max_rt = _max_positive(reconfig_times, default=1.0)
    max_rs = _max_positive(reconfig_spreads, default=1.0)
    max_rc = _max_positive(reconfig_costs, default=1.0)

    avg_deterministic_reconfig_time = (
        safe_div(sum(reconfig_times) / len(reconfig_times), max_rt) if reconfig_times else 0.0
    )
    avg_fuzzy_span = (
        safe_div(sum(reconfig_spreads) / len(reconfig_spreads), max_rs) if reconfig_spreads else 0.0
    )
    avg_reconfig_cost = (
        safe_div(sum(reconfig_costs) / len(reconfig_costs), max_rc) if reconfig_costs else 0.0
    )

    cost_ref = max(1.0, float(sum(env.cost_add_dict.values()) + sum(env.cost_rem_dict.values())))
    cumulative_cost_rate = safe_div(machine_obj.reconfig_cost_sum, cost_ref)

    return [
        is_busy,
        is_processing,
        is_reconfigure,
        remain_proc_time,
        remain_reconfig_time,
        *module_onehot,
        installed_module_rate,
        remain_operation_num,
        proc_avg_time,
        task_machine_mismatch,
        avg_deterministic_reconfig_time,
        avg_fuzzy_span,
        avg_reconfig_cost,
        cumulative_cost_rate,
    ]


def _module_feature(env, a: int) -> List[float]:
    module_onehot = [0.0] * env.module_count
    module_onehot[a] = 1.0

    task_num = safe_div(len(env.kind_task_a_dict[a]), len(env.kind_task_tuple))

    add_times = [
        tf.Triangular_fuzzy_time(env.time_add_dict[A], (0, 0, 0))
        for A in env.module_tuple
    ]
    add_spreads = [
        tf.delta_fuzzy_time(env.time_add_dict[A], (0, 0, 0))
        for A in env.module_tuple
    ]
    rem_times = [
        tf.Triangular_fuzzy_time((0, 0, 0), env.time_rem_dict[A])
        for A in env.module_tuple
    ]
    rem_spreads = [
        tf.delta_fuzzy_time((0, 0, 0), env.time_rem_dict[A])
        for A in env.module_tuple
    ]

    avg_add_time = safe_div(tf.Triangular_fuzzy_time(env.time_add_dict[a], (0, 0, 0)), _max_positive(add_times))
    avg_add_span = safe_div(tf.delta_fuzzy_time(env.time_add_dict[a], (0, 0, 0)), _max_positive(add_spreads))
    avg_rem_time = safe_div(tf.Triangular_fuzzy_time((0, 0, 0), env.time_rem_dict[a]), _max_positive(rem_times))
    avg_rem_span = safe_div(tf.delta_fuzzy_time((0, 0, 0), env.time_rem_dict[a]), _max_positive(rem_spreads))

    avg_add_cost = safe_div(env.cost_add_dict[a], max(1.0, max(env.cost_add_dict.values())))
    avg_rem_cost = safe_div(env.cost_rem_dict[a], max(1.0, max(env.cost_rem_dict.values())))

    module_rate = safe_div(
        len(tuple(m for m in env.machine_tuple if env.machine_dict[m].module == a)),
        env.machine_count,
    )

    unfinished_ops = [o for o in env.kind_task_tuple if env.kind_task_dict[o].unprocessed_count > 0]
    unfinished_need_a = [o for o in unfinished_ops if a in env.module_rj_dict[o]]
    uncompleted_opt_rate = safe_div(len(unfinished_need_a), max(1, len(unfinished_ops)))

    return [
        *module_onehot,
        task_num,
        avg_add_time,
        avg_rem_time,
        avg_add_span,
        avg_rem_span,
        avg_add_cost,
        avg_rem_cost,
        module_rate,
        uncompleted_opt_rate,
    ]


def _build_o2o(env) -> Tuple[List[List[int]], List[List[float]]]:
    edges = [[], []]
    attrs = []

    for r in env.kind_tuple:
        max_task = max(env.task_r_dict[r])
        for j in env.task_r_dict[r]:
            if j == max_task:
                continue
            src = env.kind_task_tuple.index((r, j))
            dst = env.kind_task_tuple.index((r, j + 1))
            predecessor_completed = 1.0 if env.kind_task_dict[(r, j)].processed_count > 0 else 0.0
            remaining_load = safe_div(
                sum(env.kind_task_dict[(r, jj)].unprocessed_count for jj in env.task_r_dict[r] if jj >= j),
                max(1.0, sum(env.kind_task_dict[(rr, jj)].unprocessed_count for rr, jj in env.kind_task_tuple)),
            )
            edges[0].append(src)
            edges[1].append(dst)
            attrs.append([
                1.0,
                safe_div(1.0, max_task + 1),
                predecessor_completed,
                remaining_load,
            ])

    return edges, attrs


def _build_o2m(env) -> Tuple[List[List[int]], List[List[float]]]:
    edges = [[], []]
    attrs = []

    all_proc_times = []
    for a in env.module_tuple:
        all_proc_times.extend(float(v) for v in env.time_arj_dict[a].values())
    max_proc_time = _max_positive(all_proc_times, default=1.0)

    for o_idx, o in enumerate(env.kind_task_tuple):
        r, j = o
        for m in env.machine_rj_dict[o]:
            machine_obj = env.machine_dict[m]
            current_module = machine_obj.module

            base_feasible = 1.0
            current_feasible = 0.0
            proc_time_norm = 0.0

            if current_module is not None and o in env.kind_task_a_dict[current_module]:
                current_feasible = 1.0
                proc_time_norm = safe_div(env.time_rja_dict[o][current_module], max_proc_time)

            feasible_future_modules = [
                a for a in env.machine_module_dict[m]
                if a in env.module_rj_dict[o]
            ]
            future_config_rate = safe_div(len(feasible_future_modules), max(1, len(env.machine_module_dict[m])))

            ready_match = 1.0 if env.kind_task_dict[o].buffer > 0 and current_feasible > 0 else 0.0

            edges[0].append(o_idx)
            edges[1].append(m)
            attrs.append([
                base_feasible,
                current_feasible,
                proc_time_norm,
                future_config_rate,
                ready_match,
            ])

    return edges, attrs


def _build_o2a(env) -> Tuple[List[List[int]], List[List[float]]]:
    edges = [[], []]
    attrs = []

    all_proc_times = []
    for a in env.module_tuple:
        all_proc_times.extend(float(v) for v in env.time_arj_dict[a].values())
    max_proc_time = _max_positive(all_proc_times, default=1.0)

    for o_idx, o in enumerate(env.kind_task_tuple):
        candidate_modules = env.module_rj_dict[o]
        best_time = min(float(env.time_rja_dict[o][a]) for a in candidate_modules)
        for a in candidate_modules:
            proc_time = float(env.time_rja_dict[o][a])
            support_weight = safe_div(best_time, proc_time)
            proc_time_norm = safe_div(proc_time, max_proc_time)

            edges[0].append(o_idx)
            edges[1].append(a)
            attrs.append([
                1.0,
                support_weight,
                proc_time_norm,
                safe_div(len(candidate_modules), env.module_count),
            ])

    return edges, attrs


def _build_m2a(env) -> Tuple[List[List[int]], List[List[float]]]:
    edges = [[], []]
    attrs = []

    add_time_values = [
        tf.Triangular_fuzzy_time(env.time_add_dict[a], (0, 0, 0))
        for a in env.module_tuple
    ]
    add_span_values = [
        tf.delta_fuzzy_time(env.time_add_dict[a], (0, 0, 0))
        for a in env.module_tuple
    ]
    rem_time_values = [
        tf.Triangular_fuzzy_time((0, 0, 0), env.time_rem_dict[a])
        for a in env.module_tuple
    ]
    rem_span_values = [
        tf.delta_fuzzy_time((0, 0, 0), env.time_rem_dict[a])
        for a in env.module_tuple
    ]

    max_add_time = _max_positive(add_time_values, 1.0)
    max_add_span = _max_positive(add_span_values, 1.0)
    max_rem_time = _max_positive(rem_time_values, 1.0)
    max_rem_span = _max_positive(rem_span_values, 1.0)
    max_add_cost = max(1.0, max(float(v) for v in env.cost_add_dict.values()))
    max_rem_cost = max(1.0, max(float(v) for v in env.cost_rem_dict.values()))

    for m in env.machine_tuple:
        for a in env.machine_module_dict[m]:
            is_install = 1.0 if env.machine_dict[m].module == a else 0.0
            can_install = 1.0

            add_time = tf.Triangular_fuzzy_time(env.time_add_dict[a], (0, 0, 0))
            add_span = tf.delta_fuzzy_time(env.time_add_dict[a], (0, 0, 0))
            rem_time = tf.Triangular_fuzzy_time((0, 0, 0), env.time_rem_dict[a])
            rem_span = tf.delta_fuzzy_time((0, 0, 0), env.time_rem_dict[a])

            edges[0].append(m)
            edges[1].append(a)
            attrs.append([
                is_install,
                can_install,
                safe_div(add_time, max_add_time),
                safe_div(add_span, max_add_span),
                safe_div(rem_time, max_rem_time),
                safe_div(rem_span, max_rem_span),
                safe_div(env.cost_add_dict[a], max_add_cost),
                safe_div(env.cost_rem_dict[a], max_rem_cost),
            ])

    return edges, attrs


def graph_tensor_from_env(env, device: torch.device) -> GraphTensor:
    op_features = [
        _operation_feature(env, r, j)
        for (r, j) in env.kind_task_tuple
    ]
    machine_features = [
        _machine_feature(env, m)
        for m in env.machine_tuple
    ]
    module_features = [
        _module_feature(env, a)
        for a in env.module_tuple
    ]

    o2o_edges, o2o_attr = _build_o2o(env)
    o2m_edges, o2m_attr = _build_o2m(env)
    o2a_edges, o2a_attr = _build_o2a(env)
    m2a_edges, m2a_attr = _build_m2a(env)

    if len(o2o_attr) == 0:
        o2o_edges = [[], []]
        o2o_attr = []

    graph = GraphTensor(
        op_x=to_float_tensor(op_features, device),
        machine_x=to_float_tensor(machine_features, device),
        module_x=to_float_tensor(module_features, device),
        o2o_edge_index=to_long_tensor(o2o_edges, device),
        o2m_edge_index=to_long_tensor(o2m_edges, device),
        o2a_edge_index=to_long_tensor(o2a_edges, device),
        m2a_edge_index=to_long_tensor(m2a_edges, device),
        o2o_edge_attr=to_float_tensor(o2o_attr if o2o_attr else torch.zeros((0, 4)), device),
        o2m_edge_attr=to_float_tensor(o2m_attr if o2m_attr else torch.zeros((0, 5)), device),
        o2a_edge_attr=to_float_tensor(o2a_attr if o2a_attr else torch.zeros((0, 4)), device),
        m2a_edge_attr=to_float_tensor(m2a_attr if m2a_attr else torch.zeros((0, 8)), device),
        op_count=len(env.kind_task_tuple),
        machine_count=env.machine_count,
        module_count=env.module_count,
    )

    return graph