# ============================================================
# 文件名：ablation/run_ablation.py
# 消融实验启动入口
#
# 使用方法：
#   1. 修改 ablation_config.py 中的 SCHEDULE 列表
#   2. （可选）修改 NUM_ROUNDS 等全局参数
#   3. 启动 Visdom：python -m visdom.server
#   4. 运行：python ablation/run_ablation.py
#
# 命令行参数（可选，覆盖 config 中的默认值）：
#   --num_workers 4       并行采样进程数
#   --batch_size 64       PPO mini-batch 大小
#   --seed 42             固定随机种子（-1 为随机）
#   --no_visdom           不使用 Visdom 可视化
# ============================================================
import os
import sys
import time
import argparse
import json
import subprocess
from datetime import datetime
from typing import List, Dict

# 确保可以从项目根目录导入
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from ablation.ablation_config import (
    SCHEDULE,
    NUM_ROUNDS,
    NUM_WORKERS,
    UPDATE_BATCH_SIZE,
    MP_START_METHOD,
    ROLLOUT_DEVICE,
    AUTO_START_VISDOM,
    VISDOM_PORT,
    VISDOM_ENV,
    DATA_PATH,
    ABLATION_SAVE_DIR,
    ScheduleItem,
)
from ablation.ablation_trainer import (
    train_one_round,
    is_port_open,
)
from utils import ensure_dir
from config import config


def precompute_all_references(schedule, args):
    """
    预训练阶段：为每个 (variant, file_name) 对训练纯 Makespan 和纯 Cost，
    将实际训练得到的最优值作为混合权重训练的标准化参考值。

    这样确保了 reward normalize 使用的 MAKESPAN_REF 和 COST_REF
    来自该变体在该实例上的真实单目标最优解，而非启发式估算。

    Returns:
        ref_cache: Dict[(variant, file_name), (makespan_ref, cost_ref)]
        pretrain_results: List[Dict]
    """
    # 收集所有唯一的 (variant, file_name, episodes) 元组
    unique_pairs: dict = {}
    for item in schedule:
        episodes = getattr(item, 'episodes', 500)
        for fn in item.file_names:
            key = (item.variant, fn)
            unique_pairs[key] = max(unique_pairs.get(key, episodes), episodes)

    ref_cache: dict = {}
    pretrain_results: list = []

    if not unique_pairs:
        print("  [预训练] 无需要预训练的 (variant, instance) 对")
        return ref_cache, pretrain_results

    print("\n" + "=" * 90)
    print(f"  预训练阶段：为 {len(unique_pairs)} 个 (variant, instance) 对训练纯目标")
    print("=" * 90)

    for idx, ((variant, fn), episodes) in enumerate(unique_pairs.items(), 1):
        print(f"\n{'─'*70}")
        print(f"  [{idx}/{len(unique_pairs)}] 预训练 {variant} @ {fn} — 纯Makespan & 纯Cost")
        print(f"{'─'*70}")

        # --- 纯 Makespan ---
        print(f"\n  ▶ 预训练纯Makespan: αt=1.0 αc=0.0")
        try:
            result_m = train_one_round(
                variant=variant, alpha_t=1.0, alpha_c=0.0,
                file_name=fn, episodes=episodes, reward_normalize=False,
                makespan_ref=1.0, cost_ref=1.0,
                num_workers=args.num_workers, update_batch_size=args.batch_size,
                mp_start_method=MP_START_METHOD, rollout_device=ROLLOUT_DEVICE,
                data_path=DATA_PATH, save_dir=args.save_dir,
                description=f"PRE-Makespan-{variant}-{fn}", visdom_env=VISDOM_ENV,
                seed=args.seed,
            )
            makespan_ref = max(result_m.get("best_makespan", 1.0), 1.0)
            pretrain_results.append({**result_m, "pretrain": True, "objective": "pure_makespan"})
        except Exception as e:
            print(f"  ⚠ 纯Makespan预训练失败: {e}，使用估算值")
            makespan_ref, _ = _estimate_refs_from_name(fn)
            result_m = None

        # --- 纯 Cost ---
        print(f"\n  ▶ 预训练纯Cost: αt=0.0 αc=1.0")
        try:
            result_c = train_one_round(
                variant=variant, alpha_t=0.0, alpha_c=1.0,
                file_name=fn, episodes=episodes, reward_normalize=False,
                makespan_ref=1.0, cost_ref=1.0,
                num_workers=args.num_workers, update_batch_size=args.batch_size,
                mp_start_method=MP_START_METHOD, rollout_device=ROLLOUT_DEVICE,
                data_path=DATA_PATH, save_dir=args.save_dir,
                description=f"PRE-Cost-{variant}-{fn}", visdom_env=VISDOM_ENV,
                seed=args.seed,
            )
            cost_ref = max(result_c.get("best_cost", 1.0), 1.0)
            pretrain_results.append({**result_c, "pretrain": True, "objective": "pure_cost"})
        except Exception as e:
            print(f"  ⚠ 纯Cost预训练失败: {e}，使用估算值")
            _, cost_ref = _estimate_refs_from_name(fn)
            result_c = None

        ref_cache[(variant, fn)] = (makespan_ref, cost_ref)
        print(f"  ✓ [{variant} @ {fn}] refs: makespan_ref={makespan_ref:.1f}, cost_ref={cost_ref:.1f}")

    return ref_cache, pretrain_results


def _estimate_refs_from_name(file_name: str):
    """
    根据算例参数估算参考值（仅用于预训练失败时的回退）。

    算例命名：M{m}_A{a}_R{r}_J{j}
    """
    parts = file_name.split('_')
    M = int(parts[0][1:])
    A = int(parts[1][1:])
    R = int(parts[2][1:])
    J = int(parts[3][1:])

    total_ops = R * J * 7.5
    avg_proc = 55.0
    makespan_ref = total_ops * avg_proc / max(M, 1) * 1.8
    cost_ref = M * A * 30.0 * 0.6

    return max(makespan_ref, 100.0), max(cost_ref, 100.0)


def start_visdom_server(port: int = 8097):
    """如果 Visdom 未启动则自动启动。"""
    if is_port_open(port=port):
        print(f"Visdom server 已在运行：http://127.0.0.1:{port}")
        return None

    print(f"正在启动 Visdom server：http://127.0.0.1:{port}")
    process = subprocess.Popen(
        [sys.executable, "-m", "visdom.server", "-port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if is_port_open(port=port):
            print(f"Visdom server 启动成功：http://127.0.0.1:{port}")
            return process
        time.sleep(1)
    print("⚠ 警告：Visdom server 启动超时，训练将继续运行。")
    return process


def format_weight(w: float) -> str:
    w = float(w)
    if abs(w - round(w)) < 1e-9:
        return str(int(round(w)))
    return f"{w:.2f}".rstrip("0").rstrip(".")


def main():
    parser = argparse.ArgumentParser(description="HiFAMR-BMAPPO 消融实验轮训")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--batch_size", type=int, default=UPDATE_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no_visdom", action="store_true")
    parser.add_argument("--save_dir", type=str, default=ABLATION_SAVE_DIR)
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    args = parser.parse_args()

    ensure_dir(args.save_dir)

    # ============================================================
    # 启动 Visdom
    # ============================================================
    visdom_process = None
    if AUTO_START_VISDOM and not args.no_visdom:
        visdom_process = start_visdom_server(VISDOM_PORT)

    # ============================================================
    # 阶段1：预训练纯目标（获取标准化参考值）
    # ============================================================
    ref_cache, pretrain_results = precompute_all_references(SCHEDULE, args)
    all_results: List[Dict] = list(pretrain_results)

    # ============================================================
    # 阶段2：混合权重训练（主训练循环）
    # ============================================================
    # 只处理 reward_normalize=True 的混合权重项（纯目标已在预训练完成）
    mixed_items = [item for item in SCHEDULE if item.reward_normalize]
    total_rounds = len(mixed_items) * args.rounds

    print("\n" + "=" * 90)
    print(f"  消融实验混合权重训练开始")
    print(f"  总轮数：{total_rounds}（{len(mixed_items)} 个混合权重 × {args.rounds} 轮循环）")
    print(f"  并行 Worker：{args.num_workers}")
    print(f"  Batch Size：{args.batch_size}")
    print(f"  保存目录：{args.save_dir}")
    print("=" * 90)

    round_idx = 0
    for cycle in range(args.rounds):
        for item_idx, item in enumerate(mixed_items):
            round_idx += 1

            print(f"\n{'─' * 90}")
            print(f"  Round {round_idx}/{total_rounds} (循环 {cycle+1}/{args.rounds}, 配置 {item_idx+1}/{len(mixed_items)})")
            print(f"  Variant: {item.variant}")
            print(f"  Weights: αt={item.alpha_t}, αc={item.alpha_c}")
            print(f"  Instances: {item.file_names}")
            print(f"  Description: {item.description or '(无)'}")
            print(f"{'─' * 90}")

            # 对本轮每个算例执行训练
            for fn in item.file_names:
                print(f"\n  ▶ 训练算例: {fn}")

                # 从预训练缓存获取该 (variant, file_name) 的真实参考值
                key = (item.variant, fn)
                if key in ref_cache:
                    makespan_ref, cost_ref = ref_cache[key]
                else:
                    # 防御性回退（理论上预训练已覆盖所有 pair）
                    makespan_ref, cost_ref = _estimate_refs_from_name(fn)
                    print(f"  ⚠ 未找到预训练ref，使用估算值: makespan_ref={makespan_ref:.1f}, cost_ref={cost_ref:.1f}")

                try:
                    result = train_one_round(
                        variant=item.variant,
                        alpha_t=item.alpha_t,
                        alpha_c=item.alpha_c,
                        file_name=fn,
                        episodes=item.episodes,
                        reward_normalize=True,
                        makespan_ref=makespan_ref,
                        cost_ref=cost_ref,
                        num_workers=args.num_workers,
                        update_batch_size=args.batch_size,
                        mp_start_method=MP_START_METHOD,
                        rollout_device=ROLLOUT_DEVICE,
                        data_path=DATA_PATH,
                        save_dir=args.save_dir,
                        description=f"{item.description} | {fn}",
                        visdom_env=VISDOM_ENV,
                        seed=args.seed,
                    )
                    result["round"] = round_idx
                    result["cycle"] = cycle + 1
                    all_results.append(result)

                except Exception as e:
                    print(f"  ✗ 训练失败: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "variant": item.variant,
                        "file_name": fn,
                        "error": str(e),
                        "round": round_idx,
                    })

    # ============================================================
    # 打印汇总
    # ============================================================
    print("\n" + "=" * 90)
    print("  消融实验完成！")
    print("=" * 90)

    # 分组显示
    pretrain = [r for r in all_results if r.get("pretrain")]
    mixed = [r for r in all_results if not r.get("pretrain")]

    if pretrain:
        print(f"\n  ── 预训练纯目标结果（{len(pretrain)} 项）──")
        print(f"  {'Variant':<20s} {'目标':>12s} {'Instance':<18s} {'Makespan':>10s} {'Cost':>10s}")
        print(f"  {'─'*20} {'─'*12} {'─'*18} {'─'*10} {'─'*10}")
        for r in pretrain:
            if "error" in r:
                print(f"  {r['variant']:<20s} {r.get('objective','?'):>12s} {r['file_name']:<18s} {'ERROR: '+r['error']}")
            else:
                print(
                    f"  {r['variant']:<20s} "
                    f"{r.get('objective','?'):>12s} "
                    f"{r['file_name']:<18s} "
                    f"{r['best_makespan']:>10.1f} "
                    f"{r['best_cost']:>10.1f}"
                )

    if mixed:
        print(f"\n  ── 混合权重训练结果（{len(mixed)} 项）──")
        print(f"  {'Variant':<20s} {'αt':>5s} {'αc':>5s} {'Instance':<18s} {'Makespan':>10s} {'Cost':>10s} {'Ref_m':>8s} {'Ref_c':>8s}")
        print(f"  {'─'*20} {'─'*5} {'─'*5} {'─'*18} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
        for r in mixed:
            if "error" in r:
                print(f"  {r['variant']:<20s} {'─':>5s} {'─':>5s} {r['file_name']:<18s} {'ERROR: '+r['error']}")
            else:
                key = (r['variant'], r['file_name'])
                ref_m, ref_c = ref_cache.get(key, (0, 0))
                print(
                    f"  {r['variant']:<20s} "
                    f"{format_weight(r['alpha_t']):>5s} "
                    f"{format_weight(r['alpha_c']):>5s} "
                    f"{r['file_name']:<18s} "
                    f"{r['best_makespan']:>10.1f} "
                    f"{r['best_cost']:>10.1f} "
                    f"{ref_m:>8.1f} "
                    f"{ref_c:>8.1f}"
                )

    print(f"\n  模型目录: {args.save_dir}")

    return all_results


if __name__ == "__main__":
    main()
