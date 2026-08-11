# -*- coding: utf-8 -*-
"""
独立 Pareto 前沿绘图脚本

功能：
1. 读取同一个算例下 5 组权重训练得到的 .pt 文件
2. 打印每组权重下：
   - 最小 Makespan 及其对应 Cost
   - 最小 Cost 及其对应 Makespan
3. 计算并打印全局最小 Makespan / Cost
4. 计算 Pareto 非支配点
5. 保存 Pareto 图 png / pdf 和 summary csv

用法示例：
python plot_pareto_front.py ^
    --checkpoint_dir "F:\Flexible Reconfigurable Manufacturing System\HiFAMR-BMAPPO\checkpoints" ^
    --file_name M4_A3_R4_J2 ^
    --last_n 10
"""

from pathlib import Path
import argparse

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


WEIGHT_CONFIGS = [
    (1.0, 0.0, r"$w_m=1.00,\ w_c=0.00$"),
    (0.0, 1.0, r"$w_m=0.00,\ w_c=1.00$"),
    (0.25, 0.75, r"$w_m=0.25,\ w_c=0.75$"),
    (0.5, 0.5, r"$w_m=0.50,\ w_c=0.50$"),
    (0.75, 0.25, r"$w_m=0.75,\ w_c=0.25$"),
]


def format_weight(w: float) -> str:
    """保持和训练保存的 checkpoint 文件名一致。"""
    w = float(w)
    if abs(w - round(w)) < 1e-9:
        return str(int(round(w)))
    return f"{w:.2f}".rstrip("0").rstrip(".")


def checkpoint_name(alpha_t: float, alpha_c: float, file_name: str) -> str:
    return (
        f"{format_weight(alpha_t)}makespan_"
        f"{format_weight(alpha_c)}cost_"
        f"{file_name}.pt"
    )


def to_list(x):
    """将 tensor / list / tuple 转成普通 Python list。"""
    if torch.is_tensor(x):
        return x.detach().cpu().flatten().tolist()

    if isinstance(x, (list, tuple)):
        return [v.item() if hasattr(v, "item") else v for v in x]

    raise TypeError(f"不支持的数据类型：{type(x)}")


def is_pareto_efficient(points):
    """
    计算非支配点。
    points: [(makespan, cost), ...]
    两个目标都是越小越好。
    """
    efficient = []

    for i, (m_i, c_i) in enumerate(points):
        dominated = False

        for j, (m_j, c_j) in enumerate(points):
            if i == j:
                continue

            if (
                m_j <= m_i
                and c_j <= c_i
                and (m_j < m_i or c_j < c_i)
            ):
                dominated = True
                break

        efficient.append(not dominated)

    return efficient


def load_records(checkpoint_dir: Path, file_name: str, last_n: int = 10):
    records = []
    summary_records = []

    for alpha_t, alpha_c, label in WEIGHT_CONFIGS:
        pt_path = checkpoint_dir / checkpoint_name(alpha_t, alpha_c, file_name)

        try:
            data = torch.load(pt_path, map_location="cpu")

            if "all_makespan" not in data or "all_cost" not in data:
                print(f"跳过 {pt_path.name}：缺少 all_makespan 或 all_cost")
                continue

            makespan_all = to_list(data["all_makespan"])
            cost_all = to_list(data["all_cost"])

            if last_n is not None and last_n > 0:
                makespan = makespan_all[-last_n:]
                cost = cost_all[-last_n:]
            else:
                makespan = makespan_all
                cost = cost_all

            if len(makespan) == 0 or len(cost) == 0:
                print(f"跳过 {pt_path.name}：makespan 或 cost 为空")
                continue

            min_makespan = min(makespan)
            min_cost = min(cost)

            min_makespan_idx = makespan.index(min_makespan)
            min_cost_idx = cost.index(min_cost)

            cost_at_min_makespan = cost[min_makespan_idx]
            makespan_at_min_cost = makespan[min_cost_idx]

            print(f"\n已读取：{pt_path.name}")
            print(f"权重配置：{label}")
            print(
                f"  最小 Makespan = {min_makespan:.4f}, "
                f"对应 Cost = {cost_at_min_makespan:.4f}"
            )
            print(
                f"  最小 Cost     = {min_cost:.4f}, "
                f"对应 Makespan = {makespan_at_min_cost:.4f}"
            )

            summary_records.append({
                "Weight Configuration": label,
                "File": pt_path.name,
                "Min Makespan": float(min_makespan),
                "Cost at Min Makespan": float(cost_at_min_makespan),
                "Min Cost": float(min_cost),
                "Makespan at Min Cost": float(makespan_at_min_cost),
            })

            for m, c in zip(makespan, cost):
                records.append({
                    "Makespan": float(m),
                    "Cost": float(c),
                    "Weight Configuration": label,
                    "File": pt_path.name,
                })

        except Exception as e:
            print(f"读取 {pt_path.name} 出错：{e}")

    df = pd.DataFrame(records)
    summary_df = pd.DataFrame(summary_records)

    if df.empty:
        raise ValueError("没有有效数据可绘制 Pareto 图")

    return df, summary_df


def plot_pareto_front(
    checkpoint_dir,
    file_name: str,
    output_dir=None,
    last_n: int = 10,
    show: bool = False,
):
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir) if output_dir is not None else checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "figure.titlesize": 15,
        "figure.facecolor": "w",
        "font.family": "Times New Roman",
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.bottom": False,
        "ytick.left": False,
    })

    df, summary_df = load_records(
        checkpoint_dir=checkpoint_dir,
        file_name=file_name,
        last_n=last_n,
    )

    print(f"\n成功读取 {len(df)} 个数据点")

    print("\n========== 各权重配置最优结果汇总 ==========")
    print(summary_df.to_string(index=False))

    global_min_makespan_row = df.loc[df["Makespan"].idxmin()]
    global_min_cost_row = df.loc[df["Cost"].idxmin()]

    print("\n========== 全局最优结果 ==========")
    print(
        f"全局最小 Makespan = {global_min_makespan_row['Makespan']:.4f}, "
        f"对应 Cost = {global_min_makespan_row['Cost']:.4f}, "
        f"权重 = {global_min_makespan_row['Weight Configuration']}"
    )
    print(
        f"全局最小 Cost     = {global_min_cost_row['Cost']:.4f}, "
        f"对应 Makespan = {global_min_cost_row['Makespan']:.4f}, "
        f"权重 = {global_min_cost_row['Weight Configuration']}"
    )

    # ======================
    # 计算非支配 Pareto 前沿
    # ======================
    points = list(zip(df["Makespan"].values, df["Cost"].values))
    df["Pareto Efficient"] = is_pareto_efficient(points)

    pareto_df = df[df["Pareto Efficient"]].copy()
    pareto_df = pareto_df.sort_values(by="Makespan")

    print("\n========== Pareto 非支配点 ==========")
    print(
        pareto_df[
            ["Makespan", "Cost", "Weight Configuration", "File"]
        ].to_string(index=False)
    )

    # ======================
    # 绘图
    # ======================
    sns.set_style("white")

    g = sns.JointGrid(
        data=df,
        x="Makespan",
        y="Cost",
        height=8,
        ratio=5,
        space=0.2,
    )

    sns.scatterplot(
        data=df,
        x="Makespan",
        y="Cost",
        hue="Weight Configuration",
        palette="Set1",
        alpha=0.75,
        s=80,
        ax=g.ax_joint,
    )

    # # 画 Pareto 非支配前沿线
    # if len(pareto_df) >= 2:
    #     g.ax_joint.plot(
    #         pareto_df["Makespan"],
    #         pareto_df["Cost"],
    #         linestyle="--",
    #         linewidth=2,
    #         marker="o",
    #         label="Pareto Front",
    #     )

    # 边缘分布：样本太少或重复点过多时 kde 可能失败，所以单独 try
    try:
        sns.kdeplot(
            data=df,
            x="Makespan",
            hue="Weight Configuration",
            palette="Set1",
            fill=True,
            alpha=0.35,
            legend=False,
            ax=g.ax_marg_x,
            warn_singular=False,
        )
    except Exception as e:
        print(f"上方 KDE 绘制失败，跳过：{e}")

    try:
        sns.kdeplot(
            data=df,
            y="Cost",
            hue="Weight Configuration",
            palette="Set1",
            fill=True,
            alpha=0.35,
            legend=False,
            ax=g.ax_marg_y,
            warn_singular=False,
        )
    except Exception as e:
        print(f"右侧 KDE 绘制失败，跳过：{e}")

    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend(
        handles=handles,
        labels=labels,
        title="Weight Configuration",
        loc="upper right",
        fontsize=11,
        title_fontsize=13,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    g.fig.suptitle(
        f"Pareto Front: Makespan and Cost ({file_name})",
        y=1.02,
        fontsize=15,
    )

    g.ax_joint.set_xlabel("Makespan", fontsize=15)
    g.ax_joint.set_ylabel("Cost", fontsize=15)

    png_path = output_dir / f"{file_name}_pareto_front.png"
    pdf_path = output_dir / f"{file_name}_pareto_front.pdf"
    csv_path = output_dir / f"{file_name}_pareto_summary.csv"
    pareto_csv_path = output_dir / f"{file_name}_pareto_points.csv"

    g.fig.savefig(png_path, bbox_inches="tight", dpi=300)
    g.fig.savefig(pdf_path, bbox_inches="tight")
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    pareto_df.to_csv(pareto_csv_path, index=False, encoding="utf-8-sig")

    print("\nPareto 图已保存：")
    print(png_path)
    print(pdf_path)
    print("\n汇总表已保存：")
    print(csv_path)
    print("\nPareto 非支配点已保存：")
    print(pareto_csv_path)

    if show:
        plt.show()
    else:
        plt.close(g.fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="5 个权重 checkpoint 所在目录",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="算例名称，例如 M4_A3_R6_J3",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="图片和 csv 保存目录；默认与 checkpoint_dir 相同",
    )
    parser.add_argument(
        "--last_n",
        type=int,
        default=10,
        help="每个 .pt 取最后 N 个训练周期的数据；设为 0 表示使用全部周期",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否显示图形窗口",
    )

    args = parser.parse_args()

    plot_pareto_front(
        checkpoint_dir=args.checkpoint_dir,
        file_name=args.file_name,
        output_dir=args.output_dir,
        last_n=args.last_n,
        show=args.show,
    )


if __name__ == "__main__":
    # python "plot_pareto_front.py" --checkpoint_dir "F:\Flexible Reconfigurable Manufacturing System\HiFAMR-BMAPPO\checkpoint" --file_name M4_A5_R8_J4 --last_n 20
    main()
