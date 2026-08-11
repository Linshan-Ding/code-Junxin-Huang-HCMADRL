# =========================
# 文件名：gant_plot.py
# =========================
import os
import torch
import matplotlib.pyplot as plt


CHECKPOINT_PATH = r"F:\Flexible Reconfigurable Manufacturing System\HiFAMR-Sync-MAGRL\HiFAMR-Sync-MAGRL\checkpoints\0.5makespan_0.5cost_M4_A3_R4_J2.pt"


def load_gantt_data(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint 文件不存在：{checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    schedule_dict = checkpoint.get("schedule_dict", {})
    module_change_dict = checkpoint.get("module_change_dict", {})
    metrics = checkpoint.get("metrics", {})
    episode = checkpoint.get("episode", None)

    return schedule_dict, module_change_dict, metrics, episode


def normalize_machine_keys(schedule_dict, module_change_dict):
    machine_set = set()

    for m in schedule_dict.keys():
        machine_set.add(int(m))

    for m in module_change_dict.keys():
        machine_set.add(int(m))

    machine_list = sorted(machine_set)

    new_schedule_dict = {}
    new_module_change_dict = {}

    for m in machine_list:
        new_schedule_dict[m] = schedule_dict.get(m, schedule_dict.get(str(m), []))
        new_module_change_dict[m] = module_change_dict.get(m, module_change_dict.get(str(m), []))

    return machine_list, new_schedule_dict, new_module_change_dict


def parse_operation(item):

    if len(item) == 6:
        r, j, batch_id, start_time, end_time, module = item
    elif len(item) == 5:
        r, j, start_time, end_time, module = item
        batch_id = None
    elif len(item) == 4:
        r, j, start_time, end_time = item
        batch_id = None
        module = None
    else:
        return None

    start_time = float(start_time)
    end_time = float(end_time)
    duration = end_time - start_time

    if duration <= 0:
        return None

    return {
        "kind": int(r),
        "task": int(j),
        "batch": None if batch_id is None else int(batch_id),
        "module": None if module is None else int(module),
        "start": start_time,
        "end": end_time,
        "duration": duration,
    }


def parse_reconfiguration(item):
    """
    模块切换记录兼容两种格式：
    1. (target_module, start_time, end_time, old_module)
    2. (target_module, start_time, end_time)
    """
    if len(item) == 4:
        target_module, start_time, end_time, old_module = item
    elif len(item) == 3:
        target_module, start_time, end_time = item
        old_module = None
    else:
        return None

    start_time = float(start_time)
    end_time = float(end_time)
    duration = end_time - start_time

    if duration <= 0:
        return None

    return {
        "target_module": int(target_module),
        "old_module": None if old_module is None else int(old_module),
        "start": start_time,
        "end": end_time,
        "duration": duration,
    }


def operation_label(op):
    if op["batch"] is None:
        return f"{op['kind']},{op['task']}"
    return f"({op['kind']},{op['batch']},{op['task']})"


def reconfiguration_label(rec):
    if rec["old_module"] is None:
        return f"A{rec['target_module']}"
    return f"A{rec['old_module']}→A{rec['target_module']}"


def print_gantt_info(machine_list, schedule_dict, module_change_dict, metrics, episode, checkpoint_path):
    total_operation_count = sum(len(v) for v in schedule_dict.values())
    total_reconfig_count = sum(len(v) for v in module_change_dict.values())

    print("=" * 100)
    print("甘特图信息")
    print("=" * 100)
    print(f"checkpoint: {checkpoint_path}")
    print(f"episode: {episode}")
    print(f"metrics: {metrics}")
    print(f"工序记录数量: {total_operation_count}")
    print(f"模块切换记录数量: {total_reconfig_count}")
    print("=" * 100)

    timeline = []

    for m in machine_list:
        print(f"\nMachine {m}")
        print("-" * 100)

        print("模块切换:")
        reconfig_list = []
        for item in module_change_dict.get(m, []):
            rec = parse_reconfiguration(item)
            if rec is not None:
                reconfig_list.append(rec)
                timeline.append({
                    "type": "RECONFIG",
                    "machine": m,
                    "start": rec["start"],
                    "end": rec["end"],
                    "duration": rec["duration"],
                    "label": reconfiguration_label(rec),
                })

        reconfig_list.sort(key=lambda x: x["start"])

        if len(reconfig_list) == 0:
            print("  无")
        else:
            for idx, rec in enumerate(reconfig_list, start=1):
                print(
                    f"  [{idx:03d}] "
                    f"{reconfiguration_label(rec):<15} "
                    f"start={rec['start']:.4f}, "
                    f"end={rec['end']:.4f}, "
                    f"duration={rec['duration']:.4f}"
                )

        print("工序加工:")
        operation_list = []
        for item in schedule_dict.get(m, []):
            op = parse_operation(item)
            if op is not None:
                operation_list.append(op)
                timeline.append({
                    "type": "OPERATION",
                    "machine": m,
                    "start": op["start"],
                    "end": op["end"],
                    "duration": op["duration"],
                    "label": operation_label(op),
                })

        operation_list.sort(key=lambda x: x["start"])

        if len(operation_list) == 0:
            print("  无")
        else:
            for idx, op in enumerate(operation_list, start=1):
                module_text = "" if op["module"] is None else f", module=M{op['module']}"
                batch_text = "" if op["batch"] is None else f", batch=B{op['batch']}"

                print(
                    f"  [{idx:03d}] "
                    f"{operation_label(op):<16} "
                    f"start={op['start']:.4f}, "
                    f"end={op['end']:.4f}, "
                    f"duration={op['duration']:.4f}"
                    f"{batch_text}"
                    f"{module_text}"
                )

    print("\n" + "=" * 100)
    print("全局时间线")
    print("=" * 100)

    timeline.sort(key=lambda x: (x["start"], x["end"], x["machine"], x["type"]))

    if len(timeline) == 0:
        print("无时间线记录")
    else:
        for idx, item in enumerate(timeline, start=1):
            print(
                f"[{idx:04d}] "
                f"t={item['start']:.4f} -> {item['end']:.4f} "
                f"duration={item['duration']:.4f} | "
                f"Machine {item['machine']} | "
                f"{item['type']:<9} | "
                f"{item['label']}"
            )

    print("=" * 100)


def plot_gantt(machine_list, schedule_dict, module_change_dict, episode):
    if len(machine_list) == 0:
        print("没有机器记录，无法绘制甘特图。")
        return

    fig_height = max(10, len(machine_list) * 1.2)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    y_pos_dict = {m: idx for idx, m in enumerate(machine_list)}

    operation_y_offset = 0.28
    reconfig_y_offset = -0.28

    # 块高度加大，字体大小不变
    bar_height = 0.28

    max_time = 0.0

    for m in machine_list:
        y_base = y_pos_dict[m]

        for item in module_change_dict.get(m, []):
            rec = parse_reconfiguration(item)
            if rec is None:
                continue

            max_time = max(max_time, rec["end"])

            ax.barh(
                y=y_base + reconfig_y_offset,
                width=rec["duration"],
                left=rec["start"],
                height=bar_height,
                color="gray",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.85,
            )

            ax.text(
                x=rec["start"] + rec["duration"] / 2.0,
                y=y_base + reconfig_y_offset,
                s=reconfiguration_label(rec),
                va="center",
                ha="center",
                color="black",
                fontsize=8,
            )

        for item in schedule_dict.get(m, []):
            op = parse_operation(item)
            if op is None:
                continue

            max_time = max(max_time, op["end"])

            ax.barh(
                y=y_base + operation_y_offset,
                width=op["duration"],
                left=op["start"],
                height=bar_height,
                color="tab:blue",
                edgecolor="black",
                linewidth=0.8,
                alpha=0.9,
            )

            ax.text(
                x=op["start"] + op["duration"] / 2.0,
                y=y_base + operation_y_offset,
                s=operation_label(op),
                va="center",
                ha="center",
                color="white",
                fontsize=8,
            )

    ax.set_yticks([y_pos_dict[m] for m in machine_list])
    ax.set_yticklabels([f"Machine {m}" for m in machine_list])

    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")

    title = "Gantt Chart "
    if episode is not None:
        title += f" | Episode {episode}"
    ax.set_title(title)

    ax.set_xlim(0, max_time * 1.05 if max_time > 0 else 1)

    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    ax.barh([], [], color="tab:blue", edgecolor="black", label="Operation")
    ax.barh([], [], color="gray", edgecolor="black", label="Module Reconfiguration")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def main():
    schedule_dict, module_change_dict, metrics, episode = load_gantt_data(CHECKPOINT_PATH)

    machine_list, schedule_dict, module_change_dict = normalize_machine_keys(
        schedule_dict,
        module_change_dict,
    )

    total_operation_count = sum(len(v) for v in schedule_dict.values())
    total_reconfig_count = sum(len(v) for v in module_change_dict.values())

    if total_operation_count == 0 and total_reconfig_count == 0:
        print("没有甘特图数据。请确认 .pt 文件中包含 schedule_dict 和 module_change_dict。")
        return

    print_gantt_info(
        machine_list=machine_list,
        schedule_dict=schedule_dict,
        module_change_dict=module_change_dict,
        metrics=metrics,
        episode=episode,
        checkpoint_path=CHECKPOINT_PATH,
    )

    plot_gantt(
        machine_list=machine_list,
        schedule_dict=schedule_dict,
        module_change_dict=module_change_dict,
        episode=episode,
    )


if __name__ == "__main__":
    main()