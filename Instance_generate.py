# -*- coding: utf-8 -*-
"""
随机生成训练实例

修改要求：
1. 所有模块必须至少被一台机器选择；
2. 每台机器可选模块数量为 1-4 个；
3. 所有模块必须至少被一道工序选择；
4. 每道工序可选模块数量为 1-4 个；
5. 写出 CSV 后反读校验，防止旧逻辑或写文件过程导致漏模块。
"""
from random import randint, uniform
import ast
import csv
import os

import numpy as np

# 英文显示问题
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')  # 设置英文字体

# 中文显示问题
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="SimHei.ttf", size=12)  # 指定中文字体和字号
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 报错问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Instance():
    """
    问题特点
    文件命名方式 M10_A5_S1
    """
    def __init__(self, M=None, A=None, R=None, J=None):
        # 问题特点
        self.machine_count = M  # 机器数
        self.machine_tuple = tuple(m for m in range(self.machine_count))  # 机器元组
        self.order_count = 1  # 新到达的订单总数
        self.order_tuple = tuple(s for s in range(self.order_count))  # 订单元组
        self.module_count = A  # 模块数
        self.module_tuple = tuple(a for a in range(self.module_count))  # 模块元组
        self.kind_count = R  # 工件类型数
        self.kind_tuple = tuple(r for r in range(self.kind_count))  # 工件类型元组
        self.task_count = J  # 每种工件类型的工序数
        self.file_name = 'M' + str(M) + '_A' + str(A) + '_R' + str(R) + '_J' + str(J)  # 算例文件夹名

        self.task_r_dict, self.machine_rj_dict, self.kind_task_m_dict, self.time_rja_dict, self.count_sr_dict, self.time_arrive_s_dict, \
        self.kind_task_tuple, self.time_arj_dict, self.module_rj_dict, self.machine_module_dict, self.time_add_dict, self.time_rem_dict, \
        self.cost_add_dict, self.cost_rem_dict, self.machine_module_rj_dict, self.module_machine_dict, self.test, self.test_rja = self.process_information()

        self.sum_kind_task_count = tuple(sum(values) for values in zip(*self.count_sr_dict.values()))

    """每道工序的加工时间"""
    @property
    def t_rja(self):
        return randint(10, 100)

    """每个订单中包含的每种工件类型的工件数量"""
    @property
    def N_sr(self):
        return randint(5, 10)  # (5,50)

    """三角模糊数时间"""
    @property
    def tfn_t(self):
        l = randint(10, 50)
        m = l + randint(10, 30)
        u = m + randint(10, 30)
        return (l, m, u)

    """三角模糊数成本"""
    @property
    def cost(self):
        return randint(10, 50)

    """订单到达时间间隔"""
    @property
    def t_si(self):
        return uniform(100, 200)

    def _generate_module_selection_cover_all(self, select_keys, module_keys, min_select=1, max_select=4):
        """
        为机器或工序生成可选模块集合。

        保证：
        1. 每个 key 至少选择 min_select 个模块，最多选择 max_select 个模块；
        2. 所有模块 module_keys 都至少被一个 key 选择；
        3. 返回值为 {key: tuple(sorted(modules))}。
        """
        select_keys = tuple(select_keys)
        module_keys = tuple(module_keys)

        if len(select_keys) == 0:
            raise ValueError("select_keys 不能为空")
        if len(module_keys) == 0:
            raise ValueError("module_keys 不能为空")
        if min_select < 1:
            raise ValueError("min_select 必须 >= 1")
        if max_select < min_select:
            raise ValueError("max_select 必须 >= min_select")
        if len(module_keys) > len(select_keys) * max_select:
            raise ValueError(
                f"模块数量 {len(module_keys)} 超过容量 {len(select_keys) * max_select}，"
                f"无法保证每个模块都被选择且每个对象最多选择 {max_select} 个模块"
            )

        result = {key: set() for key in select_keys}

        # 第一步：先强制覆盖所有模块。
        # 构造容量槽位，每个 key 最多出现 max_select 次。
        slots = []
        for key in select_keys:
            slots.extend([key] * max_select)
        np.random.shuffle(slots)

        modules = list(module_keys)
        np.random.shuffle(modules)

        for module, key in zip(modules, slots):
            result[key].add(module)

        # 第二步：保证每个 key 至少 min_select 个模块。
        for key in select_keys:
            while len(result[key]) < min_select:
                candidates = [a for a in module_keys if a not in result[key]]
                if not candidates:
                    break
                result[key].add(int(np.random.choice(candidates)))

        # 第三步：随机补充，但不超过 max_select，也不超过模块总数。
        upper_bound = min(max_select, len(module_keys))
        for key in select_keys:
            target_size = randint(max(min_select, len(result[key])), upper_bound)
            while len(result[key]) < target_size:
                candidates = [a for a in module_keys if a not in result[key]]
                if not candidates:
                    break
                result[key].add(int(np.random.choice(candidates)))

        # 最后再检查一次全集覆盖和数量范围。
        selected_union = set(a for values in result.values() for a in values)
        missing_modules = set(module_keys) - selected_union
        if missing_modules:
            raise ValueError(f"模块未被覆盖: {sorted(missing_modules)}")

        for key, values in result.items():
            if not (min_select <= len(values) <= max_select):
                raise ValueError(f"{key} 选择模块数量错误: {sorted(values)}")

        return {key: tuple(sorted(values)) for key, values in result.items()}

    def _check_internal_cover(self, module_rj_dict, machine_module_dict):
        """检查内存中的机器/工序模块覆盖情况。"""
        all_modules = set(self.module_tuple)

        process_union = set(a for modules in module_rj_dict.values() for a in modules)
        machine_union = set(a for modules in machine_module_dict.values() for a in modules)

        if process_union != all_modules:
            raise ValueError(
                f"工序 module_selectable 没有覆盖全部模块，缺失: {sorted(all_modules - process_union)}"
            )
        if machine_union != all_modules:
            raise ValueError(
                f"机器 module_selectable 没有覆盖全部模块，缺失: {sorted(all_modules - machine_union)}"
            )

        for key, modules in module_rj_dict.items():
            if not (1 <= len(modules) <= 4):
                raise ValueError(f"工序 {key} 的模块数不是 1-4: {modules}")

        for key, modules in machine_module_dict.items():
            if not (1 <= len(modules) <= 4):
                raise ValueError(f"机器 {key} 的模块数不是 1-4: {modules}")

    def process_information(self):
        """生成实例文件数据，并存入csv文件"""
        # 机器、工件信息
        task_r_dict = {r: tuple(j for j in range(self.task_count)) for r in self.kind_tuple}  # [r]对应工序元组
        kind_task_tuple = tuple((r, j) for r in self.kind_tuple for j in task_r_dict[r])  # 工序类型元组

        # 关键修改 1：工序选择模块，必须覆盖全部模块，每道工序 1-4 个模块。
        module_rj_dict = self._generate_module_selection_cover_all(
            select_keys=kind_task_tuple,
            module_keys=self.module_tuple,
            min_select=1,
            max_select=4
        )

        # 关键修改 2：机器选择模块，必须覆盖全部模块，每台机器 1-4 个模块。
        machine_module_dict = self._generate_module_selection_cover_all(
            select_keys=self.machine_tuple,
            module_keys=self.module_tuple,
            min_select=1,
            max_select=4
        )

        # 先检查，防止后续用到错误数据。
        self._check_internal_cover(module_rj_dict, machine_module_dict)

        kind_task_a_dict = {
            a: tuple((r, j) for (r, j) in kind_task_tuple if a in module_rj_dict[(r, j)])
            for a in self.module_tuple
        }  # 模块对应工序元组

        module_machine_dict = {
            a: tuple(m for m in self.machine_tuple if a in machine_module_dict[m])
            for a in self.module_tuple
        }  # 模块对应机器元组

        # (r,j) 可选择机器：只要机器拥有该工序可选模块中的任意一个模块，就可加工。
        machine_rj_dict = {}
        for (r, j) in kind_task_tuple:
            selectable_machines = []
            for a in module_rj_dict[(r, j)]:
                for m in self.machine_tuple:
                    if a in machine_module_dict[m]:
                        selectable_machines.append(m)
            machine_rj_dict[(r, j)] = tuple(sorted(set(selectable_machines)))

        # 因为每个工序至少有 1 个模块，且所有模块都至少被 1 台机器覆盖，理论上不会为空；这里再硬校验。
        for (r, j), machines in machine_rj_dict.items():
            if len(machines) == 0:
                raise ValueError(f"工序 {(r, j)} 没有可选机器，请检查模块覆盖逻辑")

        machine_module_rj_dict = {
            (r, j): tuple(
                (m, a)
                for m in machine_rj_dict[(r, j)]
                for a in machine_module_dict[m]
                if a in module_rj_dict[(r, j)]
            )
            for (r, j) in kind_task_tuple
        }  # (r,j)对应的(m,a)元组

        time_rja_dict = {
            (r, j): {a: self.t_rja for a in module_rj_dict[(r, j)]}
            for (r, j) in kind_task_tuple
        }  # 工序在不同模块下的加工时间

        kind_task_m_dict = {
            m: tuple((r, j) for (r, j) in kind_task_tuple if m in machine_rj_dict[(r, j)])
            for m in self.machine_tuple
        }

        time_arj_dict = {
            a: {(r, j): time_rja_dict[(r, j)][a] for (r, j) in kind_task_a_dict[a]}
            for a in self.module_tuple
        }

        time_add_dict = {a: self.tfn_t for a in self.module_tuple}  # 模块的装载时间
        time_rem_dict = {a: self.tfn_t for a in self.module_tuple}  # 模块的卸载时间
        cost_add_dict = {a: self.cost for a in self.module_tuple}  # 模块的装载成本
        cost_rem_dict = {a: self.cost for a in self.module_tuple}  # 模块的卸载成本

        # 订单信息
        count_sr_dict = {s: tuple(self.N_sr for r in range(len(task_r_dict))) for s in self.order_tuple}  # [s][r]工件类型的数量
        time_interval_list = [self.t_si for s in range(self.order_count - 1)]  # 各订单的间隔时间
        time_interval_list.insert(0, 0)
        time_arrive_s_dict = {s: int(sum(time_interval_list[:s + 1])) for s in self.order_tuple}  # 各订单的到达时间

        # 检验是否存在模块未被机器/工序选择的情况
        test = set(a for modules in machine_module_dict.values() for a in modules)
        test_rja = set(a for modules in module_rj_dict.values() for a in modules)

        return task_r_dict, machine_rj_dict, kind_task_m_dict, time_rja_dict, count_sr_dict, time_arrive_s_dict, \
               kind_task_tuple, time_arj_dict, module_rj_dict, machine_module_dict, time_add_dict, time_rem_dict, \
               cost_add_dict, cost_rem_dict, machine_module_rj_dict, module_machine_dict, test, test_rja

    @staticmethod
    def _parse_tuple_cell(value):
        """读取 CSV 中形如 '(1, 2)' 或 '(1,)' 的元组字符串。"""
        parsed = ast.literal_eval(value)
        if isinstance(parsed, int):
            return (parsed,)
        return tuple(parsed)

    def _validate_written_csv(self, base_dir):
        """写完文件后反读 CSV 校验，确认文件内容本身满足覆盖约束。"""
        all_modules = set(self.module_tuple)

        process_file = os.path.join(base_dir, self.file_name, 'process_data.csv')
        process_union = set()
        with open(process_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                modules = self._parse_tuple_cell(row['module_selectable'])
                if not (1 <= len(modules) <= 4):
                    raise ValueError(f"CSV中工序模块数不是1-4: {row}")
                process_union.update(modules)

        if process_union != all_modules:
            raise ValueError(
                f"CSV中工序 module_selectable 没有覆盖全部模块，缺失: {sorted(all_modules - process_union)}"
            )

        machine_file = os.path.join(base_dir, self.file_name, 'machine_data.csv')
        machine_union = set()
        with open(machine_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                modules = self._parse_tuple_cell(row['module_selectable'])
                if not (1 <= len(modules) <= 4):
                    raise ValueError(f"CSV中机器模块数不是1-4: {row}")
                machine_union.update(modules)

        if machine_union != all_modules:
            raise ValueError(
                f"CSV中机器 module_selectable 没有覆盖全部模块，缺失: {sorted(all_modules - machine_union)}"
            )

    def write_file(self):
        """写入csv文件"""
        base_dir = '../HiFAMR-BMAPPO/data'
        os.makedirs(os.path.join(base_dir, self.file_name), exist_ok=True)  # 新建实例文件夹
        file_csv = {'based_data.csv': ['kind_count', 'machine_count', 'module_count', 'order_count'],
                    'process_data.csv': ['kind', 'task', 'machine_selectable', 'module_selectable', 'process_time'],
                    'machine_data.csv': ['machine', 'module_selectable'],
                    'module_data.csv': ['module', 'time_add_fuzzy', 'time_rem_fuzzy', "cost_add", "cost_rem"],
                    'order_data.csv': ['order', 'time_arrive', 'kind_number']}

        for csv_name, header in file_csv.items():
            data_file = os.path.join(base_dir, self.file_name, csv_name)
            with open(data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                rows = []  # 初始化写入数据
                if csv_name == 'based_data.csv':
                    rows.append([self.kind_count, self.machine_count, self.module_count, self.order_count])
                elif csv_name == 'process_data.csv':
                    for r in self.kind_tuple:
                        for j in self.task_r_dict[r]:
                            time_machine_tuple = tuple(
                                self.time_rja_dict[(r, j)][a]
                                for a in self.module_rj_dict[(r, j)]
                            )
                            rows.append([r, j, self.machine_rj_dict[(r, j)], self.module_rj_dict[(r, j)], time_machine_tuple])
                elif csv_name == 'module_data.csv':
                    for a in self.module_tuple:
                        rows.append([a, self.time_add_dict[a], self.time_rem_dict[a], self.cost_add_dict[a], self.cost_rem_dict[a]])
                elif csv_name == 'machine_data.csv':
                    for m in self.machine_tuple:
                        rows.append([m, self.machine_module_dict[m]])
                else:
                    for s in self.order_tuple:
                        rows.append([s, self.time_arrive_s_dict[s], self.count_sr_dict[s]])
                writer.writerows(rows)

        # 关键修改 3：写完后读取 CSV 再校验，确认落盘文件真的覆盖全部模块。
        self._validate_written_csv(base_dir)
        print("写入完成")


if __name__ == '__main__':
    file_name_list = []
    M_list = [4, 8, 12]
    R_list = [4, 6, 8]
    J_list = [2, 3, 4]
    RJ_list = [(R_list[n], J_list[n]) for n in range(3)]
    rate_list = [0.75, 1, 1.25]

    for m in M_list:
        for rate_value in rate_list:
            for (r, j) in RJ_list:
                instance_object = Instance(m, round(rate_value * m), r, j)
                instance_object.write_file()
                file_name_list.append(instance_object.file_name)

    print(file_name_list)
