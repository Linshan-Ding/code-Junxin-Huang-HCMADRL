# -*- coding: utf-8 -*-
import copy
import csv, os, re


class Data():
    def __init__(self, path=None, file_name=None):
        self.file_name = file_name  # 算例文件名
        self.csv_name_list = ['based_data.csv', 'process_data.csv','machine_data.csv', 'module_data.csv','order_data.csv']
        self.path = path  # 路径
        # 基础数据信息 工件类型数， 机器数， 订单数, 延期度
        self.kind_count, self.machine_count, self.module_count,self.order_count = self.read(self.csv_name_list[0])
        self.machine_tuple = tuple(m for m in range(self.machine_count))  # 机器元组
        self.order_tuple = tuple(s for s in range(self.order_count))  # 订单元组
        self.kind_tuple = tuple(r for r in range(self.kind_count))  # 工件类型元组
        self.module_tuple = tuple(a for a in range(self.module_count))  # 模块元组
        # 订单信息 订单中各类型工件数量， 订单到达时间，
        self.count_sr_dict, self.time_arrive_s_dict = self.read(self.csv_name_list[4])
        # 加工信息 工序元组索引，可选机器元组索引，可选模块元组索引，在各模块上的加工时间
        self.task_r_dict, self.machine_rj_dict, self.module_rj_dict,self.time_rja_dict = self.read(self.csv_name_list[1])
        # 机器信息 可选择模块元组索引
        self.machine_module_dict = self.read(self.csv_name_list[2])
        # 模块信息 模块装载时间，模块卸载时间，模块装载成本，模块卸载成本
        self.time_add_dict, self.time_rem_dict, self.cost_add_dict, self.cost_rem_dict = self.read(self.csv_name_list[3])
        self.kind_task_m_dict, self.kind_task_tuple, self.time_arj_dict,self.kind_task_a_dict = self.process()
        self.machine_module_rj_dict = {(r, j): tuple(
            (m, a) for m in self.machine_rj_dict[(r, j)] for a in self.module_rj_dict[(r, j)] if
            a in self.machine_module_dict[m]) for (r, j) in self.kind_task_tuple}  # (r,j)对应的(m,a)元组
        self.module_machine_dict = {a: tuple(m for m in self.machine_tuple if a in self.machine_module_dict[m]) for a in self.module_tuple}  # 模块对应机器元组
        self.sum_kind_task_count =  tuple(sum(values) for values in zip(*self.count_sr_dict.values()))
    def process(self):
        """生成额外索引"""
        kind_task_tuple = tuple((r, j) for r in self.kind_tuple for j in self.task_r_dict[r])  # 工序类型元组
        kind_task_m_dict = {m: tuple((r, j) for (r, j) in kind_task_tuple if m in self.machine_rj_dict[(r, j)]) for m in self.machine_tuple}
        kind_task_a_dict = {a: tuple((r, j) for (r, j) in kind_task_tuple if a in self.module_rj_dict[(r, j)]) for a in self.module_tuple}
        time_arj_dict = {a: {(r, j): self.time_rja_dict[(r, j)][a] for (r, j) in kind_task_a_dict[a]} for a in self.module_tuple}

        return kind_task_m_dict, kind_task_tuple, time_arj_dict,kind_task_a_dict

    def str_int_tuple(self, s):
        """字符串中提取数字并转为int类型元组"""
        nums = [int(i) for i in re.findall(r'\d+', s)]
        return tuple(nums)

    def str_int(self, s):
        """字符串中提取所有数字，如果是单个数字返回int，多个数字返回tuple"""
        nums = [int(i) for i in re.findall(r'\d+', s)]
        if len(nums) == 1:
            return nums[0]  # 单个数字返回int
        else:
            return tuple(nums)  # 多个数字返回tuple

    def read(self, csv_name):
        """读取文件数据"""
        data_file = os.path.join(self.path, self.file_name, csv_name)
        with open(data_file, 'r') as f:
            reader = csv.reader(f)
            rows = []
            for row in reader:
                rows.append(row)
        if csv_name == 'based_data.csv':
            kind_count = self.str_int(rows[1][0])
            machine_count = self.str_int(rows[1][1])
            module_count = self.str_int(rows[1][2])
            order_count = self.str_int(rows[1][3])
            return kind_count, machine_count,module_count,order_count
        elif csv_name == 'process_data.csv':
            task_r_dict = {kind: [] for kind in self.kind_tuple}
            machine_rj_dict = {}
            module_rj_dict = {}
            time_rja_dict = {}
            for row in rows[1:]:
                kind = self.str_int(row[0])
                task = self.str_int(row[1])
                machine_str = self.str_int_tuple(row[2])
                module_str = self.str_int_tuple(row[3])
                time_str = self.str_int_tuple(row[4])
                task_r_dict[kind].append(task)
                machine_rj_dict[(kind, task)] = machine_str
                module_rj_dict[(kind, task)] = module_str
                time_rja_dict[(kind, task)] = time_str
            # 预处理数据
            for key, value in task_r_dict.items():
                task_r_dict[key] = tuple(value)
            for kind in self.kind_tuple:
                for task in task_r_dict[kind]:
                    time_module = {}
                    for module, time in zip(module_rj_dict[(kind, task)], time_rja_dict[(kind, task)]):
                        time_module[module] = time
                    time_rja_dict[(kind, task)] = copy.deepcopy(time_module)
            return task_r_dict, machine_rj_dict, module_rj_dict,time_rja_dict
        elif csv_name == 'machine_data.csv':
            machine_module_dict = {}
            for row in rows[1:]:
                machine = self.str_int(row[0])
                module_str = self.str_int_tuple(row[1])
                machine_module_dict[machine] = module_str
            return machine_module_dict
        elif csv_name == 'module_data.csv':
            time_add_dict = {}
            time_rem_dict = {}
            cost_add_dict = {}
            cost_rem_dict = {}
            for row in rows[1:]:
                module = self.str_int(row[0])
                # print(module)
                time_add = self.str_int(row[1])
                # print(time_add)
                time_rem = self.str_int(row[2])
                cost_add = self.str_int(row[3])
                cost_rem = self.str_int(row[4])
                time_add_dict[module] = time_add
                time_rem_dict[module] = time_rem
                cost_add_dict[module] = cost_add
                cost_rem_dict[module] = cost_rem
            return time_add_dict, time_rem_dict, cost_add_dict, cost_rem_dict
        else:
            count_sr_dict = {}
            time_arrive_s_dict = {}
            for row in rows[1:]:
                order = self.str_int(row[0])
                time_arrive = self.str_int(row[1])
                kind_count = self.str_int_tuple(row[2])
                count_sr_dict[order] = kind_count
                time_arrive_s_dict[order] = time_arrive
            return count_sr_dict, time_arrive_s_dict


# 测试
# if __name__ == '__main__':
    # file_name = 'M8_A6_S3'
    # path = 'D:\Python project\HCMAGRL\data'
    # file_name_list \
    #     = ['M8_A4_S1', 'M8_A6_S1', 'M8_A8_S1', 'M8_A4_S3', 'M8_A6_S3', 'M8_A8_S3', 'M8_A4_S5', 'M8_A6_S5', 'M8_A8_S5',
    #        'M12_A6_S1', 'M12_A9_S1', 'M12_A12_S1', 'M12_A6_S3', 'M12_A9_S3', 'M12_A12_S3', 'M12_A6_S5', 'M12_A9_S5', 'M12_A12_S5',
    #        'M16_A8_S1', 'M16_A12_S1', 'M16_A16_S1', 'M16_A8_S3', 'M16_A12_S3', 'M16_A16_S3', 'M16_A8_S5', 'M16_A12_S5', 'M16_A16_S5']
    #
    # instance = Data(path, file_name)
    # print("kind_task_tuple:", instance.kind_task_tuple)
    # print("time_arj_dict:", instance.time_arj_dict)
    # print("kind_task_a_dict:", instance.kind_task_a_dict)
    # print("count_sr_dict:", instance.count_sr_dict)
    # print("sum_kind_task_count:", instance.sum_kind_task_count)


