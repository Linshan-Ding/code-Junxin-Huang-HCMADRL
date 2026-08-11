"""
柔性作业车间调度基础类定义
"""
from docplex.mp.model import Model
from Instance_generate import Instance
from MO_DFRMS_instance_read import Data




class Order():
    """订单对象"""
    def __init__(self, s, arrive, count_kind):
        # 基本属性
        self.order_node = s
        self.time_arrive = arrive  # 到达时间
        self.count_kind = count_kind  # 包含的各种工件的数量
    def __repr__(self):
        return f"Order(s={self.order_node}, arrive={self.time_arrive}, count_kind={self.count_kind})"

class Kind():
    """工件类型类"""
    def __init__(self, r):
        self.kind_node = r  # 工件类型编号
        self.number = 0  # 该工件类型的工件数量



class Tasks():
    """定义工序o_rj类"""
    def __init__(self, r, j):
        # 基本属性
        self.kind = r  # 所属工件
        self.task = j  # 所属工序
        self.machine_tuple = None  # 可选加工机器编号
        self.module_tuple = None  # 可选加工模块编号
        self.machine_module_tuple = None  # 可选加工(m,a)元组
        # 附加属性
        self.unprocessed_count = 0 #该工序未完成工序数
        self.total_count = 0 #该工序总需完成工序数
        self.buffer = 0 #该工序的缓冲区内工序数量，即就绪工序的数量
        self.buffer_list = [] #该工序的缓冲区内工件列表
        self.processing_list = [] #正在加工该工序的工件列表
        self.processed_count = 0 #该工序已完成工序数
        # 流体相关属性
        self.fluid_process_rate_m_dict = {}  # 流体中被当前机器加工的速率
        self.fluid_time_sum = None  # 流体模型中该工序的加工时间
        self.fluid_rate_sum = None  # 流体模型中加工该工序的速率
        self.fluid_number = None  # 处于该工序段的流体数量
        self.fluid_unprocessed_number = None  # 未被加工的流体数
        self.fluid_unprocessed_number_start = None  # 订单到达时刻未被加工的流体数

class Job():
    def __init__(self,r,j,n):
        self.kind = r  # 所属工件
        self.opt = j  # 所属工序
        self.batch = n  # 批次

class Machine():
    """机器类"""
    def __init__(self, m):
        # 基本属性
        self.machine_node = m  # 机器编号
        self.kind_task_tuple = None  # 可选加工工序类型元组
        # 附加属性
        self.proc_state = 0  # 机器状态,是否加工中
        self.module = None  # 机器当前模块
        self.reconfig_state = 0  # 是否在重构
        self.proc_time_end = 0  # 机器完工时间
        self.reconfig_time_end = 0 #机器重构完成时间
        self.reconfig_time = 0 #机器重构时间
        self.proc_time = 0 #机器加工时间
        self.reconfig_cost = 0 #重构成本
        self.reconfig_cost_sum = 0  # 累计重构成本
        self.job_object = None  # 机器正在处理的工序对象
        self.module_object = None #机器正在重构的模块对象
        self.remain_proc_time = 0 #机器正在加工的工序剩余加工时间
        self.remain_reconfig_time = 0 #机器正在重构的模块剩余重构时间
        # 流体附加属性
        self.time_ratio_rj_dict = {}  # 流体解中分配给各工序类型的时间比例
        self.fluid_process_rate_rj_dict = {}  # 流体解中加工各工序类型的速率



class MO_DFRMS(Instance, Data):
    """柔性作业车间调度类"""
    def __init__(self, use_instance=True, **kwargs):
        if use_instance:
            Instance.__init__(self, **kwargs)
        else:
            Data.__init__(self, **kwargs)

        self.fluid_completed_time  = 0  # 流体解完工时间
        # 实例化工件类型、工件、工序类型、工序和机器对象字典
        self.kind_task_dict = {(r, j): Tasks(r, j) for r in self.kind_tuple for j in self.task_r_dict[r]}  # 工序类型对象字典
        self.order_dict = {s: Order(s, self.time_arrive_s_dict[s], self.count_sr_dict[s]) for s in self.order_tuple}  # 对象订单字典
        self.machine_dict = {m: Machine(m) for m in self.machine_tuple}  # 机器对象字典
        self.kind_dict = {r: Kind(r) for r in self.kind_tuple}  # 工件类型对象字典
        self.job_dict = {}
        self.process_rate_a_rj_dict = {a: {(r, j): 1 / self.time_arj_dict[a][(r, j)] for (r, j) in self.kind_task_a_dict[a]} for a in self.module_tuple}  # 模块加工流体速率
        # 初始化参数对象中的列表和字典
        self.reset_parameter()
        # 初始化各对象属性# 新订单到达后更新各字典对象
        self.reset_object_add(self.order_dict[0])
        # print(self.order_dict[0])
        # print("成功定义DFRMS类")

    def reset_parameter(self):
        """初始化各字典和参数"""
        for r in self.kind_tuple:
            self.kind_dict[r].number = 0
        for (r, j), kind_task_object in self.kind_task_dict.items():
            kind_task_object.machine_tuple = self.machine_rj_dict[(r, j)]  # 可选加工机器编号元组
            kind_task_object.module_tuple = self.module_rj_dict[(r, j)]  # 可选加工模块编号元组
            kind_task_object.machine_module_tuple = self.machine_module_rj_dict[(r, j)]  # 可选加工(m,a)元组
            kind_task_object.total_count = self.sum_kind_task_count[r]  # 该工序总需完成工序数
            kind_task_object.buffer = 0  # 该工序的缓冲区内工序数量，即就绪工序的数量
            kind_task_object.unprocessed_count = 0 #该工序未完成工序数
            kind_task_object.processed_count = 0 #该工序已完成工序数
            kind_task_object.buffer_list = [] #该工序的缓冲区内工件列表
            kind_task_object.processing_list = [] #正在加工该工序的工件列表
        for m, machine_object in self.machine_dict.items():
            a = machine_object.module
            machine_object.kind_task_tuple = self.kind_task_a_dict[a] if a is not None else tuple()  # 可选加工工序类型元组
            machine_object.proc_state = 0  # 机器状态
            machine_object.reconfig_state = 0 #是否在重构
            machine_object.module = None   # 机器当前模块
            machine_object.proc_time_end = 0  # 机器完工时间
            machine_object.reconfig_time_end = 0  # 机器重构完成时间
            machine_object.proc_time = 0  # 机器加工时间
            machine_object.reconfig_time = 0  # 机器重构时间
            machine_object.reconfig_cost = 0  # 重构成本
            machine_object.reconfig_cost_sum = 0  # 该机器的累计重构成本
            machine_object.job_object = None
            machine_object.module_object = None
            machine_object.remain_proc_time = 0
            machine_object.remain_reconfig_time = 0
        # self.machine_dict[0].module = self.machine_module_dict[0][0]  # 测试：初始时刻机器0装载初始模块


    def reset_fluid_parameter(self):
        """初始化流体属性参数"""
        for (r, j), kind_task_object in self.kind_task_dict.items():
            kind_task_object.fluid_process_rate_m_dict = {}  # 被各机器加工的速率
        for m, machine_object in self.machine_dict.items():
            machine_object.time_ratio_rj_dict = {}  # 流体解中分配给各工序类型的时间比例
            machine_object.fluid_process_rate_rj_dict = {}  # 流体解中加工各工序类型的速率
            machine_object.unprocessed_rj_dict = {}  # 未被m加工的工序o_rj的总数 (r,j)

    def reset_object_add(self, order_object):
        """
        :param order_object: 新到达的订单对象
        :return: 添加工序对象和机器对象+更新流体模型和属性
        对于装配工序而言每道工序的批量都是不同的
        """
        for r in self.kind_tuple:
            num_start = self.kind_dict[r].number  # 该工件类型当前工件数量+1即新工件的编号
            num_end = num_start + order_object.count_kind[r]  # 新工件的编号范围
            for n in range(num_start, num_end):
                for j in self.task_r_dict[r]:
                    job_object = Job(r,j,n)  # 实例化工件对象
                    self.job_dict[(r,j,n)] = job_object
                self.kind_task_dict[(r, 0)].buffer_list.append(self.job_dict[(r,0, n)])  # 订单到达时刻每类工序就绪工件列表
        for r in self.kind_tuple:
            self.kind_dict[r].number = self.kind_dict[r].number + order_object.count_kind[r]  # 更新工件类型对象字典
        # 更新工件类型字典、工序类型对象字典、工序对象字典、工件对象字典
        for (r,j),kind_task_object  in self.kind_task_dict.items():
            kind_task_object.unprocessed_count = kind_task_object.unprocessed_count + order_object.count_kind[r] #订单到达时刻每类工序未完成工序数
        for r in self.kind_tuple:
            self.kind_task_dict[(r,0)].buffer = self.kind_task_dict[(r,0)].buffer +  order_object.count_kind[r]
        # 初始化流体属性
        for (r, j), kind_task_object in self.kind_task_dict.items():
            kind_task_object.fluid_number = kind_task_object.buffer  # 处于该工序段的流体数量,就绪集内工序数量
            kind_task_object.fluid_unprocessed_number = kind_task_object.unprocessed_count  # 未被加工的流体数
            kind_task_object.fluid_unprocessed_number_start = kind_task_object.unprocessed_count  # 订单到达时刻未被加工的流体数量

        # 求解流体模型更新流体模型属性
        # x = self.dynamic_fluid_model()
        # # 初始化流体属性
        # self.reset_fluid_parameter()
        # # 基于流体解更新流体属性
        # self.update_fluid_parameter(x)

    def get_avail_task(self):
        #获取求解时刻系统可行工序路径
        avail_machine_tuple = tuple(m for m in self.machine_tuple if self.machine_dict[m].module is not None)
        kind_task_list = []
        for m in avail_machine_tuple:
            a = self.machine_dict[m].module
            for o in self.kind_task_a_dict[a]:
                kind_task_list.append(o)
        kind_task_list = list(set(kind_task_list)) #筛选重复工序
        rj_list = [
            (r, j) for (r, j) in kind_task_list
            if self.kind_task_dict[(r, j)].buffer > 0
        ]
        # 用于存储最终结果，避免重复
        result_list = []
        for (r, j) in rj_list:
            idx = self.task_r_dict[r].index(j)
            # 从 idx 开始连续添加
            current_idx = idx
            while current_idx < len(self.task_r_dict[r]):
                current_task = self.task_r_dict[r][current_idx]
                if (r, current_task) in kind_task_list:
                    if (r, current_task) not in result_list:  # 去重
                        result_list.append((r, current_task))
                    current_idx += 1  # 继续下一个工序
                else:
                    break  # 遇到不在 kind_task_list 中的工序，停止

        rj_list = result_list
        avail_kind_task_tuple = tuple(set(rj_list))


        machine_rj_dict = {m : tuple(o for o in avail_kind_task_tuple if o in self.kind_task_a_dict[self.machine_dict[m].module]) for m in avail_machine_tuple}
        # module_rj_dict = {self.machine_dict[m].module: machine_rj_dict[m] for m in avail_machine_tuple}
        return avail_machine_tuple,avail_kind_task_tuple,machine_rj_dict

    def dynamic_fluid_model(self):
        """
        动态流体模型求解
        发生订单到达或者重构时，重新求解流体模型
        """
        # 初始化模型对象
        model = Model('LP')
        # 获取当前重构点的可加工工序类型和对应机器
        avail_machine_tuple,avail_kind_task_tuple,machine_rj_dict = self.get_avail_task()
        # 定义决策变量
        var_list = {(m, (r, j)) for m in machine_rj_dict.keys() for (r, j) in machine_rj_dict[m]}
        # 定义决策变量上下界
        X = model.continuous_var_dict(var_list, lb=0, ub=1, name='X')
        # 各流体初始未加工数量
        fluid_number = {(r, j): self.kind_task_dict[(r, j)].fluid_unprocessed_number_start
                        for (r, j) in avail_kind_task_tuple}
        # print(fluid_number)
        #各流体加工速率
        process_rate_rj_sum = {}
        for (r,j) in avail_kind_task_tuple:
            process_rate_rj_sum[(r, j)] = sum(X[m, (r, j)] * self.process_rate_a_rj_dict[self.machine_dict[m].module][(r, j)]
                                           for m in avail_machine_tuple if (r,j) in  machine_rj_dict[m])
        # 定义目标函数
        model.maximize(model.min(process_rate_rj_sum[(r, j)]/fluid_number[(r, j)] for (r, j) in avail_kind_task_tuple))
        # 添加约束条件
        # 机器利用率约束
        model.add_constraints(model.sum(X[m, (r, j)] for (r, j) in machine_rj_dict[m] ) <= 1
                              for m in avail_machine_tuple)
        # 解的可行性约束
        model.add_constraints(process_rate_rj_sum[(r, j)] >= process_rate_rj_sum[(r, j+1)] for (r,j) in avail_kind_task_tuple if (r,j+1) in avail_kind_task_tuple )
        # 求解模型
        solution = model.solve()
        x = solution.get_value_dict(X)
        for key in var_list:
            if key not in x:
                x[key] = 0.0
        # 输出流体完工时间
        process_rate_rj_sum = {}
        for (r, j) in avail_kind_task_tuple:
            process_rate_rj_sum[(r, j)] = sum(
                x[m, (r, j)] * self.process_rate_a_rj_dict[self.machine_dict[m].module][(r, j)]
                for m in avail_machine_tuple if (r, j) in machine_rj_dict[m])
        self.fluid_completed_time = max(fluid_number[(r, j)] / process_rate_rj_sum[(r, j)] for (r, j) in avail_kind_task_tuple)
        # print("流体完工时间：", self.fluid_completed_time)
        # print(solution)
        return x

    def update_fluid_parameter(self, x):
        """基于流体解更新流体参数"""
        avail_machine_tuple, avail_kind_task_tuple, machine_rj_dict = self.get_avail_task()
        for (m, (r, j)), rate in x.items():
            machine_object = self.machine_dict[m]
            kind_task_object = self.kind_task_dict[(r, j)]
            machine_object.time_ratio_rj_dict[(r, j)] = rate  # 流体解中分配给各工序类型的时间比例
            kind_task_object.fluid_process_rate_m_dict[m] = rate*self.process_rate_a_rj_dict[machine_object.module][(r, j)]
            machine_object.fluid_process_rate_rj_dict[(r, j)] = rate*self.process_rate_a_rj_dict[machine_object.module][(r, j)]
        for (r, j) in avail_kind_task_tuple:
            kind_task_object = self.kind_task_dict[(r,j)]
            kind_task_object.fluid_rate_sum = sum(kind_task_object.fluid_process_rate_m_dict.values())  # 工序类型处理速率
            kind_task_object.fluid_time_sum = 1/kind_task_object.fluid_rate_sum  # 工序类型的加工时间


    def display_fluid_solution(self):
        """展示流体解的调度结果"""
        print("\n流体解的调度结果：")
        for m, machine_object in self.machine_dict.items():
            print(f"\n机器 {m} 的调度结果：")
            print("各工序类型的时间比例：", machine_object.time_ratio_rj_dict)
            print("各工序类型的加工速率：", machine_object.fluid_process_rate_rj_dict)
        for (r, j), kind_task_object in self.kind_task_dict.items():
            print(f"\n工序类型 ({r}, {j}) 的调度结果：")
            print("各机器的加工速率：", kind_task_object.fluid_process_rate_m_dict)
            print("总加工速率：", kind_task_object.fluid_rate_sum)
            print("加工时间：", kind_task_object.fluid_time_sum)
            print("流体数量：", kind_task_object.fluid_number)
            print("未加工流体数量：", kind_task_object.fluid_unprocessed_number)
            print("订单到达时刻未加工流体数量：", kind_task_object.fluid_unprocessed_number_start)


# 测试环境
if __name__ == '__main__':
    file_name = 'M8_A10_S3'
    file_name_list = ['M8_A6_S1', 'M8_A8_S1', 'M8_A10_S1', 'M8_A6_S3', 'M8_A8_S3', 'M8_A10_S3', 'M8_A6_S5', 'M8_A8_S5', 'M8_A10_S5', 'M12_A9_S1', 'M12_A12_S1', 'M12_A15_S1', 'M12_A9_S3', 'M12_A12_S3', 'M12_A15_S3', 'M12_A9_S5', 'M12_A12_S5', 'M12_A15_S5', 'M16_A12_S1', 'M16_A16_S1', 'M16_A20_S1', 'M16_A12_S3', 'M16_A16_S3', 'M16_A20_S3', 'M16_A12_S5', 'M16_A16_S5', 'M16_A20_S5']
    path = r'F:\Flexible Reconfigurable Manufacturing System\HiFAMR-Sync-MAGRL\data'
    dfrms_object = MO_DFRMS(use_instance=False, file_name=file_name, path=path)
    x = dfrms_object.dynamic_fluid_model()
    dfrms_object.update_fluid_parameter(x)
    dfrms_object.reset_object_add(dfrms_object.order_dict[1])
    print(dfrms_object.order_dict[1])
    x = dfrms_object.dynamic_fluid_model()
    dfrms_object.update_fluid_parameter(x)
    # dfrms_object.display_fluid_solution()
