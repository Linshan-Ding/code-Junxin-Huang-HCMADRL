import random


import Triangular_fuzzy as tf
from graph_state import GraphState
from config import config


class MO_DFRMS_Environment(GraphState):
    def __init__(self, use_instance=False, **kwargs):
        super().__init__(use_instance=use_instance, **kwargs)
        self.step_count = 0  # 决策步
        self.gap_step_time = 0 #相邻决策点时间增量
        self.gap_step_cost = 0 #相邻决策点成本值的增量
        self.step_time = 0  # 决策时间点
        self.opt_count = 0 #本次事件内完成的工序数
        self.order_arrive_time = 0 #订单到达时间
        self.order_object_list = [] #未处理订单
        self.upper_reward =0  # 上层智能体即时奖励
        self.lower_reward = 0 #  下层智能体即时奖励
        self.term_reward = 0 #   终局即时奖励
        self.upper_reward_sum = 0  # 上层智能体累计奖励
        self.lower_reward_sum = 0 #  下层智能体累计奖励
        self.term_reward_sum = 0 #   终局累计奖励
        self.done = False  # 是否为终止状态
        self.completion_time = 0  # 当前时间点完工时间
        self.completion_time_last = 0  # 上一时间步完工时间
        # 状态属性
        self.state = None  # 图状态
        self.fluid_solution = None
        # 记录甘特图
        self.schedule_log = []

    def reset(self):
            """重置环境状态"""
            self.order_object_list = [self.order_dict[s] for s in self.order_tuple]  # 未到达订单对象列表
            self.reset_parameter()  # 初始化参数对象中的列表和字典
            self.reset_object_add(self.order_object_list[0])  # 新订单到达后更新各字典对象
            self.order_object_list.remove(self.order_object_list[0])  # 更新未到达订单对象列表
            # 重置回报相关属性
            self.upper_reward = 0  # 上层智能体即时奖励
            self.lower_reward = 0  # 下层智能体即时奖励
            self.term_reward = 0 # 终局即时奖励
            self.upper_reward_sum =0 # 上层智能体累计奖励
            self.lower_reward_sum = 0  # 下层智能体累计奖励
            self.term_reward_sum = 0  # 终局累计奖励
            self.fluid_solution = None #流体模型求解结果
            # 重置时间步和当前时间点
            self.step_count = 0
            self.step_time = 0
            # 重置上一时间步和当前时间步观察到的状态
            self.state = None # 状态
            self.done = False  # 是否为终止状态
            self.gap_step_time = 0 #相邻决策点时间增量
            self.gap_step_cost = 0 #相邻决策点成本值的增量


    def upper_agent_actions(self):
        """上层智能体决策机器选择模块"""
        #处于加工或者重构的制造单元不可选
        avail_machine_list = [m for m in self.machine_tuple]
        for m in self.machine_tuple:
            machine_object = self.machine_dict[m]
            if machine_object.reconfig_state == 1 or machine_object.proc_state == 1 :
                avail_machine_list.remove(m)
        #(m,a)机器模块对动作，包括不改变自身现在模块
        ma_actions = list((m,a) for m in avail_machine_list for a in self.machine_module_dict[m])
        ma_actions = [(m, a) for m, a in ma_actions
                      if any(self.kind_task_dict[o].buffer > 0 for o in self.kind_task_a_dict[a])]
        return tuple(ma_actions)

    def lower_agent_actions(self, m, a):
        """下层智能体决策机器选择哪种工序"""
        machine_list = list(self.machine_tuple)
        # 上层agent决策是否为空动作
        if self.machine_dict[m].module != a:
            machine_list.remove(m)
        # 处于重构状态或者加工状态或者无模块机器不可选
        for m in machine_list[:]:
            machine_object = self.machine_dict[m]
            if machine_object.reconfig_state == 1 or machine_object.proc_state == 1 or machine_object.module is None:
                machine_list.remove(m)
        # print("下层智能体可选机器列表：", machine_list)
        machine_rj_dict = {}
        if machine_list:
            for m in machine_list:
                a = self.machine_dict[m].module
                machine_rj_dict[m] = tuple(o for o in self.kind_task_a_dict[a] if self.kind_task_dict[o].buffer > 0)
            # 生成可用的操作元组
            mrj_actions = tuple((m, r, j) for m in machine_rj_dict for (r, j) in machine_rj_dict[m])
            return mrj_actions
        else:
            return tuple()

    def get_idle_machine(self):
        """无空闲机器并且机器无可加工工序推进仿真时间钟"""
        machine_list = list(self.machine_tuple)
        for m,machine_object in self.machine_dict.items():
            if machine_object.reconfig_state == 1 or machine_object.proc_state == 1 :
                machine_list.remove(m)
        for m in machine_list[:]:
            a = self.machine_dict[m].module
            if a is None:
                continue
            module_list = list(self.machine_module_dict[m])
            if len(module_list) == 1:
                a = module_list[0]
                if all(self.kind_task_dict[o].buffer ==  0 for o in self.kind_task_a_dict[a]):
                    machine_list.remove(m)
            else:
                if all(self.kind_task_dict[o].buffer == 0 for a in module_list for o in self.kind_task_a_dict[a]):
                    machine_list.remove(m)
        return machine_list

    def fluid_model_state(self):
        """流体模型状态"""
        state = False #false表示流体模型不可解，true表示流体模型可解
        m_list = list( m for m in self.machine_tuple if self.machine_dict[m].module is not None)
        for m in m_list:
            a = self.machine_dict[m].module
            if any(self.kind_task_dict[o].buffer > 0 for o in self.kind_task_a_dict[a]):
                state = True
                break
        return state

    def graph_state(self):
        current_state = {
            'operation_to_operation': self.operation_to_operation_edge_index,
            'operation_to_machine': self.operation_to_machine_edge_index,
            'operation_to_module': self.operation_to_module_edge_index,
            'machine_to_module': self.machine_to_module_edge_index,
            'operation_node_features': self.rj_feature_dict,
            'machine_node_features': self.m_feature_dict,
            'module_node_features': self.a_feature_dict,
            'operation_to_machine_edge_feat':self.operation_to_machine_edge_feat,
            'operation_to_module_edge_feat':self.operation_to_module_edge_feat,
            'machine_to_module_edge_feat':self.machine_to_module_edge_feat
        }
        return current_state

    def step(self, action):
        """
        输入动作 (m_u,a,m_l,r,j) , (m_u,a)
        """
        # print("执行动作：", action)
        step_time_before = self.step_time
        cost_sum_before = sum(
            self.machine_dict[m].reconfig_cost_sum
            for m in self.machine_tuple
        )
        self.opt_count = 0
        m_u, a_u = action[0],action[1]
        machine_object = self.machine_dict[m_u]
        a = machine_object.module
        #更新机器相关属性
        if machine_object.module == a_u:
            machine_object.reconfig_cost_sum = machine_object.reconfig_cost_sum + 0
            machine_object.reconfig_cost =  0
            machine_object.reconfig_time = 0
        if machine_object.module != a_u :
            machine_object.reconfig_state = 1
            machine_object.module_object = a_u
            if a is not None:
                machine_object.reconfig_time_end = self.step_time + tf.Triangular_fuzzy_time(self.time_rem_dict[a],self.time_add_dict[a_u])
                machine_object.reconfig_time = tf.Triangular_fuzzy_time(self.time_rem_dict[a],self.time_add_dict[a_u])
                machine_object.reconfig_cost_sum =machine_object.reconfig_cost_sum + self.cost_rem_dict[a] + self.cost_add_dict[a_u]
                machine_object.reconfig_cost = self.cost_rem_dict[a] + self.cost_add_dict[a_u]
            else:
                machine_object.reconfig_time_end = self.step_time + tf.Triangular_fuzzy_time((0,0,0),self.time_add_dict[a_u])
                machine_object.reconfig_time = tf.Triangular_fuzzy_time((0,0,0),self.time_add_dict[a_u])
                machine_object.reconfig_cost_sum =machine_object.reconfig_cost_sum + 0 + self.cost_add_dict[a_u]
                machine_object.reconfig_cost = 0 + self.cost_add_dict[a_u]

        if len(action) == 5:

            (m_l,r,j) = (action[2],action[3],action[4])
            machine_object = self.machine_dict[m_l]
            kind_task_object = self.kind_task_dict[(r,j)]
            job_object = kind_task_object.buffer_list[0]
            n = job_object.batch
            a = machine_object.module
            #更新机器属性
            machine_object.proc_state = 1
            machine_object.job_object = (r,j,n)
            machine_object.proc_time_end = self.step_time + self.time_arj_dict[a][(r,j)]
            machine_object.proc_time = self.time_arj_dict[a][(r,j)]
            #更新工序属性
            kind_task_object.unprocessed_count -=1
            kind_task_object.buffer -= 1
            kind_task_object.buffer_list.remove(job_object)
            kind_task_object.processing_list.append(job_object)
        # 仿真时间钟推进
        while len(self.upper_agent_actions()) == 0:
            # print("无空闲机器，推进仿真时间钟")
            # 找最近加工完成/重构完成事件
            proc_time_list = [self.machine_dict[m].proc_time_end for m in self.machine_tuple if
                              self.machine_dict[m].proc_time_end > self.step_time]

            reconfig_time_list = [self.machine_dict[m].reconfig_time_end for m in self.machine_tuple if
                                  self.machine_dict[m].reconfig_time_end > self.step_time]
            # print("加工完成事件列表：", proc_time_list)
            # print("重构完成事件列表：", reconfig_time_list)
            proc_time_point = min(proc_time_list) if proc_time_list else float("inf")
            reconfig_time_point = min(reconfig_time_list) if reconfig_time_list else float("inf")
            self.step_time = min(proc_time_point, reconfig_time_point)

            for m, machine_object in self.machine_dict.items():
                machine_object.remain_proc_time = max(0, machine_object.proc_time_end - self.step_time)
                machine_object.remain_reconfig_time = max(0, machine_object.reconfig_time_end - self.step_time)
            # 订单到达时刻
            if len(self.order_object_list) > 0 and self.order_object_list[0].time_arrive <= self.step_time:
                # print("___________未加工完时新订单到达____________")
                order_object = self.order_object_list[0]
                self.order_object_list.remove(order_object)
                self.reset_object_add(order_object)
                self.order_arrive_time = order_object.time_arrive
            if len(self.order_object_list) > 0 and sum(
                    kind_task_object.unprocessed_count for (r, j), kind_task_object in
                    self.kind_task_dict.items()) == 0:
                # print("**********直接移动到下一个时间点***********")
                order_object = self.order_object_list[0]
                self.order_object_list.remove(order_object)
                self.reset_object_add(order_object)
                if self.fluid_model_state():
                    # 求解流体模型更新流体模型属性
                    self.fluid_solution = self.dynamic_fluid_model()
                    self.reset_fluid_parameter()
                    self.update_fluid_parameter(self.fluid_solution)
                self.order_arrive_time = order_object.time_arrive
                self.step_time = self.order_arrive_time  # 时钟移动到新订单到达时间点
            # 加工完成时刻
            if self.step_time == proc_time_point :
                for m,machine_object in self.machine_dict.items():
                    machine_object.remain_proc_time = max(0,machine_object.proc_time_end - self.step_time)
                    if machine_object.proc_time_end == self.step_time:
                        self.opt_count += 1
                        machine_object.proc_state = 0
                        (r,j,n) = machine_object.job_object
                        # print("加工完成事件：机器",m,"工序",(r,j))
                        self.kind_task_dict[(r,j)].total_count -=1
                        self.kind_task_dict[(r,j)].processed_count +=1
                        self.kind_task_dict[(r, j)].processing_list.remove(self.job_dict[(r, j, n)])
                        if (r, j + 1) in self.kind_task_tuple:
                            self.kind_task_dict[(r, j+1)].buffer += 1
                            self.kind_task_dict[(r, j+1)].buffer_list.append(self.job_dict[(r, j+1, n)])
            # 重构完成时刻
            if self.step_time == reconfig_time_point:
                for m, machine_object in self.machine_dict.items():
                    if machine_object.reconfig_time_end == self.step_time:
                        machine_object.reconfig_state = 0
                        # print("重构完成事件：机器", m, "模块", machine_object.module_object)
                        machine_object.module = machine_object.module_object

            if sum(kind_task_object.total_count for (r,j),kind_task_object in self.kind_task_dict.items()) == 0:
                self.done = True
                break
        self.step_count += 1
        # 本次决策步结束后的累计成本
        cost_sum_after = sum(
            self.machine_dict[m].reconfig_cost_sum
            for m in self.machine_tuple
        )
        # 相邻决策点时间增量
        self.gap_step_time = float(self.step_time - step_time_before)
        # 相邻决策点成本增量
        self.gap_step_cost = float(cost_sum_after - cost_sum_before)
        time_term = float(self.gap_step_time)
        cost_term = float(self.gap_step_cost)

        if getattr(config, "REWARD_NORMALIZE", False):
            time_term = time_term / max(float(config.MAKESPAN_REF), 1e-8)
            cost_term = cost_term / max(float(config.COST_REF), 1e-8)

        share_reward = (
                config.alpha_t * time_term
                + config.alpha_c * cost_term
        )

        self.upper_reward = -share_reward
        self.lower_reward = -share_reward

        self.upper_reward_sum += self.upper_reward   # 上层智能体累计奖励
        self.lower_reward_sum +=  self.lower_reward  # 下层智能体累计奖励
        self.state = self.graph_state()  # 当前图状态
        return self.state, self.upper_reward, self.lower_reward,  self.done




if __name__ == '__main__':
    file_name = 'M8_A6_S1'
    file_name_list = ['M8_A6_S1', 'M8_A8_S1', 'M8_A10_S1', 'M8_A6_S3', 'M8_A8_S3', 'M8_A10_S3', 'M8_A6_S5', 'M8_A8_S5',
                      'M8_A10_S5', 'M12_A9_S1', 'M12_A12_S1', 'M12_A15_S1', 'M12_A9_S3', 'M12_A12_S3', 'M12_A15_S3',
                      'M12_A9_S5', 'M12_A12_S5', 'M12_A15_S5', 'M16_A12_S1', 'M16_A16_S1', 'M16_A20_S1', 'M16_A12_S3',
                      'M16_A16_S3', 'M16_A20_S3', 'M16_A12_S5', 'M16_A16_S5', 'M16_A20_S5']
    path = r'F:\Flexible Reconfigurable Manufacturing System\HiFAMR-Sync-MAGRL\data'
    env = MO_DFRMS_Environment(use_instance=False, file_name=file_name, path=path)
    # print("kind_task_a_dict:",env.kind_task_a_dict)
    # print("module_rj_dict",env.module_rj_dict)
    # print("machine_module_dict",env.machine_module_dict)
    # print("machine_module_rj_dict",env.machine_module_rj_dict)
    # 随机测试环境
    # random.seed(0)

    cost_sum = 0
    # while env.step_count <= 80:
    for i in range(10):
        env.reset()
        while not env.done:
            # print("==========================================")
            # print("当前时间点：", env.step_time, "当前决策步：", env.step_count)
            # for o in env.kind_task_tuple:
            #     print(o,":",env.kind_task_dict[o].buffer, end=" ")
            for m in env.machine_tuple:
                machine_object = env.machine_dict[m]
                # print("\n机器",m,":模块",machine_object.module,"加工状态",machine_object.proc_state,"重构状态",machine_object.reconfig_state,end=" ")
            action_upper_list = list(env.upper_agent_actions())
            action_upper = random.choice(action_upper_list)
            action_lower_list =  list(env.lower_agent_actions(action_upper[0],action_upper[1]))
            if action_lower_list:
                action_lower = random.choice(action_lower_list)
                action = tuple(action_upper + action_lower)
                env.step(action)
            else:
                env.step(tuple(action_upper))
        cost_sum += sum(tuple(env.machine_dict[m].reconfig_cost_sum for m in env.machine_tuple ))
        print("测试轮次：",i+1,"总成本：",sum(tuple(env.machine_dict[m].reconfig_cost_sum for m in env.machine_tuple )),"总时间：",env.step_time)
    print("平均成本：",cost_sum/10)

