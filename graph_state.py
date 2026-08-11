

from class_MO_DFRMS import MO_DFRMS
import Triangular_fuzzy as tf
"""
3类节点特征：
工序节点,制造单元节点,模块节点
4类边特征：
"""


# 图状态类，包含节点特征和边特征
class GraphState(MO_DFRMS):
    def __init__(self, use_instance=False, **kwargs):
        super().__init__(use_instance=use_instance, **kwargs)

        #节点特征
        self.rj_feature_dict = {self.kind_task_tuple.index((r,j)) : self.operation_node_feature(r,j) for (r,j) in self.kind_task_tuple}
        self.m_feature_dict = {m + len(self.kind_task_tuple):self.machine_node_feature(m) for m in self.machine_tuple}
        self.a_feature_dict = {a + len(self.kind_task_tuple + self.machine_tuple):self.module_node_feature(a) for a in self.module_tuple}

        #(src,dst)
        self.operation_to_operation_edge_index,\
        self.operation_to_machine_edge_index,\
        self.operation_to_module_edge_index, \
        self.machine_to_module_edge_index = self.build_edge_index()

        #feat
        self.operation_to_machine_edge_feat = self.o2m_feat()
        self.operation_to_module_edge_feat = self.o2a_feat()
        self.machine_to_module_edge_feat = self.m2a_feat()


    """COO格式(src,dst,feat)替换邻接矩阵"""
    def build_edge_index(self):

        operation_to_operation_edge_index = None
        operation_to_machine_edge_index = None
        operation_to_module_edge_index = None
        machine_to_module_edge_index = None

        # o2o
        o2o_src = []
        o2o_dst = []
        for  r in self.kind_tuple:
            for j in self.task_r_dict[r]:
                if j != max(self.task_r_dict[r]):
                    idx = self.kind_task_tuple.index((r,j))
                    o2o_src.append(idx)
        for i in o2o_src:
            o2o_dst.append(i + 1)
        operation_to_operation_edge_index = [o2o_dst,o2o_src]

        #o2m
        o2m_src = []
        o2m_dst = []
        for o in self.kind_task_tuple:
            for n in range(len(self.machine_rj_dict[o])):
                o2m_src.append(self.kind_task_tuple.index(o))
        for o in self.kind_task_tuple:
            for m in self.machine_rj_dict[o]:
                o2m_dst.append(m + len(self.kind_task_tuple))
        operation_to_machine_edge_index = [o2m_src,o2m_dst]

        #o2a
        o2a_src = []
        o2a_dst = []
        for o in self.kind_task_tuple:
            for n in range(len(self.module_rj_dict[o])):
                o2a_src.append(self.kind_task_tuple.index(o))
        for o in self.kind_task_tuple:
            for a in self.module_rj_dict[o]:
                o2a_dst.append(a + len(self.kind_task_tuple)+len(self.machine_tuple))
        operation_to_module_edge_index = [o2a_src,o2a_dst]

        #m2a
        m2a_src = []
        m2a_dst = []
        for m in self.machine_tuple:
            for n in range(len(self.machine_module_dict[m])):
                m2a_src.append(len(self.kind_task_tuple) + self.machine_tuple.index(m))
        for m in self.machine_tuple:
            for a in self.machine_module_dict[m]:
                m2a_dst.append(a + len(self.kind_task_tuple)+len(self.machine_tuple))
        machine_to_module_edge_index = [m2a_src,m2a_dst]

        return operation_to_operation_edge_index,operation_to_machine_edge_index,operation_to_module_edge_index,machine_to_module_edge_index

    """工序节点状态特征"""
    def operation_node_feature(self,r,j):
        # 进度状态特征
        job_now_num = self.kind_task_dict[(r,j)].buffer / max(tuple(self.kind_task_dict[o].buffer for o in self.kind_task_tuple)) #处于该工序类型阶段的工件数归一化
        processing_rate = self.kind_task_dict[(r,j)].processed_count /   self.kind_task_dict[(r,j)].unprocessed_count +  self.kind_task_dict[(r,j)].processed_count #该工序的完成率

        #时间紧迫度特征
        pre_rj_num = j / (max(self.task_r_dict[r])+1) #前述工序类型数归一化
        remain_rj_num = (max(self.task_r_dict[r])-j) / (max(self.task_r_dict[r])+1) #后续工序类型数归一化
        # 柔性与模块需求特征
        machine_num = len(self.machine_rj_dict[(r, j)]) / self.machine_count # 可加工制造单元机器数归一化
        num = 0
        proc_time = []
        for m , machine_object in self.machine_dict.items():
            if machine_object.module is not None:
                if (r,j) in self.kind_task_a_dict[machine_object.module]:
                    num += 1
                    proc_time.append(self.time_rja_dict[(r, j)][machine_object.module])
        machine_module_num = num / self.machine_count # 可加工制造单元机器数归一化,即当前工序可加工制造单元机器数与总机器数之比，动态更新
        proc_min_time = min(proc_time) / max(proc_time) if proc_time else 0 # 最短加工时间归一化,即当前工序在当前配置上的最短加工时间与该工序在当前模配置上的最长加工时间之比，动态更新
        proc_avg_time = sum(proc_time) / len(proc_time) / max(proc_time) if proc_time else 0 # 平均加工时间归一化,即当前工序在当前配置上的平均加工时间与该工序在当前模配置上的最长加工时间之比，动态更新


        feature = [ job_now_num,processing_rate,pre_rj_num,remain_rj_num, machine_num, machine_module_num, proc_min_time,proc_avg_time]
        return feature

    """制造单元节点状态特征"""
    def machine_node_feature(self,m):
        a = self.machine_dict[m].module
        #运行状态特征
        is_idle = 1  if self.machine_dict[m].proc_state == 1 or self.machine_dict[m].reconfig_state == 1 else 0 # 是否空闲,0空，1占（重构/加工）
        is_processing = self.machine_dict[m].proc_state # 是否正在加工
        is_reconfigure = self.machine_dict[m].reconfig_state # 是否正在重构
        remain_proc_time = self.machine_dict[m].remain_proc_time # 剩余加工时间归一化,即正在加工的工序的剩余加工时间与总加工时间之比，动态更新
        remain_reconfig_time = self.machine_dict[m].remain_reconfig_time # 剩余重构时间归一化,即正在重构的机器的剩余重构时间与总加工时间之比，动态更新

        #配置状态特征
        module_num  = [0]* self.module_count
        if self.machine_dict[m].module is not None:
            module_num[self.machine_dict[m].module] = 1
        else:
            module_num = module_num
        #负荷与匹配特征
        if a is not None:
            opt_tuple = tuple(o for  o in self.kind_task_a_dict[a] if self.kind_task_dict[o].unprocessed_count > 0) # 当前配置可完成的工序类型集合
            remain_operation_num = len(opt_tuple) / len(self.kind_task_tuple)
            proc_avg_time =sum(tuple(self.time_rja_dict[o][a] for o in opt_tuple if self.kind_task_dict[o].unprocessed_count > 0)) / len(opt_tuple) / max(tuple(self.time_rja_dict[o][a] for o in opt_tuple if self.kind_task_dict[o].unprocessed_count > 0) )
        else:
            remain_operation_num = 0 # 当前制造单元上剩余加工工序数与总工序数之比，动态更新
            proc_avg_time = 0 # 平均加工时间归一化,即当前制造单元上所有可完成的工序的平均加工时间与所有工序在当前配置上的最长加工时间之比，动态更新
        task_machine_mismatch = len(self.kind_task_a_dict[self.machine_dict[m].module])/ len(self.kind_task_tuple) if self.machine_dict[m].module is not None else 0 #当前配置可执行工序类型数归一化

        #模糊重构相关特征
        avg_deterministic_reconfig_time = None #候选模块平均确定化重构时间归一化
        avg_fuzzy_span = None #候选模块平均模糊跨度归一化
        avg_reconfig_cost = None #候选配置重构成本归一化
        selectable_module = None
        reconfig_time = []
        reconfig_cost = []
        delta = []
        if self.machine_dict[m].module is None:
            selectable_module = self.machine_module_dict[m]
            for a in selectable_module:
                reconfig_time.append(tf.Triangular_fuzzy_time(self.time_add_dict[a],(0,0,0)))
                reconfig_cost.append(self.cost_add_dict[a])
                delta.append(tf.delta_fuzzy_time(self.time_add_dict[a],(0,0,0)))
        else:
            selectable_module = tuple( x for x in self.machine_module_dict[m] if x != self.machine_dict[m].module)
            for a in selectable_module:
                for A in self.machine_module_dict[m] :
                        reconfig_time.append(tf.Triangular_fuzzy_time(self.time_add_dict[a],self.time_rem_dict[A]))
                        delta.append(tf.delta_fuzzy_time(self.time_add_dict[a], self.time_rem_dict[A]))
                reconfig_cost.append(self.cost_add_dict[a]+self.cost_rem_dict[self.machine_dict[m].module])
        avg_deterministic_reconfig_time = sum(reconfig_time) / len(reconfig_time) / max(reconfig_time) if len(reconfig_time) !=0 else 0
        avg_fuzzy_span = sum(delta) / len(delta) / max(delta)   if len(delta) !=0 else 0
        avg_reconfig_cost = sum(reconfig_cost) / len(reconfig_cost) / max(reconfig_cost) if len(reconfig_cost) !=0 else 0

        feature = [is_idle, is_processing, is_reconfigure, remain_proc_time, remain_reconfig_time]+module_num+[
                    remain_operation_num, proc_avg_time, task_machine_mismatch, avg_deterministic_reconfig_time,
                    avg_fuzzy_span,avg_reconfig_cost]

        return feature

    """模块节点状态特征"""
    def module_node_feature(self,a):

        #类型特征
        module_type_onehot = [0] * self.module_count
        module_type_onehot[a] = 1
        task_num = len(self.kind_task_a_dict[a]) / len(self.kind_task_tuple)

        #重构属性特征
        avg_add_time = tf.Triangular_fuzzy_time(self.time_add_dict[a],(0,0,0)) / max(tuple(tf.Triangular_fuzzy_time(self.time_add_dict[A],(0,0,0)) for A in self.module_tuple))
        avg_add_span = tf.delta_fuzzy_time(self.time_add_dict[a],(0,0,0)) /  max(tuple(tf.delta_fuzzy_time(self.time_add_dict[A],(0,0,0)) for A in self.module_tuple))
        avg_rem_time = tf.Triangular_fuzzy_time((0,0,0),self.time_rem_dict[a]) / max(tuple(tf.Triangular_fuzzy_time((0,0,0),self.time_rem_dict[A]) for A in self.module_tuple))
        avg_rem_span = tf.delta_fuzzy_time((0,0,0),self.time_rem_dict[a]) / max(tuple(tf.delta_fuzzy_time((0,0,0),self.time_rem_dict[A]) for A in self.module_tuple))
        avg_add_cost = self.cost_add_dict[a] / max(self.cost_add_dict.values())
        avg_rem_cost = self.cost_rem_dict[a] / max(self.cost_rem_dict.values())

        #当前系统使用特征
        module_rate = len(tuple(m for m in self.machine_tuple if self.machine_dict[m].module == a)) / self.machine_count
        opt_a_list = list(self.kind_task_a_dict[a])
        opt_list = list(self.kind_task_tuple)
        for o in opt_a_list:
            if self.kind_task_dict[o].unprocessed_count == 0:
                opt_a_list.remove(a)
        for o in opt_list:
            if self.kind_task_dict[o].unprocessed_count  == 0:
                opt_list.remove(a)
        uncompleted_opt_rate = len(opt_a_list) / len(opt_list)

        feature = module_type_onehot + [task_num,avg_add_time,avg_rem_time,avg_add_span,avg_rem_span,avg_add_cost,avg_rem_cost,module_rate,uncompleted_opt_rate]

        return feature


    """制造单元-工序边"""
    def o2m_feat(self):

        o2m_feat = []
        for n in range(len(self.operation_to_machine_edge_index[0])):
            o =self.operation_to_machine_edge_index[0][n]
            (r, j) = self.kind_task_tuple[o]
            m = self.operation_to_machine_edge_index[1][n] - len(self.kind_task_tuple)
            proc_time = 0
            if self.machine_dict[m].module is not None :
                proc_time = self.time_arj_dict[self.machine_dict[m].module][(r,j)] / max(self.time_arj_dict[self.machine_dict[m].module].values()) if (r,j) in self.kind_task_a_dict[self.machine_dict[m].module] else 0
            o2m_feat.append([proc_time])

        return o2m_feat

    """工序-模块边"""
    def o2a_feat(self):

        o2a_feat = []

        time_set = {value for a in self.module_tuple for value in self.time_arj_dict[a].values()}
        for n in range(len(self.operation_to_module_edge_index[0])):
            o_idx = self.operation_to_module_edge_index[0][n]
            o = self.kind_task_tuple[o_idx]
            a = self.operation_to_module_edge_index[1][n] - len(self.kind_task_tuple) -len(self.machine_tuple)
            proc_time = self.time_arj_dict[a][o] / max(time_set)
            o2a_feat.append([proc_time])

        return o2a_feat

    """制造单元-模块边"""
    def m2a_feat(self):

        m2a_feat = []
        for n in range(len(self.machine_to_module_edge_index[0])):
            m = self.machine_to_module_edge_index[0][n] -len(self.kind_task_tuple)
            a = self.machine_to_module_edge_index[1][n] -len(self.kind_task_tuple)- len(self.machine_tuple)
            is_install = 1 if self.machine_dict[m].module == a else 0
            can_install= 1
            m2a_feat.append([is_install,can_install])

        return m2a_feat

def display(file_name = None ,path =None):
    node_feat = Node_feature(use_instance=False, file_name=file_name, path=path)
    # 打印节点特征字典
    print("=" * 50)
    print("工序节点特征 (rj_feature_dict):")
    for node_id, features in node_feat.rj_feature_dict.items():
        print(f"  节点 {node_id}: {features}")

    print("\n" + "=" * 50)
    print("制造单元节点特征 (m_feature_dict):")
    for node_id, features in node_feat.m_feature_dict.items():
        print(f"  节点 {node_id}: {features}")

    print("\n" + "=" * 50)
    print("模块节点特征 (a_feature_dict):")
    for node_id, features in node_feat.a_feature_dict.items():
        print(f"  节点 {node_id}: {features}")

    # 打印边索引
    print("\n" + "=" * 50)
    print("工序-工序边索引 (operation_to_operation_edge_index):")
    print(f"  {node_feat.operation_to_operation_edge_index}")

    print("\n工序-制造单元边索引 (operation_to_machine_edge_index):")
    print(f"  {node_feat.operation_to_machine_edge_index}")

    print("\n工序-模块边索引 (operation_to_module_edge_index):")
    print(f"  {node_feat.operation_to_module_edge_index}")

    print("\n制造单元-模块边索引 (machine_to_module_edge_index):")
    print(f"  {node_feat.machine_to_module_edge_index}")

    # 打印边特征

    print("\n工序-制造单元边特征 (operation_to_machine_edge_feat):")
    for i, feat in enumerate(node_feat.operation_to_machine_edge_feat):
        print(f"  边 {i}: {feat}")

    print("\n工序-模块边特征 (operation_to_module_edge_feat):")
    for i, feat in enumerate(node_feat.operation_to_module_edge_feat):
        print(f"  边 {i}: {feat}")

    print("\n制造单元-模块边特征 (machine_to_module_edge_feat):")
    for i, feat in enumerate(node_feat.machine_to_module_edge_feat):
        print(f"  边 {i}: {feat}")

if __name__ == '__main__' :

        file_name = 'M8_A6_S1'
        path = r'F:\Flexible Reconfigurable Manufacturing System\HiFAMR-Sync-MAGRL\data'
        display(file_name = file_name,path = path)
