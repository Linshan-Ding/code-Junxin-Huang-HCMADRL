# 工业案例实施验证记录

验证日期：2026-09-26。解释器：`E:\anaconda3\envs\python3.13\python.exe`（Python 3.13.7），运行与恢复检查使用 CPU。

## 已完成

- 下载未修改的开放获取原论文，对照第 15 页核验订单、20 个加工时间、切换成本和零件需求表。来源及 SHA-256 见 `data/industrial_motor/source/manifest.json`。
- 生成正式 16 件／80 道任务与 smoke 4 件／20 道任务的独立输入，校验整数时间刻度、全部兼容关系及成本拟合约束。
- `python -m industrial_case run --profile smoke` 完成 HCMAGRL、Flat、MLP 各 2 次 PPO 迭代，以及 SPT、SetupGreedy、TwoStage、NSGA-II。7 份评估均完成，轨迹校验全部通过。NSGA-II 使用 32 次评估。
- `python -m unittest industrial_case.test_pipeline -v` 的 8 项测试覆盖：数据及成本拟合、完整订单规则调度与重放、异常轨迹拒绝、固定批次顺序、HV、正式导出限制、断点恢复和 Windows spawn 采样一致性。
- 中断后恢复的模型张量与不中断训练逐项相等；检查点包含优化器和随机数状态。两个独立进程的 rollout 与相同种子的串行 rollout 指标一致。
- 检查新的 `--help`、既有训练入口 `--help` 和三个 README 的本地链接。当前 `docplex` 已安装，旧评估入口缺失 `train.py` 的问题仍保留。
- 原消融包导入不存在的 DDQN 等模块会阻断 Flat/MLP；已仅改为按需导入，未补造缺失算法或更改现有模型。
- 论文生成了案例参数表、成本拟合表、完整模块表、示意图和待实验占位。正文、标题、图注、表格及新参考文献已检查为蓝色，图内保留配色。
- 使用本机既有 Tectonic 0.17 编译整篇稿件成功；`paper_assets/scripts/check.py` 返回 `all checks passed`。检查器保留原标题／类文件的两项 overfull 提示（约 7.6 pt 和 117.1 pt），未修改原前置内容。
- Pareto／敏感性和甘特图绘图代码以真实 smoke 输出在临时目录中验证过；这些 smoke 图没有导入论文。

## 验证边界

**未启动 75 次正式训练，也没有正式工业案例性能结论。** 稿件中的结果表、运行时间表和结果图保持明确的待实验状态。单元测试中的完整订单规则计算只用于验证可行性，不构成完整方法比较。

首次实施只验证了 CPU 训练与恢复。随后做过 CPU／CUDA 的少量内存更新计时，见下节；尚未验证完整 CUDA 训练及 CUDA 断点恢复。没有安装或调用 CPLEX；单订单案例不进入流体求解分支。只有公开文献参数得到逐项核对，模块设计、成本分配及模糊时间仍是公开声明的重构假设。

本机实测包版本为 `torch 2.11.0+cu128`、`numpy 2.3.3`、`scipy 1.16.2`、`pandas 2.3.2`、`matplotlib 3.10.6`、`docplex 2.32.264`、`pymoo 0.6.2`。这些是记录值，不是声明的兼容版本范围；运行清单保存完整环境信息。

## README 与零参数入口重构验证（2026-09-26）

本次只新增 `scripts/industrial/` 的入口、集中运行设置及测试，并重写两份 README 的工业案例部分。实验预算仍来自 `industrial_case.common.profile`；网络、PPO、数据转换、评估、统计和签名计算均未修改。

### 实际执行

- 新增入口测试 **10 项通过**，约 30 秒。其中实际从一个与仓库无关、含空格的临时目录启动 `run_00_smoke.py`，在独立结果根目录获得 3 个检查点、7 份可行评估和完整聚合。
- 再次运行真实 smoke 入口，逐文件 SHA-256 确认检查点、训练历史、评估、搜索和聚合内容均未改写；两次启动日志分别保存。测试产物在临时目录内清理，仓库原 smoke 结果保留。
- 在临时目录故意修改运行签名，真实入口以非零退出码拒绝复用，原结果内容保持不变。未完成 full 结果和 smoke 正式导出均被拒绝，临时论文源码未被修改。
- 原有 `industrial_case.test_pipeline` 的 **8 项测试再次通过**，约 9 秒，包含完整订单规则审计、断点恢复张量一致性及 Windows spawn 采样一致性。
- 新 `run_08_compile_check.py` 真实编译了当前待实验稿件，并运行 `check.py`，退出码 0、`all checks passed.`。实际使用当前应用提供的 Tectonic 0.17.0，入口日志隔离到临时目录；未执行正式导出。
- 编译器仍报告原稿 overfull、样式文件编码、字体配置和 PDF 版本提示；检查器保留两项 overfull 提示。论文源码和工业实验数字未改动，没有将“检查通过”解释为所有排版提示消失。

### 静态检查与模拟调度

- 核对 10 个零参数脚本与 README 命令逐字一致、十节结构完整、本地链接有效；Python 文件通过 AST 语法解析。
- 测试核对主方法 25 次、消融 50 次、基线 32 份评估和学习方法 75 份评估的入口覆盖，使用原 profile 计算预期输出，不新建一套协议。
- 全阶段顺序与方法选择、CPU 默认值、`--resume`／结果目录转发、编译器优先级通过单元测试；这些是调度验证，**不是执行 75 次正式训练**。
- 模拟上游失败时，一键入口停止后续阶段并保留非零退出码及错误日志；实际子进程测试验证了含空格路径和标准输出记录。额外参数（包括对零参数入口误传 `--help`）会直接拒绝，不启动实验；需要帮助时查底层 CLI。
- 缺少编译器和编译失败路径通过模拟测试；编译失败时不会调用后续论文检查。产物缺失即使子进程退出码为 0，也会使入口失败。

### 耗时估计来源与边界

此前在本机 Ryzen 9 8945HX／RTX 5070 Ti Laptop 上进行了独立内存计时：完整 16 件／80 道任务、平衡权重，每种方法在 CPU 和 CUDA 各执行 3 轮，每轮 4 条 rollout 和一次 PPO 更新；不保存模型、不生成正式指标。剔除首轮进程启动后，CPU 每轮约为 HCMAGRL 5.04 秒、Flat 2.08 秒、MLP 2.29 秒。按 25 次／方法、500 轮／次线性外推训练约 32.7 小时；README 建议预留 36–48 小时。

这一估计只基于每方法两轮稳定计时，未覆盖各权重及训练后期行为，也未包含全部写盘开销，不能当作完整实验实测或耗时上界。CUDA 短测总时间与 CPU 接近，当前零参数入口默认 CPU。本次重构没有重跑计时、启动正式训练或生成工业案例性能结论。

复查入口测试：

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' -B -m unittest discover -s scripts/industrial -p test_runners.py -v
& 'E:\anaconda3\envs\python3.13\python.exe' -B -m unittest industrial_case.test_pipeline -v
```

## 接续命令

在代码仓库根目录执行：

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_all.py
```

此命令会执行正式训练，需由使用者主动启动；本次验收未运行。成功后分别执行，每条成功后再继续：

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_07_export_paper.py
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_08_compile_check.py
```

配置或执行代码变更后，在 `scripts/industrial/_settings.py` 设置新的 `RESULT_ROOT`（底层对应 `--output`）；不要修改旧运行清单以绕过签名检查。详细参数、覆盖行为与论文编译说明见[根 README](../README.md#工业电机案例)和[论文工具链 README](../paper_assets/README.md#工业案例论文导出)。
