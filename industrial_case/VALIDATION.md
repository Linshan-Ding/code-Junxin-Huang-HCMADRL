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

未验证 CUDA 训练；本机能检测到 GPU 不等于已完成该路径的运行验证。没有安装或调用 CPLEX；单订单案例不进入流体求解分支。只有公开文献参数得到逐项核对，模块设计、成本分配及模糊时间仍是公开声明的重构假设。

本机实测包版本为 `torch 2.11.0+cu128`、`numpy 2.3.3`、`scipy 1.16.2`、`pandas 2.3.2`、`matplotlib 3.10.6`、`docplex 2.32.264`、`pymoo 0.6.2`。这些是记录值，不是声明的兼容版本范围；运行清单保存完整环境信息。

## 接续命令

在代码仓库根目录执行：

```powershell
$Python = 'E:\anaconda3\envs\python3.13\python.exe'
$env:HCMAGRL_PAPER = 'D:\Python project\Junxin_Huang_HCMAGRL_RMS_FRT'
& $Python -m industrial_case run --profile full --resume
if ($LASTEXITCODE -ne 0) { throw '正式运行尚未完成' }
& $Python -m industrial_case export-paper --profile full
if ($LASTEXITCODE -ne 0) { throw '正式论文导出未完成' }
```

配置或执行代码变更后使用新的 `--output` 根目录；不要修改旧运行清单以绕过签名检查。详细参数、覆盖行为与论文编译说明见[根 README](../README.md#工业电机案例)和[论文工具链 README](../paper_assets/README.md#工业案例论文导出)。
