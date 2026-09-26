# HCMAGRL：可重构制造系统的调度与重构联合优化

本仓库包含分层协同多智能体图强化学习方法的实现、27 个制造系统算例、实验结果，以及论文图表生成工具。优化目标是同时降低最大完工时间（makespan）和模块重构成本；模块装卸时间使用三角模糊数表示。

方法以工序、机器和模块构成异构图，利用带边注意力的图编码器提取状态特征。上层策略选择机器与模块的重构动作，下层策略选择加工调度动作；训练采用多个 worker 并行采样、主进程统一执行 mini-batch PPO 更新。

论文使用 **HCMAGRL** 名称；仓库名与原始结果目录保留 `HCMADRL`，代码中还保留 `HiFAMR-BMAPPO`、`HiFAMRSyncMAGRL` 等历史工程名和类名。论文中的制造单元（manufacturing cell）对应代码中的 `Machine`。阅读与运行时请保留实际文件名和标识。

## 阅读导航

- [目录与数据](#目录与数据)：核心模块、算例格式和已有实验结果。
- [环境准备](#环境准备)：Python 解释器、依赖与求解器前提。
- [快速开始](#快速开始)：准备输出目录、启动 Visdom、运行单算例训练。
- [权重实验与消融实验](#权重实验与消融实验)：批量训练配置及输出。
- [可视化与结果](#可视化与结果)：Pareto 图、甘特图、检查点与评估结果。
- [当前限制](#当前限制)：需要先调整的路径和暂不能直接运行的入口。
- [论文图表生成指南](paper_assets/README.md)：利用已有结果重生成论文图表与表格。
- [工业电机案例](#工业电机案例)：公开数据重构、独立训练与评估、论文蓝色修订产物。

希望查看现有实验结果的读者，可以从 `result/` 和 `paper_assets/figures/data/` 开始，无需先训练模型。工业案例已提供独立的数据、训练、评估和论文产物流程；原有 27 个合成算例的旧入口仍存在下文所列限制。

## 工业电机案例

本节是**补充实验完整复现手册**，按“自检 → 数据 → 训练 → 基线 → 评测 → 聚合 → 论文产物”执行。初次使用建议逐步运行；确认自检通过后，可用第 9 节的一键入口生成全部实验数据。本文所需的每项产物及其生成入口见第 10 节。

### 1. 问题与假设

使用 [Zhao 等公开论文](https://doi.org/10.1371/journal.pone.0348884) 中常州 AMEC&GBM 电机装配案例的参数，属于**基于公开工业案例重构的计算实验**，未进行现场采集或部署验证。保留 4 类产品、5 个工位、订单 `[5,4,4,3]`、20 个加工时间，共 80 道实际加工任务。五个工位映射为制造单元，每个单元配置四种产品专用逻辑模块，共 20 种模块，模块只兼容对应工位。

| 参数类别 | 内容 |
| --- | --- |
| 公开值 | 产品、订单量、工位加工时间、产品间换型成本矩阵；保留原始 PDF 和提取表 |
| 推导值 | 将非对角换型成本拟合为“卸载成本＋安装成本”，两组分量总和相等；保留拟合矩阵、残差 |
| 假设值 | 成本平均分配至五工位；安装／卸载模态时间为工位平均加工时间的 0.6／0.4 倍，三角模糊数为模态的 `(0.8,1,1.2)` 倍 |

初始配置为空、订单统一在零时刻释放、缓冲无限；不采用原文的固定投产间隔、零件供给频率及周期末回切约束。加工时间沿用原文 **time-unit**，不解释为分钟；读取器使用 1000 倍整数时间刻度，结果还原为 time-unit，成本统一为 CNY。初始安装成本是推导值，装卸时间不是实测工人数据。详情见[来源与数据说明](data/industrial_motor/README.md)。

本案例是固定路线、工位独立重构场景，其表现不能单独证明全部柔性路由能力。**当前流程是在工业案例上训练和评估，不是零样本迁移。** 旧模型的模块 one-hot 输入宽度可能与 20 模块案例不兼容，目前也未找到原合成算例的正式策略文件；因此本节不提供冻结旧策略直接迁移的命令。

### 2. 环境配置与计算预算

本节命令使用 PowerShell、本机解释器 `E:\anaconda3\envs\python3.13\python.exe`。其他设备替换解释器及仓库路径；实验脚本名和步骤不变。先在终端进入代码仓库：

```powershell
Set-Location 'D:\Python project\code-Junxin-Huang-HCMADRL'
& 'E:\anaconda3\envs\python3.13\python.exe' --version
& 'E:\anaconda3\envs\python3.13\python.exe' -c "import torch,numpy,scipy,pandas,matplotlib,visdom,docplex,pymoo; print('依赖导入成功'); print('CUDA available:', torch.cuda.is_available())"
```

仅在依赖缺失时安装，不必在每次运行前升级已有环境：

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' -m pip install torch numpy scipy pandas matplotlib visdom docplex pymoo
```

这些是包名，不是已验证的版本范围；本机版本及验证边界见[验证记录](industrial_case/VALIDATION.md)。新流程无需启动 Visdom 服务；`visdom`、`docplex` 仍由既有模块导入。本案例为单订单，不调用流体求解分支，无需 CPLEX 运行时。论文编译工具仅在最后的编译步骤需要，缺失不会阻止训练。

运行设置集中在 [scripts/industrial/_settings.py](scripts/industrial/_settings.py)：

| 设置 | 默认值与含义 |
| --- | --- |
| `RESULT_ROOT` | 当前仓库的 `result/industrial_motor`；自动再加 `smoke/` 或 `full/`，不要填到 profile 子目录 |
| `DEVICE` | `cpu`；控制 PPO 更新和最终评估，rollout worker 始终使用 CPU |
| `PAPER_ROOT` | 优先读取已设置的 `HCMAGRL_PAPER`，否则使用兄弟目录 `Junxin_Huang_HCMAGRL_RMS_FRT` |
| `LATEX_COMPILER` | `None`，自动查找现有 Tectonic／latexmk；也可填写可执行文件的绝对路径 |

相对设置路径均以代码仓库为基准。一般不需要编辑设置文件；需要隔离实验时，只更换 `RESULT_ROOT`。实验种子、权重和预算仍以现有 [profile 定义](industrial_case/common.py) 为唯一来源。

正式协议为 HCMAGRL、Flat、MLP × 五组时间权重 `0、0.25、0.5、0.75、1` × 五个种子 `42—46`，共 **75 次训练**；每次 500 次迭代、每轮 4 条 rollout，共 150,000 条训练 rollout。各运行依次执行，不能把 4 个 rollout worker 理解为同时进行 4 次独立训练。

| 阶段 | 本机时间预算 |
| --- | --- |
| 依赖已就绪时的 smoke | 分钟级；只有 4 件／20 道任务 |
| HCMAGRL：25 次正式训练 | 线性外推约 17.5 小时 |
| Flat、MLP：各 25 次 | 线性外推约 7.2、8.0 小时 |
| 全部基线 | 建议留 10–20 分钟 |
| 最终评估、聚合、导出、编译 | 合计建议留 10–30 分钟 |
| 完整流程 | 训练外推约 33 小时，安排上预留 **36–48 小时** |

以上来自 Ryzen 9 8945HX／RTX 5070 Ti Laptop 本机完整 80 道任务、平衡权重下每方法 3 轮的短测，排除了首轮启动开销，**不是完整训练实测或耗时上界**。不同权重、后期决策步数和持续负载会改变时间；可用正式前 20–50 轮重新估算。CUDA 短测总耗时与 CPU 接近，本节默认 CPU；正式训练及 CUDA 断点恢复均未在本次文档验收中运行。

### 3. 冒烟自检

**前置条件：**完成依赖检查。该入口自动准备和验证数据，随后执行三种学习方法各 2 次迭代，以及全部规则和 32 次 NSGA-II 评估。

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_00_smoke.py
```

**输出：**`result/industrial_motor/smoke/` 内的 3 个检查点、训练历史、7 份评估、搜索档案、4 个聚合 CSV 和清单；启动日志位于 `result/industrial_motor/runner_logs/`。

**成功判据：**进程退出码 0，结尾出现 `[FINISHED]`，审计行显示 7 份记录。本次已验证的 smoke 中 7 份均可行；若自己的运行出现不可行结果，应先查看评估记录的 `errors`／`termination` 再投入正式算力。文件存在性检查不能替代轨迹审计。

**耗时与重跑：**分钟级；再次执行会恢复或跳过匹配的已完成训练／评估，并重新聚合。smoke 只能验证流程，**不得用于论文数字或收敛结论**。

### 4. 数据准备

**前置条件：**仓库已有 `data/industrial_motor/source/` 中的公开 PDF 和来源清单；首次自检已执行过此步，可以安全再次核验。

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_01_prepare_data.py
```

**输出：**公开提取表、成本拟合残差、`derived.json`，以及 `MOTOR_FULL/` 和 `MOTOR_SMOKE/` 下各五种 CSV。

**成功判据：**退出码 0，输出加工任务数 80／20、兼容关系、整数时间刻度和源文件 SHA-256 校验通过。**耗时：**通常秒级。**重跑行为：**按照相同公开参数重生成派生文件；公开提取值已被改动时拒绝覆盖。该入口不会重新下载或替换源 PDF；来源文件缺失时应先恢复仓库原文件，按需下载仍使用底层 `prepare --fetch` 接口。

### 5. 训练主方法

**前置条件：**自检和数据核验通过，确定结果根目录与设备；长时间运行期间保持机器供电和唤醒。

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_02_train_hcmagrl.py
```

**输出：**`full/checkpoints/HCMAGRL_w*_s*.pt` 和 `full/training/HCMAGRL_w*_s*.json`，各 25 份，另有配置／代码／输入签名和固定 SPT 归一化参考值。

**成功判据：**25 次运行均完成第 500 轮，检查点标记完成，脚本退出码 0。进度中的 `feasible 4/4` 表示该轮四条采样轨迹的可行数；不表示最终策略已经完成独立评估。保存和评估的是**最终迭代模型**，不是按测试成绩挑选的模型。

**耗时与重跑：**外推约 17.5 小时。每轮原子更新模型、优化器、随机状态及历史；中断后运行同一命令从最后完整检查点继续。已完成的匹配运行由原训练器核验并跳过，不自动缩短预算。

### 6. 基线与消融

**前置条件：**数据核验通过；所有步骤使用同一个结果根目录。建议依次执行，不同时开启多个正式入口。

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_03_train_ablations.py
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_04_baselines.py
```

**输出与成功判据：**Flat／MLP 各 25 个完成的最终检查点及训练历史；基线生成 32 份评估记录、5 份 NSGA-II 搜索档案及 5 份搜索历史 CSV，各脚本退出码 0。

| 方法 | 正式执行次数及统计单位 |
| --- | --- |
| Flat、MLP | 各五权重 × 五训练种子，500 轮／次 |
| SPT、TwoStage | 各计算一次 |
| SetupGreedy | 每权重一次，共 5 次 |
| NSGA-II | 五搜索种子，每种子 10,000 次调度评估；从解集中按五权重选代表解，共 25 份评估 |

确定性规则的 7 份结果不复制成五个独立样本。NSGA-II 每种子的评估预算与单个学习方法五权重训练的 rollout 数对齐，但不意味着相同计算成本。TwoStage 先按成本枚举四类产品的 24 种批次顺序，再生成该顺序下的可行调度。

**耗时与重跑：**Flat／MLP 合计外推约 15.2 小时；基线建议留 10–20 分钟。学习方法支持逐轮恢复；基线跳过匹配的已有结果。NSGA-II 只在一个种子的搜索结束后保存完整档案，**中断正在进行的种子时，该种子需重新搜索**。

### 7. 评测与实验数据生成

**前置条件：**三种学习方法的 75 个最终检查点已完成。基线评估已由第 6 节生成，无需重复搜索。

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_05_evaluate.py
```

**输出：**`full/evaluations/` 新增 75 份学习方法记录，与基线合计 **107 份**。每份包含完工时间、拟合成本、原矩阵重算成本、重构次数、可行性、执行时间，以及加工／安装／卸载完整事件轨迹。

**成功判据：**75 次评估均有记录、脚本退出码 0；最终完整性和跨方法汇总检查在第 8 节执行。独立审计核对任务覆盖、工序先后、单元占用、安装状态以及目标值。达到 `20 × 实际工序数` 决策上限的轨迹按未完成记录，不伪装为可行解。

**耗时与重跑：**分钟级；跳过已有评估，聚合时再次检查记录身份及签名。**107 份记录齐全不等于 107 份都可行**，应阅读可行率和失败原因；不要删除失败样本来改善统计。

### 8. 统计聚合与论文接续

**前置条件：**三种学习方法及全部基线评估完成。

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_06_aggregate.py
```

**输出：**`full/aggregate/` 中的 `detail.csv`、`summary.csv`、`portfolios.csv`、`sensitivity.csv` 和 `manifest.json`。**成功判据：**退出码 0，完整性清单覆盖 107 份评估；缺失记录、错误签名或不足预算会报错。**耗时与重跑：**通常秒级至分钟级；重新核验原始记录，并覆盖当前运行的派生聚合文件。

统计口径固定如下：

- 平衡权重下报告完工时间、成本、重构次数、执行时间和可行率；均值／样本标准差按独立训练或搜索种子统计，并保留有效样本数。
- HV 使用每种子的五权重代表解集合；共同 SPT 值归一化，统一参考点为全部方法可行代表解各坐标最大值的 1.1 倍。规则只有实际产生的解，不补造重复点。
- 同时保留原换型矩阵按工位五等分的成本重算结果；初始安装成本仍采用推导值，周期末不回切。
- 时间敏感性针对平衡权重调度，保持单元活动顺序、工序优先关系及每次重构的工件就绪条件，按装卸时间倍率 `0.5、1、2` 重算最早开工时刻。倍率 1 也可能消除原空闲；这是固定调度方案重放，不是随机工人环境下的策略鲁棒性测试。

全部正式记录通过检查后，再单独执行论文导出和编译：

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_07_export_paper.py
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_08_compile_check.py
```

第 07 步生成工业案例独立命名的表格、三张图、结果宏和蓝色结果文字；固定种子 42、平衡权重的 HCMAGRL 与 TwoStage 甘特图不择优挑选。第 08 步编译 `main.pdf`，保留日志，并运行 `check.py`。各自退出码为 0 才算该步骤成功，最后仍需目检 PDF。

导出缺少正式数据时在写论文产物前失败；不能用 smoke 替代。编译入口也可独立检查当前待实验草稿。详细编译器查找、输入输出和覆盖范围见[工业案例论文导出](paper_assets/README.md#工业案例论文导出)。

### 9. 启动、恢复与排障

**一键生成全部实验数据：**完成第 2 节环境检查后，可使用以下命令替代逐步执行。它顺序执行 00—06，任何一步失败立即停止；已完成训练会核验后跳过。

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_all.py
```

该命令包含正式训练，建议预留 36–48 小时；只生成实验数据，不自动写入论文。之后按第 8 节运行 07、08。逐条手动运行时，每条结束后先检查退出状态和日志；普通 PowerShell 粘贴多条命令本身不提供跨步骤失败即停保证，需要自动停止时使用 `run_all.py`。

**PyCharm：**右键目标 `run_*.py` → Run；或在 Run → Edit Configurations 中新建 Python 配置：

| 字段 | 填写内容 |
| --- | --- |
| Script path | 项目内对应的 `scripts/industrial/run_*.py`；全流程选择 `run_all.py` |
| Parameters | 留空 |
| Python interpreter | `E:\anaconda3\envs\python3.13\python.exe` |
| Working directory | `D:\Python project\code-Junxin-Huang-HCMADRL` |
| Environment variables | 无必填项；子进程自动启用 UTF-8、无缓冲输出和禁写字节码 |

入口按自身文件位置定位仓库，即使 IDE 工作目录不同也能启动。所有实验循环都在 Python 内部完成，脚本不接收命令行参数；高级调参与单方法运行继续使用 `python -m industrial_case`，其参数含义见各子命令的 `--help`。

**恢复与输出隔离：**

- 中断后使用同一入口、同一结果目录和相同设备恢复。训练恢复模型、优化器和随机状态；正在写入但未完成的临时检查点不作为完成结果。
- 修改实验代码、数据或预算后，设置新的 `RESULT_ROOT`；原 CLI 会拒绝不匹配的签名。不要删除旧记录或手改 manifest 绕过检查。
- 本节新增脚本只封装 CLI，不改变既有实验签名计算。修改 README 不要求重训。
- 默认只有一个实验运行依次执行，单个正式运行内部有 4 个 CPU rollout worker；本次入口不提供额外并发数字参数。
- 每次启动日志独立保存到 `RESULT_ROOT/runner_logs/`，其中包含命令、错误、耗时和产物目录。无论首次执行还是恢复，都检查末尾状态。

| 情况 | 处理 |
| --- | --- |
| `ModuleNotFoundError` | 按第 2 节使用同一解释器检查／安装缺失包 |
| 源 PDF 或清单缺失 | 恢复公开来源文件；底层下载接口仅补缺失 PDF，不替换已有来源 |
| `signature/device mismatch` | 检查设置；恢复旧运行保持原设备，实验版本改变使用新结果根目录 |
| 找不到最终检查点 | 完成第 5、6 节训练；只有 smoke 模型不能执行正式评估 |
| `Missing ... required evaluations` | 按错误提示补齐对应训练／基线／评测阶段，再聚合 |
| 正式导出拒绝 smoke／预算不足 | 完成正式协议，不更改标记伪装为 full |
| 找不到编译器或 `main.tex` | 修正 `_settings.py` 中的论文／编译器路径；不影响已保存实验数据 |
| `check.py` 报错 | 阅读最新 `main.log`；先修复编译或引用问题再检查 |

### 10. 产物对照表

下表默认结果根目录为 `result/industrial_motor`；设置自定义根目录后，所有入口会统一使用新位置。`full/` 与 `smoke/` 始终隔离。

| 数据／产物 | 生成或核验入口 | 论文用途 |
| --- | --- | --- |
| `data/industrial_motor/source/` 中 PDF、manifest | 01 核验仓库已保存来源 | 工业背景、来源引用 |
| `extracted/published.json`、`extracted/cost_fit.csv`、`derived.json` | 01 | 参数来源表、拟合残差与完整模块补充表 |
| `instances/MOTOR_FULL/`、`MOTOR_SMOKE/` 各五种 CSV | 01；00 也准备 | 正式／自检输入，80／20 道任务 |
| `full/manifest.json`、`references.json` | 02—05 首次运行建立，后续核验 | 协议、环境、代码／数据签名与共同参考值 |
| `full/checkpoints/HCMAGRL_*.pt`、`training/HCMAGRL_*.json` | 02，各 25 份 | 主方法最终策略与训练耗时 |
| `full/checkpoints/Flat_*.pt`、`MLP_*.pt` 及训练历史 | 03，各 25 份 | 消融策略与训练耗时 |
| `full/nsga/seed_*.json`、`seed_*_history.csv` | 04，各 5 份 | 搜索档案、完整评估历史和耗时 |
| `full/evaluations/`：规则 7 份、NSGA-II 25 份 | 04 | 方法比较和基线事件轨迹 |
| `full/evaluations/`：学习方法 75 份 | 05 | 方法比较、原矩阵成本重算、甘特图与敏感性输入 |
| `full/aggregate/detail.csv` | 06 | 107 份逐次结果，含失败记录 |
| `full/aggregate/summary.csv` | 06 | 平衡权重比较表、运行时间及成本模型补充表 |
| `full/aggregate/portfolios.csv` | 06 | 各方法各种子的 HV 和非支配解数量 |
| `full/aggregate/sensitivity.csv` | 06 | 平衡权重、固定活动顺序下的三倍率时间敏感性 |
| `full/aggregate/manifest.json` | 06 | 完整性、输入记录校验和、共同 HV 参考点 |
| `paper_assets/figures/data/industrial/` | 07 复制聚合数据 | 与本次论文导出对应的数据副本 |
| 论文 `tables/industrial_case.tex`、`industrial_cost_fit.tex`、`industrial_modules.tex` | 07 | 来源参数、成本拟合、完整模块表 |
| 论文 `tables/industrial_results.tex`、`industrial_runtime.tex` | 07 | 主比较表、成本重算和运行时间补充表 |
| 论文 `tables/industrial_result_figures.tex`、`industrial_findings.tex`、`macros/industrial-results.tex` | 07 | 图引用、真实结果文字及结果就绪标记 |
| 论文 `figures/fig_industrial_system.pdf` | 07 | 五工位及逻辑模块示意图 |
| 论文 `figures/fig_industrial_performance.pdf` | 07 | Pareto 与时间敏感性组合图 |
| 论文 `figures/fig_industrial_gantt.pdf` | 07 | 预指定 HCMAGRL／TwoStage 对照甘特图 |
| `paper_assets/figures/_proofs/industrial/*.png` | 07 | 目检稿，不作为实验输入 |
| 论文 `main.pdf`、`main.log` | 08 | 稿件、编译和一致性检查 |
| `runner_logs/` | 每次入口启动 | 执行命令、时间与错误诊断 |

完整数据要求保留原始评估记录，**仅有聚合 CSV 不能通过工业案例正式导出**。每份数据的生成命令都已在以上步骤给出；现有底层 CLI、事件审计和正式导出门槛继续有效。本次入口验收只运行独立目录中的 smoke，正式结果仍待完整实验。

## 目录与数据

```text
code-Junxin-Huang-HCMADRL/
├── config.py                   # 网络、PPO、权重与运行配置
├── agent.py                    # 分层智能体、轨迹缓冲区与 PPO 更新
├── graph_encoder.py            # 异构图编码与边注意力
├── graph_state.py / rl_state.py # 环境图状态与张量表示
├── upper_actor.py / upper_critic.py
├── lower_actor.py / lower_critic.py
├── env.py / class_MO_DFRMS.py    # 制造系统仿真、事件推进与流体模型
├── Triangular_fuzzy.py          # 三角模糊时间运算
├── MO_DFRMS_instance_read.py    # 算例 CSV 读取
├── Instance_generate.py        # 随机算例生成（运行前需调整输出路径）
├── train_visdom.py              # 单算例训练与 Visdom 曲线
├── run_weight_sweep.py          # 两组单目标 + 三组混合权重训练
├── evaluate.py                 # 旧评估入口，当前有缺失依赖与配置
├── plot_pareto_front.py         # 从训练检查点绘制权重对比图
├── gant_plot.py                 # 从检查点显示甘特图
├── ablation/                   # 消融变体、调度配置与训练器
├── data/                       # 27 个算例，每个目录包含 5 个 CSV
├── checkpoints/                # 本地模型输出目录，需与训练配置对齐
├── result/                     # 已有训练日志、评估 CSV、Excel 与对比图
└── paper_assets/               # 论文数据聚合、表格、图与一致性检查
```

### 算例格式

算例命名为 `M{M}_A{A}_R{R}_J{J}`：`M` 为机器数，`A` 为模块数，`R` 为工件类型数，`J` 为每种工件的工序数。例如 `M4_A3_R4_J2` 表示 4 台机器、3 个模块、4 类工件，每类 2 道工序；实际加工任务数还取决于订单中的工件数量。

现有算例覆盖 `M = 4、8、12`，每个机器规模包含三种模块数量和三组 `(R, J)` 组合，共 27 个。完整列表见 [config.py](config.py) 的 `new_file_names`。

每个 `data/<算例名>/` 目录包含：

| 文件 | 内容 |
| --- | --- |
| `based_data.csv` | 工件类型数、机器数、模块数、订单数 |
| `process_data.csv` | 工序、可选机器、可选模块与加工时间 |
| `machine_data.csv` | 每台机器可安装的模块 |
| `module_data.csv` | 模块装卸的三角模糊时间及装卸成本 |
| `order_data.csv` | 订单到达时间与各类工件数量 |

训练默认读取已有 CSV。直接执行 [Instance_generate.py](Instance_generate.py) 会生成同名随机算例并写文件；其输出路径目前是相对当前工作目录的 `../HiFAMR-BMAPPO/data`。只有需要生成新数据时才调整该路径并运行，同名文件会被覆盖。

## 环境准备

### 解释器与工作目录

下面的训练和可视化命令均在**仓库根目录**执行，使用 PowerShell。本机默认解释器如下；其他环境请替换 `$Python` 和仓库路径。每个新终端都需要设置自己的 `$Python`。

```powershell
$Python = 'E:\anaconda3\envs\python3.13\python.exe'
Set-Location 'D:\Python project\code-Junxin-Huang-HCMADRL'
$env:PYTHONUTF8 = '1'
& $Python --version
```

`PYTHONUTF8` 让 Python 在 Windows 下按 UTF-8 读写文本，也适用于后续论文工具链。本机解释器是 Python 3.13；仓库没有依赖锁文件或完整兼容性矩阵，不能将该版本视为所有实验已验证的统一环境。

### 按用途安装依赖

| 用途 | Python 包 | 说明 |
| --- | --- | --- |
| 主模型与消融训练 | `torch`、`numpy`、`matplotlib`、`visdom`、`docplex` | 环境导入链会加载 `docplex` 和 `matplotlib` |
| 检查点 Pareto 图 | `torch`、`pandas`、`matplotlib`、`seaborn`、`scipy` | `scipy` 用于核密度绘图 |
| 论文聚合与数据图 | `numpy`、`pandas`、`scipy`、`openpyxl`、`matplotlib`、`seaborn` | 不需要先加载训练模型 |
| 论文示意图与稿件编译 | 无额外 Python 包 | 需要提供 `pdflatex`、`latexmk` 等命令的 LaTeX 环境，详见子目录指南 |

按需执行，所有包操作都使用同一个解释器：

```powershell
# 训练
& $Python -m pip install torch numpy matplotlib visdom docplex

# 结果分析与论文数据图
& $Python -m pip install pandas seaborn scipy openpyxl
```

以上命令只列出依赖，不锁定版本，也不指定 CUDA 构建。主进程根据 `torch.cuda.is_available()` 选择 CUDA 或 CPU；并行 rollout worker 默认使用 CPU。

`docplex` 是当前训练入口的必需导入依赖。若执行路径调用 `dynamic_fluid_model()`，还必须具备可用的 CPLEX 求解运行时；仅能导入 `docplex` 不等于已经具备求解能力。

## 快速开始

### 准备检查点路径

运行前，在 [config.py](config.py) 的 `config` 类中将 `SAVE_DIR` 设置为本机可写目录。例如，使用仓库下的 `checkpoints/`：

```python
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
```

这是读者运行前需要做的配置调整。仓库当前值仍为 `D:\HCMADRL\checkpoints`。虽然训练入口提供 `--save_dir`，实际检查点路径仍由 `config.SAVE_DIR` 构造，因此只传 `--save_dir` 不会改变模型保存位置。

### 启动 Visdom

在单独的 PowerShell 终端运行并保持服务开启：

```powershell
$Python = 'E:\anaconda3\envs\python3.13\python.exe'
& $Python -m visdom.server -port 8097
```

浏览器打开 [Visdom](http://127.0.0.1:8097)，选择 `MOFRMS_Training` 环境。单算例训练入口不会自动启动该服务。

### 运行单算例

回到已设置解释器和工作目录的训练终端。先用 2 个训练迭代、1 个 worker 检查流程：

```powershell
$env:ALPHA_T = '0.5'
$env:ALPHA_C = '0.5'
$env:REWARD_NORMALIZE = '0'

& $Python .\train_visdom.py --path .\data --file_name M4_A3_R4_J2 --episodes 2 --num_workers 1 --seed 42
```

完成配置并确认依赖可用后，可改为常规训练：

```powershell
& $Python .\train_visdom.py --path .\data --file_name M4_A3_R4_J2 --episodes 500 --num_workers 4 --update_batch_size 64 --mp_start_method spawn --rollout_device cpu --seed 42
```

这里的一个 `episode` 对应一次训练迭代：每个 worker 采样一条轨迹，然后合并样本更新模型；曲线记录各 worker 的均值。两次迭代只用于检查运行流程，不能用于判断模型收敛。

训练结束后保存一个检查点，例如 `checkpoints/0.5makespan_0.5cost_M4_A3_R4_J2.pt`。同目录下相同权重与算例再次训练会覆盖同名文件；短流程检查和正式实验需要保留各自结果时，应分别设置 `config.SAVE_DIR`。

### 常用配置

| 配置入口 | 当前默认值 | 作用 |
| --- | --- | --- |
| `--path` / `HIFAMR_DATA_PATH` | 当前工作目录下的 `data/` | 算例根目录；命令行参数优先 |
| `--file_name` | `M4_A3_R4_J2` | 算例目录名 |
| `--episodes` | `500` | 训练迭代数 |
| `--num_workers` / `HIFAMR_NUM_WORKERS` | `4` | 并行采样进程数 |
| `--update_batch_size` / `HIFAMR_UPDATE_BATCH_SIZE` | `64` | PPO mini-batch 大小 |
| `--mp_start_method` / `HIFAMR_MP_START_METHOD` | `spawn` | 多进程启动方式 |
| `--rollout_device` / `HIFAMR_ROLLOUT_DEVICE` | `cpu` | 并行 worker 使用的设备 |
| `--seed` | `-1` | 非负数启用固定随机种子，`-1` 不固定 |
| `ALPHA_T`、`ALPHA_C` | `0.5`、`0.5` | makespan 与成本的权重 |
| `REWARD_NORMALIZE` | `0` | 设为 `1` 时使用参考值归一化奖励 |
| `MAKESPAN_REF`、`COST_REF` | `1.0`、`1.0` | 归一化参考值，混合权重实验应使用对应单目标训练结果 |

环境变量应在启动 Python 前设置。网络维度、学习率、PPO 参数及更新是否打乱样本等配置见 [config.py](config.py)。

## 权重实验与消融实验

### 五组权重实验

[run_weight_sweep.py](run_weight_sweep.py) 针对一个算例依次执行：

1. 训练 `(ALPHA_T, ALPHA_C) = (1, 0)` 和 `(0, 1)`，不归一化奖励。
2. 从检查点提取对应目标的最优观测值，作为 `MAKESPAN_REF` 和 `COST_REF`。
3. 训练 `(0.25, 0.75)`、`(0.5, 0.5)`、`(0.75, 0.25)`，开启归一化。
4. 调用 Pareto 绘图脚本，读取每组最后 10 个训练迭代的数据。

运行前需修改该脚本顶部的 `PROJECT_DIR`，当前仍指向旧的 `F:\Flexible Reconfigurable Manufacturing System\HiFAMR-BMAPPO`。同时确认 `DATA_PATH`、`SAVE_DIR`、`FILE_NAME` 与实验配置；**该脚本的 `SAVE_DIR` 必须和 `config.SAVE_DIR` 指向同一目录**，否则训练完成后会找不到检查点。

当前脚本默认算例为 `M4_A4_R4_J2`，每组 500 次迭代、4 个 worker、batch size 128。它没有命令行参数解析器，实验设置通过脚本顶部常量调整：

```powershell
& $Python .\run_weight_sweep.py
```

该入口会检查 8097 端口并尝试启动 Visdom。输出为五个无变体前缀的 `.pt` 文件及同目录下的 Pareto 图和汇总 CSV。

### 消融实验

消融入口为 [ablation/run_ablation.py](ablation/run_ablation.py)，配置集中在 [ablation/ablation_config.py](ablation/ablation_config.py)。

| 变体标识 | 含义 | 已有结果目录 |
| --- | --- | --- |
| `original` | 原始分层图强化学习模型 | `result/HCMADRL/` |
| `flat_mappo` | 扁平化单层级策略 | `result/Flat_DRL/` |
| `mlp_encoder` | 使用 MLP 替代图编码器 | `result/MLP_Encoder/` |
| `homo_gnn` | 使用同质图替代异构图 | `result/HomoGNN/` |
| `no_attn` | 使用均值聚合替代边注意力 | `result/No_Attn/` |

表中的 `result/` 目录用于识别已有实验记录；当前消融训练器保存到 `checkpoints/ablation/`，不会自动重写这些结果目录。

当前生效的 `SCHEDULE` 只为 `original` 在 `M12_A15_R4_J2` 上生成五组权重任务，每组 500 次迭代，**不会默认运行所有变体或全部 27 个算例**。例如，要为一个小算例配置原始模型和四个变体，可将配置文件末尾的 `SCHEDULE` 替换为：

```python
SCHEDULE = make_ablation_comparison_schedule(
    file_names=["M4_A3_R4_J2"],
    episodes=500,
)
```

运行命令：

```powershell
& $Python .\ablation\run_ablation.py --num_workers 4 --batch_size 64 --seed 42 --rounds 1 --save_dir .\checkpoints\ablation
```

训练器先为每个“变体 + 算例”执行两组单目标训练，再运行混合权重任务；`--rounds` 重复混合权重循环，单目标预训练在循环前执行。若预训练失败，代码会记录异常并使用估算参考值继续，复现实验时应检查日志中的失败与回退信息。

消融输出名为 `<variant>_<权重>makespan_<权重>cost_<算例>.pt`，例如 `original_0.5makespan_0.5cost_M4_A3_R4_J2.pt`。文件名不包含轮次，多轮运行会覆盖同名检查点。此入口的 `--save_dir` 会传给训练器并控制实际输出；`--no_visdom` 目前只跳过自动启动服务，训练器仍会尝试创建 Visdom 客户端。

## 可视化与结果

### 检查点内容

主训练检查点包含 `model_state_dict`、加工轨迹 `schedule_dict`、模块切换轨迹 `module_change_dict`，以及以下训练记录：

- `all_makespan`、`all_cost`：各迭代的 worker 均值。
- `all_episode_runtime`、`all_episode_steps`：迭代耗时与平均步数。
- `all_upper_reward_sum`、`all_lower_reward_sum`、`all_upper_lower_reward_sum`：上下层奖励统计。
- `all_done_rate`、`all_worker_metrics`：完成率与各 worker 的指标。

模型权重按平均上下层奖励和选择；轨迹取对应迭代中优先完成、回报较高的 worker。`all_*` 保存完整训练历史，不是独立评估集结果；各迭代平均目标值也不一定对应某一条实际调度轨迹。训练摘要中的最小 makespan 和最小 cost 可能来自不同迭代。

### Pareto 图

完成上述五组权重训练后，在仓库根目录运行：

```powershell
& $Python .\plot_pareto_front.py --checkpoint_dir .\checkpoints --file_name M4_A4_R4_J2 --last_n 10 --output_dir .\checkpoints\pareto
```

`--checkpoint_dir` 必须指向实际模型目录；`--last_n 0` 使用全部迭代，`--show` 额外显示窗口，未指定 `--output_dir` 时输出到检查点目录。脚本查找无变体前缀的五个文件，不能直接识别消融入口保存的文件名。

输出为 `<算例>_pareto_front.png`、`<算例>_pareto_front.pdf`、`<算例>_pareto_summary.csv` 和 `<算例>_pareto_points.csv`。脚本以训练均值记录计算非支配点，并绘制按权重分组的散点及边缘核密度；当前根脚本中的前沿连接线被注释。论文基于独立评估 CSV 的跨方法对比图，请使用 [paper_assets/README.md](paper_assets/README.md) 中的流程。

### 甘特图

先将 [gant_plot.py](gant_plot.py) 顶部的 `CHECKPOINT_PATH` 改为实际 `.pt` 文件路径，再运行：

```powershell
& $Python .\gant_plot.py
```

该脚本没有检查点路径命令行参数；它读取检查点中的加工与重构轨迹，打印概要并弹出图形窗口，不自动保存图像。

### 已有结果与独立评估

| 位置 | 内容 |
| --- | --- |
| `result/HCMADRL/` 及四个消融结果目录 | 已保存的训练日志、worker 指标与配置快照 |
| `result/DDQN/`、`result/EDQN/`、`result/SAC/`、`result/TD3/` | 强化学习基线的已有结果；论文图表将 `TD3` 映射为 `D-DRL` |
| [result/eval/eval_detail.csv](result/eval/eval_detail.csv) | 按方法、算例、权重与 seed 记录 makespan 和 cost |
| [result/eval/eval_summary.csv](result/eval/eval_summary.csv) | 各组合的 seed 数、均值、标准差、最小值与最大值 |
| [result/HCMADRL.xlsx](result/HCMADRL.xlsx) | 多目标指标及公开基准对比数据 |
| `result/pareto/` | 已有跨方法对比图片 |
| [paper_assets/figures/data/](paper_assets/figures/data/) | 论文图表与表格读取的聚合 CSV |

[evaluate.py](evaluate.py) 当前不能直接用于重新评估：它导入的 `train.py` 不在仓库中，且引用了未定义的 `config.MODEL_NAME` 与 `config.SEED`；默认算例 `M8_A6_S1` 也不属于当前数据集。即使补齐这些接口，该脚本现有实现也只打印评估指标，不导出上述评估 CSV。

因此，原合成算例的现有结果可以直接用于分析与论文图表重生成；从头复现这一套旧实验还需要补齐独立评估、结果导出及基线训练流程。仓库提供基线结果，并不包含 DDQN、EDQN、SAC、TD3 的完整训练入口。工业案例使用上文独立流程，不依赖这个旧评估入口。

## 当前限制

| 情况 | 运行前应确认的内容 |
| --- | --- |
| 主训练的保存路径 | 实际由 `config.SAVE_DIR` 决定，不能仅依靠 `--save_dir` 修改 |
| 权重扫描找不到脚本或检查点 | 调整旧 `PROJECT_DIR`，并对齐扫描脚本与主训练的保存目录 |
| 甘特图、随机算例生成写错位置 | 分别检查 `CHECKPOINT_PATH` 与生成器的 `base_dir` |
| 训练导入时报缺少 `docplex` | 用运行训练的同一个解释器安装；需要流体求解时另检查 CPLEX 运行时 |
| 独立评估失败 | `evaluate.py` 尚未适配当前仓库接口，详见上节 |
| 原始结果聚合找不到代码仓库 | 显式设置 `HCMAGRL_CODE`；当前默认仍为 `/home/user/code-Junxin-Huang-HCMADRL` |

原 README 核验时训练入口曾因缺少 `docplex` 在导入阶段失败；工业案例实施时已在本机默认解释器安装该依赖。工业案例的小规模训练、独立评估和轨迹核验另有运行记录，但原有 27 个算例的完整实验没有重新运行，旧入口的上述接口和路径限制仍然存在。
