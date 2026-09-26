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

希望查看现有实验结果的读者，可以从 `result/` 和 `paper_assets/figures/data/` 开始，无需先训练模型。重新训练、独立评估和生成论文产物是不同步骤；当前仓库尚未将它们接成完整的一键复现流程。

## 工业电机案例

`industrial_case/` 提供一套独立流程，使用 [Zhao 等公开论文](https://doi.org/10.1371/journal.pone.0348884) 中常州 AMEC&GBM 电机装配案例的参数。论文来源是公开文献，未进行现场采集或部署验证。`data/industrial_motor/source/` 保留未改动的开放获取论文及 SHA-256；`extracted/published.json` 对应原文第 15 页的三张表和订单，`derived.json` 保存转换参数及来源分类。

保留 4 类产品、5 个工位、`[5,4,4,3]` 件订单和全部 20 个加工时间，生成 80 道实际加工任务。工位映射为制造单元，每个单元可装 4 种产品专用逻辑模块，共 20 种模块。这是适配本文模型的重构：逻辑模块、无限缓冲、空初始配置与统一零时刻释放是建模假设；原文固定投产间隔、零件频率及周期末回切约束不纳入。

原始产品间成本通过非负加性最小二乘转换为卸载加安装成本，固定两组分量总和相等后平均分配至五个工位。`extracted/cost_fit.csv` 保留每项误差。模糊装卸时间是根据工位平均加工时间构造的假设值，**不是实测工人时间**。输出时间单位沿用原文未具体命名的 time-unit，内部按 1000 倍整数刻度读取；成本由千元转为 CNY。评估另用原成本矩阵按工位五等分重算成本，初始安装仍用推导值。

### 命令与依赖

以下命令从代码仓库根目录运行；其他环境替换解释器和两个仓库路径。新流程无需 Visdom 服务，也不调用单订单之外的流体求解功能；环境仍需导入 `docplex`，本案例无需 CPLEX 运行时。

```powershell
$Python = 'E:\anaconda3\envs\python3.13\python.exe'
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
Set-Location 'D:\Python project\code-Junxin-Huang-HCMADRL'
$env:HCMAGRL_PAPER = 'D:\Python project\Junxin_Huang_HCMAGRL_RMS_FRT'

& $Python -m pip install torch numpy scipy pandas matplotlib visdom docplex pymoo
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败' }

# 已含原始 PDF；需要重新获取缺失的来源文件时才加 --fetch（需要网络）
& $Python -m industrial_case prepare
if ($LASTEXITCODE -ne 0) { throw '案例准备失败' }
& $Python -m industrial_case validate
if ($LASTEXITCODE -ne 0) { throw '数据检查失败' }

# 4 件产品、20 道任务；三种模型各 2 次迭代；NSGA-II 32 次评估
& $Python -m industrial_case run --profile smoke --resume
if ($LASTEXITCODE -ne 0) { throw '小规模流程检查失败' }

# 正式运行耗时较长：75 次训练，500 次迭代/次，4 个 CPU rollout worker
# PPO 更新默认 CPU；可明确加 --device cuda，恢复训练时保持同一设备
& $Python -m industrial_case run --profile full --resume
if ($LASTEXITCODE -ne 0) { throw '完整实验未完成；请检查日志后恢复' }

& $Python -m industrial_case export-paper --profile full
if ($LASTEXITCODE -ne 0) { throw '论文导出被阻止；检查完整性及输入签名' }
```

只生成参数表、示意图和“完整实验待运行”说明时使用 `export-paper --profile full --draft`。该命令不读取 smoke 数值，并拒绝将已经导出的正式结果退回草稿。正式导出仅在全套运行记录存在且通过签名、预算、轨迹核验后执行；失败或不可行运行保留并列明，不静默删除。

可分别调用 `train`、`baselines`、`evaluate`、`aggregate` 和 `export-paper`，每个入口支持 `--help`。`run` 顺序执行校验、训练、基线、评估和聚合，**不自动修改论文**。`train --stop-after 1` 可在指定绝对迭代数暂停以测试恢复，随后使用 `train --resume` 继续；这不会缩短正式实验要求。`train --methods HCMAGRL` 可仅运行一种学习方法，论文导出仍要求全套方法。

### 实验设置、输出与覆盖规则

正式学习方法为 HCMAGRL、Flat、MLP；基线为 SPT、SetupGreedy、成本优先的 TwoStage 和 NSGA-II。学习及搜索种子固定为 `42—46`，时间权重为 `0、0.25、0.5、0.75、1`。SPT 和 TwoStage 各运行一次，SetupGreedy 每个权重一次；确定性输出不会复制成五个统计样本。参考值由事前 SPT 调度确定，各模型保存并评估最终迭代参数。

NSGA-II 每种子使用 10,000 次环境评估，保留搜索档案，并按相同五组权重选代表解。HV 按每种子的代表解集合计算，以共同 SPT 参考值归一化，参考点为全部方法可行代表解坐标最大值的 1.1 倍。报告样本标准差，并分别记录训练／搜索时间与一次调度执行时间；相同 rollout 数不代表相同实际耗时。每条轨迹最多允许 `20 × 实际工序数` 个决策步，达到上限按未完成报告。

| 位置 | 内容 |
| --- | --- |
| `data/industrial_motor/instances/MOTOR_FULL/` | 正式案例的五种输入 CSV |
| `data/industrial_motor/instances/MOTOR_SMOKE/` | 每类 1 件产品的小规模案例 |
| `result/industrial_motor/<profile>/manifest.json` | 配置、环境版本、输入及代码签名 |
| `.../checkpoints/`、`.../training/` | 可恢复模型、优化器、随机状态与逐轮 worker 指标 |
| `.../evaluations/` | 独立评估指标及完整加工、装卸事件轨迹，包括失败记录 |
| `.../nsga/` | 搜索档案、优先级编码、完整目标评估历史 |
| `.../aggregate/` | 明细、汇总、每种子 HV、时间敏感性及完整性清单 |
| `paper_assets/figures/data/industrial/` | 正式导出时复制的聚合数据 |

`--output <目录>` 改变结果根目录，仍自动添加 `smoke/` 或 `full/` 子目录。配置、数据或执行代码变更后，旧输出会被拒绝复用，应选新目录；同一签名下用 `--resume` 恢复原训练或跳过已完成运行。每次迭代原子更新本次运行检查点，聚合命令覆盖本次运行的派生 CSV。`prepare` 重生成此案例的派生输入，公开提取值不一致时拒绝覆盖。下载命令只补缺失源 PDF，不替换已有来源文件。

实施时完成了 7 份可行的 smoke 评估和 8 项自动测试，正式训练未启动；具体环境、验证命令和边界见[验证记录](industrial_case/VALIDATION.md)。可用 `& $Python -m unittest industrial_case.test_pipeline -v` 重新验证。

敏感性分析保持既有调度的单元活动顺序、工序优先关系及触发每次重构的工件就绪条件，按装卸耗时倍率 `0.5、1、2` 重算最早可行时间。倍率 1 也是按固定顺序重新计算最早开工时间，不强制保留原调度中的空闲；该分析属于调度重放，不等同于策略在随机工人环境中的鲁棒性。甘特图固定采用 HCMAGRL 的种子 42、平衡权重与确定性 TwoStage 对照。论文产物和编译流程见[工业案例论文导出](paper_assets/README.md#工业案例论文导出)。

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

因此，现有结果可以直接用于分析与论文图表重生成；从头复现全部论文实验还需要补齐独立评估、结果导出及基线训练流程。仓库提供基线结果，并不包含 DDQN、EDQN、SAC、TD3 的完整训练入口。

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
