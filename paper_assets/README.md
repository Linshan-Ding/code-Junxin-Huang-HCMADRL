# paper_assets：论文数据、图表与表格生成

[返回仓库使用指南](../README.md)

这里保存论文图表的生成工具与聚合数据。训练方法、依赖和实验入口见根 README；本指南说明如何从**已有实验结果**重生成论文产物，不需要先训练模型。

论文仓库 `Junxin_Huang_HCMAGRL_RMS_FRT` 保存稿件源码、类文件、参考文献，以及编译所需的图、表格与宏。代码仓库负责生成这些产物；论文编译读取生成后的 PDF 和 TeX 文件，不直接读取原始实验 CSV。

## 工业案例论文导出

工业电机案例使用独立的 `industrial_case` 流程。数据来源、完整训练／基线／评测步骤、断点恢复和计算预算见[根 README 的十节复现手册](../README.md#工业电机案例)。本节负责将正式结果转换为论文产物；不通过旧的 `build_data.py` 或 `make_tables.py` 聚合。

```text
公开来源与派生参数 → MOTOR_FULL 输入
    → 75 次学习方法训练＋规则／TwoStage／NSGA-II
    → 107 份评估记录与事件轨迹
    → full/aggregate：detail、summary、portfolios、sensitivity、manifest
    → run_07_export_paper.py：正式门槛检查＋图表／表格／宏／蓝色结果文字
    → run_08_compile_check.py：main.pdf＋main.log＋check.py
```

### 导出前检查

- 已完成根 README 第 3—8 节，或 `scripts/industrial/run_all.py`，保留全部正式记录。`run_all.py` 只生成实验数据，不自动写入论文。
- 在 [运行设置](../scripts/industrial/_settings.py) 中确认 `RESULT_ROOT` 和 `PAPER_ROOT`。前者与训练时一致且不包含 `full/` 后缀；后者必须包含 `main.tex`。
- `PAPER_ROOT` 默认使用 `HCMAGRL_PAPER`，未设置时指向代码仓库的兄弟目录 `Junxin_Huang_HCMAGRL_RMS_FRT`。新入口不依赖旧工具的 `HCMAGRL_CODE` 默认值。
- 正式导出重新核验原始运行记录、签名、预算和轨迹，**仅保留聚合 CSV 不足以重新导出工业案例**。这一点与下文可直接利用已有聚合 CSV 重绘的原合成实验不同。

### 第 07 步：导出图表、表格和结果宏

从**代码仓库根目录**运行；其他设备替换解释器与仓库路径：

```powershell
Set-Location 'D:\Python project\code-Junxin-Huang-HCMADRL'
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_07_export_paper.py
```

**输入：**`RESULT_ROOT/full/` 的清单、参考值、107 份正式评估和数据来源参数。**成功判据：**退出码 0、日志中显示正式导出完成及 `[FINISHED]`，并核对下表。**耗时：**通常秒级至分钟级；此步骤不训练模型、不搜索新解。**重跑行为：**重新审计并覆盖本次工业案例的派生产物。

| 输出位置 | 内容 |
| --- | --- |
| `paper_assets/figures/data/industrial/` | 本次正式聚合 CSV 和完整性清单副本 |
| 论文 `tables/industrial_case.tex`、`industrial_cost_fit.tex`、`industrial_modules.tex` | 来源参数、成本拟合、完整模块表 |
| 论文 `tables/industrial_results.tex`、`industrial_runtime.tex` | 方法比较、运行时间和原矩阵成本重算表 |
| 论文 `tables/industrial_result_figures.tex`、`industrial_findings.tex` | 结果图引用和由实际数据生成的比较文字 |
| 论文 `macros/industrial-results.tex` | 数值宏及正式结果就绪标记 |
| 论文 `figures/fig_industrial_system.pdf` | 五工位和逻辑模块示意图 |
| 论文 `figures/fig_industrial_performance.pdf` | Pareto／固定调度时间敏感性组合图 |
| 论文 `figures/fig_industrial_gantt.pdf` | HCMAGRL 种子 42、平衡权重与 TwoStage 甘特图 |
| 本目录 `figures/_proofs/industrial/*.png` | 三张图的目检稿 |

导出使用已有 NumPy／Pandas／Matplotlib 环境，示意图也是 Matplotlib 矢量图，不需要先编译 TikZ。PDF 最终宽度为 164.6 mm。新增案例只有少量代表解，绘制实际非支配点，不对稀疏点做核密度估计；原合成实验 Pareto 联合图的特殊样式继续保留。

覆盖范围限于工业案例独立命名的表格、宏、图片及上述工业聚合目录，原实验产物不受影响。新增／修改的论文文字、表格、图注为蓝色，图内保留算法配色。记录齐全仍可能存在不可行运行，应同时查看可行率和有效样本数，不删除失败记录。

### 第 08 步：编译、检查与目检

**前置条件：**论文源码及其引用产物齐备。正式实验完成后编译正式稿；也可独立运行本步骤检查当前“待完整实验”草稿。

```powershell
& 'E:\anaconda3\envs\python3.13\python.exe' scripts/industrial/run_08_compile_check.py
```

编译器查找顺序为：`_settings.py` 中的显式 `LATEX_COMPILER` → PATH 中的 Tectonic → 本机已有捆绑 Tectonic → PATH 中的 `latexmk`。显式路径错误会直接报错；未找到工具时提示设置路径，不自动安装。Tectonic 保留日志和中间文件，首次使用可能需要联网补齐宏包。

**输出：**论文目录的 `main.pdf`、`main.log` 及其他编译中间文件；启动日志位于 `RESULT_ROOT/runner_logs/`。**成功判据：**编译成功后才调用现有 `check.py`；其输出 `all checks passed.`，整个入口退出码为 0。**耗时与重跑：**有缓存时通常秒级至分钟级；重新编译并检查当前稿件，不更改实验记录。

最后打开 PDF，目检案例参数表、方法比较表、三张图、补充表和蓝色修订。`check.py` 不验证科学结论，且 `note` 提示不会导致失败；本机原稿标题／类文件的 overfull 提示需要与新增案例页面的版面问题区分。

PyCharm 可直接右键上述脚本运行，参数留空，解释器使用本机项目 Python。跨平台已选定正确 Python 环境时，可在 Bash 进入代码仓库后执行同名 `python scripts/industrial/run_07_export_paper.py`、`python scripts/industrial/run_08_compile_check.py`；循环、路径解析和编译操作均由 Python 处理。

### 草稿、失败与验证边界

当前正式实验尚未执行，已有稿件保留“待完整实验”说明。零参数第 07 步始终要求正式数据，不会自动退回草稿，也不读取 smoke 数字。

如需重新生成来源表、示意图和待实验占位，底层高级接口仍支持 `python -m industrial_case export-paper --profile full --draft`；使用自定义结果目录时保持与运行设置一致。草稿模式拒绝覆盖已就绪的正式结果。普通复现流程无需执行这个接口。

缺失正式运行、签名不匹配或预算不足时，正式导出在写入论文产物前失败；修复相应上游阶段后重跑。编译失败则阅读本次日志，保留已导出的图表。当前入口验收只执行独立目录的 smoke、失败门槛测试及现有草稿编译，详情见[验证记录](../industrial_case/VALIDATION.md)。

## 生成流程与输入输出

```text
代码仓库 data/、result/（日志、评估 CSV、HCMADRL.xlsx）
    └── scripts/build_data.py
          └── paper_assets/figures/data/*.csv
                ├── scripts/make_tables.py → 论文仓库 tables/*.tex、macros/results.tex
                └── figures/src/plot_*.py  → 论文仓库 figures/*.pdf
                                          → 本目录 figures/_proofs/*.png

figures/src/fig-*.tex + figures/scifig-preamble.tex
    └── pdflatex → 示意图 PDF → 复制到论文仓库 figures/

论文仓库 main.tex + 生成产物
    └── latexmk → main.pdf、main.log
          └── scripts/check.py → 一致性检查结果
```

| 工具 | 输入 | 输出与副作用 |
| --- | --- | --- |
| [scripts/build_data.py](scripts/build_data.py) | `HCMAGRL_CODE` 下的算例、训练日志、评估明细及 Excel | 覆盖本目录 `figures/data/` 下对应的聚合 CSV |
| [scripts/make_tables.py](scripts/make_tables.py) | 本目录的聚合 CSV | 覆盖论文仓库 `tables/` 下生成的表格和 `macros/results.tex` |
| `figures/src/plot_*.py` | 本目录的聚合 CSV；部分示例数值直接定义在绘图脚本中 | 覆盖论文仓库同名 PDF，以及本目录 `figures/_proofs/` 中的 PNG |
| `figures/src/fig-*.tex` | 示意图源码及共享 TikZ 前导 | 编译目录内的 PDF、日志等中间文件；复制步骤覆盖论文仓库同名 PDF |
| [scripts/check.py](scripts/check.py) | 论文源码、图文件、参考文献与最新 `main.log` | 打印检查结果，不修改稿件 |

输入数据主要包括：

- `data/<算例>/`：生成实例维度与统计信息。
- `result/<方法>/<实验>/log.csv`：生成收敛曲线和运行时间统计。
- `result/eval/eval_detail.csv`：生成评估聚合、标量评分、非支配点及统计检验结果。
- `result/HCMADRL.xlsx` 的 `RL`、`Ablation`、`VS DABC` 工作表：提供多目标指标、胜率及公开 RMSSP 基准数据。

聚合目录已包含 `instances.csv`、`convergence.csv`、`eval_cells.csv`、`cell_scores.csv`、`pareto_points.csv`、`stats_pairwise.csv`、`stats_summary.csv`、`indicators.csv`、`indicator_winrate.csv`、`benchmark_rmssp.csv` 和 `runtime.csv`。**仅重绘现有图表时，可以跳过 `build_data.py`，直接使用这些文件。**

重新聚合不会自动运行模型，也不会从新训练的 `.pt` 生成评估 CSV。若替换了实验数据，需要先按输入格式准备相应日志、评估明细与 Excel；部分聚合函数会跳过缺失的训练日志，因此脚本运行成功也不能单独证明实验覆盖完整。

## 环境与路径

### 依赖

Python 数据工具需要 `numpy`、`pandas`、`scipy`、`openpyxl`、`matplotlib`、`seaborn`。论文数据图不依赖 PyTorch、Visdom 或 CPLEX。`check.py` 只使用 Python 标准库。

示意图需要 `pdflatex` 及 `standalone`、TikZ、AMS 数学包；论文编译需要 `latexmk` 和论文源码使用的 LaTeX 包。数据图本身不要求先安装 LaTeX。字体设置见 [figures/scifig_style.py](figures/scifig_style.py) 和各绘图脚本。

### 路径变量的实际含义

| 环境变量 | 当前默认值 | 影响范围 |
| --- | --- | --- |
| `HCMAGRL_CODE` | `/home/user/code-Junxin-Huang-HCMADRL`，是固定的旧路径 | 仅控制 `build_data.py` 读取的代码仓库根目录 |
| `HCMAGRL_PAPER` | 相对于工具所在仓库的兄弟目录 `Junxin_Huang_HCMAGRL_RMS_FRT` | 表格、宏、PDF 的输出目录，以及论文检查的默认目录 |

`HCMAGRL_CODE` 应指向包含 `data/` 和 `result/` 的**代码仓库根目录**，不是 `result/` 本身。它不改变聚合 CSV 的输出位置；聚合 CSV 始终写入运行中的这份工具链所在的 `figures/data/`。绘图和表格脚本也始终从该目录读取 CSV。

下面的示例显式设置两条路径。命令会写入或覆盖上述生成产物；论文仓库应已存在且包含完整稿件源码。

## PowerShell 操作流程

### 设置环境

以下使用本机默认解释器；其他环境请替换解释器及两个仓库路径：

```powershell
$Python = 'E:\anaconda3\envs\python3.13\python.exe'
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:HCMAGRL_CODE = 'D:\Python project\code-Junxin-Huang-HCMADRL'
$env:HCMAGRL_PAPER = 'D:\Python project\Junxin_Huang_HCMAGRL_RMS_FRT'
Set-Location (Join-Path $env:HCMAGRL_CODE 'paper_assets')

& $Python -m pip install numpy pandas scipy openpyxl matplotlib seaborn
```

每个新终端需要重新设置这些变量。下列步骤默认顺序执行，除示意图和论文编译的临时目录切换外，当前工作目录保持为 `paper_assets/`。

### 聚合原始结果（按需）

仅在需要重新从原始数据生成聚合 CSV 时执行；使用仓库已有 CSV 可跳过此步：

```powershell
& $Python .\scripts\build_data.py
if ($LASTEXITCODE -ne 0) { throw '数据聚合失败，请先检查输入文件和路径。' }
```

### 生成表格与结果宏

```powershell
& $Python .\scripts\make_tables.py
if ($LASTEXITCODE -ne 0) { throw '表格生成失败。' }
```

生成的表格覆盖实例、主要目标结果、统计检验、排名、多目标指标、基准对比与运行时间。正文中的结果宏写入论文仓库 `macros/results.tex`；这些生成文件的修改应从上游数据与生成脚本入手，否则重生成时会被覆盖。

### 生成数据图

```powershell
Get-ChildItem -LiteralPath .\figures\src -Filter 'plot_*.py' |
    Sort-Object Name |
    ForEach-Object {
        & $Python $_.FullName
        if ($LASTEXITCODE -ne 0) { throw "绘图失败：$($_.Name)" }
    }
```

PDF 写入 `$env:HCMAGRL_PAPER\figures`，PNG 目检稿写入本目录 `figures/_proofs/`。只重绘某一类图时，也可单独运行相应 `plot_*.py`；这些脚本没有通用的输入输出路径命令行参数。

### 编译并复制示意图

示意图通过相对路径载入 `../scifig-preamble.tex`，因此从 `figures/src/` 编译：

```powershell
$FigureOutput = Join-Path $env:HCMAGRL_PAPER 'figures'
New-Item -ItemType Directory -Path $FigureOutput -Force | Out-Null
Push-Location .\figures\src
try {
    Get-ChildItem -Filter 'fig-*.tex' | Sort-Object Name | ForEach-Object {
        & pdflatex -interaction=nonstopmode -halt-on-error $_.Name
        if ($LASTEXITCODE -ne 0) { throw "示意图编译失败：$($_.Name)" }
        Copy-Item -LiteralPath ($_.BaseName + '.pdf') -Destination $FigureOutput -Force
    }
}
finally {
    Pop-Location
}
```

对应五张示意图：`fig-system`、`fig-framework`、`fig-graph-state`、`fig-action-mask`、`fig-attention`。此步骤会在源目录留下 PDF、`.aux`、`.log` 等编译产物。

### 编译论文并检查

```powershell
Push-Location $env:HCMAGRL_PAPER
try {
    & latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw '论文编译失败，请检查 main.log。' }
}
finally {
    Pop-Location
}

& $Python .\scripts\check.py --paper $env:HCMAGRL_PAPER --venue aei
if ($LASTEXITCODE -ne 0) { throw '论文一致性检查未通过。' }
```

## Bash 等价流程

在 Bash 中执行以下流程，先把两个路径改为实际仓库路径。`python3` 应指向已安装上述依赖的解释器；Windows 的 `E:\...\python.exe` 路径不适用于此示例。

```bash
set -e
export PYTHONUTF8=1
export HCMAGRL_CODE='/path/to/code-Junxin-Huang-HCMADRL'
export HCMAGRL_PAPER='/path/to/Junxin_Huang_HCMAGRL_RMS_FRT'
cd "$HCMAGRL_CODE/paper_assets"

# 仅当需要更新聚合数据时，取消下一行注释。
# python3 scripts/build_data.py
python3 scripts/make_tables.py
for script in figures/src/plot_*.py; do
    python3 "$script"
done

mkdir -p "$HCMAGRL_PAPER/figures"
(
    cd figures/src
    for source in fig-*.tex; do
        pdflatex -interaction=nonstopmode -halt-on-error "$source"
        cp "${source%.tex}.pdf" "$HCMAGRL_PAPER/figures/"
    done
)
(
    cd "$HCMAGRL_PAPER"
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
)
python3 scripts/check.py --paper "$HCMAGRL_PAPER" --venue aei
```

## 图表风格与命名

[figures/scifig_style.py](figures/scifig_style.py) 为常规数据图设置颜色、线型、标记、字号和印刷尺寸；`save()` 统一输出论文 PDF 与本地 PNG。示意图使用 [figures/scifig-preamble.tex](figures/scifig-preamble.tex) 中的共享 TikZ 风格。

[figures/src/plot_pareto_joint.py](figures/src/plot_pareto_joint.py) 移植自仓库根目录的 `plot_pareto_front.py`。正文使用的两张联合分布图保留 seaborn JointGrid 风格、Set1 配色与填充核密度，输出 `fig_pareto_joint_rl.pdf` 和 `fig_pareto_joint_abl.pdf`。与根脚本相比，这里按方法分组、从聚合评估 CSV 取数据、每张图放置两个算例，并绘制非支配前沿连接线；具体差异写在脚本文档字符串中。

另一个 [figures/src/plot_pareto.py](figures/src/plot_pareto.py) 输出 `fig_pareto_rl.pdf` 与 `fig_pareto_ablation.pdf`，属于常规多面板图。两种脚本输出名不同，批量运行不会互相覆盖。

方法名映射在 `scripts/build_data.py` 中定义，包括 `HCMADRL → HCMAGRL`、`TD3 → D-DRL` 及四种消融名称。原始结果目录仍使用旧名称，重命名目录前需要同步处理数据读取逻辑。

## 一致性检查与验证边界

`check.py` 覆盖 13 类检查：标签与引用、浮动体引用、图文件存在性、参考文献键与引用、图片缩放、编译日志中的未定义引用和溢出、浮动体定位参数，以及摘要、highlights、关键词和标题约束。

检查器默认使用 `aei` 规则；`--venue` 还接受 `cie`、`elsevier`、`ieee-trans`、`jms`、`rcim`。这些是脚本内置规则，不能代替投稿时的期刊要求核对。`--paper` 可覆盖 `HCMAGRL_PAPER`，例如：

```powershell
& $Python .\scripts\check.py --paper $env:HCMAGRL_PAPER --venue aei
```

需要先成功编译论文，生成最新的 `main.log`；缺少日志会报 `FAIL`。出现 `FAIL` 时退出码为 1，无失败时打印 `all checks passed.` 并以 0 退出。部分问题只输出 `note`，例如未引用的文献条目、较大的 overfull hbox 和标题长度提示，仍应人工检查。该脚本不验证实验统计结论，也不能替代 PDF 版面与 PNG 图像的目检。

原合成实验文档最初核验了路径、参数及生成关系，没有重新聚合其数据。工业案例入口本次另有独立 smoke、恢复与错误门槛测试，并通过新编译入口检查了当前待实验稿件；没有启动正式训练或写入正式结果。环境与数据版本变化后，应按相应步骤重新验证实际输出。
