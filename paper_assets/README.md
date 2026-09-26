# paper_assets：论文数据、图表与表格生成

[返回仓库使用指南](../README.md)

这里保存论文图表的生成工具与聚合数据。训练方法、依赖和实验入口见根 README；本指南说明如何从**已有实验结果**重生成论文产物，不需要先训练模型。

论文仓库 `Junxin_Huang_HCMAGRL_RMS_FRT` 保存稿件源码、类文件、参考文献，以及编译所需的图、表格与宏。代码仓库负责生成这些产物；论文编译读取生成后的 PDF 和 TeX 文件，不直接读取原始实验 CSV。

## 工业案例论文导出

新增工业电机案例使用独立的 `industrial_case` 流程。来源、重构假设、训练预算和完整运行命令见[根 README](../README.md#工业电机案例)，不通过旧的 `build_data.py` 或 `make_tables.py` 聚合。

```text
data/industrial_motor/source/ + extracted/ + derived.json
    → instances/MOTOR_FULL/、MOTOR_SMOKE/
    → industrial_case train / baselines / evaluate
    → result/industrial_motor/<profile>/evaluations/、nsga/
    → industrial_case aggregate → 本次运行 aggregate/*.csv
    → industrial_case export-paper --profile full
        → paper_assets/figures/data/industrial/*
        → 论文 tables/industrial_*.tex、macros/industrial-results.tex
        → 论文 figures/fig_industrial_*.pdf
        → 本目录 figures/_proofs/industrial/*.png
```

入口从**代码仓库根目录**运行，默认论文目录仍为兄弟目录，也可通过 `--paper` 覆盖 `HCMAGRL_PAPER`：

```powershell
$Python = 'E:\anaconda3\envs\python3.13\python.exe'
$env:PYTHONUTF8 = '1'
$env:HCMAGRL_CODE = 'D:\Python project\code-Junxin-Huang-HCMADRL'
$env:HCMAGRL_PAPER = 'D:\Python project\Junxin_Huang_HCMAGRL_RMS_FRT'
Set-Location $env:HCMAGRL_CODE

# 不需要训练结果：生成来源参数表、成本拟合、模块表、案例示意图及蓝色待实验说明
& $Python -m industrial_case export-paper --profile full --draft
if ($LASTEXITCODE -ne 0) { throw '工业案例草稿生成失败' }

# 全部正式运行完成后执行；完整性检查不通过时不修改稿件产物
& $Python -m industrial_case export-paper --profile full
if ($LASTEXITCODE -ne 0) { throw '工业案例正式导出失败' }

Set-Location $env:HCMAGRL_PAPER
latexmk -pdf main.tex
if ($LASTEXITCODE -ne 0) { throw '论文编译失败' }
Set-Location $env:HCMAGRL_CODE
& $Python .\paper_assets\scripts\check.py --paper $env:HCMAGRL_PAPER
```

Bash 下对应入口为 `python -m industrial_case export-paper --profile full`，先 `export HCMAGRL_PAPER=/absolute/path/to/paper` 并进入代码仓库根目录；随后在论文目录运行 `latexmk -pdf main.tex`。如果用 `--output` 保存实验，导出命令需要传入相同结果根目录。

本机实施验证使用已存在的 Tectonic 编译成功，PATH 中没有 `latexmk`。使用 Tectonic 时，在论文目录执行 `tectonic -X compile --keep-logs --keep-intermediates main.tex`；Windows 若未加入 PATH，用可执行文件的绝对路径替换 `tectonic`。保留日志是后续 `check.py` 的前提，首次使用可能需要联网下载缺失宏包。新小节验证状态见[验证记录](../industrial_case/VALIDATION.md)。

导出需要 `numpy`、`matplotlib`、`pandas` 所在的既有分析环境；正式流程依赖详见根 README。示意图及数据图使用 Matplotlib 输出矢量 PDF，不要求先编译 TikZ，宽度为最终版面的 164.6 mm。已有 Pareto 联合图的特殊样式继续保留；新增案例只有少量代表点，采用散点显示非支配解，不对稀疏样本绘制核密度曲线。

导出仅覆盖 `industrial_*` 表格、独立结果宏、`fig_industrial_*` 图片及上述工业聚合目录，保留原实验产物。草稿导出会拒绝覆盖已完成的正式结果；`smoke` 始终禁止用于论文。正式导出重新检查原始运行记录而非仅信任聚合 CSV，因此重绘时也需保留相应运行目录。全文新增修订文字、表格与图注为蓝色，图内保留算法区分色；完整训练前不会产生伪造结果图或改进百分比。

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

本次 README 更新核验了源码中的路径、参数及生成关系，并运行了 `check.py --help`；没有执行数据重聚合、覆盖论文产物或编译稿件。环境与数据版本变化后，应按上述步骤重新验证实际输出。
