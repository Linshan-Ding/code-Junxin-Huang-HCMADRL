# paper_assets — 论文图表与数字的生成链

论文仓库（`Junxin_Huang_HCMAGRL_RMS_FRT`）只保留编译稿件必须的文件：`.tex`、类文件、
参考文献和 18 张图 PDF。生成这些图表与数字的工具链在这里，紧挨着它读取的原始实验结果。

```
paper_assets/
├── scripts/
│   ├── build_data.py    result/ 与 HCMADRL.xlsx → figures/data/*.csv（聚合，可复现）
│   ├── make_tables.py   figures/data/*.csv → 论文仓库的 tables/*.tex 与 macros/results.tex
│   └── check.py         稿件一致性 + 投稿硬性限制（13 项）
└── figures/
    ├── scifig_style.py  全组图风格契约；save() 把 PDF 写进论文仓库，PNG 目检稿留在 _proofs/
    ├── src/*.py         每张数据图一个脚本
    └── data/*.csv       聚合结果表，正文每个数字的来源
```

## 重生成

论文仓库默认取兄弟目录，用 `$HCMAGRL_PAPER` 覆盖；原始结果目录用 `$HCMAGRL_CODE` 覆盖。

```bash
cd paper_assets
python3 scripts/build_data.py            # 原始结果 → figures/data/*.csv
python3 scripts/make_tables.py           # CSV → 论文仓库 tables/ 与 macros/
for f in figures/src/plot_*.py; do python3 "$f"; done   # → 论文仓库 figures/*.pdf
cd ../../Junxin_Huang_HCMAGRL_RMS_FRT && latexmk -pdf main.tex
cd - && python3 scripts/check.py         # 13 项检查，零 FAIL 才算过
```

`figures/src/plot_pareto_joint.py` 是仓库根目录 `plot_pareto_front.py` 的移植：正文图 11、12
沿用该脚本的绘图风格（seaborn JointGrid、Set1、填充核密度），而非其余图的全组风格契约，
脚本文档字符串里逐项写明了移植时改了什么、为什么。
