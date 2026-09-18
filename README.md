# CRSP Daily 收益预测实验

本项目用 2024 年 CRSP Daily 数据研究股票横截面收益预测。当前主实验采用 **7–11 月逐月 walk-forward**：扩展训练集、上月验证集、当月测试集；12 月信号与实现收益留作未评估的 lockbox。信号在 (t) 日收盘后生成，在 (t+1) 日收盘成交，收益对应 (t+1) 收盘至 (t+2) 收盘。

## 主要产物

| 路径 | 内容 |
| --- | --- |
| `doc/pipeline.md` | 研究问题、特征、时间口径与评价协议 |
| `src/run_crsp_monthly_walkforward_2024.py` | 5 折月度实验入口 |
| `output/crsp_monthly_walkforward_2024/` | 每折配置、逐行预测、IC、组合收益与汇总指标 |
| `result/crsp_monthly_walkforward_2024.md` | 月度实验报告 |
| `presentation/main.pdf` | 14 页简洁 Beamer 汇报，包含原始日频数据切片与逐日图表 |
| `presentation/main.tex`、`presentation/make_figures.py` | PDF 源文件及图表生成脚本 |
| `presentation/experiment_summary.md` | 幻灯片与实验结果的对应关系 |
| `presentation/speaker_notes_zh.md` | 每页中文讲稿 |

主实验在 105 个测试交易日得到树模型平均 Rank IC 0.0128、Ridge 0.0110。组合指标只在三个策略均有完整入选持仓收益的 63 个共同日期计算；净 Sharpe 均为负。具体方法、缺失情况和局限见报告。

## 数据

| 路径 | 内容 |
| --- | --- |
| `data/crsp_dsf_2024.csv.gz` | 2024 年 CRSP Daily，2,400,962 个证券日、10,503 个 PERMNO、252 个交易日 |
| `data/crsp_dsedelist_2024.csv.gz` | 2024 年非空退市收益；由 `src/download_crsp_dsedelist_2024.py` 从 WRDS 提取 |
| `output/crsp_filter_2024/dsenames_2024_snapshot.csv` | 带生效日期的名称、股类与交易所历史快照 |

特征使用截至信号日的 20 个市场交易日滚动窗口。普通股与交易所资格按当时有效的名称区间匹配；TAQ 全年采样结果不用于历史预测股票池。输入文件 SHA-256、缺失值及收益拼接计数保存在主实验的 `run_metadata.json`。

## 复现主实验和 PDF

在项目根目录运行。以下为本机验证过的 Python 环境，需要 pandas、NumPy、scikit-learn、SciPy 和 Matplotlib；PDF 编译需要 `pdflatex`。

```bash
OPENBLAS_NUM_THREADS=1 /home/wangb/miniforge3/bin/python src/run_crsp_monthly_walkforward_2024.py
/home/wangb/miniforge3/bin/python src/audit_crsp_daily_2024.py
MPLBACKEND=Agg /home/wangb/miniforge3/bin/python presentation/make_figures.py
pdflatex -interaction=nonstopmode -halt-on-error -output-directory presentation presentation/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory presentation presentation/main.tex
```

若需重新从 WRDS 获取退市收益，可先运行 `src/download_crsp_dsedelist_2024.py`。这一步需要 WRDS 权限。

## 其他研究文件

- `src/run_crsp_benchmark_2024.py`、`output/crsp_benchmark_2024/`、`result/crsp_benchmark_2024.md`：较早的单次时间切分实验，供审计；当前 PDF 使用月度实验结果。
- `doc/filter.md`、`src/run_crsp_filter_2024.py`、`output/crsp_filter_2024/`：CRSP → TAQ 容量采样流程，与预测实验分开。其已有结果可用 `.wrds-venv/bin/python src/run_crsp_filter_2024.py` 复现。
- `notebook/crsp_daily_learning_and_client_story.ipynb`：探索性分析。
