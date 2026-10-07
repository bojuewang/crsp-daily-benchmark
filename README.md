# CRSP Daily: reproducible cross-sectional return research

[中文说明](README.zh.md) · [Design](doc/pipeline.md) · [Validation](doc/validation.md) · [Recorded results](result/crsp_monthly_walkforward_2024.md)

This Python project asks whether nonlinear models improve cross-sectional stock-return prediction over simple baselines, and whether predictive gains survive portfolio construction and costs. It uses real 2024 CRSP daily data through WRDS, effective-dated security eligibility, eight rolling features, five monthly chronological test folds, pandas/NumPy and scikit-learn.

**Recorded conclusion:** the tree model modestly improves average Rank IC, but all strategies have negative net Sharpe on the common complete-date sample. Missing outcomes materially limit the portfolio comparison. This is research evidence, not evidence of profitable trading.

## Recorded evidence

These numbers come from the existing [monthly report](result/crsp_monthly_walkforward_2024.md) at commit `bd86235de9aec79aff0091c4163790bc86e27357`. They were **not regenerated** during the 2026-10-07 update: licensed inputs and cached predictions are excluded from Git.

| Item | Repository-reported result |
| --- | --- |
| Raw data | 2,400,962 security-days; 10,503 PERMNOs; 252 dates in 2024 |
| Out-of-sample evaluation | July–November; 105 signal dates; 286,046 rows per model; 285,882 valid labels |
| Mean daily Rank IC | Reversal 0.00590; Ridge 0.01104; gradient boosting tree 0.01276 |
| Regression MSE | Zero 0.00131959; Ridge 0.00131761; tree 0.00131746 |
| Net Sharpe on 63 common complete dates | Reversal −6.350; Ridge −1.527; tree −2.852 |
| Auxiliary logistic regression | ROC-AUC 0.50317; near-random discrimination |

The one-year sample, missing outcomes and observed test results prevent strong generalization claims. December is excluded by the existing evaluation protocol; this update has not evaluated it.

## Experiment

- **Question:** does model complexity improve prediction and cost-adjusted portfolios under identical folds? This is a retrospective question, not a claimed preregistration.
- **Universe:** effective-dated common shares on NYSE/AMEX/NASDAQ with signal-date absolute price at least USD 5. Annual survival and TAQ capacity screens are excluded from prediction membership.
- **Features:** eight statistics over 20 market days; absent security dates are retained in the calendar. Preprocessing is fit inside the training window.
- **Timing:** features after signal-date close; modeled fill at next trading-day close; return through the following close. Applicable regular and delisting returns are compounded.
- **Models:** zero-return loss baseline, reversal ranking baseline, Ridge, `HistGradientBoostingRegressor`, auxiliary fixed-C logistic regression. No neural-network results are claimed here.
- **Selection:** expanding training, previous-month validation MSE, current-month testing, two-day label embargo, refit on train plus validation.
- **Portfolio proxy:** top/bottom deciles, +1/−1 aggregate side weights (gross exposure 2), 10 bps per unit of absolute target-weight change. Future fill availability affects candidate selection; this is an explicit limitation.

## Run checks without licensed data

From the root, using Python 3.11 or 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

Small synthetic fixtures check timing, eligibility, missing dates, delistings, purging, costs and artifact corruption. They are not investment-performance evidence. The tested environment and verification boundaries are in [doc/validation.md](doc/validation.md).

## Reproduce the real-data study

Requires WRDS entitlement to CRSP. Data, credentials and derived row-level outputs are not redistributed or committed.

```bash
# Use your own account; omit to let WRDS prompt for a username.
export WRDS_USERNAME=your_wrds_username
.venv/bin/python src/download_crsp_dsf_2024.py
.venv/bin/python src/download_crsp_dsedelist_2024.py
.venv/bin/python src/download_crsp_names_2024.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python src/run_crsp_monthly_walkforward_2024.py
.venv/bin/python src/audit_crsp_daily_2024.py
.venv/bin/python src/verify_walkforward_outputs.py --require-inputs
```

Inputs: `data/crsp_dsf_2024.csv.gz`, `data/crsp_dsedelist_2024.csv.gz`, and `output/crsp_filter_2024/dsenames_2024_snapshot.csv`. The standalone names downloader avoids running the separate annual TAQ filter. Data revisions and dependency versions can change results: compare generated SHA-256 hashes and retain your installed versions. `requirements.txt` provides version ranges, not an exact historical lock.

The monthly runner writes predictions, fold choices, daily IC, portfolio series, aggregate metrics, classification results and input hashes to `output/crsp_monthly_walkforward_2024/`. It overwrites named artifacts; archive an existing run before rerunning.

The verifier recomputes selected saved regression/IC/portfolio statistics and checks available input hashes. Without `--require-inputs`, missing inputs are reported explicitly. Passing it does not independently validate model fits, classification or historical execution.

## Inspect and present

| Path | Purpose |
| --- | --- |
| `src/run_crsp_benchmark_2024.py` | Shared features, labels and portfolio accounting; earlier single-split run |
| `src/run_crsp_monthly_walkforward_2024.py` | Five-fold fitting and stitched evaluation |
| `tests/test_research_contract.py` | Synthetic mechanics and corruption checks |
| `src/verify_walkforward_outputs.py` | Saved-output consistency and input-hash checks |
| `result/crsp_monthly_walkforward_2024.md` | Existing numerical report, coverage and limitations |
| `presentation/main.pdf` | Existing presentation of the monthly experiment |
| `doc/portfolio_evidence.md` | Project selection and application evidence boundaries |

After a real-data run, figures and the existing PDF can be rebuilt:

```bash
MPLBACKEND=Agg .venv/bin/python presentation/make_figures.py
pdflatex -interaction=nonstopmode -halt-on-error -output-directory presentation presentation/main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory presentation presentation/main.tex
```

Multi-year testing, uncertainty estimates and fully executable failed-fill rules are proposed extensions, not completed results. The earlier single-split and annual CRSP→TAQ capacity studies remain separate exploratory work.
