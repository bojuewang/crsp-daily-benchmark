# Stock Return Prediction

## Problem

Use daily CRSP to train model and predict cross-sectional stock returns. Key is to figure out the upgraded complexity of model relating to improvement of **Rank IC** and **portfolio Sharpe**.

```text
CRSP → design matrix → ERM → walk-forward tests → IC / Rank IC → portfolio → PnL
```

## Data and target

| Data | Available in this repository |
| --- | --- |
| CRSP daily stock file | `data/crsp_dsf_2024.csv.gz`; 2,400,962 security-day rows, 10,503 PERMNOs, 252 trading days (Jan 2–Dec 31, 2024) |
| Daily fields | `permno`, `date`, `prc`, `vol`, `shrout`, `ret`, `retx`, `openprc` |
| Name history | `output/crsp_filter_2024/dsenames_2024_snapshot.csv`; effective dates, `shrcd`, `exchcd` |
| Existing filter output | 702,841 stock-days after common-share, exchange, and $5 daily price screens |

| Research definition | Baseline rule |
| --- | --- |
| Panel key | (`permno`, trading date) |
| Predictor | $X_{i,t}\in\mathbb R^8$: eight features computed from the trailing 20 trading days (defined below). |
| Daily universe | Common shares (`SHRCD` 10 or 11), major U.S. exchanges (`EXCHCD` 1, 2, or 3), and $|\mathrm{prc}_{i,t}|\ge5$. Any further liquidity or size screen uses information known by $t$; never filter on future survival. |
| Signal and execution | Build features after the close on $t$; trade at the close on $t+1$. |
| Holding period | Close of $t+1$ through close of $t+2$. |
| Major task: regression | $Y_{i,t}=R_{i,(t+1,t+2]}$. The available proxy is `ret` dated $t+2$; where both returns apply, total return is $(1+\mathrm{ret})(1+\mathrm{dlret})-1$. |
| Minor task: classification | $Z_{i,t}=\mathbf 1\{Y_{i,t}>0\}$. |

## Mathematical formulation

Let $\mathcal D$ be the ordered set of trading days, $\mathcal P$ the set of CRSP `permno` identifiers, and $\mathcal F_t$ the information available after the close on $t\in\mathcal D$. In the current file, $\mathcal D=\mathcal D_{2024}$. Here $t+1$ and $t+2$ denote the next two trading days. The main objects are:

| Object | Definition |
| --- | --- |
| $\mathcal U_t\subseteq\mathcal P$ | Stocks satisfying the point-in-time universe rules using $\mathcal F_t$ only. |
| $\mathcal M=\{\text{naive},\text{linear},\text{tree},\text{neural}\}$ | Model families; $\mathcal H_m$ is the hypothesis class for family $m$. |

### Index and value domains

Write the 2024 dates as $\mathcal D_{2024}=\{d_1<\cdots<d_{252}\}$, with $d_1=$ January 2 and $d_{252}=$ December 31. The security index $i$ is a **PERMNO value**, not a row number: $i\in\mathcal P_{2024}$, where $|\mathcal P_{2024}|=10{,}503$ and the observed identifiers run from 10026 to 93436 with gaps. At each date, $\mathcal U_{d_j}\subseteq\mathcal P_{2024}$ and $0\le|\mathcal U_{d_j}|\le10{,}503$; the actual daily sizes depend on the stated filters.

The notebook `notebook/crsp_daily_learning_and_client_story.ipynb` already computes eight **full-year, security-level summaries**. To make those features usable for daily prediction without future leakage, the baseline recomputes the same summaries on the trailing 20 market trading days $\{d_{j-19},\ldots,d_j\}$. Within that window, let $W_{i,d_j}$ contain the observed dates for security $i$ with $|\mathrm{prc}_{i,s}|\ge5$. Define $P_{i,s}=|\mathrm{prc}_{i,s}|$, $Q_{i,s}=\mathrm{vol}_{i,s}$, $H_{i,s}=1000\,\mathrm{shrout}_{i,s}$, and $R_{i,s}=\mathrm{ret}_{i,s}$. The implementable baseline is $X_{i,t}=(x^{(1)}_{i,t},\ldots,x^{(8)}_{i,t})^{\mathsf T}$:

| Coordinate | Rolling definition over $s\in W_{i,t}$ | Value domain |
| --- | --- | --- |
| $x^{(1)}$: `trading_days` | Number of distinct dates | Integer from 10 to 20 for valid feature rows |
| $x^{(2)}$: `med_price` | $\operatorname{median}(P_{i,s})$ | At least $5$ when defined |
| $x^{(3)}$: `adv` | $\operatorname{mean}(Q_{i,s})$ over nonmissing volume, including zero | Nonnegative |
| $x^{(4)}$: `med_dollar_volume` | $\operatorname{median}(P_{i,s}Q_{i,s})$ where $Q_{i,s}>0$ | Positive |
| $x^{(5)}$: `adv_dollar` | $\operatorname{mean}(P_{i,s}Q_{i,s})$ where $Q_{i,s}>0$ | Positive |
| $x^{(6)}$: `med_turnover` | $\operatorname{median}(Q_{i,s}/H_{i,s})$ where $Q_{i,s},H_{i,s}>0$ | Positive |
| $x^{(7)}$: `annual_volatility` | $\sqrt{252}\operatorname{std}(R_{i,s})$, sample standard deviation (`ddof=1`) | Nonnegative |
| $x^{(8)}$: `med_market_cap` | $\operatorname{median}(P_{i,s}H_{i,s})$ where $H_{i,s}>0$ | Positive |

Require at least 10 valid observations for each summary and a complete 20-market-day lookback; otherwise the predictor is unavailable. The original annual `trading_days >= 200` screen must not be applied at each historical date.

| Symbol | Permitted range in the current design |
| --- | --- |
| $t=d_j$ | Baseline signal date with $20\le j\le250$: 20 days of available market history and two later days for the holding-period label. Missing security history may remove more $(i,t)$ pairs. |
| $a\in\{1,\ldots,8\}$ | Baseline feature coordinate; $p=8$ for the notebook-derived rolling feature set. |
| $x^{(a)}_{i,t}$ | A real value in the coordinate-specific domain above; invalid or insufficient observations make the baseline row unavailable. |
| $Y_{i,t},\hat Y^{(m)}_{i,t}$ | Real-valued return and prediction; a valid simple realized return is at least $-1$. $Z_{i,t}\in\{0,1\}$. |
| $k,r,L,n_k$ | Positive integers for fold, training-row, sequence length, and training-row count; $1\le r\le n_k$. Their maxima depend on the chosen splits and features. |

These are **domains**, not empirical feature bounds. In the raw 2024 file, nonmissing `ret` ranges from $-0.995707$ to $14.457832$, `vol` from $0$ to $1{,}922{,}596{,}684$, and `shrout` from $10$ to $24{,}568{,}840$ (thousands of shares). Derived-feature ranges must be measured after the rolling features and daily universe are built. Missing raw values remain missing until the specified exclusion or training-only imputation step.

For fold $k$, let $\mathcal T_k^{\mathrm{train}}$ be its training signal dates and $A_k^{\mathrm{train}}=\{(i,t):t\in\mathcal T_k^{\mathrm{train}},\ i\in\mathcal U_t,\ X_{i,t}\text{ and }Y_{i,t}\text{ are valid}\}$. The number of training rows is

$$
n_k=|A_k^{\mathrm{train}}|
=\sum_{t\in\mathcal T_k^{\mathrm{train}}}\sum_{i\in\mathcal U_t}
\mathbf 1\{X_{i,t}\text{ valid and }Y_{i,t}\text{ valid}\}.
$$

Thus $n_k$ varies by fold; it cannot be a single numeric constant until the training dates and validity rules are applied. Choose a fixed row order $(i_r,t_r)_{r=1}^{n_k}$ and form the **design matrix** and label vector

$$
\mathbf X_k^{\mathrm{train}}=
\begin{bmatrix}X_{i_1,t_1}^{\mathsf T}\\ \vdots\\ X_{i_{n_k},t_{n_k}}^{\mathsf T}\end{bmatrix}
\in\mathbb R^{n_k\times 8},\qquad
\mathbf y_k^{\mathrm{train}}=
\begin{bmatrix}Y_{i_1,t_1}\\ \vdots\\ Y_{i_{n_k},t_{n_k}}\end{bmatrix}
\in\mathbb R^{n_k}.
$$

Here $f\in\mathcal H_m$ is a candidate prediction function $f:\mathbb R^8\to\mathbb R$, not an existing file or a specific model architecture. For example, Ridge uses $f(x)=\beta_0+x^{\mathsf T}\beta$; a tree model uses a nonlinear function. Training produces

$$
\hat f_{m,k}\in\underset{f\in\mathcal H_m}{\arg\min}\;
\frac{1}{n_k}\sum_{r=1}^{n_k}\ell\!\left(f(X_{i_r,t_r}),Y_{i_r,t_r}\right)
+\lambda_m\Omega_m(f),
$$

where $\ell$ is the training loss and $\Omega_m$ is an optional complexity penalty. Classification instead fits $g:\mathbb R^8\to[0,1]$ to $Z_{i,t}$. Sequence models replace each row with a history window $X^{(L)}_{i,t}\in\mathbb R^{L\times8}$, giving a training tensor in $\mathbb R^{n_k\times L\times8}$; $L$ and its extra warm-up period must be specified before that experiment.

Select hyperparameters on the later validation observations, then issue $\hat Y^{(m)}_{i,t}=\hat f_{m,k}(X_{i,t})$ only for $t$ in the test block $\mathcal T_k^{\mathrm{test}}$. All training and validation labels must have become observable before the first test signal date. The stitched out-of-sample index is $\mathcal T^{\mathrm{OOS}}=\bigcup_k\mathcal T_k^{\mathrm{test}}$; the lockbox is excluded from every fit and selection decision.

At each $t\in\mathcal T^{\mathrm{OOS}}$, rank valid scores within $\mathcal U_t$. Let $L_{m,t}$ and $S_{m,t}$ be its top and bottom deciles. Target weights are $+1/|L_{m,t}|$ on $L_{m,t}$, $-1/|S_{m,t}|$ on $S_{m,t}$, and zero elsewhere. Under a prespecified rule for failed fills and delistings, the executed weights $w^{(m)}_{i,t}$ yield

$$
r^{(m),\mathrm{net}}_t
=\sum_{i\in\mathcal U_t}w^{(m)}_{i,t}Y_{i,t}-C_t\!\left(w^{(m)}_t,w^{(m)}_{t-1}\right),
$$

where $C_t$ is the stated transaction-cost model. Prediction quality is the daily cross-sectional $\operatorname{Spearman}(\hat Y^{(m)}_{i,t},Y_{i,t})$ on observations with valid realized returns; portfolio quality is the Sharpe ratio of $\{r^{(m),\mathrm{net}}_t:t\in\mathcal T^{\mathrm{OOS}}\}$. The empirical question is whether a more complex $\mathcal H_m$ improves both measures relative to simpler families under identical folds, universe rules, and costs. Valid-return coverage and failed fills are reported separately so that missing outcomes do not silently change the comparison.

## Research pipeline

1. **Data and features:** Join the 2024 name-history snapshot by effective date, audit missingness, and obtain delisting returns before final evaluation. First implement the eight 20-day rolling features defined above. Treat reversal, momentum, liquidity, and beta as later additions with separately specified formulas. Fit imputation and scaling only on each training window.
2. **Models:** Start with zero prediction and simple reversal, then OLS/Ridge/Elastic Net and XGBoost or LightGBM. Add logistic regression for classification. Evaluate MLP, GRU/LSTM, and Transformer only after the simpler pipeline works. Use regression losses such as MSE or Huber and binary cross-entropy for classification; reserve Sharpe and drawdown for portfolio evaluation. The constant zero predictor is a loss baseline, not an IC or portfolio baseline.
3. **Validation:** Use chronological `Train → Validation → Test` windows. Record split dates, sample counts, data versions, configurations, and seeds.
4. **Prediction evaluation:** Compute daily cross-sectional Pearson IC and Spearman Rank IC, then report their means, ICIR, hit rates, distributions, and rolling series. For classification, also report cross-entropy, accuracy, balanced accuracy, and ROC-AUC.
5. **Portfolio evaluation:** Report gross and cost-adjusted returns, annualized return and volatility, Sharpe, maximum drawdown, and turnover. State the cost model and any optional Fitness formula explicitly.

## Deliverables and current scope

Deliver a reproducible feature panel, fold-level predictions and metadata, model comparison, stitched out-of-sample IC series, portfolio returns, and a concise conclusion that separates predictive power from net tradable performance.

The present files support data checks and a provisional end-to-end smoke test. Final return labels and portfolio results require delisting-return data; a credible multi-year walk-forward study and independent lockbox also require earlier and later daily history plus matching name history. The annual TAQ sampling rules in `doc/filter.md` use full-year statistics and must not be reused as a historical daily prediction universe.
