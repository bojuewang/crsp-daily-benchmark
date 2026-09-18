---
title: "2024 CRSP → TAQ budget-aware sampling protocol"
---

# 2024 CRSP → TAQ budget-aware sampling protocol

## Objective

Define a reproducible 2024 equity universe from CRSP Daily, then choose a market-cap/liquidity-balanced subset whose *predicted* full-year TAQ (`Trades + NBBO`) size is at most **225 GB**. The remaining 25 GB from the 250 GB storage allocation is an error buffer, not part of the packing budget.

This is not a `top_n` or top-dollar-volume screen. The most active names can generate a disproportionate number of quote records and exhaust the TAQ budget while reducing cross-sectional coverage.

## Data and point-in-time identifiers

Use `crsp.dsf` for daily price, return, volume, and shares outstanding. `crsp.dsf` alone does **not** contain `SHRCD` or `EXCHCD` (the local `data/crsp_dsf_2024.csv.gz` is an example). Join the CRSP names/history table (normally `crsp.dsenames`) by `permno` and effective date before applying the universe rule:

```sql
ON d.permno = n.permno
AND d.date BETWEEN n.namedt AND n.nameendt
```

Treat open-ended `nameendt` as current/future. Do not join names only by `permno`: classifications and exchange membership may change during the year.

## Daily eligibility and measures

At each daily observation retain only common stocks listed on the three main US exchanges:

```text
shrcd ∈ {10, 11}
exchcd ∈ {1, 2, 3}
price = abs(prc) ≥ 5
```

CRSP can encode a negative `prc` when it is based on a bid/ask quote; use `abs(prc)` throughout rather than dropping those observations. Define:

```text
dollar_volume_it = abs(prc_it) * vol_it
market_cap_it     = abs(prc_it) * shrout_it * 1,000
turnover_it       = vol_it / (shrout_it * 1,000)
```

`vol` is in shares and `shrout` is in thousands of shares. Observations with missing/non-positive prices, volumes, or shares outstanding do not contribute to the corresponding statistic; report their counts.

For each `permno`, calculate over the retained daily observations:

| Field | Definition |
| --- | --- |
| `trading_days` | number of distinct CRSP dates |
| `med_price` | median `abs(prc)` |
| `adv` | mean `vol` |
| `med_dollar_volume` | median daily dollar volume |
| `adv_dollar` | mean daily dollar volume |
| `med_turnover` | median daily turnover |
| `annual_volatility` | standard deviation of daily `ret` × sqrt(252) |
| `med_market_cap` | median daily market cap |

The stock-level eligibility screen is:

```text
trading_days       ≥ 200
med_dollar_volume  ≥ $5,000,000
```

The daily $5 price condition is intentionally applied *before* aggregation. It therefore requires a stock to satisfy the liquidity and day-count rules using economically usable ($5+) days, rather than allowing a low-price period to inflate its trading-day count.

## Remove activity extremes

Among stock-level eligible names, calculate the 5th and 95th percentiles of `adv_dollar`, and retain names inside the inclusive interval:

```text
q05(adv_dollar) ≤ adv_dollar ≤ q95(adv_dollar)
```

This removes both illiquid tail names and TAQ-heavy super-active names. Record the actual percentile cutoffs and excluded PERMNOs. A 10th–90th variant is permitted only as a documented sensitivity analysis, not as an unrecorded substitution.

## Activity proxy and TAQ-size calibration

Before a TAQ pilot is available, rank remaining candidates with the dimensionless activity proxy:

```text
S = 0.50 z(log(adv))
  + 0.30 z(log(adv_dollar))
  + 0.15 z(med_turnover)
  + 0.05 z(annual_volatility)
```

Each z score is computed across the post-extreme-filter candidate set. The proxy is for ranking and allocation only; **it is not GB**.

For the production selection, download a pilot of 20–30 names spanning the market-cap and liquidity cells, measure the compressed on-disk TAQ size using the same files, schema, compression, and partitioning planned for production, and fit:

```text
log(TAQ_GB_i) = β0 + β1 log(adv_i) + β2 log(adv_dollar_i)
                + β3 med_turnover_i + β4 annual_volatility_i + ε_i
```

Use the fitted model to create `predicted_taq_gb`. Check that predictions are positive and inspect holdout residuals; if the pilot is too small for a stable multivariable fit, use a simpler, documented model rather than treating `S` as a size estimate.

## Stratification and budget packing

Create quintiles of `med_market_cap` and `adv_dollar` in the candidate set. Sample/allocate across their 5 × 5 cells so that no small group of mega-cap, high-activity names dominates. A practical starting allocation is roughly 180 names: 30 each from low/medium and high liquidity in each of mega/large, mid, and small cap bands. Empty or thin cells should be recorded and reallocated deterministically.

Within each cell, use a fixed random seed and rank by the activity proxy (or other pre-registered rule). Pack candidates while preserving the target cell allocation and enforcing the budget:

```python
budget_gb = 225.0
used_gb = 0.0
selected = []

for stock in allocation_order:
    predicted = stock.predicted_taq_gb
    if predicted > 0 and used_gb + predicted <= budget_gb:
        selected.append(stock.permno)
        used_gb += predicted
```

The expected final size is `used_gb ≤ 225 GB`; the provisioned storage is approximately 250 GB. Do not replace this rule with a fixed stock count. For full-year Trades + NBBO, expect a first production run to contain roughly 150–200 names, but the prediction-and-packing result is authoritative.

## Required reproducibility outputs

Persist the following with every run:

1. CRSP/name-history extract version and query dates.
2. Rule counts after each daily and stock-level filter.
3. Feature table keyed by `permno`, including missing-value counts.
4. Quantile cutoffs, quintile breakpoints, allocation, and random seed.
5. Pilot PERMNOs, measured sizes, model specification/coefficients, and validation error.
6. Final selected PERMNOs with `predicted_taq_gb`, cumulative predicted GB, and actual downloaded GB.

Record actual cumulative downloaded size throughout the run for calibration and
capacity monitoring. It is an observed diagnostic, not a hard stop: the
experiment's storage budget is supplied by the run configuration and may vary
with the available hardware.
