# Validation and interpretation

## Status on 2026-10-07

Numerical claims come from the existing monthly report at commit `bd86235de9aec79aff0091c4163790bc86e27357`. Git excludes the licensed inputs and prediction caches under `data/` and `output/`. Historical input hashes are not available for inspection here. Old results are **repository-reported**, not independently reproduced in this update.

Fifteen unittest checks passed with Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0 and SciPy 1.17.0. WRDS downloads and the full CRSP run were not executed. Test fixtures are synthetic and have no financial interpretation.

## Checked behavior

| Check | Failure detected |
| --- | --- |
| Mutate future prices, volumes and returns | Earlier features depending on future data |
| Remove one security-day | Market-day rolling window compressing missing dates |
| Inspect signal/entry/exit dates | Incorrect holding-period label |
| Change eligibility interval | Future eligibility affecting membership |
| Combine regular and delisting returns | Adding rather than compounding returns |
| Remove entry price | Failed fill retaining a valid label |
| Check five folds; inject a delayed label | Unobserved labels leaking across blocks; December test outcomes |
| Hand-calculate deciles and initial cost | Incorrect exposure or turnover convention |
| Remove selected versus unselected outcomes | Missing holding returns silently treated as zero |
| Duplicate predictions; alter reported Rank IC | Corrupted saved outputs passing checks |
| Require missing inputs | Artifact consistency confused with input verification |

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/verify_walkforward_outputs.py
.venv/bin/python src/verify_walkforward_outputs.py --require-inputs
```

The verifier checks unique prediction keys, identical targets/coverage across models, date chronology, all test months, December exclusion, daily IC and portfolio files, the common complete-date intersection, aggregate regression/IC/portfolio statistics and available input hashes. It does not reconstruct fits, check classification or every fold-metric file, certify historical timestamps, or establish significance. Full reproduction requires source version, licensed inputs, output artifacts and an environment record.

## Limits that change interpretation

1. **Missing-date selection:** only 63 of 105 dates enter common portfolio statistics. Missing outcomes may be informative. Annualizing the filtered sequence does not establish a realizable continuous-year return.
2. **Future fill conditioning:** `portfolio()` drops failed future entry fills and reranks survivors. An executable policy requires intended weights chosen at signal time and a predeclared failed-fill rule. Changing this rule requires a new experiment.
3. **Accounting proxy:** aggregate exposure is +1 long and −1 short. Costs use target-weight changes, including initial entry, without full drift-adjusted execution, financing, borrow, spread, impact, terminal liquidation or missing-date position accounting.
4. **Uncertainty:** five monthly folds within one year do not establish multi-regime stability. Existing results include no confidence intervals or significance claim. Classification is near random.
5. **Selection history:** July–November outcomes have been observed; further tuning is exploratory. December is excluded by code, but independent lockbox status also depends on the researcher's prior access and decisions.
6. **Data history:** effective-dated names mitigate universe leakage but do not certify every vendor revision. Missing delisting returns remain missing. Annual TAQ screens must not determine historical prediction membership.

## Next experiment, not a completed result

Freeze the hypothesis, code/input hashes, features, parameter search and executable fill rules before further tuning. Run multi-year chronological folds, cover the entire test calendar, report missingness and all baselines, and prespecify daily-level uncertainty estimates such as a block bootstrap. Keep an independent holdout outside selection. These extensions are not claimed as implemented by this update.
