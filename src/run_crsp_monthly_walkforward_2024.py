"""Monthly expanding-window evaluation of the 2024 CRSP benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, log_loss,
                             mean_absolute_error, mean_squared_error, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_crsp_benchmark_2024 import (ROOT, DSF, NAMES, DELIST, FEATURES, SEED, COST_BPS,
                                     build_panel, digest, ic_metrics, portfolio, summary)

OUT = ROOT / "output" / "crsp_monthly_walkforward_2024"
TEST_MONTHS = ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11"]


def regression_model(kind, value):
    if kind == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=value))
    return HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=value, learning_rate=.05,
                                         min_samples_leaf=200, random_state=SEED)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, audit = build_panel()
    dates = pd.Index(sorted(x.date.unique()))
    folds, predictions, class_predictions = [], [], []
    for month in TEST_MONTHS:
        test_month = pd.Period(month, freq="M")
        val_month = test_month - 1
        test_start = dates[dates.to_period("M") == test_month].min()
        val_start = dates[dates.to_period("M") == val_month].min()
        test_start_pos, val_start_pos = dates.get_loc(test_start), dates.get_loc(val_start)
        train_end = dates[val_start_pos - 3]
        val_end = dates[test_start_pos - 3]
        test_end = dates[dates.to_period("M") == test_month].max()
        # Reserve all December realized outcomes for the December lockbox.
        if month == "2024-11":
            test_end = x.loc[x.exit_date.lt("2024-12-01") & x.date.dt.to_period("M").eq(test_month), "date"].max()
        train = x.date.le(train_end) & x.y.notna()
        val = x.date.between(val_start, val_end) & x.y.notna()
        test = x.date.between(test_start, test_end)
        assert x.loc[train, "exit_date"].max() < x.loc[val, "date"].min()
        assert x.loc[val, "exit_date"].max() < x.loc[test, "date"].min()
        assert not (month == "2024-11" and x.loc[test, "exit_date"].max() >= pd.Timestamp("2024-12-01"))
        params = {}
        base = x.loc[test, ["permno", "date", "entry_date", "exit_date", "fill_ok", "y", "reversal"]].copy()
        base["fold"] = month
        for kind, values in {"ridge": [0.1, 10.0, 1000.0], "tree": [3, 7]}.items():
            trials = []
            for value in values:
                model = regression_model(kind, value)
                model.fit(x.loc[train, FEATURES], x.loc[train, "y"])
                mse = mean_squared_error(x.loc[val, "y"], model.predict(x.loc[val, FEATURES]))
                trials.append((mse, value))
            best_loss, best_value = min(trials)
            params[kind] = {"parameter": best_value, "validation_mse": best_loss, "trials": trials}
            model = regression_model(kind, best_value)
            fit = train | val
            model.fit(x.loc[fit, FEATURES], x.loc[fit, "y"])
            z = base.drop(columns="reversal").copy()
            z["model"] = kind
            z["score"] = model.predict(x.loc[test, FEATURES])
            predictions.append(z)
        for kind, score in [("zero", 0), ("reversal", base.reversal)]:
            z = base.drop(columns="reversal").copy()
            z["model"] = kind
            z["score"] = score
            predictions.append(z)
        classifier = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=1.0, max_iter=200, random_state=SEED))
        fit = train | val
        classifier.fit(x.loc[fit, FEATURES], x.loc[fit, "y"].gt(0).astype(int))
        c = base[["fold", "permno", "date", "y"]].copy()
        c["prob_up"] = classifier.predict_proba(x.loc[test, FEATURES])[:, 1]
        class_predictions.append(c)
        folds.append({"fold": month, "train_first": str(x.loc[train, "date"].min().date()),
                      "train_last": str(train_end.date()), "validation_first": str(val_start.date()),
                      "validation_last": str(val_end.date()), "test_first": str(test_start.date()),
                      "test_last": str(test_end.date()), "train_rows": int(train.sum()),
                      "validation_rows": int(val.sum()), "test_rows": int(test.sum()),
                      "selection": params})
        print("completed", month, "test rows", int(test.sum()), flush=True)
    pred = pd.concat(predictions, ignore_index=True).sort_values(["date", "model", "permno"])
    assert not pred.duplicated(["model", "permno", "date"]).any()
    pred.to_csv(OUT / "predictions.csv.gz", index=False, compression="gzip")
    cp = pd.concat(class_predictions, ignore_index=True).sort_values(["date", "permno"])
    cp.to_csv(OUT / "classification_predictions.csv.gz", index=False, compression="gzip")
    fold_metrics = []
    for (fold, model), g in pred.groupby(["fold", "model"]):
        v = g.y.notna() & g.score.notna()
        row = {"fold": fold, "model": model, "n": len(g), "valid_labels": int(v.sum()),
               "mse": mean_squared_error(g.loc[v, "y"], g.loc[v, "score"]),
               "mae": mean_absolute_error(g.loc[v, "y"], g.loc[v, "score"])}
        if model != "zero":
            daily = ic_metrics(g)
            row.update({"ic_mean": daily.ic.mean(), "rank_ic_mean": daily.rank_ic.mean(),
                        "rank_ic_days": len(daily)})
        fold_metrics.append(row)
    pd.DataFrame(fold_metrics).to_csv(OUT / "fold_metrics.csv", index=False)
    summaries, portfolios = [], {}
    for kind in ["zero", "reversal", "ridge", "tree"]:
        z = pred.loc[pred.model.eq(kind)].copy()
        v = z.y.notna() & z.score.notna()
        if kind == "zero":
            summaries.append({"model": kind, "n_predictions": len(z), "n_valid_labels": int(v.sum()),
                              "mse": mean_squared_error(z.loc[v, "y"], z.loc[v, "score"]),
                              "mae": mean_absolute_error(z.loc[v, "y"], z.loc[v, "score"])})
            continue
        daily, port = ic_metrics(z), portfolio(z)
        daily.to_csv(OUT / f"ic_{kind}.csv", index=False)
        port.to_csv(OUT / f"portfolio_{kind}.csv", index=False)
        portfolios[kind] = (z, daily, port)
    common = set.intersection(*(set(p.loc[p.net.notna(), "date"]) for _, _, p in portfolios.values()))
    pd.DataFrame({"date": sorted(common)}).to_csv(OUT / "common_portfolio_dates.csv", index=False)
    for kind, (z, daily, port) in portfolios.items():
        selected = port.loc[port.date.isin(common)]
        summaries.append({"model": kind, **summary(z, daily, selected),
                          "all_test_port_days": len(port), "all_test_missing_port_days": int(port.net.isna().sum())})
    pd.DataFrame(summaries).to_csv(OUT / "model_comparison.csv", index=False)
    cv = cp.loc[cp.y.notna()]
    true, prob = cv.y.gt(0).astype(int), cv.prob_up
    classification = {"n": len(cv), "positive_rate": true.mean(), "log_loss": log_loss(true, prob),
                      "accuracy": accuracy_score(true, prob.ge(.5)),
                      "balanced_accuracy": balanced_accuracy_score(true, prob.ge(.5)),
                      "roc_auc": roc_auc_score(true, prob)}
    pd.DataFrame([classification]).to_csv(OUT / "classification_metrics.csv", index=False)
    metadata = {"protocol": "monthly expanding train / preceding-month validation / current-month test",
                "months": TEST_MONTHS, "seed": SEED, "cost_bps_per_unit_turnover": COST_BPS,
                "folds": folds, "data_audit": audit, "common_portfolio_dates": len(common),
                "december_lockbox": "All December signals and realized returns untouched",
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [DSF, NAMES, DELIST]}}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=float) + "\n")
    print(json.dumps({"comparison": summaries, "classification": classification,
                      "common_portfolio_dates": len(common)}, indent=2, default=float))


if __name__ == "__main__":
    main()
