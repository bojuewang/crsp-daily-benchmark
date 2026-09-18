"""One-year, chronological CRSP benchmark following doc/pipeline.md."""
from __future__ import annotations

import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "crsp_benchmark_2024"
DSF = ROOT / "data" / "crsp_dsf_2024.csv.gz"
NAMES = ROOT / "output" / "crsp_filter_2024" / "dsenames_2024_snapshot.csv"
DELIST = ROOT / "data" / "crsp_dsedelist_2024.csv.gz"
FEATURES = ["trading_days", "med_price", "adv", "med_dollar_volume", "adv_dollar",
            "med_turnover", "annual_volatility", "med_market_cap"]
SEED = 20240918
COST_BPS = 10


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_panel():
    d = pd.read_csv(DSF, usecols=["permno", "date", "prc", "vol", "shrout", "ret"], parse_dates=["date"])
    n = pd.read_csv(NAMES, usecols=["permno", "namedt", "nameendt", "shrcd", "exchcd"],
                    parse_dates=["namedt", "nameendt"])
    assert not d.duplicated(["permno", "date"]).any()
    dates = np.sort(d.date.unique())
    date_index = pd.Index(dates)
    d["day"] = date_index.get_indexer(d.date)
    # Full calendar grid makes rolling(20) mean 20 market days, including absent security dates.
    idx = pd.MultiIndex.from_product([np.sort(d.permno.unique()), range(len(dates))], names=["permno", "day"])
    p = d.set_index(["permno", "day"])[["prc", "vol", "shrout", "ret"]].reindex(idx)
    p["date"] = dates[p.index.get_level_values("day")]
    # Effective-date join; no annual eligibility or survival screen.
    names = p.reset_index()[["permno", "day", "date"]].merge(n, on="permno", how="left")
    names = names.loc[names.date.ge(names.namedt) & names.date.le(names.nameendt.fillna(pd.Timestamp("2262-04-11")))]
    assert not names.duplicated(["permno", "day"]).any(), "overlapping name intervals"
    good = names.shrcd.isin([10, 11]) & names.exchcd.isin([1, 2, 3])
    eligible = pd.Series(False, index=idx)
    eligible.loc[pd.MultiIndex.from_frame(names.loc[good, ["permno", "day"]])] = True
    price = p.prc.abs()
    eligible &= price.ge(5)
    valid = price.ge(5)
    volume = p.vol.where(valid & p.vol.ge(0))
    positive_volume = p.vol.where(valid & p.vol.gt(0))
    shares = p.shrout.mul(1000).where(p.shrout.gt(0))
    grp = lambda s: s.groupby(level="permno").rolling(20, min_periods=1)
    # Drop duplicate group level introduced by groupby.rolling.
    roll = lambda s, op, minp=10: getattr(s.groupby(level="permno").rolling(20, min_periods=minp), op)().droplevel(0)
    x = pd.DataFrame(index=idx)
    x["trading_days"] = roll(price.where(valid).notna().astype(float), "sum", 20)
    x["med_price"] = roll(price.where(valid), "median")
    x["adv"] = roll(volume, "mean")
    dollar = (price * positive_volume).where(valid)
    x["med_dollar_volume"] = roll(dollar, "median")
    x["adv_dollar"] = roll(dollar, "mean")
    x["med_turnover"] = roll((positive_volume / shares).where(valid), "median")
    x["annual_volatility"] = roll(p.ret.where(valid), "std") * np.sqrt(252)
    x["med_market_cap"] = roll((price * shares).where(valid), "median")
    x["reversal"] = -p.ret
    x["date"] = p.date
    x["eligible"] = eligible
    x = x.loc[x.eligible & x[FEATURES].notna().all(axis=1) & x.trading_days.ge(10)].copy()
    x = x.loc[x.index.get_level_values("day") <= len(dates)-3]
    # Signal at close t, fill at close t+1, earn return dated t+2.
    x["entry_day"] = x.index.get_level_values("day") + 1
    x["exit_day"] = x.index.get_level_values("day") + 2
    x["entry_date"] = dates[x.entry_day]
    x["exit_date"] = dates[x.exit_day]
    entry = p[["prc"]].rename(columns={"prc": "entry_prc"}).reset_index()
    x = x.reset_index().merge(entry, left_on=["permno", "entry_day"], right_on=["permno", "day"],
                              how="left", suffixes=("", "_entry")).drop(columns="day_entry")
    x["fill_ok"] = x.entry_prc.abs().gt(0)
    outcome = p[["ret"]].reset_index().rename(columns={"ret": "exit_ret", "day": "exit_day"})
    x = x.merge(outcome, on=["permno", "exit_day"], how="left")
    dl = pd.read_csv(DELIST, parse_dates=["dlstdt"])
    assert not dl.duplicated(["permno", "dlstdt"]).any()
    x = x.merge(dl[["permno", "dlstdt", "dlret"]], left_on=["permno", "exit_date"],
                right_on=["permno", "dlstdt"], how="left")
    r, dr = x.exit_ret, x.dlret
    x["y"] = np.where(r.notna() & dr.notna(), (1+r)*(1+dr)-1, r.fillna(dr))
    x.loc[~x.fill_ok, "y"] = np.nan
    audit = {"raw_rows": len(d), "raw_permnos": d.permno.nunique(), "market_days": len(dates),
             "feature_eligible_rows": len(x), "entry_fill_missing": int((~x.fill_ok).sum()),
             "exit_ret_only": int((r.notna() & dr.isna() & x.fill_ok).sum()),
             "exit_dlret_only": int((r.isna() & dr.notna() & x.fill_ok).sum()),
             "exit_both": int((r.notna() & dr.notna() & x.fill_ok).sum()),
             "exit_both_missing": int((r.isna() & dr.isna() & x.fill_ok).sum()),
             "valid_labels": int(x.y.notna().sum())}
    assert (x.entry_day == x.day + 1).all() and (x.exit_day == x.day + 2).all()
    assert x.loc[r.notna() & dr.notna() & x.fill_ok, "y"].notna().all()
    return x, audit


def ic_metrics(frame):
    daily = []
    for date, g in frame.groupby("date"):
        z = g[["score", "y"]].dropna()
        if len(z) < 20: continue
        daily.append({"date": date, "n": len(z), "ic": z.score.corr(z.y),
                      "rank_ic": z.score.corr(z.y, method="spearman")})
    return pd.DataFrame(daily)


def portfolio(frame):
    rows, prev = [], {}
    for date, g in frame.groupby("date", sort=True):
        g = g.loc[g.fill_ok & g.score.notna()].sort_values(["score", "permno"])
        if len(g) < 20: continue
        k = max(1, len(g)//10)
        short, long = g.head(k), g.tail(k)
        selected = pd.concat([short, long])
        weights = {int(i): w for i, w in zip(short.permno, np.repeat(-1/k, k))}
        weights.update({int(i): w for i, w in zip(long.permno, np.repeat(1/k, k))})
        turnover = sum(abs(weights.get(i, 0)-prev.get(i, 0)) for i in weights.keys() | prev.keys())
        missing = int(selected.y.isna().sum())
        gross = sum(weights[int(i)]*y for i, y in zip(selected.permno, selected.y)) if not missing else np.nan
        rows.append({"date": date, "n_candidates": len(g), "n_per_side": k, "missing_selected_returns": missing,
                     "turnover": turnover, "gross": gross, "net": gross-turnover*COST_BPS/10000 if not missing else np.nan})
        prev = weights
    return pd.DataFrame(rows)


def summary(pred, daily, port):
    y, s = pred.y, pred.score
    valid = y.notna() & s.notna()
    p = port.dropna(subset=["net"])
    out = {"n_predictions": len(pred), "n_valid_labels": int(valid.sum()),
           "mse": mean_squared_error(y[valid], s[valid]), "mae": mean_absolute_error(y[valid], s[valid]),
           "ic_mean": daily.ic.mean(), "rank_ic_mean": daily.rank_ic.mean(),
           "icir": daily.ic.mean()/daily.ic.std(ddof=1),
           "rank_icir": daily.rank_ic.mean()/daily.rank_ic.std(ddof=1),
           "rank_ic_hit_rate": (daily.rank_ic > 0).mean(), "n_ic_days": len(daily),
           "n_port_days": len(port), "n_valid_port_days": len(p),
           "n_missing_port_days": int(port.net.isna().sum()), "mean_turnover": port.turnover.mean()}
    for col in ["gross", "net"]:
        v = p[col]
        wealth = pd.concat([pd.Series([1.0]), (1+v).cumprod()], ignore_index=True)
        out[col+"_ann_return"] = wealth.iloc[-1]**(252/len(v))-1
        out[col+"_ann_vol"] = v.std(ddof=1)*np.sqrt(252)
        out[col+"_sharpe"] = v.mean()/v.std(ddof=1)*np.sqrt(252)
        out[col+"_max_drawdown"] = (wealth/wealth.cummax()-1).min()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, audit = build_panel()
    # Two trading-day embargo before each following block: all prior labels are known.
    train = x.date.le("2024-06-26") & x.y.notna()
    val = x.date.between("2024-07-01", "2024-08-28") & x.y.notna()
    test = x.date.between("2024-09-03", "2024-11-27")
    lockbox = x.date.ge("2024-12-02")
    assert x.loc[train, "exit_date"].max() < x.loc[val, "date"].min()
    assert x.loc[val, "exit_date"].max() < x.loc[test, "date"].min()
    # Hyperparameters chosen on validation only. No December outcomes are inspected.
    configs = {"ridge": [0.1, 10.0, 1000.0], "tree": [3, 7]}
    fitted, choice = {}, {}
    for kind, values in configs.items():
        trials = []
        for v in values:
            if kind == "ridge":
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=v))
            else:
                model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=v, learning_rate=.05,
                                                       min_samples_leaf=200, random_state=SEED)
            model.fit(x.loc[train, FEATURES], x.loc[train, "y"])
            score = model.predict(x.loc[val, FEATURES])
            trials.append((mean_squared_error(x.loc[val, "y"], score), v))
        best_loss, best = min(trials)
        choice[kind] = {"validation_mse": best_loss, "parameter": best, "all_trials": trials}
        if kind == "ridge":
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=best))
        else:
            model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=best, learning_rate=.05,
                                                   min_samples_leaf=200, random_state=SEED)
        fit = train | val
        model.fit(x.loc[fit, FEATURES], x.loc[fit, "y"])
        fitted[kind] = model
    base = x.loc[test, ["permno", "day", "date", "entry_date", "exit_date", "fill_ok", "y", "reversal"]].copy()
    predictions, summaries, daily_by_model, port_by_model = [], [], {}, {}
    for kind in ["zero", "reversal", "ridge", "tree"]:
        z = base.copy()
        z["model"] = kind
        z["score"] = (0 if kind == "zero" else z.reversal if kind == "reversal"
                      else fitted[kind].predict(x.loc[test, FEATURES]))
        predictions.append(z.drop(columns="reversal"))
        if kind == "zero":
            # Constant scores have no cross-sectional ranking; loss baseline only.
            valid = z.y.notna()
            summaries.append({"model": kind, "n_predictions": len(z), "n_valid_labels": int(valid.sum()),
                              "mse": mean_squared_error(z.loc[valid, "y"], z.loc[valid, "score"]),
                              "mae": mean_absolute_error(z.loc[valid, "y"], z.loc[valid, "score"])})
            continue
        daily = ic_metrics(z)
        port = portfolio(z)
        daily.insert(0, "model", kind)
        port.insert(0, "model", kind)
        daily.to_csv(OUT / f"ic_{kind}.csv", index=False)
        port.to_csv(OUT / f"portfolio_{kind}.csv", index=False)
        daily_by_model[kind] = daily
        port_by_model[kind] = port
    common_dates = set.intersection(*(set(p.loc[p.net.notna(), "date"]) for p in port_by_model.values()))
    pd.DataFrame({"date": sorted(common_dates)}).to_csv(OUT / "common_portfolio_dates.csv", index=False)
    for kind in ["reversal", "ridge", "tree"]:
        z = next(p for p in predictions if p.model.iloc[0] == kind)
        daily, port = daily_by_model[kind], port_by_model[kind]
        comparable_port = port.loc[port.date.isin(common_dates)].copy()
        summaries.append({"model": kind, **summary(z, daily, comparable_port),
                          "all_test_port_days": len(port), "all_test_missing_port_days": int(port.net.isna().sum())})
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(OUT / "predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(summaries).to_csv(OUT / "model_comparison.csv", index=False)
    classifier = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                               LogisticRegression(C=1.0, max_iter=200, random_state=SEED))
    fit = train | val
    classifier.fit(x.loc[fit, FEATURES], x.loc[fit, "y"].gt(0).astype(int))
    c = x.loc[test, ["permno", "date", "y"]].copy()
    c["prob_up"] = classifier.predict_proba(x.loc[test, FEATURES])[:, 1]
    c.to_csv(OUT / "classification_predictions.csv.gz", index=False, compression="gzip")
    cv = c.loc[c.y.notna()]
    true = cv.y.gt(0).astype(int)
    prob = cv.prob_up
    classification_metrics = {"n": len(cv), "positive_rate": true.mean(),
                              "log_loss": log_loss(true, prob),
                              "accuracy": accuracy_score(true, prob.ge(.5)),
                              "balanced_accuracy": balanced_accuracy_score(true, prob.ge(.5)),
                              "roc_auc": roc_auc_score(true, prob)}
    pd.DataFrame([classification_metrics]).to_csv(OUT / "classification_metrics.csv", index=False)
    # Compact reproducible feature panel (including valid training/validation/test rows).
    x.loc[train | val | test, ["permno", "date", "entry_date", "exit_date", *FEATURES, "reversal", "fill_ok", "y"]].to_csv(
        OUT / "feature_panel.csv.gz", index=False, compression="gzip")
    meta = {"protocol": "doc/pipeline.md", "seed": SEED, "cost_bps_per_unit_turnover": COST_BPS,
            "label": "close(t+1) to close(t+2), CRSP ret at t+2 combined with dlret when available",
            "split": {"train": ["2024-01-02", "2024-06-26"], "validation": ["2024-07-01", "2024-08-28"],
                      "test": ["2024-09-03", "2024-11-27"], "untouched_lockbox": ["2024-12-02", "2024-12-31"]},
            "split_counts": {"train": int(train.sum()), "validation": int(val.sum()), "test": int(test.sum()),
                             "lockbox_feature_rows": int(lockbox.sum())},
            "audit": audit, "selection": choice, "classification": {"model": "logistic_regression", "C": 1.0},
            "common_portfolio_dates": len(common_dates),
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [DSF, NAMES, DELIST]},
            "limitations": ["Only 2024 data; no multiyear walk-forward", "December lockbox not evaluated",
                            "Portfolio metrics use only dates with complete selected returns for all models"]}
    (OUT / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=float) + "\n")
    print(json.dumps({"audit": audit, "split_counts": meta["split_counts"], "model_comparison": summaries,
                      "classification_metrics": classification_metrics},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
