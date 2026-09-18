"""Reproducibly build the 2024 CRSP candidate universe in doc/filter.md.

This program deliberately stops before final TAQ packing unless a measured TAQ
pilot CSV is supplied.  The activity proxy is not a GB estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
YEAR = 2024
SEED = 20240917
DSF = ROOT / "data" / f"crsp_dsf_{YEAR}.csv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_names(output: Path) -> None:
    """Snapshot only dsenames intervals that overlap the target calendar year."""
    import wrds

    db = wrds.Connection(wrds_username="wangbojue")
    try:
        names = db.raw_sql(
            """
            SELECT permno, namedt, nameendt, shrcd, exchcd, ticker, comnam
            FROM crsp.dsenames
            WHERE namedt <= %(year_end)s
              AND (nameendt IS NULL OR nameendt >= %(year_start)s)
            ORDER BY permno, namedt
            """,
            params={"year_start": "2024-01-01", "year_end": "2024-12-31"},
            date_cols=["namedt", "nameendt"],
        )
    finally:
        db.close()
    names.to_csv(output, index=False)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    return (series - series.mean()) / std if std and np.isfinite(std) else pd.Series(0.0, index=series.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "crsp_filter_2024")
    parser.add_argument("--names-file", type=Path, help="Existing dsenames snapshot; bypasses WRDS download.")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    names_path = args.names_file or out / "dsenames_2024_snapshot.csv"
    if not names_path.exists():
        download_names(names_path)

    dsf = pd.read_csv(DSF, parse_dates=["date"], low_memory=False)
    names = pd.read_csv(names_path, parse_dates=["namedt", "nameendt"])
    names["nameendt"] = names["nameendt"].fillna(pd.Timestamp("2262-04-11"))
    merged = dsf.merge(names, on="permno", how="inner", validate="many_to_many")
    merged = merged.loc[(merged.date >= merged.namedt) & (merged.date <= merged.nameendt)].copy()
    # Overlapping name intervals should not create duplicate security-days.
    merged = merged.sort_values(["permno", "date", "namedt"]).drop_duplicates(["permno", "date"], keep="last")

    price = merged.prc.abs()
    daily_base = merged.shrcd.isin([10, 11]) & merged.exchcd.isin([1, 2, 3]) & (price >= 5)
    retained = merged.loc[daily_base].copy()
    retained["abs_prc"] = retained.prc.abs()
    retained["valid_price"] = retained.abs_prc.gt(0) & retained.abs_prc.notna()
    retained["valid_vol"] = retained.vol.gt(0) & retained.vol.notna()
    retained["valid_shrout"] = retained.shrout.gt(0) & retained.shrout.notna()
    retained["vol_for_adv"] = retained.vol.where(retained.valid_vol)
    retained["ret_for_volatility"] = pd.to_numeric(retained.ret, errors="coerce")
    retained["dollar_volume"] = np.where(retained.valid_price & retained.valid_vol, retained.abs_prc * retained.vol, np.nan)
    retained["market_cap"] = np.where(retained.valid_price & retained.valid_shrout, retained.abs_prc * retained.shrout * 1000, np.nan)
    retained["turnover"] = np.where(retained.valid_vol & retained.valid_shrout, retained.vol / (retained.shrout * 1000), np.nan)

    def missing_count(x: pd.Series) -> int:
        return int(x.isna().sum())

    features = retained.groupby("permno", sort=True).agg(
        trading_days=("date", "nunique"), med_price=("abs_prc", "median"), adv=("vol_for_adv", "mean"),
        med_dollar_volume=("dollar_volume", "median"), adv_dollar=("dollar_volume", "mean"),
        med_turnover=("turnover", "median"), annual_volatility=("ret_for_volatility", lambda x: x.std(ddof=1) * np.sqrt(252)),
        med_market_cap=("market_cap", "median"), missing_price=("valid_price", lambda x: int((~x).sum())),
        missing_volume=("valid_vol", lambda x: int((~x).sum())), missing_shares=("valid_shrout", lambda x: int((~x).sum())),
        missing_return=("ret", missing_count),
    ).reset_index()
    stock_ok = (features.trading_days >= 200) & (features.med_dollar_volume >= 5_000_000)
    eligible = features.loc[stock_ok].copy()
    q05, q95 = eligible.adv_dollar.quantile([.05, .95])
    candidates = eligible.loc[eligible.adv_dollar.between(q05, q95, inclusive="both")].copy()
    excluded = eligible.loc[~eligible.adv_dollar.between(q05, q95, inclusive="both"), ["permno", "adv_dollar"]].copy()
    excluded["reason"] = np.where(excluded.adv_dollar < q05, "below_q05_adv_dollar", "above_q95_adv_dollar")

    # Values must be positive for log; all candidates should satisfy this from the liquidity rule.
    candidates["activity_proxy"] = (.50 * zscore(np.log(candidates.adv)) + .30 * zscore(np.log(candidates.adv_dollar))
                                    + .15 * zscore(candidates.med_turnover) + .05 * zscore(candidates.annual_volatility))
    candidates["market_cap_quintile"] = pd.qcut(candidates.med_market_cap.rank(method="first"), 5, labels=False) + 1
    candidates["liquidity_quintile"] = pd.qcut(candidates.adv_dollar.rank(method="first"), 5, labels=False) + 1
    candidates = candidates.sort_values(["market_cap_quintile", "liquidity_quintile", "activity_proxy", "permno"], ascending=[True, True, False, True])

    counts = {
        "raw_dsf_rows": len(dsf), "rows_after_point_in_time_name_join": len(merged),
        "daily_common_main_exchange_price_ge_5": len(retained), "stock_featured": len(features),
        "stock_eligible_days_and_liquidity": len(eligible), "post_activity_extremes_candidates": len(candidates),
    }
    metadata = {
        "protocol": "doc/filter.md", "year": YEAR, "random_seed": SEED,
        "run_utc": datetime.now(timezone.utc).isoformat(), "dsf_file": str(DSF), "dsf_sha256": sha256(DSF),
        "names_file": str(names_path), "names_sha256": sha256(names_path), "names_query": "crsp.dsenames overlapping 2024; joined permno and date BETWEEN namedt/nameendt",
        "activity_extreme_cutoffs": {"q05_adv_dollar": q05, "q95_adv_dollar": q95}, "counts": counts,
        "retained_daily_missing_or_nonpositive_counts": {
            "price": int((~retained.valid_price).sum()), "volume": int((~retained.valid_vol).sum()),
            "shares_outstanding": int((~retained.valid_shrout).sum()), "return": int(retained.ret_for_volatility.isna().sum()),
        },
        "quintile_breakpoints": {
            "med_market_cap": candidates.med_market_cap.quantile([.2, .4, .6, .8]).to_dict(),
            "adv_dollar": candidates.adv_dollar.quantile([.2, .4, .6, .8]).to_dict(),
        },
        "taq_status": "No measured pilot supplied: predicted_taq_gb and final budget packing intentionally not produced.",
    }
    features.to_csv(out / "feature_table_all_retained_stocks.csv", index=False)
    candidates.to_csv(out / "candidate_universe.csv", index=False)
    excluded.to_csv(out / "activity_extreme_exclusions.csv", index=False)
    all_cells = pd.MultiIndex.from_product([range(1, 6), range(1, 6)], names=["market_cap_quintile", "liquidity_quintile"])
    candidates.groupby(["market_cap_quintile", "liquidity_quintile"]).size().reindex(all_cells, fill_value=0).rename("candidate_count").reset_index().to_csv(out / "cell_counts.csv", index=False)
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=float) + "\n")
    print(json.dumps(metadata, indent=2, default=float))


if __name__ == "__main__":
    main()
