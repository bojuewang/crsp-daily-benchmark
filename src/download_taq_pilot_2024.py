"""Download a deterministic, stratified 2024 TAQ Trades + NBBO pilot from WRDS.

Files are gzip-compressed CSV, partitioned by PERMNO and feed.  This is the
same layout whose file sizes are later used for the TAQ size calibration.
"""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import wrds

ROOT = Path(__file__).resolve().parents[1]
FILTER_DIR = ROOT / "output" / "crsp_filter_2024"
YEAR, SEED = 2024, 20240917


def pilot_permnos(candidates: pd.DataFrame, n: int) -> pd.DataFrame:
    """One median-activity name per non-empty cell, then deterministic extras."""
    ordered = candidates.sort_values(["market_cap_quintile", "liquidity_quintile", "activity_proxy", "permno"])
    chosen_index = ordered.groupby(["market_cap_quintile", "liquidity_quintile"]).apply(lambda x: x.index[len(x) // 2])
    cell_first = ordered.loc[chosen_index.to_numpy()]
    extra = ordered.loc[~ordered.permno.isin(cell_first.permno)].sort_values(["activity_proxy", "permno"])
    return pd.concat([cell_first, extra.head(n - len(cell_first))], ignore_index=True).sort_values("permno")


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    has_data = path.exists() and path.stat().st_size > 0
    with gzip.open(path, "at", newline="") as f:
        frame.to_csv(f, index=False, header=not has_data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "taq_2024_pilot")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    if not 20 <= args.n <= 30:
        raise ValueError("The protocol requires a 20–30 name pilot.")
    candidates = pd.read_csv(FILTER_DIR / "candidate_universe.csv")
    selected = pilot_permnos(candidates, args.n)
    selected.to_csv(args.output_dir / "pilot_permnos.csv", index=False) if args.output_dir.exists() else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "pilot_permnos.csv", index=False)

    db = wrds.Connection(wrds_username="wangbojue")
    log: list[dict] = []
    try:
        trading_dates = pd.read_csv(ROOT / "data" / "crsp_dsf_2024.csv.gz", usecols=["date"]).date.unique()
        days = pd.DatetimeIndex(pd.to_datetime(trading_dates)).sort_values()
        days = days[(days >= pd.Timestamp(args.start)) & (days <= pd.Timestamp(args.end))]
        permnos = tuple(map(int, selected.permno))
        for day in days:
            stamp = day.strftime("%Y%m%d")
            # Link table makes the CRSP-PERMNO to TAQ-symbol mapping point-in-time.
            links = db.raw_sql(
                """SELECT DISTINCT ON (permno) permno, sym_root, sym_suffix
                   FROM wrdsapps_link_crsp_taqm.taqmclink
                   WHERE date = %(day)s AND permno IN %(permnos)s
                   ORDER BY permno, match_lvl DESC NULLS LAST""",
                params={"day": day.date(), "permnos": permnos},
            )
            if links.empty:
                log.append({"date": stamp, "status": "no_links"})
                continue
            def sql_literal(value: object) -> str:
                return "NULL" if pd.isna(value) else "'" + str(value).replace("'", "''") + "'"
            values = ", ".join("(%d, %s, %s)" % (r.permno, sql_literal(r.sym_root), sql_literal(r.sym_suffix)) for r in links.itertuples(index=False))
            for feed, table in (("trades", f"ctm_{stamp}"), ("nbbo", f"complete_nbbo_{stamp}")):
                sql = f"""WITH links(permno, sym_root, sym_suffix) AS (VALUES {values})
                    SELECT l.permno, t.* FROM taqm_2024.{table} AS t
                    JOIN links AS l ON t.sym_root = l.sym_root
                       AND t.sym_suffix IS NOT DISTINCT FROM l.sym_suffix"""
                frame = db.raw_sql(sql)
                for permno, one in frame.groupby("permno"):
                    append_csv(args.output_dir / feed / f"permno={int(permno)}.csv.gz", one)
                log.append({"date": stamp, "feed": feed, "linked_permnos": len(links), "rows": len(frame)})
                print(stamp, feed, f"{len(frame):,} rows", flush=True)
    finally:
        db.close()
    pd.DataFrame(log).to_csv(args.output_dir / "download_log.csv", index=False)
    sizes = []
    for path in args.output_dir.glob("*/*.csv.gz"):
        sizes.append({"feed": path.parent.name, "permno": int(path.stem.split("=")[1].split(".")[0]), "compressed_bytes": path.stat().st_size})
    pd.DataFrame(sizes).groupby("permno", as_index=False).compressed_bytes.sum().assign(measured_taq_gb=lambda x: x.compressed_bytes / 1e9).to_csv(args.output_dir / "pilot_measured_sizes.csv", index=False)
    (args.output_dir / "run_metadata.json").write_text(json.dumps({"year": YEAR, "seed": SEED, "n": args.n, "start": args.start, "end": args.end, "layout": "per-PERMNO gzip CSV; feeds=trades,nbbo", "run_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
