"""Download the latest complete CRSP daily stock-file year from WRDS."""

from pathlib import Path
import gzip
import pandas as pd
import wrds


YEAR = 2024
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / f"crsp_dsf_{YEAR}.csv.gz"


def main() -> None:
    db = wrds.Connection(wrds_username="wangbojue")
    try:
        with gzip.open(OUTPUT, "wt", newline="") as handle:
            for month in range(1, 13):
                start = pd.Timestamp(YEAR, month, 1)
                end = start + pd.offsets.MonthBegin(1)
                data = db.raw_sql(
                    """
                    SELECT *
                    FROM crsp.dsf
                    WHERE date >= %(start)s AND date < %(end)s
                    ORDER BY date, permno
                    """,
                    params={"start": start.date(), "end": end.date()},
                    date_cols=["date"],
                )
                data.to_csv(handle, index=False, header=(month == 1))
                print(f"{start:%Y-%m}: {len(data):,} rows")
    finally:
        db.close()

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
