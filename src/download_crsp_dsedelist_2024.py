"""Download nonmissing 2024 CRSP daily delisting returns from WRDS."""
from pathlib import Path
import os

import wrds

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "crsp_dsedelist_2024.csv.gz"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection(wrds_username=os.environ.get("WRDS_USERNAME"))
    try:
        data = db.raw_sql(
            """SELECT permno, dlstdt, dlret, dlstcd FROM crsp.dsedelist
               WHERE dlstdt BETWEEN '2024-01-01' AND '2024-12-31'
                 AND dlret IS NOT NULL
               ORDER BY permno, dlstdt""",
            date_cols=["dlstdt"],
        )
    finally:
        db.close()
    if data.duplicated(["permno", "dlstdt"]).any():
        raise ValueError("Duplicate delisting return keys")
    data.to_csv(OUTPUT, index=False, compression="gzip")
    print(f"Wrote {len(data):,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
