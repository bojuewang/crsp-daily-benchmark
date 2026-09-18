"""Recompute point-in-time daily sample counts for the presentation."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "crsp_monthly_walkforward_2024" / "data_audit.csv"


def main():
    d = pd.read_csv(ROOT / "data/crsp_dsf_2024.csv.gz",
                    usecols=["permno", "date", "prc", "ret", "shrout"], parse_dates=["date"])
    n = pd.read_csv(ROOT / "output/crsp_filter_2024/dsenames_2024_snapshot.csv",
                    usecols=["permno", "namedt", "nameendt", "shrcd", "exchcd"],
                    parse_dates=["namedt", "nameendt"])
    n["nameendt"] = n.nameendt.fillna(pd.Timestamp("2262-04-11"))
    j = d.merge(n, on="permno", how="left")
    j = j.loc[j.date.ge(j.namedt) & j.date.le(j.nameendt)]
    if j.duplicated(["permno", "date"]).any():
        raise ValueError("Overlapping effective-name intervals")
    common = j.shrcd.isin([10, 11]) & j.exchcd.isin([1, 2, 3])
    price = j.prc.abs().ge(5)
    rows = [
        ("Raw DSF", len(d)),
        ("Effective-date name match", len(j)),
        ("Common shares on main exchanges", int(common.sum())),
        ("Above plus signal-day abs(price) >= $5", int((common & price).sum())),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["stage", "security_day_rows"]).to_csv(OUT, index=False)
    print("Date range:", d.date.min().date(), "to", d.date.max().date())
    print("PERMNOs:", d.permno.nunique(), "trading dates:", d.date.nunique())
    print("Negative PRC rows:", int(d.prc.lt(0).sum()))
    print(pd.DataFrame(rows, columns=["stage", "security_day_rows"]).to_string(index=False))


if __name__ == "__main__":
    main()
