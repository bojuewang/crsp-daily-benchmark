"""Fetch effective-dated name history without running the annual TAQ filter."""
from pathlib import Path

from run_crsp_filter_2024 import download_names

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    output = ROOT / "output/crsp_filter_2024/dsenames_2024_snapshot.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"Already exists: {output}; move it explicitly before refreshing.")
    download_names(output)
    print(f"Wrote {output}")
