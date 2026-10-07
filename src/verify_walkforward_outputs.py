"""Verify saved monthly artifacts without fitting models or inventing results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_crsp_benchmark_2024 import ROOT, digest, ic_metrics, portfolio, summary
from run_crsp_monthly_walkforward_2024 import TEST_MONTHS

STRATEGIES = ("reversal", "ridge", "tree")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_numeric(actual, expected, label):
    require(len(actual) == len(expected), f"{label}: row count mismatch")
    require(set(expected.columns) <= set(actual.columns), f"{label}: missing columns")
    for col in expected:
        if col == "date":
            ok = pd.to_datetime(actual[col]).reset_index(drop=True).equals(
                pd.to_datetime(expected[col]).reset_index(drop=True))
        else:
            ok = np.allclose(actual[col], expected[col], equal_nan=True, rtol=1e-8, atol=1e-12)
        require(ok, f"{label}: mismatch in {col}")


def verify(output, root=ROOT, require_inputs=False):
    output, root = Path(output), Path(root)
    meta = json.loads((output / "run_metadata.json").read_text())
    require(meta["months"] == TEST_MONTHS, "Unexpected test-month protocol")
    require(meta["cost_bps_per_unit_turnover"] == 10, "Unexpected transaction cost")
    pred = pd.read_csv(output / "predictions.csv.gz", parse_dates=["date", "entry_date", "exit_date"])
    needed = {"model", "permno", "date", "entry_date", "exit_date", "score", "y", "fill_ok", "fold"}
    require(needed <= set(pred), "Missing prediction columns")
    require(not pred.duplicated(["model", "permno", "date"]).any(), "Duplicate prediction keys")
    require(set(pred.model) == {"zero", *STRATEGIES}, "Unexpected model set")
    require(not np.isinf(pred.score).any(), "Infinite scores")
    require(pred.loc[~pred.model.eq("reversal"), "score"].notna().all(), "Missing fitted-model scores")
    require((pred.date < pred.entry_date).all() and (pred.entry_date < pred.exit_date).all(), "Invalid signal/entry/exit chronology")
    require(pred.exit_date.lt("2024-12-01").all(), "December lockbox used")
    require(pred.date.dt.strftime("%Y-%m").eq(pred.fold).all(), "Fold/date mismatch")
    require(pred.fold.isin(TEST_MONTHS).all(), "Signal outside test months")
    require(set(pred.fold) == set(TEST_MONTHS), "Missing test folds")
    require(pred.fill_ok.isin([True, False]).all(), "Invalid fill flags")
    require(pred.loc[~pred.fill_ok, "y"].isna().all(), "Failed fills have labels")
    reference = pred.loc[pred.model.eq("zero"), ["permno", "date", "entry_date", "exit_date", "fill_ok", "y", "fold"]].sort_values(["date", "permno"]).reset_index(drop=True)
    for kind in STRATEGIES:
        keys = pred.loc[pred.model.eq(kind), reference.columns].sort_values(["date", "permno"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(keys, reference, check_dtype=False)
    require(pred.loc[pred.model.eq("zero"), "score"].eq(0).all(), "Zero baseline changed")
    all_port, checks = {}, []
    comparison = pd.read_csv(output / "model_comparison.csv").set_index("model")
    require(set(comparison.index) == {"zero", *STRATEGIES} and comparison.index.is_unique, "Invalid comparison model keys")
    for kind in STRATEGIES:
        z = pred.loc[pred.model.eq(kind)].sort_values(["date", "permno"])
        daily, port = ic_metrics(z), portfolio(z)
        compare_numeric(pd.read_csv(output / f"ic_{kind}.csv"), daily, f"IC {kind}")
        compare_numeric(pd.read_csv(output / f"portfolio_{kind}.csv"), port, f"portfolio {kind}")
        all_port[kind] = (z, daily, port)
    common = set.intersection(*(set(p.loc[p.net.notna(), "date"]) for _, _, p in all_port.values()))
    saved = pd.read_csv(output / "common_portfolio_dates.csv", parse_dates=["date"])
    require(not saved.date.duplicated().any() and set(saved.date) == common, "Common complete-date mismatch")
    require(len(common) == meta["common_portfolio_dates"], "Metadata common-date count mismatch")
    require(len(common) >= 2, "Insufficient complete dates for performance statistics")
    for kind, (z, daily, port) in all_port.items():
        stats = summary(z, daily, port.loc[port.date.isin(common)])
        for key, expected in stats.items():
            require(key in comparison, f"Missing comparison metric {key}")
            require(np.isclose(comparison.loc[kind, key], expected, equal_nan=True, rtol=1e-8, atol=1e-12), f"{kind}: comparison mismatch in {key}")
        checks.append({"model": kind, "predictions": len(z), "valid_labels": int(z.y.notna().sum()), "ic_days": len(daily), "missing_portfolio_days": int(port.net.isna().sum())})
    input_checks = {}
    for name, expected_hash in meta["source_sha256"].items():
        path = (root / name).resolve()
        require(path.is_relative_to(root.resolve()), "Input path outside project")
        if path.exists():
            require(digest(path) == expected_hash, f"Input hash mismatch: {name}")
            input_checks[name] = "matched"
        else:
            require(not require_inputs, f"Missing licensed input: {name}")
            input_checks[name] = "unavailable; artifact consistency only"
    return {"status": "passed", "scope": "saved regression, IC and portfolio artifact consistency; model fits and classification not independently checked", "test_days": pred.date.nunique(), "common_complete_days": len(common), "models": checks, "inputs": input_checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/crsp_monthly_walkforward_2024")
    parser.add_argument("--require-inputs", action="store_true", help="Also require all licensed inputs and matching hashes")
    args = parser.parse_args()
    try:
        result = verify(args.output_dir, require_inputs=args.require_inputs)
    except (ValueError, AssertionError, KeyError, FileNotFoundError) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(json.dumps(result, indent=2, default=int))


if __name__ == "__main__":
    main()
