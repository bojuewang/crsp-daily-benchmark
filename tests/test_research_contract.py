"""Small synthetic fixtures check mechanics, never investment performance."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_crsp_benchmark_2024 as b
from run_crsp_monthly_walkforward_2024 import fold_masks
from verify_walkforward_outputs import verify


class PanelContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dates = pd.bdate_range("2024-01-02", periods=45)
        self.raw = pd.DataFrame({"permno": 10001, "date": self.dates, "prc": np.linspace(10, 12, 45), "vol": 1000., "shrout": 100., "ret": np.linspace(-.01, .01, 45)})
        # Second security supplies the market calendar if first-security observations disappear.
        self.raw = pd.concat([self.raw, self.raw.assign(permno=10002)], ignore_index=True)
        self.names = pd.DataFrame({"permno": [10001, 10002], "namedt": ["2024-01-01"]*2, "nameendt": ["2024-12-31"]*2, "shrcd": [10, 10], "exchcd": [1, 1]})
        self.delist = pd.DataFrame(columns=["permno", "dlstdt", "dlret"])

    def panel(self):
        self.raw.to_csv(self.root / "dsf.gz", index=False)
        self.names.to_csv(self.root / "names.csv", index=False)
        self.delist.to_csv(self.root / "dl.gz", index=False)
        with patch.multiple(b, DSF=self.root / "dsf.gz", NAMES=self.root / "names.csv", DELIST=self.root / "dl.gz"):
            return b.build_panel()[0]

    def test_future_mutation_preserves_earlier_features(self):
        before = self.panel()
        cutoff = self.dates[25]
        self.raw.loc[self.raw.date.gt(cutoff), ["prc", "vol", "ret"]] = [1000., 99999., .7]
        after = self.panel()
        cols = ["permno", "date", *b.FEATURES, "reversal"]
        pd.testing.assert_frame_equal(before.loc[before.date.le(cutoff), cols], after.loc[after.date.le(cutoff), cols])

    def test_label_uses_second_future_market_day(self):
        row = self.panel().query("permno == 10001").iloc[0]
        self.assertEqual(row.date, self.dates[19])
        self.assertEqual(row.entry_date, self.dates[20])
        self.assertEqual(row.exit_date, self.dates[21])
        self.assertAlmostEqual(row.y, self.raw.loc[self.raw.permno.eq(10001) & self.raw.date.eq(self.dates[21]), "ret"].iloc[0])

    def test_missing_security_date_is_not_compressed(self):
        self.raw = self.raw.loc[~(self.raw.permno.eq(10001) & self.raw.date.eq(self.dates[10]))]
        row = self.panel().query("permno == 10001").iloc[0]
        self.assertEqual(row.date, self.dates[19])
        self.assertEqual(row.trading_days, 19)

    def test_effective_name_interval_is_respected(self):
        self.names.loc[self.names.permno.eq(10001), "namedt"] = str(self.dates[25].date())
        self.assertEqual(self.panel().query("permno == 10001").date.min(), self.dates[25])

    def test_delisting_return_compounds(self):
        self.delist = pd.DataFrame({"permno": [10001], "dlstdt": [self.dates[21]], "dlret": [-.5]})
        row = self.panel().query("permno == 10001").iloc[0]
        ret = self.raw.loc[self.raw.permno.eq(10001) & self.raw.date.eq(self.dates[21]), "ret"].iloc[0]
        self.assertAlmostEqual(row.y, (1+ret)*.5-1)

    def test_failed_fill_keeps_row_with_missing_label(self):
        self.raw.loc[self.raw.permno.eq(10001) & self.raw.date.eq(self.dates[20]), "prc"] = np.nan
        row = self.panel().query("permno == 10001").iloc[0]
        self.assertFalse(row.fill_ok)
        self.assertTrue(pd.isna(row.y))


class FoldContract(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2024-01-02", "2024-12-31")
        self.x = pd.DataFrame({"date": dates[:-2], "exit_date": dates[2:], "y": .01})

    def test_all_folds_purge_unobserved_labels(self):
        for month in ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11"]:
            train, val, test, *_ = fold_masks(self.x, month)
            self.assertLess(self.x.loc[train, "exit_date"].max(), self.x.loc[val, "date"].min())
            self.assertLess(self.x.loc[val, "exit_date"].max(), self.x.loc[test, "date"].min())
            self.assertLess(self.x.loc[test, "exit_date"].max(), pd.Timestamp("2024-12-01"))
            self.assertFalse((train & val | train & test | val & test).any())

    def test_delayed_label_is_rejected(self):
        self.x.loc[0, "exit_date"] = pd.Timestamp("2024-08-01")
        with self.assertRaisesRegex(ValueError, "Training labels overlap"):
            fold_masks(self.x, "2024-07")


class PortfolioContract(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"date": pd.Timestamp("2024-07-01"), "permno": range(30), "score": range(30), "fill_ok": True, "y": np.arange(30)/1000})

    def test_decile_weights_and_initial_cost(self):
        result = b.portfolio(self.frame()).iloc[0]
        self.assertEqual(result.n_per_side, 3)
        self.assertAlmostEqual(result.turnover, 2.)
        self.assertAlmostEqual(result.gross, .028-.001)
        self.assertAlmostEqual(result.net, result.gross-.002)

    def test_missing_selected_return_invalidates_day(self):
        frame = self.frame()
        frame.loc[29, "y"] = np.nan
        result = b.portfolio(frame).iloc[0]
        self.assertEqual(result.missing_selected_returns, 1)
        self.assertTrue(pd.isna(result.net))

    def test_missing_unselected_return_keeps_day(self):
        frame = self.frame()
        frame.loc[15, "y"] = np.nan
        self.assertFalse(pd.isna(b.portfolio(frame).iloc[0].net))


class ArtifactContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        frames = []
        for date in pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-03", "2024-08-01", "2024-09-03", "2024-10-01", "2024-11-01"]):
            for model in ["zero", "reversal", "ridge", "tree"]:
                frames.append(pd.DataFrame({"date": date, "entry_date": date+pd.Timedelta(days=1), "exit_date": date+pd.Timedelta(days=2), "permno": range(30), "model": model, "fold": date.strftime("%Y-%m"), "fill_ok": True, "score": 0 if model == "zero" else np.arange(30)/1000, "y": np.arange(30)/1000 + np.sin(np.arange(30)*(date.month+date.day))*.01}))
        self.pred = pd.concat(frames, ignore_index=True)
        self.pred.to_csv(self.out / "predictions.csv.gz", index=False)
        rows = []
        for model in ["reversal", "ridge", "tree"]:
            z = self.pred.loc[self.pred.model.eq(model)]
            daily, port = b.ic_metrics(z), b.portfolio(z)
            daily.to_csv(self.out / f"ic_{model}.csv", index=False)
            port.to_csv(self.out / f"portfolio_{model}.csv", index=False)
            rows.append({"model": model, **b.summary(z, daily, port)})
        rows.append({"model": "zero"})
        pd.DataFrame(rows).to_csv(self.out / "model_comparison.csv", index=False)
        pd.DataFrame({"date": self.pred.date.unique()}).to_csv(self.out / "common_portfolio_dates.csv", index=False)
        (self.out / "run_metadata.json").write_text(json.dumps({"months": ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11"], "cost_bps_per_unit_turnover": 10, "common_portfolio_dates": 7, "source_sha256": {"data/absent.csv": "not-a-real-hash"}}))

    def test_consistent_artifacts_pass(self):
        self.assertEqual(verify(self.out, root=self.out)["common_complete_days"], 7)

    def test_duplicate_predictions_are_rejected(self):
        pd.concat([self.pred, self.pred.iloc[[0]]]).to_csv(self.out / "predictions.csv.gz", index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            verify(self.out, root=self.out)

    def test_changed_report_metric_is_rejected(self):
        table = pd.read_csv(self.out / "model_comparison.csv")
        table.loc[table.model.eq("tree"), "rank_ic_mean"] = -.75
        table.to_csv(self.out / "model_comparison.csv", index=False)
        with self.assertRaisesRegex(ValueError, "rank_ic_mean"):
            verify(self.out, root=self.out)

    def test_strict_input_mode_rejects_missing_inputs(self):
        with self.assertRaisesRegex(ValueError, "Missing licensed input"):
            verify(self.out, root=self.out, require_inputs=True)


if __name__ == "__main__":
    unittest.main()
