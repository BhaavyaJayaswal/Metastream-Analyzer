import unittest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.analysis import (
    compute_error_distribution,
    build_analysis_df,
    compute_region_stats,
)


class TestComputeErrorDistribution(unittest.TestCase):

    def test_returns_expected_keys(self):
        result = compute_error_distribution([1, 2, 3], [0.1, 0.2, 0.3])
        self.assertEqual(set(result.keys()), {"abs_mean", "rel_mean", "rel_max"})

    def test_abs_mean_correct(self):
        result = compute_error_distribution([0, 2, 4], [0, 0, 0])
        self.assertAlmostEqual(result["abs_mean"], 2.0)

    def test_rel_max_correct(self):
        result = compute_error_distribution([1, 1, 1], [0.1, 0.2, 0.5])
        self.assertAlmostEqual(result["rel_max"], 0.5)


class TestBuildAnalysisDf(unittest.TestCase):

    def test_column_names(self):
        n = 10
        df = build_analysis_df(
            list(range(n)), np.ones(n), np.ones(n) * 1.1,
            np.ones(n) * 0.1, np.ones(n) * 0.01,
        )
        self.assertEqual(
            set(df.columns),
            {"timestamp", "actual_tp", "predicted_tp", "abs_error", "rel_error"},
        )


class TestComputeRegionStats(unittest.TestCase):

    def test_duration_correctness(self):
        n = 20
        df = pd.DataFrame({
            "timestamp":    np.arange(n, dtype=float),
            "actual_tp":    np.ones(n) * 1e6,
            "predicted_tp": np.ones(n) * 1.1e6,
            "abs_error":    np.ones(n) * 1e5,
            "rel_error":    np.ones(n) * 0.1,
        })
        stats = compute_region_stats(df, [(0, 10)])
        self.assertAlmostEqual(stats[0]["duration"], 10.0)


if __name__ == "__main__":
    unittest.main()
