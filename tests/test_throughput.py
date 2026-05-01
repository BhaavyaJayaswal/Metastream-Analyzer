import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.throughput import (
    compute_abs_error,
    compute_rel_error,
    compute_rolling_error,
    detect_error_regions,
)


class TestComputeAbsError(unittest.TestCase):

    def test_known_value(self):
        """|100 - 130| == 30"""
        self.assertAlmostEqual(compute_abs_error([100], [130])[0], 30.0)

    def test_zero_when_exact(self):
        """error is zero when prediction equals actual"""
        self.assertAlmostEqual(compute_abs_error([500], [500])[0], 0.0)

    def test_always_non_negative(self):
        """output is always >= 0"""
        rng = np.random.default_rng(0)
        actual    = rng.uniform(0, 1e6, 100)
        predicted = rng.uniform(0, 1e6, 100)
        self.assertTrue(np.all(compute_abs_error(actual, predicted) >= 0))


class TestComputeRelError(unittest.TestCase):

    def test_known_value(self):
        """|200 - 250| / 200 == 0.25"""
        self.assertAlmostEqual(compute_rel_error([200], [250])[0], 0.25, places=6)

    def test_zero_actual_no_division_by_zero(self):
        """actual==0 uses eps guard; result must be finite"""
        self.assertTrue(np.isfinite(compute_rel_error([0], [100])[0]))

    def test_zero_when_exact(self):
        self.assertAlmostEqual(compute_rel_error([300], [300])[0], 0.0, places=9)


class TestComputeRollingError(unittest.TestCase):

    def test_output_length_matches_input(self):
        """output length == input length"""
        self.assertEqual(len(compute_rolling_error([0.1] * 50, window=10)), 50)

    def test_first_value_with_min_periods(self):
        """first output equals first input (min_periods=1)"""
        self.assertAlmostEqual(compute_rolling_error([0.4, 0.2, 0.6], window=5)[0], 0.4)

    def test_constant_series_unchanged(self):
        self.assertTrue(np.allclose(compute_rolling_error([0.3] * 30, window=10), 0.3))


class TestDetectErrorRegions(unittest.TestCase):

    def test_single_contiguous_region(self):
        """one region for one contiguous spike at indices 3-7"""
        rel_error = [0.1] * 3 + [0.5] * 5 + [0.1] * 4
        self.assertEqual(detect_error_regions(rel_error, threshold=0.3), [(3, 7)])

    def test_no_regions_when_all_below_threshold(self):
        """empty list when error never exceeds threshold"""
        self.assertEqual(detect_error_regions([0.1] * 20, threshold=0.5), [])

    def test_region_extends_to_last_index(self):
        """region running to end of array is closed at last index"""
        regions = detect_error_regions([0.1] * 5 + [0.9] * 5, threshold=0.5)
        self.assertEqual(regions[0][1], 9)

    def test_multiple_disjoint_regions(self):
        regions = detect_error_regions([0.9, 0.1, 0.9, 0.1, 0.9], threshold=0.5)
        self.assertEqual(len(regions), 3)


if __name__ == "__main__":
    unittest.main()