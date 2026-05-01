import unittest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.dataset import split_train_test, balance_training_dataset, prepare_features


def _make_df(n_rows, game_name="GameA"):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "game":               game_name,
        "current_vmaf":       np.random.choice([60, 70, 80], n_rows),
        "current_throughput": rng.uniform(1e5, 5e6, n_rows),
        "needed_vmaf":        np.random.choice([70, 80, 90], n_rows),
        "needed_throughput":  rng.uniform(1e5, 5e6, n_rows),
    })


class TestSplitTrainTest(unittest.TestCase):

    def test_approximate_split_ratio(self):
        """test set has ~50% of rows when test_size=0.5"""
        _, df_test = split_train_test({"GameA": _make_df(100)}, test_size=0.5)
        self.assertGreaterEqual(len(df_test), 40)
        self.assertLessEqual(len(df_test), 60)

    def test_raises_on_empty_datasets(self):
        """ValueError raised when datasets dict is empty"""
        with self.assertRaises(ValueError):
            split_train_test({}, test_size=0.5)

    def test_combined_size_correct(self):
        """Train + test row counts sum to total dataset size"""
        df = _make_df(100)
        df_train, df_test = split_train_test({"GameA": df}, test_size=0.5)
        self.assertAlmostEqual(len(df_train) + len(df_test), 100, delta=2)


class TestBalanceTrainingDataset(unittest.TestCase):

    def test_equalises_game_counts(self):
        """smaller game is oversampled to match the larger"""
        df = pd.concat([
            _make_df(10, "SmallGame"),
            _make_df(30, "BigGame"),
        ]).reset_index(drop=True)
        balanced = balance_training_dataset(df)
        counts = balanced.groupby("game").size()
        self.assertEqual(counts["SmallGame"], counts["BigGame"])
        self.assertEqual(counts["BigGame"], 30)


class TestPrepareFeatures(unittest.TestCase):

    def test_feature_columns(self):
        """X_train has exactly the 3 expected feature columns"""
        df = _make_df(50)
        X_train, _, _, _ = prepare_features(df, df)
        self.assertEqual(
            set(X_train.columns),
            {"current_vmaf", "current_throughput", "needed_vmaf"},
        )

    def test_label_length_matches_features(self):
        df = _make_df(50)
        X_train, y_train, _, _ = prepare_features(df, df)
        self.assertEqual(len(y_train), len(X_train))


if __name__ == "__main__":
    unittest.main()