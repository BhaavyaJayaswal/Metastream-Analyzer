import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.models import RandomForestModel, GradientBoostingModel

EXPECTED_EVAL_KEYS = {"model_name", "MAE", "RMSE", "R2", "Max Error", "Median Error"}
RNG = np.random.default_rng(42)


def _synthetic_data(n_train=80, n_test=20):
    X_train = RNG.uniform(0, 1, (n_train, 3))
    y_train = X_train[:, 0] * 1e6 + RNG.normal(0, 1e4, n_train)
    X_test  = RNG.uniform(0, 1, (n_test,  3))
    y_test  = X_test[:, 0]  * 1e6 + RNG.normal(0, 1e4, n_test)
    return X_train, y_train, X_test, y_test


class TestRandomForestModel(unittest.TestCase):

    def setUp(self):
        X_train, y_train, self.X_test, self.y_test = _synthetic_data()
        self.model = RandomForestModel(n_estimators=10, random_state=42)
        self.model.train(X_train, y_train)

    def test_predict_output_length(self):
        """predict() returns array with length == number of test samples"""
        self.assertEqual(len(self.model.predict(self.X_test)), len(self.X_test))

    def test_evaluate_keys(self):
        """evaluate() returns all expected metric keys"""
        self.assertEqual(
            set(self.model.evaluate(self.X_test, self.y_test).keys()),
            EXPECTED_EVAL_KEYS,
        )

    def test_mae_non_negative(self):
        """MAE is always >= 0"""
        self.assertGreaterEqual(self.model.evaluate(self.X_test, self.y_test)["MAE"], 0)

    def test_r2_at_most_one(self):
        """R2 <= 1.0"""
        self.assertLessEqual(self.model.evaluate(self.X_test, self.y_test)["R2"], 1.0)

    def test_model_name_correct(self):
        result = self.model.evaluate(self.X_test, self.y_test)
        self.assertEqual(result["model_name"], "Random Forest")


class TestGradientBoostingModel(unittest.TestCase):

    def setUp(self):
        X_train, y_train, self.X_test, self.y_test = _synthetic_data()
        self.model = GradientBoostingModel(n_estimators=20, random_state=42)
        self.model.train(X_train, y_train)

    def test_predict_output_length(self):
        """prediction output shape matches test set size"""
        self.assertEqual(len(self.model.predict(self.X_test)), len(self.X_test))

    def test_mae_non_negative(self):
        """MAE >= 0"""
        self.assertGreaterEqual(self.model.evaluate(self.X_test, self.y_test)["MAE"], 0)

    def test_r2_at_most_one(self):
        """R2 <= 1.0"""
        self.assertLessEqual(self.model.evaluate(self.X_test, self.y_test)["R2"], 1.0)


if __name__ == "__main__":
    unittest.main()