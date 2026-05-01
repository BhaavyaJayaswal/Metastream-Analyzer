from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class BaseModel:
    def __init__(self, model, name):
        self.model = model
        self.name = name

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        max_err = np.max(np.abs(y_test - y_pred))
        med_err = np.median(np.abs(y_test - y_pred))
        return {
            "model_name": self.name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Max Error": max_err,
            "Median Error": med_err,
        }


class RandomForestModel(BaseModel):
    def __init__(self, **params):
        model = RandomForestRegressor(**params)
        super().__init__(model, "Random Forest")


class GradientBoostingModel(BaseModel):
    def __init__(self, **params):
        model = GradientBoostingRegressor(**params)
        super().__init__(model, "Gradient Boosting")


class XGBoostModel(BaseModel):
    def __init__(self, **params):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed.")
        model = xgb.XGBRegressor(**params)
        super().__init__(model, "XGBoost")