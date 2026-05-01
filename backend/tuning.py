import itertools
import numpy as np
from backend.models import RandomForestModel, GradientBoostingModel, XGBoostModel


def _build_model(model_type, params):
    if model_type == "Random Forest":
        return RandomForestModel(**params)
    elif model_type == "Gradient Boosting":
        return GradientBoostingModel(**params)
    else:
        return XGBoostModel(**params)


def generate_param_grid(model_type, ranges):
    keys = list(ranges.keys())
    values = [ranges[k] for k in keys]
    grid = []
    for combo in itertools.product(*values):
        grid.append(dict(zip(keys, combo)))
    return grid


def run_grid_search(model_type, param_ranges, X_train, y_train, X_test, y_test, progress_callback=None):
    grid = generate_param_grid(model_type, param_ranges)
    total = len(grid)

    best_model = None
    best_params = None
    best_score = np.inf
    best_results = None

    for i, params in enumerate(grid, start=1):
        model = _build_model(model_type, params)
        model.train(X_train, y_train)
        results = model.evaluate(X_test, y_test)
        mae = results["MAE"]

        if mae < best_score:
            best_score = mae
            best_model = model
            best_params = params
            best_results = results

        if progress_callback:
            progress_callback(i, total, best_score, params)

    return best_model, best_params, best_score, best_results, total