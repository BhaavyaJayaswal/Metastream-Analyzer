import numpy as np
import pandas as pd


def compute_error_distribution(abs_error, rel_error):
    abs_error = np.array(abs_error)
    rel_error = np.array(rel_error)
    return {
        "abs_mean": np.mean(abs_error),
        "rel_mean": np.mean(rel_error),
        "rel_max": np.max(rel_error),
    }


def compute_region_stats(df, regions):
    """
    Takes list of (start_idx, end_idx) regions and a df with:
    - timestamp
    - actual_tp
    - predicted_tp
    - abs_error
    - rel_error

    returns a list of dictionaries with stats/region.
    """
    stats = []

    for (start, end) in regions:
        segment = df.iloc[start:end+1]

        stats.append({
            "start_time": segment["timestamp"].iloc[0],
            "end_time": segment["timestamp"].iloc[-1],
            "duration": segment["timestamp"].iloc[-1] - segment["timestamp"].iloc[0],

            "mean_abs_error": segment["abs_error"].mean(),
            "max_abs_error": segment["abs_error"].max(),

            "mean_rel_error": segment["rel_error"].mean(),
            "max_rel_error": segment["rel_error"].max(),

            "actual_mean": segment["actual_tp"].mean(),
            "predicted_mean": segment["predicted_tp"].mean(),
        })

    return stats


def build_analysis_df(timestamps, actual, predicted, abs_error, rel_error):
    """
    combine all arrays into a single Data Frame for analysis.
    """
    return pd.DataFrame({
        "timestamp": timestamps,
        "actual_tp": actual,
        "predicted_tp": predicted,
        "abs_error": abs_error,
        "rel_error": rel_error
    })
