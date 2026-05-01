import numpy as np
import pandas as pd


def calc_throughput(df, delta_t=1.0):
    """
    Compute throughput per second from raw metastream DataFrame.
    Expects columns: 'time', 'pkt_size'.
    Returns DataFrame with columns: 'timestamp', 'throughput'.
    """
    df = df.sort_values('time')
    max_time = df['time'].max()

    bins = np.arange(0, max_time + delta_t + 1e-8, delta_t)
    df['interval_end'] = pd.cut(df['time'], bins=bins, right=True, labels=bins[1:])

    bytes_per_interval = (
        df.groupby('interval_end')['pkt_size']
        .sum()
        .reindex(bins[1:], fill_value=0)
    )

    bitrates = bytes_per_interval * 8.0 / delta_t

    return pd.DataFrame({
        'timestamp': np.concatenate(([0.0], bins[1:])),
        'throughput': np.concatenate(([0.0], bitrates.values))
    })


def compute_abs_error(actual, predicted):
    """
    Compute absolute error: |actual - predicted|.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.abs(actual - predicted)


def compute_rel_error(actual, predicted, eps=1e-9):
    """
    Compute relative error: |actual - predicted| / (actual + eps).
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.abs(actual - predicted) / (actual + eps)


def compute_rolling_error(rel_error, window=20):
    """
    Compute rolling mean of relative error.
    rel_error: 1D array-like
    window: number of samples
    """
    rel_error = pd.Series(rel_error)
    return rel_error.rolling(window=window, min_periods=1).mean().values


def detect_error_regions(rel_error, threshold):
    """
    Identify continuous regions where relative error > threshold.
    Returns a list of (start_index, end_index) tuples.
    """
    rel_error = np.array(rel_error)
    above = rel_error > threshold

    regions = []
    start = None

    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            regions.append((start, i - 1))
            start = None

    if start is not None:
        regions.append((start, len(rel_error) - 1))

    return regions
