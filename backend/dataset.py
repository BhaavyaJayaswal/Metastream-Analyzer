import os
import zipfile
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from backend.loader import load_compressed_pickle
from backend.throughput import calc_throughput

valid_current = [50, 60, 70, 80, 90]
valid_pairs = []

for v in valid_current:
    if v + 10 <= 90:
        valid_pairs.append((v, v + 10))
    if v - 10 >= 50:
        valid_pairs.append((v, v - 10))


def extract_zip_to_temp(uploaded_zip):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        z.extractall(temp_dir)
    return temp_dir


def build_dataset_for_game(game_folder, game_name, delta_t=1.0):
    rows = []

    for (current_vmaf, needed_vmaf) in valid_pairs:

        path_current = os.path.join(
            game_folder,
            f"tvmaf-{current_vmaf}_tinterval-1",
            "analytix",
            "video_metrics.pkl.gz"
        )

        path_needed = os.path.join(
            game_folder,
            f"tvmaf-{needed_vmaf}_tinterval-1",
            "analytix",
            "video_metrics.pkl.gz"
        )

        if not os.path.exists(path_current) or not os.path.exists(path_needed):
            continue

        df_current = load_compressed_pickle(path_current)
        df_needed = load_compressed_pickle(path_needed)

        tp_current = calc_throughput(df_current, delta_t)
        tp_needed  = calc_throughput(df_needed, delta_t)

        merged = pd.merge(
            tp_current,
            tp_needed,
            on="timestamp",
            suffixes=("_current", "_needed")
        )

        for _, row in merged.iterrows():
            rows.append({
                "game": game_name,
                "current_vmaf": current_vmaf,
                "current_throughput": row["throughput_current"],
                "needed_vmaf": needed_vmaf,
                "needed_throughput": row["throughput_needed"],
            })

    return pd.DataFrame(rows)


def find_vmaf_root(folder):
    """
    Recursively search for the folder that contains tvmaf-* subfolders.
    """
    for root, dirs, files in os.walk(folder):
        for d in dirs:
            if d.startswith("tvmaf-") and "_tinterval-1" in d:
                return root  # this folder contains the VMAF subfolders
    return None


def build_dataset_from_uploaded_zips(uploaded_zips):
    datasets = {}

    for uploaded_zip in uploaded_zips:
        game_name = os.path.splitext(uploaded_zip.name)[0]

        temp_folder = extract_zip_to_temp(uploaded_zip)

        vmaf_root = find_vmaf_root(temp_folder)

        if vmaf_root is None:
            print(f"[WARN] No VMAF folders found in {uploaded_zip.name}")
            datasets[game_name] = pd.DataFrame()
            continue

        datasets[game_name] = build_dataset_for_game(vmaf_root, game_name)

    return datasets


def split_train_test(datasets, test_size=0.5):

    train_dfs = []
    test_dfs = []

    for game_name, df in datasets.items():
        if len(df) < 2:
            continue

        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        train_dfs.append(train_df)
        test_dfs.append(test_df)

    if not train_dfs or not test_dfs:
        raise ValueError("No data available for train/test split.")

    df_train_raw = pd.concat(train_dfs).reset_index(drop=True)
    df_test = pd.concat(test_dfs).reset_index(drop=True)

    return df_train_raw, df_test


def balance_training_dataset(df_train_raw):
    grouped = df_train_raw.groupby("game")
    max_size = grouped.size().max()

    balanced = []

    for game, df_g in grouped:
        df_balanced = df_g.sample(n=max_size, replace=True, random_state=42)
        balanced.append(df_balanced)

    return pd.concat(balanced).reset_index(drop=True)


def prepare_features(df_train, df_test):
    X_train = df_train[["current_vmaf", "current_throughput", "needed_vmaf"]]
    y_train = df_train["needed_throughput"]

    X_test = df_test[["current_vmaf", "current_throughput", "needed_vmaf"]]
    y_test = df_test["needed_throughput"]

    return X_train, y_train, X_test, y_test
