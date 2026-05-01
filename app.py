import os
import time
import threading
import numpy as np
import pandas as pd
import streamlit as st

from backend.loader import load_compressed_pickle
from backend.throughput import (
    calc_throughput,
    compute_abs_error,
    compute_rel_error,
    compute_rolling_error,
    detect_error_regions,
)
from backend.dataset import (
    extract_zip_to_temp,
    find_vmaf_root,
    build_dataset_from_uploaded_zips,
    split_train_test,
    balance_training_dataset,
    prepare_features,
)
from backend.models import RandomForestModel, GradientBoostingModel, XGBoostModel
from backend.tuning import run_grid_search
from backend.save_load import save_model, list_saved_models, load_saved_model
from backend.analysis import (
    build_analysis_df,
    compute_error_distribution,
    compute_region_stats,
)
from backend.visualization import (
    make_throughput_figure,
    make_rolling_error_figure,
    make_region_zoom_figure,
    make_cdf_figure,
)

st.set_page_config(page_title="Metastream Analyzer", layout="wide")
st.title("Metastream Analyzer")

VMAF_LEVELS = [50, 60, 70, 80, 90]

_CHART_CONFIG = {"displayModeBar": False}


@st.cache_resource
def _tuning_state():
    """dict shared between the bkg training thread and Streamlit reruns."""
    return {
        "done": True,
        "cancel": False,
        "cancelled": False,
        "progress": 0.0,
        "status": "",
        "result": (),
    }


def _stat(col, label, value):
    """metric card that replaces st.metric's huge numbers."""
    col.markdown(
        f"<div style='padding:4px 0 8px'>"
        f"<div style='font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.05em;font-weight:500'>{label}</div>"
        f"<div style='font-size:14px;font-weight:600;color:#ffffff;word-break:break-word'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


stream_tab, model_tab = st.tabs(["Stream Analyzer", "Model Analyzer"])

with stream_tab:
    st.header("Stream Analyzer")

    ctrl_col, plot_col = st.columns([1, 3])

    with ctrl_col:
        st.subheader("Controls")

        uploaded_zip_stream = st.file_uploader(
            "Upload game ZIP",
            type=["zip"],
            key="stream_zip_uploader",
            help="ZIP containing tvmaf-*/analytix/video_metrics.pkl.gz folders",
        )

        selected_vmaf = st.selectbox("VMAF Level", VMAF_LEVELS, key="stream_vmaf_level")

        st.markdown("---")
        st.markdown("**Load a trained model**")
        saved_models = list_saved_models()
        if saved_models:
            selected_model_file = st.selectbox(
                "Saved model", saved_models, key="stream_model_selector"
            )
            if st.button("Load Model", key="stream_load_model_btn"):
                st.session_state["stream_model"] = load_saved_model(selected_model_file)
                st.session_state["stream_model_name"] = selected_model_file
                st.success(f"Loaded: {selected_model_file}")
        else:
            st.info("No saved models yet. Train and save one in the Model Analyzer tab.")

        if "stream_model_name" in st.session_state:
            st.caption(f"Active model: **{st.session_state['stream_model_name']}**")

        st.markdown("---")
        threshold_pct = st.slider("Error Threshold (%)", 1, 100, 20, key="stream_threshold")
        threshold = threshold_pct / 100.0
        rolling_window = st.slider("Rolling window (s)", 5, 60, 20, key="stream_window")

        analyze_btn = st.button("Analyze Stream", key="stream_analyze_btn", type="primary")

    if analyze_btn:
        if uploaded_zip_stream is None:
            st.error("Please upload a game ZIP file.")
        elif "stream_model" not in st.session_state:
            st.error("Please load a trained model first.")
        else:
            with st.spinner("Extracting ZIP..."):
                temp_folder = extract_zip_to_temp(uploaded_zip_stream)
                vmaf_root = find_vmaf_root(temp_folder)

            if vmaf_root is None:
                st.error("No VMAF folders found in the uploaded ZIP.")
            else:
                directions = [
                    (selected_vmaf, selected_vmaf + d)
                    for d in (+10, -10)
                    if 50 <= selected_vmaf + d <= 90
                ]

                model = st.session_state["stream_model"]
                results = []

                with st.spinner("Running model predictions..."):
                    for current_vmaf, needed_vmaf in directions:
                        path_current = os.path.join(
                            vmaf_root, f"tvmaf-{current_vmaf}_tinterval-1", "analytix", "video_metrics.pkl.gz",
                        )
                        path_needed = os.path.join(
                            vmaf_root, f"tvmaf-{needed_vmaf}_tinterval-1", "analytix", "video_metrics.pkl.gz",
                        )
                        if not os.path.exists(path_current) or not os.path.exists(path_needed):
                            continue

                        df_current = load_compressed_pickle(path_current)
                        df_needed = load_compressed_pickle(path_needed)
                        tp_current = calc_throughput(df_current)
                        tp_needed = calc_throughput(df_needed)
                        merged = pd.merge(
                            tp_current, tp_needed, on="timestamp",
                            suffixes=("_current", "_needed"),
                        )
                        X_pred = pd.DataFrame({
                            "current_vmaf": current_vmaf,
                            "current_throughput": merged["throughput_current"],
                            "needed_vmaf": needed_vmaf,
                        })
                        merged["predicted_needed"] = model.predict(X_pred)

                        direction = "Upgrade" if needed_vmaf > current_vmaf else "Downgrade"
                        results.append({
                            "label": f"{current_vmaf} → {needed_vmaf}",
                            "direction": direction,
                            "current_vmaf": current_vmaf,
                            "needed_vmaf": needed_vmaf,
                            "timestamps": merged["timestamp"].values,
                            "actual_current": merged["throughput_current"].values,
                            "actual_needed": merged["throughput_needed"].values,
                            "predicted_needed": merged["predicted_needed"].values,
                        })

                if results:
                    st.session_state["stream_data"] = results
                    st.success(f"Analysis complete — {len(results)} direction(s) processed.")
                else:
                    st.error("Could not find the required VMAF stream files in the ZIP.")

    with plot_col:
        st.subheader("Visualizations")

        if "stream_data" not in st.session_state:
            st.info("Upload a ZIP, load a model, and click **Analyze Stream** to begin.")
        else:
            all_results = st.session_state["stream_data"]
            cdf_series = []
            for data in all_results:
                timestamps = data["timestamps"]
                actual_current = data["actual_current"]
                actual_needed = data["actual_needed"]
                predicted_needed = data["predicted_needed"]

                abs_error = compute_abs_error(actual_needed, predicted_needed)
                rel_error = compute_rel_error(actual_needed, predicted_needed)
                rolling_error = compute_rolling_error(rel_error, window=rolling_window)
                error_regions = detect_error_regions(rolling_error, threshold)
                analysis_df = build_analysis_df(
                    timestamps, actual_needed, predicted_needed, abs_error, rel_error
                )
                cdf_series.append((data["label"], rel_error))

                st.markdown(f"### {data['direction']}: VMAF {data['label']}")

                err_dist = compute_error_distribution(abs_error, rel_error)
                m1, m2, m3, m4 = st.columns(4)
                _stat(m1, "Mean Abs Error", f"{err_dist['abs_mean']:,.0f} bps")
                _stat(m2, "Mean Rel Error", f"{err_dist['rel_mean'] * 100:.1f}%")
                _stat(m3, "Max Rel Error", f"{err_dist['rel_max'] * 100:.1f}%")
                _stat(m4, "Error Regions", str(len(error_regions)))

                fig1 = make_throughput_figure(
                    timestamps, actual_current, actual_needed, predicted_needed,
                    error_regions=error_regions,
                )
                st.plotly_chart(fig1, use_container_width=True, config=_CHART_CONFIG)

                fig2 = make_rolling_error_figure(
                    timestamps, rolling_error, threshold,
                    error_regions=error_regions,
                )
                st.plotly_chart(fig2, use_container_width=True, config=_CHART_CONFIG)

                if error_regions:
                    st.subheader("Error Region Inspector")
                    st.caption(
                        f"{len(error_regions)} region(s) where rolling error exceeded "
                        f"{threshold_pct}%. Select one to inspect."
                    )
                    region_stats = compute_region_stats(analysis_df, error_regions)
                    region_labels = [
                        f"Region {i + 1}:  t = {s['start_time']:.1f}s – {s['end_time']:.1f}s  ({s['duration']:.1f}s)"
                        for i, s in enumerate(region_stats)
                    ]
                    safe_key = data["label"].replace(" ", "_").replace("→", "to")
                    selected_label = st.selectbox(
                        "Select a region:", region_labels, key=f"region_sel_{safe_key}"
                    )
                    region_idx = region_labels.index(selected_label)
                    s = region_stats[region_idx]

                    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
                    _stat(rc1, "Duration", f"{s['duration']:.1f}s")
                    _stat(rc2, "Mean Rel Error", f"{s['mean_rel_error'] * 100:.1f}%")
                    _stat(rc3, "Max Rel Error", f"{s['max_rel_error'] * 100:.1f}%")
                    _stat(rc4, "Actual TP (mean)", f"{s['actual_mean']:,.0f} bps")
                    _stat(rc5, "Predicted TP (mean)", f"{s['predicted_mean']:,.0f} bps")

                    start_idx, end_idx = error_regions[region_idx]
                    end_idx = min(end_idx, len(timestamps) - 1)
                    fig_zoom = make_region_zoom_figure(
                        timestamps[start_idx:end_idx + 1],
                        actual_needed[start_idx:end_idx + 1],
                        predicted_needed[start_idx:end_idx + 1],
                        selected_label,
                    )
                    st.plotly_chart(fig_zoom, use_container_width=True, config=_CHART_CONFIG)
                else:
                    st.success("No error regions detected above the threshold.")

                st.markdown("---")

            st.markdown("### CDF of Relative Error")
            cdf_percentile = st.slider(
                "Show up to percentile", 50, 100, 99,
                key="cdf_percentile",
                help="Clips the x-axis at the chosen percentile of error values, "
                     "letting you zoom into the low-error region and ignore outliers.",
            )
            all_errors = np.concatenate([err for _, err in cdf_series])
            x_max = float(np.percentile(all_errors, cdf_percentile) * 100)
            fig_cdf = make_cdf_figure(cdf_series, x_max=x_max)
            st.plotly_chart(fig_cdf, use_container_width=True, config=_CHART_CONFIG)

with model_tab:
    st.header("Model Analyzer")

    th = _tuning_state()

    _result = th["result"]
    if th["done"] and _result:
        _best_model, _best_params, _best_score, _best_results = _result
        th["result"] = ()
        st.session_state["trained_model"] = _best_model
        st.session_state["trained_model_type"] = st.session_state.get("_pending_model_type", "Unknown")
        st.session_state["trained_model_params"] = _best_params
        st.session_state["tuning_last_score"] = _best_score
        st.session_state["tuning_last_params"] = _best_params
        st.session_state["tuning_last_results"] = _best_results

    if th["cancelled"]:
        st.warning("Training was cancelled.")
        th["cancelled"] = False

    uploaded_zips = st.file_uploader(
        "Upload game ZIPs",
        type=["zip"],
        accept_multiple_files=True,
        key="model_zip_uploader",
    )

    model_type = st.selectbox(
        "Choose model",
        ["Random Forest", "Gradient Boosting", "XGBoost"],
        key="model_type_selector",
    )

    train_ratio = st.slider("Training set size (%)", 10, 90, 50, key="model_train_ratio")
    test_size = 1.0 - (train_ratio / 100.0)

    st.markdown("### Hyperparameters")
    tune = st.checkbox("Enable hyperparameter tuning", value=False, key="enable_tuning")

    base_params = {}
    param_ranges = {}

    if model_type == "Random Forest":
        if not tune:
            st.markdown("#### Random Forest Parameters")
            n_estimators = st.slider("n_estimators", 10, 300, 100, key="rf_n_est")
            max_depth = st.slider("max_depth", 2, 20, 10, key="rf_depth")
            min_samples_split = st.slider("min_samples_split", 2, 10, 2, key="rf_mss")
            base_params = {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_split": min_samples_split}
        else:
            st.markdown("#### Random Forest — Tuning Ranges")
            n_estimators_range = st.slider("n_estimators range", 10, 300, (50, 150), key="rf_n_est_r")
            max_depth_range = st.slider("max_depth range", 2, 20, (5, 15), key="rf_depth_r")
            min_samples_split_range = st.slider("min_samples_split range", 2, 10, (2, 5), key="rf_mss_r")
            param_ranges = {"n_estimators": n_estimators_range, "max_depth": max_depth_range, "min_samples_split": min_samples_split_range}

    elif model_type == "Gradient Boosting":
        if not tune:
            st.markdown("#### Gradient Boosting Parameters")
            n_estimators = st.slider("n_estimators", 10, 300, 100, key="gb_n_est")
            learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1, key="gb_lr")
            max_depth = st.slider("max_depth", 2, 10, 3, key="gb_depth")
            base_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth}
        else:
            st.markdown("#### Gradient Boosting — Tuning Ranges")
            n_estimators_range = st.slider("n_estimators range", 10, 300, (50, 150), key="gb_n_est_r")
            learning_rate_range = st.slider("learning_rate range", 0.01, 0.3, (0.05, 0.2), key="gb_lr_r")
            max_depth_range = st.slider("max_depth range", 2, 10, (3, 7), key="gb_depth_r")
            param_ranges = {"n_estimators": n_estimators_range, "learning_rate": learning_rate_range, "max_depth": max_depth_range}

    else:
        if not tune:
            st.markdown("#### XGBoost Parameters")
            n_estimators = st.slider("n_estimators", 50, 500, 200, key="xgb_n_est")
            learning_rate = st.slider("learning_rate", 0.01, 0.3, 0.1, key="xgb_lr")
            max_depth = st.slider("max_depth", 2, 12, 6, key="xgb_depth")
            subsample = st.slider("subsample", 0.5, 1.0, 0.8, key="xgb_sub")
            base_params = {"n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth, "subsample": subsample}
        else:
            st.markdown("#### XGBoost — Tuning Ranges")
            n_estimators_range = st.slider("n_estimators range", 50, 500, (100, 300), key="xgb_n_est_r")
            learning_rate_range = st.slider("learning_rate range", 0.01, 0.3, (0.05, 0.2), key="xgb_lr_r")
            max_depth_range = st.slider("max_depth range", 2, 12, (4, 8), key="xgb_depth_r")
            subsample_range = st.slider("subsample range", 0.5, 1.0, (0.6, 1.0), key="xgb_sub_r")
            param_ranges = {"n_estimators": n_estimators_range, "learning_rate": learning_rate_range, "max_depth": max_depth_range, "subsample": subsample_range}

    train_button = st.button("Train Model", key="model_train_button", type="primary")

    if not th["done"]:
        st.info("Hyperparameter tuning is running...")
        st.progress(th["progress"], text=f"{th['progress'] * 100:.0f}% complete")
        st.write(th["status"])
        if st.button("Cancel Training", key="cancel_tuning_btn"):
            th["cancel"] = True
            st.warning("Cancellation requested — stopping after the current combination...")
        time.sleep(0.5)
        st.rerun()

    if "tuning_last_score" in st.session_state:
        st.success("Hyperparameter tuning complete!")
        if "tuning_last_results" in st.session_state:
            st.subheader("Model Performance")
            st.write(st.session_state["tuning_last_results"])
        st.subheader("Best Parameters")
        st.write(st.session_state["tuning_last_params"])

    if train_button:
        if not uploaded_zips:
            st.error("Please upload at least one ZIP file.")
        else:
            with st.spinner("Building dataset from uploaded ZIPs..."):
                datasets = build_dataset_from_uploaded_zips(uploaded_zips)

            empty_zips = [name for name, df in datasets.items() if df.empty]
            if empty_zips:
                for name in empty_zips:
                    st.error(f"No VMAF folders found in {name}.zip.")
            valid_datasets = {k: v for k, v in datasets.items() if not v.empty}
            if not valid_datasets:
                st.stop()

            with st.spinner("Preparing training data..."):
                df_train_raw, df_test = split_train_test(valid_datasets, test_size=test_size)
                df_train = balance_training_dataset(df_train_raw)
                X_train, y_train, X_test, y_test = prepare_features(df_train, df_test)

            if not tune:
                if model_type == "Random Forest":
                    model = RandomForestModel(**base_params)
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingModel(**base_params)
                else:
                    model = XGBoostModel(**base_params)

                with st.spinner("Training model..."):
                    model.train(X_train, y_train)
                    results = model.evaluate(X_test, y_test)

                st.success("Training complete!")
                st.subheader("Model Performance")
                st.write(results)
                st.subheader("Parameters Used")
                st.write(base_params)

                st.session_state["trained_model"] = model
                st.session_state["trained_model_type"] = model_type
                st.session_state["trained_model_params"] = base_params

                st.session_state.pop("tuning_last_score", None)
                st.session_state.pop("tuning_last_params", None)
                st.session_state.pop("tuning_last_results", None)

            else:
                def linspace_int(a, b, n=3):
                    return list(sorted(set(int(x) for x in np.linspace(a, b, n))))

                def linspace_float(a, b, n=3):
                    return [round(float(x), 4) for x in np.linspace(a, b, n)]

                param_ranges_expanded = {}
                for k, (low, high) in param_ranges.items():
                    if isinstance(low, float) or isinstance(high, float):
                        param_ranges_expanded[k] = linspace_float(low, high)
                    else:
                        param_ranges_expanded[k] = linspace_int(low, high)

                th["done"] = False
                th["cancel"] = False
                th["cancelled"] = False
                th["progress"] = 0.0
                th["status"] = "Starting..."
                th["result"] = ()

                st.session_state["_pending_model_type"] = model_type
                st.session_state.pop("tuning_last_score", None)
                st.session_state.pop("tuning_last_params", None)
                st.session_state.pop("tuning_last_results", None)

                _mt = model_type
                _pre = param_ranges_expanded
                _Xtr, _ytr, _Xte, _yte = X_train, y_train, X_test, y_test
                _th = th

                def _progress_callback(current, total, best_score, current_params):
                    if _th["cancel"]:
                        raise InterruptedError("Cancelled by user")
                    _th["progress"] = current / total
                    _th["status"] = (
                        f"{current} / {total} combinations evaluated — "
                        f"best MAE so far: {best_score:.4f}"
                    )

                def _train():
                    try:
                        best_model, best_params, best_score, best_results, _ = run_grid_search(
                            _mt, _pre, _Xtr, _ytr, _Xte, _yte,
                            progress_callback=_progress_callback,
                        )
                        _th["result"] = (best_model, best_params, best_score, best_results)
                    except InterruptedError:
                        _th["cancelled"] = True
                    finally:
                        _th["done"] = True

                threading.Thread(target=_train, daemon=True).start()
                st.rerun()

    st.markdown("---")
    st.subheader("Save current model")

    if "trained_model" in st.session_state:
        save_note = st.text_input("Optional name/label for this model", value="", key="model_save_note")
        if st.button("Save current model", key="save_model_button"):
            path = save_model(
                st.session_state["trained_model"],
                st.session_state.get("trained_model_type", "model"),
                note=save_note,
            )
            st.success(f"Model saved to: `{path}`")
    else:
        st.info("Train a model first to enable saving.")
