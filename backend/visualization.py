import numpy as np
import plotly.graph_objects as go


def make_throughput_figure(
    timestamps, actual_current, actual_needed, predicted_needed, error_regions=None
):
    """
    Graph 1: Throughput Time Series.

    Shows actual throughput at the current VMAF level, actual throughput at the
    needed VMAF level, and the model-predicted throughput for the needed level.
    Error regions are shaded in red.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps, y=actual_current,
        name="Actual (current VMAF)",
        line=dict(color="#1f77b4", width=1.5),
    ))

    fig.add_trace(go.Scatter(
        x=timestamps, y=actual_needed,
        name="Actual (needed VMAF)",
        line=dict(color="#2ca02c", width=1.5),
    ))

    fig.add_trace(go.Scatter(
        x=timestamps, y=predicted_needed,
        name="Predicted (needed VMAF)",
        line=dict(color="#ff7f0e", width=1.5, dash="dash"),
    ))

    if error_regions:
        for i, (start, end) in enumerate(error_regions):
            t_start = timestamps[start]
            t_end = timestamps[min(end, len(timestamps) - 1)]
            fig.add_vrect(
                x0=t_start, x1=t_end,
                fillcolor="red", opacity=0.12,
                layer="below", line_width=0,
                annotation_text=f"R{i+1}" if (t_end - t_start) > 2 else "",
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="red",
            )

    fig.update_layout(
        title="Throughput Time Series",
        xaxis_title="Time (s)",
        yaxis_title="Throughput (bps)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        height=420,
        margin=dict(t=50, b=80),
    )

    return fig


def make_rolling_error_figure(timestamps, rolling_error, threshold, error_regions=None):
    """
    Graph 2: Rolling Relative Error.

    Shows the rolling mean of the relative prediction error.
    A dashed threshold line marks the acceptable bound.
    Error regions exceeding the threshold are shaded in red.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=rolling_error * 100,
        name="Rolling Rel. Error",
        line=dict(color="#9467bd", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(148,103,189,0.1)",
    ))

    fig.add_hline(
        y=threshold * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({threshold * 100:.0f}%)",
        annotation_position="top right",
        annotation_font_color="red",
    )

    if error_regions:
        for start, end in error_regions:
            t_start = timestamps[start]
            t_end = timestamps[min(end, len(timestamps) - 1)]
            fig.add_vrect(
                x0=t_start, x1=t_end,
                fillcolor="red", opacity=0.12,
                layer="below", line_width=0,
            )

    fig.update_layout(
        title="Rolling Relative Error",
        xaxis_title="Time (s)",
        yaxis_title="Relative Error (%)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        height=380,
        margin=dict(t=50, b=80),
    )

    return fig


def make_cdf_figure(series, x_max=None):
    """
    CDF of relative error.

    series: list of (label, rel_error_array) tuples — one curve per direction.
    x_max:  if given, clips the x-axis at this value in percentage so outliers don't
            compress the interesting low-error region.
    """
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (label, rel_error) in enumerate(series):
        sorted_err = np.sort(np.array(rel_error)) * 100  # convert to percent
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
        fig.add_trace(go.Scatter(
            x=sorted_err,
            y=cdf,
            name=f"VMAF {label}",
            mode="lines",
            line=dict(width=2, color=colors[idx % len(colors)]),
        ))

    layout = dict(
        title="CDF of Relative Prediction Error",
        xaxis_title="Relative Error (%)",
        yaxis_title="Cumulative Probability (%)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        height=380,
        margin=dict(t=50, b=80),
    )
    if x_max is not None:
        layout["xaxis_range"] = [0, x_max]

    fig.update_layout(**layout)
    return fig


def make_region_zoom_figure(timestamps, actual_needed, predicted_needed, region_label):
    """
    Zooms a chart for a selected error region.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps, y=actual_needed,
        name="Actual",
        line=dict(color="#2ca02c", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=timestamps, y=predicted_needed,
        name="Predicted",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=f"Zoom: {region_label}",
        xaxis_title="Time (s)",
        yaxis_title="Throughput (bps)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.25),
        height=320,
        margin=dict(t=50, b=80),
    )

    return fig
