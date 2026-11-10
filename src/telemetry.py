import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_chip_telemetry(duration_sec=120, inject_anomaly=True):
    np.random.seed(42)
    n = max(int(duration_sec * 10), 100)
    t = np.linspace(0, duration_sec, n)

    base_power = 45.0 + 3.5 * np.sin(2 * np.pi * t / 30.0) + np.random.normal(0, 0.8, n)
    temp_c = 52.0 + 0.18 * (base_power - 45.0) + 4.0 * (1 - np.exp(-t / 20.0)) + np.random.normal(0, 0.35, n)

    if inject_anomaly:
        start = int(n * 0.62)
        end = int(n * 0.82)
        spike_len = max(end - start, 1)
        temp_c[start:end] += np.linspace(4.0, 26.0, spike_len)
        base_power[start:end] += np.linspace(5.0, 22.0, spike_len)

    return pd.DataFrame({
        "Time_sec": np.round(t, 2),
        "Temperature_C": np.round(temp_c, 2),
        "Power_W": np.round(base_power, 2),
    })


def plot_telemetry_dashboard(df_tel):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=("Die Junction Temperature (°C)", "Instantaneous Power Draw (W)")
    )
    fig.add_trace(
        go.Scatter(x=df_tel["Time_sec"], y=df_tel["Temperature_C"], name="Temp (°C)", line=dict(color="#EF553B", width=2.5)),
        row=1, col=1
    )
    fig.add_hline(y=75.0, line_dash="dash", line_color="#FFA15A", annotation_text="Thermal Throttling Limit (75°C)", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=df_tel["Time_sec"], y=df_tel["Power_W"], name="Power (W)", line=dict(color="#00CC96", width=2.5)),
        row=2, col=1
    )
    fig.update_layout(height=420, margin=dict(l=30, r=20, t=40, b=30), showlegend=False)
    fig.update_xaxes(title_text="Elapsed Time (s)", row=2, col=1)
    return fig
