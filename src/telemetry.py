import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_chip_telemetry(duration_sec=1000, inject_anomaly=True, anomaly_start=450, anomaly_end=480):
    """
    Yari iletken cip stres testi sirasinda olusan termal, voltaj, akim ve guc verilerini uretir.
    """
    np.random.seed(42)
    time_sec = np.arange(0, duration_sec)

    # Normal calisma kosullari (Termal dongu, cekirdek voltaji ve akim)
    temp_c = 52.0 + 4.5 * np.sin(time_sec / 60.0) + np.random.normal(0, 0.8, duration_sec)
    voltage_v = 1.18 + 0.02 * np.cos(time_sec / 45.0) + np.random.normal(0, 0.005, duration_sec)
    current_a = 17.5 + 1.2 * np.sin(time_sec / 50.0) + np.random.normal(0, 0.3, duration_sec)

    # Termal / Voltaj kacak anomalisi enjeksiyonu
    if inject_anomaly:
        temp_c[anomaly_start:anomaly_end] += np.linspace(5, 29, anomaly_end - anomaly_start)
        voltage_v[anomaly_start:anomaly_end] += np.random.uniform(0.14, 0.21, anomaly_end - anomaly_start)
        current_a[anomaly_start:anomaly_end] += np.random.uniform(5.5, 7.5, anomaly_end - anomaly_start)

    df = pd.DataFrame({
        "Time_Sec": time_sec,
        "Temperature_C": np.round(temp_c, 2),
        "Voltage_V": np.round(voltage_v, 3),
        "Current_A": np.round(current_a, 2)
    })

    # Guc tuketimi (Watt) ve 10 saniyelik hareketli ortalama
    df["Power_W"] = np.round(df["Voltage_V"] * df["Current_A"], 2)
    df["Temp_Rolling_10s"] = np.round(df["Temperature_C"].rolling(window=10, min_periods=1).mean(), 2)
    df["Thermal_Alert"] = df["Temperature_C"] > 75.0

    return df


def plot_telemetry_dashboard(df):
    """
    Cip stres testi verilerini cok panelli interaktif Plotly grafigine donusturur.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Real-Time Die Temperature & 10s Moving Average (°C)",
            "Instantaneous Power Dissipation (Watts = V × I)"
        )
    )

    # 1. Panel: Sicaklik ve Hareketli Ortalama
    fig.add_trace(
        go.Scatter(x=df["Time_Sec"], y=df["Temperature_C"], name="Die Temp (°C)", line=dict(color="#FF4B4B", width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["Time_Sec"], y=df["Temp_Rolling_10s"], name="10s Moving Avg (°C)", line=dict(color="#FFA15A", width=2, dash="dash")),
        row=1, col=1
    )
    fig.add_hline(y=75.0, line_dash="dot", line_color="red", annotation_text="Critical Threshold (75°C)", row=1, col=1)

    # 2. Panel: Guc Tuketimi (Watt)
    fig.add_trace(
        go.Scatter(x=df["Time_Sec"], y=df["Power_W"], name="Power (W)", fill="tozeroy", line=dict(color="#00CC96", width=1.8)),
        row=2, col=1
    )

    fig.update_layout(
        height=550,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(title_text="Temp (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Power (W)", row=2, col=1)
    fig.update_xaxes(title_text="Stress Test Duration (Seconds)", row=2, col=1)

    return fig
