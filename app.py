import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.telemetry import generate_chip_telemetry, plot_telemetry_dashboard

st.set_page_config(page_title="SiliconGuard AI | Semiconductor Yield Prototype", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_ai_model():
    return joblib.load("models/siliconguard_rf.pkl")

st.title("🛡️ SiliconGuard AI — Semiconductor Yield & Telemetry Prototype")
st.caption("⚠️ **Academic & Portfolio Demo:** Built on the public UCI SECOM dataset (590 anonymized fab sensors) and synthetic thermal telemetry simulation.")

artifact = load_ai_model()
pipeline = artifact["pipeline"]
top_features = artifact["top_features"]
metrics = artifact["metrics"]

# Anonim SECOM sensorlerine temsili surec aciklamasi (Illustrative mapping)
sensor_labels = {
    "Attribute 104": "Attribute 104 [Illustrative: Etch Pressure]",
    "Attribute 34": "Attribute 34 [Illustrative: Thermal Gradient]",
    "Attribute 60": "Attribute 60 [Illustrative: Gas Flow]",
    "Attribute 32": "Attribute 32 [Illustrative: RF Power]",
    "Attribute 478": "Attribute 478 [Illustrative: Litho Focus]",
    "Attribute 520": "Attribute 520 [Illustrative: Chamber Vacuum]",
    "Attribute 131": "Attribute 131 [Illustrative: Ion Dose]",
    "Attribute 214": "Attribute 214 [Illustrative: CMP Temp]",
    "Attribute 511": "Attribute 511 [Illustrative: Oxide Thickness]",
    "Attribute 206": "Attribute 206 [Illustrative: Edge Uniformity]"
}

with st.sidebar:
    st.header("⚙️ Simulation & Threshold Controls")
    st.caption("Adjust decision threshold and simulated unit costs:")
    threshold = st.slider("Alert Probability Threshold (%)", min_value=5, max_value=50, value=15, step=1)
    scrap_cost = st.number_input("Simulated Packaging Cost Saved ($)", min_value=500, max_value=10000, value=3500, step=500)
    false_alarm_cost = st.number_input("Simulated False-Alarm Inspection Cost ($)", min_value=50, max_value=2000, value=400, step=50)
    st.divider()
    st.info(f"📊 **Dataset Baseline:** UCI SECOM has a **{metrics['total_defect_rate_pct']}%** actual defect rate across 1,567 wafers (80/20 stratified split).")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧪 Batch Prediction Demo",
    "📊 Model Validation & Metrics",
    "🗺️ 2D Wafer Map (Simulated Layout)",
    "🔍 Single-Wafer Z-Score Analysis",
    "🌡️ Synthetic Thermal Telemetry"
])

uploaded_file = st.file_uploader("Upload SECOM-formatted CSV (or use 50-sample held-out test slice)", type=["csv"])
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
else:
    df_raw = pd.read_csv("data/sample_wafer_batch.csv")

has_ground_truth = "Ground_Truth_Label" in df_raw.columns
y_true_batch = df_raw["Ground_Truth_Label"].values if has_ground_truth else None
df_features = df_raw.drop(columns=["Ground_Truth_Label"]) if has_ground_truth else df_raw

probs = pipeline.predict_proba(df_features)[:, 1]
preds = (probs * 100 >= threshold).astype(int)
wafer_ids = [f"TEST-WAFER-{1000+i}" for i in range(len(df_features))]

results_df = pd.DataFrame({
    "Sample_ID": wafer_ids,
    "Predicted_Risk_%": (probs * 100).round(2),
    "Model_Flag": ["🔴 Flagged High-Risk" if p == 1 else "🟢 Pass" for p in preds]
})

if has_ground_truth:
    results_df["Actual_SECOM_Label"] = ["Defective (1)" if y == 1 else "Healthy (0)" for y in y_true_batch]

results_sorted = results_df.sort_values(by="Predicted_Risk_%", ascending=False).reset_index(drop=True)

with tab1:
    st.subheader("Held-Out Test Batch Predictions & Simulated Cost Impact")
    flagged_count = int((preds == 1).sum())

    if has_ground_truth:
        tp = int(((preds == 1) & (y_true_batch == 1)).sum())
        fp = int(((preds == 1) & (y_true_batch == 0)).sum())
        fn = int(((preds == 0) & (y_true_batch == 1)).sum())
        net_est_savings = (tp * scrap_cost) - (fp * false_alarm_cost)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Samples Evaluated", len(results_df))
        m2.metric("Flagged by Threshold", f"{flagged_count} ({round((flagged_count/len(preds))*100, 1)}%)")
        m3.metric("True Defects Caught / False Alarms", f"{tp} TP / {fp} FP")
        m4.metric("Est. Net Simulated Impact", f"${net_est_savings:,}", help="Calculated as (True Positives × Packaging Cost) - (False Positives × Re-inspection Cost)")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Samples Evaluated", len(results_df))
        m2.metric("Flagged High-Risk", flagged_count)

    st.divider()
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("### Held-Out Test Set Predictions")
        st.dataframe(results_sorted, width="stretch", height=350)
    with col2:
        st.markdown("### Global Feature Importance (Random Forest)")
        st.caption("Computed across the entire training set (not specific to a single wafer):")
        labeled_features = [(sensor_labels.get(s, s), imp) for s, imp in top_features]
        feat_df = pd.DataFrame(labeled_features, columns=["Sensor_Feature", "Gini_Importance"])
        fig_feat = px.bar(feat_df.iloc[::-1], x="Gini_Importance", y="Sensor_Feature", orientation="h", template="plotly_dark", color="Gini_Importance", color_continuous_scale="Tealgrn")
        fig_feat.update_layout(height=390, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_feat, width="stretch")

with tab2:
    st.subheader("Model Validation Metrics (20% Stratified Held-Out Test Set)")
    st.markdown(f"Trained on **{metrics['train_size']}** wafers and evaluated on **{metrics['test_size']}** unseen wafers from UCI SECOM:")
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Test ROC-AUC Score", metrics["roc_auc"])
    k2.metric("Precision (at default 0.50 cutoff)", metrics["precision"])
    k3.metric("Recall / Sensitivity (at 0.50 cutoff)", metrics["recall"])

    st.markdown("### Confusion Matrix (Held-Out Test Set)")
    cm = np.array(metrics["confusion_matrix"])
    fig_cm = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted Label", y="Actual SECOM Label", color="Count"),
        x=["Healthy (0)", "Defective (1)"], y=["Healthy (0)", "Defective (1)"],
        template="plotly_dark"
    )
    fig_cm.update_layout(height=380, width=500)
    st.plotly_chart(fig_cm)
    st.info("ℹ️ **Engineering Note:** Because SECOM has only ~6.6% actual defects, lowering the probability threshold in the sidebar increases Recall (catching more defects) at the trade-off of higher False Positives.")

with tab3:
    st.subheader("2D Spatial Wafer Map (Simulated Coordinate Projection)")
    st.caption("⚠️ Note: UCI SECOM does not include physical X/Y die coordinates. Coordinates below are synthetically projected to demonstrate spatial clustering visualization.")
    n = len(results_df)
    angles = np.linspace(0, 6 * np.pi, n)
    radii = np.sqrt(np.linspace(0.05, 0.92, n))
    x_coords = np.round(radii * np.cos(angles) * 150, 1)
    y_coords = np.round(radii * np.sin(angles) * 150, 1)

    fig = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 200)
    fig.add_trace(go.Scatter(x=155 * np.cos(theta), y=155 * np.sin(theta), mode="lines", line=dict(color="#4A5568", width=3), name="300mm Boundary", hoverinfo="skip"))

    healthy_mask = preds == 0
    fig.add_trace(go.Scatter(x=x_coords[healthy_mask], y=y_coords[healthy_mask], mode="markers", marker=dict(size=14, color="#00CC96", symbol="square"), name="Pass"))
    fig.add_trace(go.Scatter(x=x_coords[~healthy_mask], y=y_coords[~healthy_mask], mode="markers", marker=dict(size=16, color="#FF4B4B", symbol="square", line=dict(width=1.5, color="white")), name="Flagged High-Risk"))
    fig.update_layout(template="plotly_dark", height=450, xaxis=dict(range=[-175, 175]), yaxis=dict(range=[-175, 175], scaleanchor="x", scaleratio=1))
    st.plotly_chart(fig, width="stretch")

with tab4:
    st.subheader("Single-Wafer Z-Score Deviation Analysis")
    st.caption("Compares an individual wafer's top sensor readings against the batch mean (in standard deviations, σ). This is a statistical deviation check, not SHAP.")
    selected_wafer = st.selectbox("Select Test Sample ID:", results_sorted["Sample_ID"].tolist())
    orig_idx = wafer_ids.index(selected_wafer)

    top_raw_cols = [s for s, _ in top_features]
    batch_means = df_features[top_raw_cols].mean()
    batch_stds = df_features[top_raw_cols].std().replace(0, 1)
    wafer_vals = df_features.iloc[orig_idx][top_raw_cols].fillna(batch_means)
    z_scores = ((wafer_vals - batch_means) / batch_stds).round(2)

    dev_df = pd.DataFrame({
        "Sensor": [sensor_labels.get(c, c) for c in top_raw_cols],
        "Z_Score_Sigma": z_scores.values
    }).sort_values(by="Z_Score_Sigma", key=abs, ascending=True)

    fig_dev = px.bar(dev_df, x="Z_Score_Sigma", y="Sensor", orientation="h", title=f"Statistical Deviation (Z-Score) for {selected_wafer}", template="plotly_dark", color="Z_Score_Sigma", color_continuous_scale="RdYlGn_r")
    fig_dev.add_vline(x=0, line_color="white", line_dash="dash")
    st.plotly_chart(fig_dev, width="stretch")

with tab5:
    st.subheader("Synthetic Die-Level Thermal & Power Stress Simulation")
    st.caption("⚠️ Generated via NumPy sinusoidal + Gaussian noise equations to demonstrate time-series anomaly alerting.")
    c1, c2 = st.columns([1, 2.5])
    with c1:
        inject = st.checkbox("Inject Synthetic Thermal Leak", value=True)
        duration = st.slider("Simulation Duration (Sec)", 300, 1500, 1000, 100)
        df_tel = generate_chip_telemetry(duration_sec=duration, inject_anomaly=inject)
        st.metric("Peak Simulated Temp", f"{df_tel['Temperature_C'].max()} °C")
        st.metric("Peak Simulated Power", f"{df_tel['Power_W'].max()} W")
    with c2:
        st.plotly_chart(plot_telemetry_dashboard(df_tel), width="stretch")
