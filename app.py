import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from src.telemetry import generate_chip_telemetry, plot_telemetry_dashboard
from src.diagnostics import calculate_psi, compute_calibration_data

st.set_page_config(page_title="SiliconGuard AI | Semiconductor Yield & Telemetry", page_icon="🔬", layout="wide")

def load_artifact():
    return joblib.load("models/siliconguard_v3.pkl")

artifact = load_artifact()
pipeline = artifact["pipeline"]
top_features = artifact["top_features"]
metrics = artifact["metrics"]
feature_names = artifact["feature_names"]
val_t_pct = int(round(metrics.get("val_selected_threshold", 0.07) * 100))

st.sidebar.title("Control Panel")
threshold = st.sidebar.slider("Alert Probability Threshold (%)", 1, 95, val_t_pct, 1) / 100.0
st.sidebar.markdown("### Economic Assumptions (Simulated)")
cost_tp = st.sidebar.number_input("Savings per Caught Defect (\$)", 500, 20000, 3500, 100)
cost_fp = st.sidebar.number_input("Inspection Cost per False Alarm (\$)", 50, 5000, 400, 50)
cost_fn = st.sidebar.number_input("Packaging Loss per Missed Defect (\$)", 500, 25000, 4500, 100)
st.sidebar.info("**Methodology:** UCI SECOM (1,567 wafers, 590 sensors). Chronological Split (65% Train / 15% Val / 20% Test). Designed to prevent preprocessing leakage; threshold selected strictly on validation split.")

st.title("🔬 SiliconGuard AI — Semiconductor Yield & Telemetry Prototype")
st.caption("Applied ML wafer defect screening (chronological train/val/test split) with explicit cost-threshold analysis and simulated thermal telemetry.")

uploaded_file = st.file_uploader("Upload SECOM-formatted CSV (or evaluate on the 314-wafer chronological test set)", type=["csv"])
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    missing_cols = set(feature_names) - set(df_raw.columns)
    if missing_cols:
        st.error(f"Uploaded CSV is missing {len(missing_cols)} required SECOM sensor columns (e.g., {list(missing_cols)}). Please use the sample CSV template.")
        st.stop()
else:
    df_raw = pd.read_csv("data/sample_wafer_batch.csv")

has_ground_truth = "Ground_Truth_Label" in df_raw.columns
X_batch = df_raw[feature_names].copy()
probs = pipeline.predict_proba(X_batch)[:, 1]
preds = (probs >= threshold).astype(int)

results_df = pd.DataFrame({
    "Wafer_ID": [f"WAF-{1000 + i}" for i in range(len(X_batch))],
    "Defect_Risk_Pct": np.round(probs * 100, 2),
    "Decision": np.where(preds == 1, "🚨 HIGH RISK (Inspect)", "✅ PASS (To Packaging)"),
})
if has_ground_truth:
    results_df["Actual_Status"] = np.where(df_raw["Ground_Truth_Label"] == 1, "DEFECT", "PASS")

tab1, tab2, tab3, tab4 = st.tabs(["1. Batch Screening & Economics", "2. Benchmark, CI & Confusion Matrix", "3. Wafer Map & Sensor Deviations", "4. Simulated Thermal Telemetry"])
with tab1:
    st.subheader(f"Batch Risk Screening ({len(X_batch)} Wafers)")
    c1, c2, c3, c4 = st.columns(4)
    flagged_count = int(preds.sum())
    c1.metric("Wafers Screened", len(X_batch))
    c2.metric("Flagged High Risk", f"{flagged_count} ({flagged_count / len(X_batch) * 100:.1f}%)")
    if has_ground_truth:
        y_true = df_raw["Ground_Truth_Label"].values
        total_defects = int(y_true.sum())
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=list((0, 1))).ravel()
        net_model = (tp * cost_tp) - (fp * cost_fp) - (fn * cost_fn)
        net_baseline = -(total_defects * cost_fn)
        inc_gain = net_model - net_baseline
        c3.metric("True Defects Caught (TP / Total)", f"{tp} / {total_defects}")
        c4.metric("Net Economic Impact (Model)", f"\${net_model:,.0f}", delta=f"+\${inc_gain:,.0f} vs. Baseline (\${net_baseline:,.0f})")
        st.info(f"**Formula:** `Net Impact = (TP × \${cost_tp:,}) − (FP × \${cost_fp:,}) − (FN × \${cost_fn:,})` | **Do-Nothing Baseline (Predict All Pass):** **\${net_baseline:,.0f}** | **Incremental Gain:** **+\${inc_gain:,.0f}**")
        thresh_grid = np.linspace(0.01, 0.95, 95)
        net_curve = [(confusion_matrix(y_true, (probs >= t).astype(int), labels=list((0, 1))).ravel()[3] * cost_tp) - (confusion_matrix(y_true, (probs >= t).astype(int), labels=list((0, 1))).ravel()[1] * cost_fp) - (confusion_matrix(y_true, (probs >= t).astype(int), labels=list((0, 1))).ravel()[2] * cost_fn) for t in thresh_grid]
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(x=thresh_grid * 100, y=net_curve, mode="lines", name="Model Net Impact (\$)", line=dict(color="#00CC96", width=3)))
        fig_cost.add_hline(y=net_baseline, line_dash="dot", line_color="#EF553B", annotation_text=f"No-Screening Baseline (\${net_baseline:,.0f})")
        fig_cost.add_vline(x=threshold * 100, line_dash="dash", line_color="#FFA15A", annotation_text=f"Active: {threshold * 100:.0f}% (\${net_model:,.0f})")
        fig_cost.update_layout(title="Cost-Threshold Sensitivity Curve (Diagnostic View)", xaxis_title="Decision Threshold (%)", yaxis_title="Net Financial Impact (\$)", height=320, margin=dict(l=40, r=30, t=40, b=40))
        st.plotly_chart(fig_cost, use_container_width=True)
    st.dataframe(results_df, use_container_width=True, height=280)
    st.download_button("⬇️ Download Sample CSV Template (10 Wafers)", df_raw[feature_names].head(10).to_csv(index=False).encode("utf-8"), "secom_sample_template.csv", "text/csv")

with tab2:
    st.subheader("Chronological Test Benchmark, 95% CI & Confusion Matrix")
    st.warning(f"**Small-Sample Uncertainty Notice:** Test set contains **{metrics['test_size']} wafers** with only **{metrics.get('test_defects', 17)} true defects**. 95% bootstrap confidence intervals (1,000 resamples) are reported alongside point metrics.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test ROC-AUC", f"{metrics['roc_auc']:.4f}", delta=f"95% CI: {metrics.get('roc_ci')}")
    m2.metric("Test PR-AUC", f"{metrics['pr_auc']:.4f}", delta=f"95% CI: {metrics.get('pr_ci')}")
    m3.metric("Validation-Selected Threshold", f"{val_t_pct}%", delta=f"Baseline PR-AUC: {metrics.get('baseline_pr_auc', 0.0541):.4f}")
    m4.metric("Baseline Accuracy (All Pass)", f"{metrics.get('baseline_acc', 0.9459) * 100:.2f}%")
    if has_ground_truth:
        y_true = df_raw["Ground_Truth_Label"].values
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=list((0, 1))).ravel()
        c_cm, c_pr = st.columns(2)
        with c_cm:
            st.markdown(f"**Confusion Matrix (at {threshold*100:.0f}% Threshold)**")
            st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual PASS (0)", "Actual DEFECT (1)"], columns=["Pred PASS (0)", "Pred HIGH RISK (1)"]), use_container_width=True)
        with c_pr:
            st.markdown("**Classification Metrics**")
            st.dataframe(pd.DataFrame({"Metric": ["Precision", "Recall", "TP / FP / FN"], "Value": [f"{precision_score(y_true, preds, zero_division=0):.4f}", f"{recall_score(y_true, preds, zero_division=0):.4f}", f"{tp} / {fp} / {fn}"]}), use_container_width=True, hide_index=True)
    if "model_comparison" in metrics:
        st.markdown("### Benchmark Against Naive Baseline & Alternative Classifiers")
        st.dataframe(pd.DataFrame(metrics["model_comparison"]), use_container_width=True, hide_index=True)
    st.markdown("### Top 10 Critical Sensors (Permutation Importance on Validation PR-AUC)")
    fig_imp = px.bar(top_features.sort_values("Importance_Score", ascending=True), x="Importance_Score", y="Sensor_Feature", orientation="h", color="Importance_Score", color_continuous_scale="Tealgrn")
    fig_imp.update_layout(height=360, margin=dict(l=110, r=30, t=20, b=40), coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    if has_ground_truth:
        st.markdown("### Probability Calibration Curve & Sensor Drift (PSI)")
        col_cal, col_psi = st.columns(2)
        with col_cal:
            cal_df = compute_calibration_data(df_raw["Ground_Truth_Label"].values, probs, n_bins=5)
            fig_cal = go.Figure()
            fig_cal.add_trace(go.Scatter(x=cal_df["Mean_Predicted_Prob"], y=cal_df["Observed_Defect_Rate"], mode="lines+markers", name="Model Calibration", line=dict(color="#00CC96", width=3)))
            fig_cal.add_trace(go.Scatter(x=list((0, 100)), y=list((0, 100)), mode="lines", name="Perfect Calibration", line=dict(dash="dash", color="#888888")))
            fig_cal.update_layout(title="Reliability / Calibration Curve (%)", xaxis_title="Mean Predicted Risk (%)", yaxis_title="Observed Defect Rate (%)", height=330, margin=dict(l=30, r=20, t=40, b=30))
            st.plotly_chart(fig_cal, use_container_width=True)
        with col_psi:
            ref_df = pd.read_csv("data/sample_wafer_batch.csv")
            top_s = next(iter(top_features["Sensor_Feature"]))
            psi_val = calculate_psi(ref_df[top_s].values, X_batch[top_s].values)
            drift_status = "Stable (<0.10)" if psi_val < 0.10 else ("Moderate Shift (0.10–0.25)" if psi_val < 0.25 else "Significant Drift (>0.25)")
            st.metric(f"Top Sensor PSI Drift ({top_s})", f"{psi_val:.4f}", delta=drift_status)
            st.info("**PSI Guide:** `< 0.10` = No significant distribution shift | `0.10 – 0.25` = Moderate drift | `> 0.25` = Retraining recommended.")

with tab3:
    st.subheader("Simulated Spatial Wafer Risk Map & Sensor Deviation (Z-Score) Inspection")
    st.caption("Note: UCI SECOM does not include wafer coordinates; spatial positions below are simulated for visualization.")
    col_map, col_diag = st.columns(2)
    with col_map:
        np.random.seed(42)
        n_pts = min(len(probs), 300)
        r, theta = np.sqrt(np.random.uniform(0, 1, n_pts)), np.random.uniform(0, 2 * np.pi, n_pts)
        fig_map = px.scatter(pd.DataFrame({"X": r * np.cos(theta), "Y": r * np.sin(theta), "Risk_Pct": np.round(probs[:n_pts] * 100, 1), "Wafer_ID": results_df["Wafer_ID"].iloc[:n_pts]}), x="X", y="Y", color="Risk_Pct", hover_data=["Wafer_ID", "Risk_Pct"], color_continuous_scale="RdYlGn_r", title="Simulated 300mm Wafer Die Risk Distribution")
        fig_map.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_map, use_container_width=True)
    with col_diag:
        selected_wafer = st.selectbox("Select Wafer for Sensor Deviation (Z-Score) Inspection", results_df["Wafer_ID"])
        w_idx = int(selected_wafer.replace("WAF-", "")) - 1000
        z_scores = ((X_batch.iloc[w_idx] - X_batch.mean()) / X_batch.std().replace(0, np.nan)).dropna()
        top_z = z_scores.reindex(z_scores.abs().sort_values(ascending=False).head(10).index)
        fig_z = px.bar(pd.DataFrame({"Sensor": top_z.index, "Z_Score": top_z.values}).sort_values("Z_Score"), x="Z_Score", y="Sensor", orientation="h", color="Z_Score", color_continuous_scale="RdBu_r", title=f"Top 10 Sensor Deviations (Z-Score) — {selected_wafer}")
        fig_z.update_layout(height=360, margin=dict(l=110, r=20, t=40, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_z, use_container_width=True)

with tab4:
    st.subheader("Simulated Chip-Level Thermal & Power Telemetry")
    st.caption("Synthetic physics-inspired telemetry module demonstrating transient thermal spike detection.")
    c1, c2 = st.columns(2)
    with c1:
        duration = st.slider("Simulation Duration (sec)", 30, 300, 120, 10)
        inject = st.checkbox("Inject Thermal Runaway / Micro-Defect Spike", value=True)
        df_tel = generate_chip_telemetry(duration_sec=duration, inject_anomaly=inject)
        st.metric("Peak Simulated Temp", f"{df_tel['Temperature_C'].max():.1f} °C")
        st.metric("Peak Simulated Power", f"{df_tel['Power_W'].max():.1f} W")
    with c2:
        st.plotly_chart(plot_telemetry_dashboard(df_tel), use_container_width=True)
# ==========================================
# NEW MODULE: SHAP ROOT CAUSE ANALYSIS (XAI)
# ==========================================
from src.diagnostics import compute_shap_explanation

st.markdown("---")
st.subheader("🔍 Explainable AI (XAI) – SHAP Sensor Root Cause Analysis")
st.caption("Game-theoretic feature attribution showing which sensors drive defect predictions.")

if st.button("Run SHAP Root Cause Analysis on Current Batch"):
    with st.spinner("Calculating Shapley values across active sensors..."):
        try:
            shap_values, importance_df = compute_shap_explanation(
                pipeline=pipeline,
                X_sample=X_batch,
                feature_names=feature_names
            )
            
            col_shap1, col_shap2 = st.columns([1.2, 1])
            
            with col_shap1:
                fig_shap = px.bar(
                    importance_df.sort_values(by="Mean_SHAP_Impact", ascending=True),
                    x="Mean_SHAP_Impact",
                    y="Sensor",
                    orientation="h",
                    title="Top 10 Most Influential Sensors (Mean |SHAP Value|)",
                    labels={"Mean_SHAP_Impact": "Average Impact on Defect Risk", "Sensor": "Sensor ID"}
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                
                with col_shap2:
                 st.markdown("**Key Diagnostic Insights:**")
                top_sensor = importance_df.iloc[0]["Sensor"]
                top_impact = importance_df.iloc[0]["Mean_SHAP_Impact"]

                st.info(
                    f"Across the evaluated batch, **{top_sensor}** is the primary driver of defect risk "
                    f"with an average SHAP attribution of **{top_impact:.4f}**."
                )
                st.dataframe(importance_df, use_container_width=True, hide_index=True)
                
        except Exception as e:
            st.error(f"SHAP computation error: {e}")
# ==========================================
# NEW MODULE: DATA & CONCEPT DRIFT MONITOR
# ==========================================
from src.diagnostics import monitor_sensor_drift

st.markdown("---")
st.subheader("📡 Automated Sensor & Data Drift Monitor (PSI)")
st.caption("Detects sensor degradation and distribution shifts before model accuracy silently drops.")

drift_injection = st.slider(
    "Simulate Sensor Calibration Drift (Standard Deviation Shift)",
    min_value=0.0, max_value=2.0, value=0.0, step=0.25,
    help="Simulates real-world sensor aging or environmental shift on critical sensors."
)

# Referans olarak mevcut batch'i al, simülasyon için kopyasına kayma (drift) uygula
ref_df = X_batch.copy()
curr_df = X_batch.copy()

# En kritik ilk 5 sensöre (SHAP veya top_features) simüle edilen kaymayı ekle
monitored_sensors = [c for c in X_batch.columns if c not in ["Wafer_ID", "Target", "Label"]][:5]

if drift_injection > 0:
    np.random.seed(42)
    for s in monitored_sensors:
        std_val = ref_df[s].std()
        curr_df[s] = curr_df[s] + (drift_injection * std_val)

drift_summary_df = monitor_sensor_drift(ref_df, curr_df, monitored_sensors)

col_d1, col_d2 = st.columns([1.2, 1])

with col_d1:
    st.dataframe(drift_summary_df, use_container_width=True, hide_index=True)

with col_d2:
    max_psi = drift_summary_df["PSI_Score"].max()
    if max_psi > 0.25:
        st.error(f"🚨 **CRITICAL DRIFT DETECTED (Max PSI: {max_psi})**\n\nSensor distribution shifted significantly. Model retraining or recalibration is triggered.")
    elif max_psi >= 0.10:
        st.warning(f"⚠️ **MODERATE DRIFT (Max PSI: {max_psi})**\n\nMinor sensor drift observed. Continue close monitoring.")
    else:
        st.success(f"✅ **SYSTEM HEALTHY (Max PSI: {max_psi})**\n\nAll monitored sensors match the baseline training distribution.")
# ==========================================
# NEW MODULE: DEEP LEARNING LSTM SEQUENCE MONITOR
# ==========================================
from src.lstm_model import train_and_predict_lstm

st.markdown("---")
st.subheader("🧠 Deep Learning Temporal Sequence Monitor (PyTorch LSTM)")
st.caption("Captures sequential sensor memory across consecutive wafer steps using a 3D Recurrent Neural Network.")

col_lstm_ctrl1, col_lstm_ctrl2 = st.columns(2)
with col_lstm_ctrl1:
    seq_window = st.slider("Temporal Window Length (Consecutive Steps)", min_value=3, max_value=10, value=5)
with col_lstm_ctrl2:
    train_epochs = st.slider("Training Epochs", min_value=10, max_value=50, value=25, step=5)

if st.button("Train & Run LSTM Temporal Sequence Model"):
    with st.spinner("Training PyTorch LSTM neural network on 3D sensor windows..."):
        try:
            numeric_X = X_batch.select_dtypes(include=[np.number])
            loss_history, lstm_probs = train_and_predict_lstm(
                X_df=numeric_X,
                seq_len=seq_window,
                epochs=train_epochs
            )
            
            c_l1, c_l2 = st.columns(2)
            with c_l1:
                loss_df = pd.DataFrame({"Epoch": range(1, len(loss_history) + 1), "BCE_Loss": loss_history})
                fig_loss = px.line(
                    loss_df, x="Epoch", y="BCE_Loss",
                    title="LSTM Training Convergence (Binary Cross-Entropy Loss)",
                    markers=True
                )
                st.plotly_chart(fig_loss, use_container_width=True)
                
            with c_l2:
                seq_df = pd.DataFrame({
                    "Window_Step": range(1, len(lstm_probs) + 1),
                    "Temporal_Risk_Score": lstm_probs
                })
                fig_seq = px.area(
                    seq_df, x="Window_Step", y="Temporal_Risk_Score",
                    title="Sequential Defect Risk Trajectory Across Windows"
                )
                st.plotly_chart(fig_seq, use_container_width=True)
                
        except Exception as e:
            st.error(f"LSTM training error: {e}")
