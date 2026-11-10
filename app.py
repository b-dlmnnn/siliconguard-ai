import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from src.telemetry import generate_chip_telemetry, plot_telemetry_dashboard

st.set_page_config(
    page_title="SiliconGuard AI | Semiconductor Yield & Telemetry Prototype",
    page_icon="🔬",
    layout="wide",
)

@st.cache_resource
def load_artifact():
    return joblib.load("models/siliconguard_rf.pkl")

artifact = load_artifact()
pipeline = artifact["pipeline"]
top_features = artifact["top_features"]
metrics = artifact["metrics"]
feature_names = artifact["feature_names"]

# --- SIDEBAR ---
st.sidebar.title("Control Panel")
st.sidebar.markdown("Configure decision threshold and economic parameters:")

threshold = st.sidebar.slider(
    "Alert Probability Threshold (%)",
    min_value=5,
    max_value=95,
    value=16,
    step=1,
    help="Wafers with predicted defect probability above this threshold are flagged for inspection.",
) / 100.0

st.sidebar.markdown("### Economic Assumptions (Simulated)")
cost_tp = st.sidebar.number_input("Savings per Caught Defect ($)", min_value=500, max_value=20000, value=3500, step=100)
cost_fp = st.sidebar.number_input("Inspection Cost per False Alarm ($)", min_value=16, max_value=5000, value=400, step=50)
cost_fn = st.sidebar.number_input("Packaging Loss per Missed Defect ($)", min_value=500, max_value=25000, value=4500, step=100)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Methodology & Data Scope:**\n"
    "- **Dataset:** UCI SECOM (1,567 wafers, 590 sensors, 6.64% defect rate)\n"
    "- **Validation:** Chronological Split (First 80% Train / Last 20% Future Test)\n"
    "- **Leakage Control:** VarianceThreshold, Median Imputer & Scaler fit strictly inside CV/Train pipeline."
)

# --- HEADER ---
st.title("🔬 SiliconGuard AI — Semiconductor Yield & Telemetry Prototype")
st.caption(
    "Leakage-free ML defect screening trained on real UCI SECOM fab sensor data (chronological split) "
    "paired with cost-threshold optimization and simulated thermal telemetry."
)

# --- DATA LOADING ---
uploaded_file = st.file_uploader(
    "Upload SECOM-formatted CSV (or use full 314-wafer chronological test set)",
    type=["csv"],
)
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
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

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Batch Prediction & Cost Curve",
    "2. Model Comparison & Metrics",
    "3. Wafer Map & Sensor Z-Scores",
    "4. Simulated Thermal Telemetry",
])

# ==================== TAB 1 ====================
with tab1:
    st.subheader(f"Batch Risk Screening ({len(X_batch)} Wafers)")

    c1, c2, c3, c4 = st.columns(4)
    flagged_count = int(preds.sum())
    c1.metric("Wafers Screened", len(X_batch))
    c2.metric("Flagged High Risk", f"{flagged_count} ({flagged_count / len(X_batch) * 100:.1f}%)")

    if has_ground_truth:
        y_true = df_raw["Ground_Truth_Label"].values
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        net_impact = (tp * cost_tp) - (fp * cost_fp) - (fn * cost_fn)
        c3.metric("True Defects Caught (TP / Total)", f"{tp} / {int(y_true.sum())}")
        c4.metric(
            "Net Economic Impact vs. No Screening",
            f"${net_impact:,.0f}",
            delta=f"FP: {fp} (-${fp * cost_fp:,}) | FN: {fn} (-${fn * cost_fn:,})",
        )

        # Cost-Threshold Optimization Curve
        st.markdown("### Cost-Threshold Optimization Curve")
        st.caption("Evaluates net financial impact across all thresholds (5%–95%) accounting for TP savings, FP inspection costs, and FN missed-defect losses.")
        thresh_grid = np.linspace(0.05, 0.95, 91)
        net_curve = []
        for t in thresh_grid:
            p_t = (probs >= t).astype(int)
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, p_t, labels=[0, 1]).ravel()
            net_curve.append((tp_t * cost_tp) - (fp_t * cost_fp) - (fn_t * cost_fn))

        best_idx = int(np.argmax(net_curve))
        opt_thresh = thresh_grid[best_idx] * 100
        opt_value = net_curve[best_idx]

        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=thresh_grid * 100,
            y=net_curve,
            mode="lines",
            name="Net Financial Impact ($)",
            line=dict(color="#00CC96", width=3),
        ))
        fig_cost.add_vline(
            x=threshold * 100,
            line_dash="dash",
            line_color="#FFA15A",
            annotation_text=f"Selected: {threshold * 100:.0f}% (${net_impact:,.0f})",
        )
        fig_cost.add_trace(go.Scatter(
            x=[opt_thresh],
            y=[opt_value],
            mode="markers+text",
            name="Optimal Threshold",
            marker=dict(size=11, color="#EF553B"),
            text=[f"Optimal: {opt_thresh:.0f}% (${opt_value:,.0f})"],
            textposition="top center",
        ))
        fig_cost.update_layout(
            xaxis_title="Decision Threshold (%)",
            yaxis_title="Net Financial Impact ($)",
            height=340,
            margin=dict(l=40, r=30, t=40, b=40),
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    else:
        c3.metric("Mean Defect Risk", f"{probs.mean() * 100:.1f}%")
        c4.metric("Ground Truth", "Unlabeled Upload")

    st.dataframe(results_df, use_container_width=True, height=320)

    csv_template = df_raw[feature_names].head(10).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Sample CSV Template (10 Wafers)",
        data=csv_template,
        file_name="secom_sample_template.csv",
        mime="text/csv",
    )

# ==================== TAB 2 ====================
with tab2:
    st.subheader("Chronological Test Benchmark & Model Comparison")
    st.caption(
        f"Validation Strategy: {metrics.get('split_method', 'Chronological Split')} | "
        f"Train: {metrics['train_size']} wafers | Test: {metrics['test_size']} wafers"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC (Selected Model)", f"{metrics['roc_auc']:.4f}", delta=f"+{metrics['roc_auc'] - 0.5:.4f} vs Random")
    m2.metric("PR-AUC (Imbalanced Metric)", f"{metrics['pr_auc']:.4f}", delta=f"{metrics['pr_auc'] / max(metrics.get('baseline_pr_auc', 0.054), 1e-4):.1f}x vs Baseline")
    m3.metric("Baseline Accuracy (All Pass)", f"{metrics.get('baseline_acc', 0.9459) * 100:.2f}%")
    m4.metric("Overall Dataset Defect Rate", f"{metrics['defect_rate_pct']:.2f}%")

    if "model_comparison" in metrics:
        st.markdown("### Benchmark Against Naive Baseline & Alternative Classifiers")
        st.info(
            "**Why PR-AUC matters here:** Because 93.4% of wafers pass, a naive model that predicts 'Pass' for everything "
            "achieves ~94.6% accuracy while catching 0 defects. **PR-AUC** measures true defect-detection precision across recall levels."
        )
        comp_df = pd.DataFrame(metrics["model_comparison"])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("### Top 10 Critical Sensors (Permutation Importance on PR-AUC)")
    fig_imp = px.bar(
        top_features.sort_values("Importance_Score", ascending=True),
        x="Importance_Score",
        y="Sensor_Feature",
        orientation="h",
        color="Importance_Score",
        color_continuous_scale="Tealgrn",
    )
    fig_imp.update_layout(
        xaxis_title="Mean PR-AUC Drop When Sensor Shuffled (Permutation Importance)",
        yaxis_title="SECOM Sensor ID",
        height=400,
        margin=dict(l=110, r=30, t=20, b=40),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ==================== TAB 3 ====================
with tab3:
    st.subheader("Simulated Spatial Wafer Risk Map & Single-Wafer Sensor Diagnostics")
    st.caption("Note: SECOM does not include wafer coordinates; spatial positions below are simulated for visualization.")

    col_map, col_diag = st.columns([1, 1])
    with col_map:
        np.random.seed(42)
        n_pts = min(len(probs), 300)
        r = np.sqrt(np.random.uniform(0, 1, n_pts))
        theta = np.random.uniform(0, 2 * np.pi, n_pts)
        wafer_map_df = pd.DataFrame({
            "X": r * np.cos(theta),
            "Y": r * np.sin(theta),
            "Risk_Pct": np.round(probs[:n_pts] * 100, 1),
            "Wafer_ID": results_df["Wafer_ID"].iloc[:n_pts],
        })
        fig_map = px.scatter(
            wafer_map_df,
            x="X",
            y="Y",
            color="Risk_Pct",
            hover_data=["Wafer_ID", "Risk_Pct"],
            color_continuous_scale="RdYlGn_r",
            title="Simulated 300mm Wafer Die Risk Distribution",
        )
        fig_map.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_map, use_container_width=True)

    with col_diag:
        selected_wafer = st.selectbox("Select Wafer for Root-Cause Sensor Z-Score Inspection", results_df["Wafer_ID"])
        w_idx = int(selected_wafer.replace("WAF-","")) - 1000
        row_vals = X_batch.iloc[w_idx]

        batch_means = X_batch.mean()
        batch_stds = X_batch.std().replace(0, np.nan)
        z_scores = ((row_vals - batch_means) / batch_stds).dropna()
        top_z = z_scores.reindex(z_scores.abs().sort_values(ascending=False).head(10).index)

        z_df = pd.DataFrame({"Sensor": top_z.index, "Z_Score": top_z.values}).sort_values("Z_Score")
        fig_z = px.bar(
            z_df, x="Z_Score", y="Sensor", orientation="h",
            color="Z_Score", color_continuous_scale="RdBu_r",
            title=f"Top 10 Sensor Deviations (Z-Score) — {selected_wafer}",
        )
        fig_z.update_layout(height=360, margin=dict(l=110, r=20, t=40, b=20), coloraxis_showscale=False)
        st.plotly_chart(fig_z, use_container_width=True)

# ==================== TAB 4 ====================
with tab4:
    st.subheader("Simulated Chip-Level Thermal & Power Telemetry")
    st.caption("Synthetic physics-inspired telemetry module demonstrating transient thermal spike detection.")

    c1, c2 = st.columns(2)
    with c1:
        duration = st.slider("Simulation Duration (sec)", 30, 300, 120, step=10)
        inject = st.checkbox("Inject Thermal Runaway / Micro-Defect Spike", value=True)
        df_tel = generate_chip_telemetry(duration_sec=duration, inject_anomaly=inject)
        st.metric("Peak Simulated Temp", f"{df_tel['Temperature_C'].max():.1f} °C")
        st.metric("Peak Simulated Power", f"{df_tel['Power_W'].max():.1f} W")
    with c2:
        st.plotly_chart(plot_telemetry_dashboard(df_tel), use_container_width=True)
