# 🔬 SiliconGuard AI — Semiconductor Yield & Telemetry Prototype

An end-to-end machine learning and process-analytics prototype for semiconductor wafer defect screening and chip-level thermal telemetry monitoring.

**Live Interactive App:** [https://streamlit.app](https://streamlit.app)

## Key Engineering Highlights
- **Leakage-Free ML Pipeline:** `VarianceThreshold`, `SimpleImputer(strategy="median")`, and `StandardScaler` fit strictly within the training split to prevent data leakage.
- **Chronological Validation:** Evaluated on the final 20% future slice (**314 wafers**) of the **UCI SECOM** dataset (1,567 wafers, 590 fab sensors, 6.64% defect rate) to simulate real production deployment.
- **Imbalanced Metrics & Model Benchmark:** Evaluates **PR-AUC** alongside ROC-AUC against a naive *Predict-All-Pass* baseline (`0.0541` PR-AUC). Regularized **Logistic Regression (Balanced)** achieves **`0.7964` ROC-AUC** and **`0.2068` PR-AUC** (~3.8x over baseline).
- **Cost-Threshold Optimization:** Interactive economic curve modeling True Positive savings, False Positive inspection costs, and False Negative packaging losses across thresholds (5%–95%).
- **Simulated Telemetry & Diagnostics:** Permutation-importance sensor ranking, wafer-level Z-score anomaly inspection, and physics-inspired thermal/power transient simulation.

## Limitations & Scope
- **Dataset Noise & High Dimensionality:** UCI SECOM contains 590 anonymized sensors with significant missing values and noise; predictions serve as decision-support screening rather than standalone metrology replacement.
- **Simulated Modules:** Economic cost parameters, spatial 300mm wafer coordinates, and transient thermal telemetry waveforms are simulated for prototype demonstration.
