# 🔬 SiliconGuard AI — Semiconductor Yield & Telemetry Prototype

An applied machine learning and process-analytics prototype for semiconductor wafer defect screening and chip-level thermal telemetry monitoring.

**Live Interactive App:** [https://streamlit.app](https://streamlit.app)

## 1. Problem & Data
- **Problem:** Defective wafers escaping early screening incur downstream packaging losses, while excessive false alarms create metrology bottlenecks.
- **Dataset:** Public **UCI SECOM** dataset (`1,567` wafers, `590` sensors, `6.64%` overall defect rate).

## 2. Method & Leakage Control
- **Chronological Split:** Ordered chronologically into **65% Train** (`1,018`), **15% Validation** (`235`), and **20% Future Test** (`314` wafers, `17` true defects).
- **Preprocessing:** Designed to prevent preprocessing leakage by fitting `VarianceThreshold(0.0)`, `SimpleImputer(strategy="median")`, and `StandardScaler()` strictly inside the training pipeline.
- **Model & Threshold Selection:** Regularized `LogisticRegression(class_weight="balanced", C=0.05)` was selected based on PR-AUC. The operating threshold (**7%**) was chosen strictly on the chronological validation split before evaluating on the test set.

## 3. Results & Economic Formula (Held-Out Test Set, n = 314)
- **Test ROC-AUC:** `0.6922` (95% Bootstrap CI: `[0.5602, 0.8149]`)
- **Test PR-AUC:** `0.1063` (95% Bootstrap CI: `[0.0584, 0.2060]`) vs. Naive Predict-All-Pass Baseline (`0.0541`)
- **Explicit Cost Formula:** `Net Impact = (TP × Savings_TP) − (FP × Cost_FP) − (FN × Loss_FN)` compared against the do-nothing baseline (`-Total_Defects × Loss_FN`).

## 4. Limitations & Scope
- **Modest Performance & Noisy Dataset:** UCI SECOM is noisy with only `17` defects in the test slice, leading to wide confidence intervals and modest predictive power intended for decision-support screening.
- **Z-Score Interpretation:** Z-score bars indicate statistical deviation from batch norms rather than proven physical root cause.
- **Simulated Modules:** Economic parameters, spatial 300mm wafer coordinates, and thermal telemetry waveforms are simulated for prototype demonstration.

## 5. Run Locally
```bash
git clone https://github.com
cd siliconguard-ai
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```
