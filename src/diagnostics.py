import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def calculate_psi(expected, actual, buckets=10):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    quantiles = np.linspace(0, 100, buckets + 1)
    bins = np.unique(np.percentile(expected, quantiles))
    if len(bins) < 3:
        return 0.0

    bins[0] = -np.inf
    bins[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = np.clip(exp_counts / len(expected), 1e-4, 1.0)
    act_pct = np.clip(act_counts / len(actual), 1e-4, 1.0)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return round(float(psi), 4)


def compute_calibration_data(y_true, y_probs, n_bins=6):
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({
        "Mean_Predicted_Prob": np.round(prob_pred * 100, 2),
        "Observed_Defect_Rate": np.round(prob_true * 100, 2),
    })


def compute_net_impact(y_true, y_pred, cost_tp=3500, cost_fp=400, cost_fn=4500):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return (tp * cost_tp) - (fp * cost_fp) - (fn * cost_fn)
import shap

def compute_shap_explanation(pipeline, X_sample, feature_names):
    """
    Pipeline icindeki son tahmin modeli icin SHAP degerlerini hesaplar.
    X_sample: Ham veya secilmis sensör ozelliklerini iceren DataFrame/array.
    """
    # 1. Veriyi pipeline'in on isleme adimlarindan (Imputer, Scaler vb.) gecir
    preprocessor = pipeline[:-1]
    model = pipeline[-1]
    X_transformed = preprocessor.transform(X_sample)
    
    # 2. Veri boyutunu guvenli sekilde sinirla (Hizli calismasi icin en fazla 100 satir)
    if len(X_transformed) > 100:
        X_transformed = X_transformed[:100]
        
    # 3. Modele uygun SHAP aciklayiciyi (Explainer) olustur ve degerleri hesapla
    explainer = shap.Explainer(model, X_transformed)
    shap_values = explainer(X_transformed)
    
    # Ikili siniflandirmada (0: Saglam, 1: Hatali) sadece Hatali (1) sinifinin etkisini al
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
        
    # Ortalama mutlak SHAP etkisine gore en onemli 10 sensoru sirala
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Eger VarianceThreshold ozellik sayisini azalttiysa aktif isimleri eslestir
    if hasattr(preprocessor, "get_feature_names_out"):
        active_names = preprocessor.get_feature_names_out(feature_names)
    else:
               active_names = [f"Feature_{i}" for i in range(X_transformed.shape[1])]
        
    importance_df = pd.DataFrame({
        "Sensor": active_names,
        "Mean_SHAP_Impact": mean_abs_shap
    }).sort_values(by="Mean_SHAP_Impact", ascending=False).head(10)
    
    return shap_values, importance_df
def monitor_sensor_drift(reference_df, current_df, sensor_list):
    """
    Kritik sensorlerin referans (egitim) dagilimi ile yeni gelen batch arasindaki
    PSI (Population Stability Index) kaymasini hesaplar.
    PSI < 0.10: Stabil | 0.10 - 0.25: Hafif Kayma | > 0.25: Kritik Kayma (Yeniden Egitim Gerekli)
    """
    drift_records = []
    for sensor in sensor_list:
        if sensor in reference_df.columns and sensor in current_df.columns:
            psi_val = calculate_psi(reference_df[sensor].values, current_df[sensor].values)
            if psi_val < 0.10:
                status = "🟢 Stable"
            elif psi_val <= 0.25:
                status = "🟡 Moderate Drift"
            else:
                status = "🔴 Critical Drift"
            drift_records.append({
                "Sensor": sensor,
                "PSI_Score": round(psi_val, 4),
                "Drift_Status": status
            })
    return pd.DataFrame(drift_records).sort_values(by="PSI_Score", ascending=False)

