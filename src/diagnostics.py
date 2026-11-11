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
