import os
import joblib
import numpy as np
import pandas as pd
from src.diagnostics import calculate_psi, compute_net_impact


def test_artifact_and_data_exist():
    assert os.path.exists("models/siliconguard_v3.pkl"), "Trained V3 model artifact missing"
    assert os.path.exists("data/sample_wafer_batch.csv"), "Test wafer batch CSV missing"


def test_chronological_test_shape():
    artifact = joblib.load("models/siliconguard_v3.pkl")
    df = pd.read_csv("data/sample_wafer_batch.csv")
    assert len(df) == 314, f"Expected 314 future test wafers, got {len(df)}"
    for col in artifact["feature_names"][:10]:
        assert col in df.columns


def test_cost_formula_accuracy():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    net = compute_net_impact(y_true, y_pred, cost_tp=3500, cost_fp=400, cost_fn=4500)
    assert net == (3500 - 400 - 4500)


def test_psi_drift_calculation():
    rng = np.random.RandomState(42)
    base = rng.normal(0, 1, 500)
    same = rng.normal(0, 1, 500)
    shifted = rng.normal(2.5, 1, 500)
    assert calculate_psi(base, same) < 0.10
    assert calculate_psi(base, shifted) > 0.25
