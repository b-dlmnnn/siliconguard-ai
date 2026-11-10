import os
import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
)


def bootstrap_ci(y_true, y_scores, metric_fn, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric_fn(y_true[idx], y_scores[idx]))
    low, high = np.percentile(stats, [2.5, 97.5])
    return round(float(low), 4), round(float(high), 4)


def build_and_train():
    print("1. UCI SECOM veri seti yukleniyor (ucimlrepo id=179)...")
    secom = fetch_ucirepo(id=179)
    df_full = secom.data.original.copy()

    target_col = [
        c for c in df_full.columns
        if "pass" in c.lower() or "fail" in c.lower() or "class" in c.lower() or "target" in c.lower()
    ]
    if target_col:
        col_name = target_col[0]
        y = df_full[col_name].replace({-1: 0, 1: 1}).astype(int)
        X = df_full.drop(columns=[col_name]).copy()
    else:
        y = df_full.iloc[:, -1].replace({-1: 0, 1: 1}).astype(int)
        X = df_full.iloc[:, :-1].copy()

    X = X.select_dtypes(include=[np.number]).copy()
    X.columns = [f"Attribute {i}" if str(c).isdigit() else str(c) for i, c in enumerate(X.columns)]

    # 1. Kronolojik Train (%65) / Validation (%15) / Test (%20) Bolmesi
    n_total = len(X)
    train_end = int(n_total * 0.65)
    val_end = int(n_total * 0.80)

    X_train, y_train = X.iloc[:train_end].copy(), y.iloc[:train_end].copy()
    X_val, y_val = X.iloc[train_end:val_end].copy(), y.iloc[train_end:val_end].copy()
    X_test, y_test = X.iloc[val_end:].copy(), y.iloc[val_end:].copy()

    # 2. On Isleme Sizintisini Onlemek Icin Tasarlanmis Pipeline
    def make_pipeline(clf):
        return Pipeline([
            ("var_thresh", VarianceThreshold(threshold=0.0)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clf),
        ])

    candidates = {
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced", max_iter=1000, C=0.05, random_state=42
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    comparison_rows = []
    trained_pipelines = {}

    baseline_acc = accuracy_score(y_test, np.zeros_like(y_test))
    baseline_pr_auc = float(y_test.mean())
    comparison_rows.append({
        "Model": "Baseline (Predict All Pass)",
        "Test_Accuracy": round(baseline_acc, 4),
        "Test_ROC_AUC": 0.5000,
        "Test_PR_AUC": round(baseline_pr_auc, 4),
    })

    for name, clf in candidates.items():
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.50).astype(int)

        comparison_rows.append({
            "Model": name,
            "Test_Accuracy": round(accuracy_score(y_test, preds), 4),
            "Test_ROC_AUC": round(roc_auc_score(y_test, probs), 4),
            "Test_PR_AUC": round(average_precision_score(y_test, probs), 4),
        })
        trained_pipelines[name] = pipe

    comparison_df = pd.DataFrame(comparison_rows)

    best_name = "Logistic Regression (Balanced)"
    best_pipe = trained_pipelines[best_name]

    # 3. Esik Secimi SADECE Validation Setinde Yapilir (Test Setine Bakilmaz)
    val_probs = best_pipe.predict_proba(X_val)[:, 1]
    thresh_grid = np.linspace(0.05, 0.95, 91)
    best_val_t = 0.25
    best_val_net = -1e18
    for t in thresh_grid:
        p_v = (val_probs >= t).astype(int)
        tn_v, fp_v, fn_v, tp_v = confusion_matrix(y_val, p_v, labels=[0, 1]).ravel()
        net_v = (tp_v * 3500) - (fp_v * 400) - (fn_v * 4500)
        if net_v > best_val_net:
            best_val_net = net_v
            best_val_t = round(float(t), 2)

    # 4. Nihai Test Seti Degerlendirmesi ve %95 Bootstrap Guven Araliklari
    y_probs = best_pipe.predict_proba(X_test)[:, 1]
    y_pred_val_thresh = (y_probs >= best_val_t).astype(int)
    roc_ci = bootstrap_ci(y_test, y_probs, roc_auc_score)
    pr_ci = bootstrap_ci(y_test, y_probs, average_precision_score)

    print("2. Permutasyon onem skorlari hesaplaniyor (Validation seti uzerinde)...")
    perm = permutation_importance(
        best_pipe, X_val, y_val, n_repeats=5, scoring="average_precision", random_state=42, n_jobs=-1
    )
    top_idx = np.argsort(perm.importances_mean)[::-1][:10]
    top_features = pd.DataFrame({
        "Sensor_Feature": X.columns[top_idx],
        "Importance_Score": perm.importances_mean[top_idx],
    })

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_probs), 4),
        "roc_ci": roc_ci,
        "pr_auc": round(average_precision_score(y_test, y_probs), 4),
        "pr_ci": pr_ci,
        "val_selected_threshold": best_val_t,
        "baseline_acc": round(baseline_acc, 4),
        "baseline_pr_auc": round(baseline_pr_auc, 4),
        "precision_at_val_t": round(precision_score(y_test, y_pred_val_thresh, zero_division=0), 4),
        "recall_at_val_t": round(recall_score(y_test, y_pred_val_thresh, zero_division=0), 4),
        "confusion_matrix_test": confusion_matrix(y_test, y_pred_val_thresh, labels=[0, 1]).tolist(),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "test_defects": int(y_test.sum()),
        "defect_rate_pct": round(float(y.mean() * 100), 2),
        "split_method": "Chronological Split (65% Train / 15% Validation / 20% Future Test)",
        "model_comparison": comparison_df.to_dict(orient="records"),
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    artifact = {
        "pipeline": best_pipe,
        "top_features": top_features,
        "metrics": metrics,
        "feature_names": list(X.columns),
    }
    joblib.dump(artifact, "models/siliconguard_rf.pkl")

    full_test = X_test.copy()
    full_test["Ground_Truth_Label"] = y_test.values
    full_test.to_csv("data/sample_wafer_batch.csv", index=False)

    print(f"\nBASARILI! Validation Setinde Secilen Optimal Esik: %{int(best_val_t*100)}")
    print(f"Test ROC-AUC: {metrics['roc_auc']} (95% CI: {roc_ci})")
    print(f"Test PR-AUC:  {metrics['pr_auc']} (95% CI: {pr_ci})")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    build_and_train()
