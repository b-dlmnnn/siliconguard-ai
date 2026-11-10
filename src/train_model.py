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


def build_and_train():
    print("1. UCI SECOM veri seti indiriliyor (ucimlrepo id=179)...")
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

    # Sadece sayisal sensor sutunlarini koru
    X = X.select_dtypes(include=[np.number]).copy()
    X.columns = [f"Attribute {i}" if str(c).isdigit() else str(c) for i, c in enumerate(X.columns)]

    # 2. Zamana Gore Bolme (Chronological Split: ilk %80 egitim, son %20 gelecek test)
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    # 3. Sizinti (Leakage) Onleyici On Isleme Pipeline'i
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

    # Temel Cizgi (Baseline: Hepsini Saglam Say)
    baseline_acc = accuracy_score(y_test, np.zeros_like(y_test))
    baseline_pr_auc = float(y_test.mean())
    comparison_rows.append({
        "Model": "Baseline (Predict All Pass)",
        "Accuracy": round(baseline_acc, 4),
        "ROC_AUC": 0.5000,
        "PR_AUC": round(baseline_pr_auc, 4),
    })
    best_name = "Logistic Regression (Balanced)"
    for name, clf in candidates.items():
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.50).astype(int)

        comparison_rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "ROC_AUC": round(roc_auc_score(y_test, probs), 4),
            "PR_AUC": round(average_precision_score(y_test, probs), 4),
        })
        trained_pipelines[name] = pipe

    comparison_df = pd.DataFrame(comparison_rows)

    # Ana model secimi: Random Forest (Balanced)
    best_name = "Random Forest (Balanced)"
    best_pipe = trained_pipelines[best_name]
    y_probs = best_pipe.predict_proba(X_test)[:, 1]
    y_pred_default = (y_probs >= 0.50).astype(int)

    # 4. Permutasyon Onemi (Permutation Importance)
    print("2. Permutasyon onem skorlari hesaplaniyor...")
    perm = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=5, scoring="average_precision", random_state=42, n_jobs=-1
    )
    top_idx = np.argsort(perm.importances_mean)[::-1][:10]
    top_features = pd.DataFrame({
        "Sensor_Feature": X.columns[top_idx],
        "Importance_Score": perm.importances_mean[top_idx],
    })

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_probs), 4),
        "pr_auc": round(average_precision_score(y_test, y_probs), 4),
        "baseline_acc": round(baseline_acc, 4),
        "baseline_pr_auc": round(baseline_pr_auc, 4),
        "precision_50": round(precision_score(y_test, y_pred_default, zero_division=0), 4),
        "recall_50": round(recall_score(y_test, y_pred_default, zero_division=0), 4),
        "confusion_matrix_50": confusion_matrix(y_test, y_pred_default).tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "defect_rate_pct": round(float(y.mean() * 100), 2),
        "split_method": "Chronological Split (First 80% Train / Last 20% Future Test)",
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

    # 5. Tum Test Setini (314 Wafer) Kaydet
    full_test = X_test.copy()
    full_test["Ground_Truth_Label"] = y_test.values
    full_test.to_csv("data/sample_wafer_batch.csv", index=False)

    print("\nBASARILI! V2 Bilimsel Model ve Tum Test Seti (314 wafer) kaydedildi.")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    build_and_train()
