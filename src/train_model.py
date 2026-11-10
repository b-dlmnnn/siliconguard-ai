import os
import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score

def build_and_train():
    print("1. UCI SECOM veri seti indiriliyor...")
    secom = fetch_ucirepo(id=179)
    df_full = secom.data.original.copy()

    target_col = [c for c in df_full.columns if 'pass' in c.lower() or 'fail' in c.lower() or 'class' in c.lower() or 'target' in c.lower()]
    if target_col:
        col_name = target_col[0]
        y = df_full[col_name].replace({-1: 0, 1: 1})
        X = df_full.drop(columns=[col_name])
    else:
        y = df_full.iloc[:, -1].replace({-1: 0, 1: 1})
        X = df_full.iloc[:, :-1]

    X = X.select_dtypes(include=[np.number])

    # %80 Egitim, %20 Test ayrimi (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 3),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 3),
        "confusion_matrix": cm.tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_defect_rate_pct": round(float(y.mean() * 100), 2)
    }

    importances = pipeline.named_steps['classifier'].feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    top_features = [(str(X.columns[i]), round(float(importances[i]), 4)) for i in top_indices]

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    model_artifact = {
        "pipeline": pipeline,
        "top_features": top_features,
        "feature_names": list(X.columns),
        "metrics": metrics
    }
    joblib.dump(model_artifact, "models/siliconguard_rf.pkl")

    # Test setinden 50 ornegi gercek etiketleriyle (Ground Truth) birlikte kaydet
    sample_test = X_test.head(50).copy()
    sample_test["Ground_Truth_Label"] = y_test.head(50).values
    sample_test.to_csv("data/sample_wafer_batch.csv", index=False)
    print("BASARILI: Model ve dogrulama metrikleri kaydedildi! ROC-AUC:", metrics["roc_auc"])

if __name__ == "__main__":
    build_and_train()
