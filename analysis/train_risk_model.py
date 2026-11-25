"""
Train the fire-risk model used by the website.

Data: ouladsayadyounes/WildFires - NDVI, land surface temperature (LST),
burned area and a fire / no_fire label, derived from MODIS (Terra) imagery
over Central Canada's forest region.

The model is a two-feature logistic regression (NDVI + LST). Its
coefficients and the scaler parameters are written to site/model_results.json
so the static page can compute the same probability in JavaScript.

Run from the repo root:  python analysis/train_risk_model.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WildFires_DataSet.csv"
OUT_PATH = ROOT / "site" / "model_results.json"
FEATURES = ["NDVI", "LST"]


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=FEATURES + ["CLASS"])
    df["label"] = (df["CLASS"].str.strip() == "fire").astype(int)
    print(f"{len(df)} rows, fire share = {df['label'].mean():.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES].values, df["label"].values,
        test_size=0.25, random_state=42, stratify=df["label"],
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    # Only ~23% of rows are fires. Without class weights the model predicts
    # "no fire" almost every time (recall was 0.13 in my first run), which is
    # useless for an early-warning tool. Balanced weights trade accuracy and
    # precision for recall.
    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    for k, v in metrics.items():
        print(f"{k:>16}: {v}")

    results = {
        "data_source": "https://github.com/ouladsayadyounes/WildFires",
        "n_samples_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_balance": {
            "fire": int(df["label"].sum()),
            "no_fire": int((df["label"] == 0).sum()),
        },
        "metrics": metrics,
        "model": {
            "type": "logistic_regression",
            "features": FEATURES,
            "coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        # used for the slider limits on the page
        "feature_ranges": {
            "ndvi": {"min": float(df["NDVI"].min()), "max": float(df["NDVI"].max())},
            "lst": {"min": float(df["LST"].min()), "max": float(df["LST"].max())},
        },
    }
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
