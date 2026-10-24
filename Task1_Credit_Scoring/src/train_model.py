
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "default of credit card clients.xls"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "default payment next month"
RANDOM_STATE = 42


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    # Load and validate the dataset.
    df = pd.read_excel(DATA_PATH, header=1)
    df.columns = df.columns.astype(str).str.strip()

    if TARGET not in df.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])

    # Exclude the identifier to prevent it influencing predictions.
    X = df.drop(columns=[TARGET, "ID"], errors="ignore")
    y = df[TARGET].astype(int)

    # Treat coded demographic fields as categorical.
    categorical_cols = [
        col for col in ["SEX", "EDUCATION", "MARRIAGE"]
        if col in X.columns
    ]
    numeric_cols = [
        col for col in X.columns if col not in categorical_cols
    ]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_cols,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    fitted_models = {}

    for name, classifier in candidates.items():
        pipeline = Pipeline([
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

        results.append(metrics)
        fitted_models[name] = pipeline

        print(f"\n{'=' * 55}")
        print(name)
        print("=" * 55)
        print(json.dumps(metrics, indent=2))
        print(classification_report(
            y_test,
            predictions,
            target_names=["No Default", "Default"],
            zero_division=0,
        ))

    # Select by ROC-AUC, with F1-score as a tie-breaker.
    results.sort(
        key=lambda item: (item["roc_auc"], item["f1_score"]),
        reverse=True,
    )
    best_name = results[0]["model"]
    best_model = fitted_models[best_name]
    best_predictions = best_model.predict(X_test)
    best_probabilities = best_model.predict_proba(X_test)[:, 1]

    # Save all evaluation results.
    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump({
            "dataset": DATA_PATH.name,
            "target": TARGET,
            "rows_used": int(len(df)),
            "features_used": list(X.columns),
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
            "selection_metric": "ROC-AUC, with F1-score as tie-breaker",
            "best_model": best_name,
            "results": results,
        }, file, indent=4)

    pd.DataFrame(results).to_csv(
        REPORT_DIR / "model_comparison.csv", index=False
    )

    joblib.dump(best_model, MODEL_DIR / "credit_scoring_model.joblib")

    # Save confusion matrix.
    cm = confusion_matrix(y_test, best_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Save ROC curve.
    RocCurveDisplay.from_predictions(y_test, best_probabilities)
    plt.title(f"ROC Curve — {best_name}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "roc_curve.png", dpi=150)
    plt.close()

    print(f"\nBest model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'credit_scoring_model.joblib'}")
    print(f"Saved reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()
