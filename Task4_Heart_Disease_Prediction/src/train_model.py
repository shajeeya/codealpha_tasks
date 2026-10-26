
"""Train and evaluate heart disease classification models."""

import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed.cleveland.data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def load_dataset():
    """Load and validate the Cleveland heart disease dataset."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place processed.cleveland.data in the project's data folder."
        )

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values="?",
    )

    if df.shape != (303, 14):
        raise ValueError(f"Unexpected dataset shape: {df.shape}")

    if df["target"].isna().any():
        raise ValueError("The target column contains missing values.")

    if not df["target"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Unexpected target values in the dataset.")

    X = df.drop(columns="target")
    y = (df["target"] > 0).astype(int)

    return X, y


def build_models():
    """Return candidate models with preprocessing in each pipeline."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }


def main():
    """Train, compare, evaluate, and save the selected model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []
    fitted_models = {}

    for name, classifier in build_models().items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc",
            },
            n_jobs=1,
            error_score="raise",
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        results.append({
            "model": name,
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "cv_balanced_accuracy_mean": float(
                scores["test_balanced_accuracy"].mean()
            ),
            "cv_f1_mean": float(scores["test_f1"].mean()),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_balanced_accuracy": float(
                balanced_accuracy_score(y_test, predictions)
            ),
            "test_precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "test_recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "test_f1": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "test_roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
        })

        fitted_models[name] = pipeline

    results_df = pd.DataFrame(results).sort_values(
        "cv_f1_mean",
        ascending=False,
    )

    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    best_name = str(results_df.iloc[0]["model"])
    best_model = fitted_models[best_name]
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    joblib.dump(best_model, MODEL_DIR / "heart_disease_model.joblib")

    metadata = {
        "model_name": best_name,
        "feature_names": list(X.columns),
        "target_mapping": {
            "0": "No disease",
            "1-4": "Disease present",
        },
        "dataset": "UCI Heart Disease - Cleveland",
        "test_size": 0.20,
        "random_state": 42,
        "selection_metric": "Mean cross-validation F1-score",
        "note": (
            "Educational model only; not validated for clinical diagnosis."
        ),
    }

    (RESULTS_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions,
        "probability_disease_present": probabilities,
    }).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    ConfusionMatrixDisplay(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ),
        display_labels=["No disease", "Disease present"],
    ).plot(cmap="Blues")

    plt.title(f"Heart Disease Prediction — {best_name}")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No disease", "Disease present"],
        zero_division=0,
        output_dict=True,
    )

    (RESULTS_DIR / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nCross-validation and test results:")
    print(results_df.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print("\nHeld-out test classification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No disease", "Disease present"],
        zero_division=0,
    ))
    print(f"\nModel saved to: {MODEL_DIR / 'heart_disease_model.joblib'}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
