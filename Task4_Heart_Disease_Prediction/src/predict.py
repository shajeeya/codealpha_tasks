
"""Predict heart disease status using a saved classification model."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "heart_disease_model.joblib"

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def predict_heart_disease(values: list[float]) -> tuple[int, float]:
    """Return the binary prediction and estimated disease probability."""
    if len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} feature values; got {len(values)}."
        )

    if not np.isfinite(values).all():
        raise ValueError("All provided values must be finite numbers.")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run src/train_model.py first."
        )

    model = joblib.load(MODEL_PATH)

    features = pd.DataFrame([values], columns=FEATURE_NAMES)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, 1])

    return prediction, probability


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict binary heart disease status from 13 features."
    )
    parser.add_argument(
        "features",
        nargs=13,
        type=float,
        metavar="VALUE",
        help="13 values in the documented feature order.",
    )
    args = parser.parse_args()

    try:
        prediction, probability = predict_heart_disease(args.features)
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))

    label = "Disease present" if prediction == 1 else "No disease detected"
    print(f"Prediction: {label}")
    print(f"Estimated model probability of disease: {probability:.2%}")
    print(
        "Disclaimer: Educational demonstration only; "
        "not a medical diagnosis."
    )


if __name__ == "__main__":
    main()
