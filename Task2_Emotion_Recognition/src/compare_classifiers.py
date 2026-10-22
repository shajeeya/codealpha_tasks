
from pathlib import Path
import json

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ravdess_features.npz"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    with np.load(DATA_PATH, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(str)
        groups = data["groups"].astype(int)

    if X.shape != (1440, 80):
        raise ValueError(f"Unexpected feature matrix shape: {X.shape}")

    cv = GroupKFold(n_splits=6)

    models = {
        "SVM_RBF": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1,
                gamma="scale",
                class_weight="balanced",
            ),
        ),
        "SVM_Linear": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="linear",
                C=1,
                class_weight="balanced",
            ),
        ),
        "Logistic_Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1,
                class_weight="balanced",
                max_iter=3000,
                random_state=42,
            ),
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")

        scores = cross_validate(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "macro_f1": "f1_macro",
            },
            n_jobs=1,
            error_score="raise",
        )

        results[name] = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"])),
                "fold_scores": scores[f"test_{metric}"].tolist(),
            }
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }

        for metric, values in results[name].items():
            print(
                f"  {metric}: {values['mean']:.4f} "
                f"+/- {values['std']:.4f}"
            )
        print()

    best_name = max(
        results,
        key=lambda name: results[name]["macro_f1"]["mean"],
    )

    output = {
        "evaluation": "6-fold GroupKFold grouped by actor",
        "dataset": "Original 80-feature MFCC dataset",
        "best_model_by_mean_macro_f1": best_name,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "classifier_comparison.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("Best model by mean macro F1:", best_name)
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()
