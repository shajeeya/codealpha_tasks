
from pathlib import Path
import json

import numpy as np

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_dataset(filename):
    path = DATA_DIR / filename

    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(str)
        groups = data["groups"].astype(int)

    if len(X) != 1440 or len(y) != 1440 or len(groups) != 1440:
        raise ValueError(f"Unexpected number of recordings in {filename}.")

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite features found in {filename}.")

    return X, y, groups


def evaluate(X, y, groups, cv):
    model = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1,
            gamma="scale",
            class_weight="balanced",
        ),
    )

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

    return {
        metric: {
            "mean": float(np.mean(scores[f"test_{metric}"])),
            "std": float(np.std(scores[f"test_{metric}"])),
            "fold_scores": scores[f"test_{metric}"].tolist(),
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
    }


def main():
    X_base, y_base, groups_base = load_dataset("ravdess_features.npz")
    X_enhanced, y_enhanced, groups_enhanced = load_dataset(
        "ravdess_enhanced_features.npz"
    )

    if not (
        np.array_equal(y_base, y_enhanced)
        and np.array_equal(groups_base, groups_enhanced)
    ):
        raise ValueError("The datasets have different labels or actor IDs.")

    cv = GroupKFold(n_splits=6)

    for X, y, groups in [
        (X_base, y_base, groups_base),
        (X_enhanced, y_enhanced, groups_enhanced),
    ]:
        for train_idx, test_idx in cv.split(X, y, groups):
            if set(groups[train_idx]) & set(groups[test_idx]):
                raise RuntimeError("Actor leakage detected.")

    print("Comparing the same SVM configuration on both feature sets.")
    print("Evaluation: six-fold actor-independent GroupKFold\n")

    results = {
        "evaluation": "6-fold GroupKFold grouped by actor",
        "classifier": "SVM RBF, C=1, gamma=scale, class_weight=balanced",
        "original_80_features": evaluate(X_base, y_base, groups_base, cv),
        "enhanced_174_features": evaluate(
            X_enhanced, y_enhanced, groups_enhanced, cv
        ),
    }

    for name in ("original_80_features", "enhanced_174_features"):
        print(name)
        for metric, values in results[name].items():
            print(
                f"  {metric}: {values['mean']:.4f} "
                f"+/- {values['std']:.4f}"
            )
        print()

    old_f1 = results["original_80_features"]["macro_f1"]["mean"]
    new_f1 = results["enhanced_174_features"]["macro_f1"]["mean"]

    results["macro_f1_difference_enhanced_minus_original"] = new_f1 - old_f1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "feature_comparison.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Macro F1 difference (enhanced - original): {new_f1 - old_f1:+.4f}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
