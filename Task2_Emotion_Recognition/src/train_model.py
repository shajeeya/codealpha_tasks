
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.exceptions import FitFailedWarning


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42
N_SPLITS = 6


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = DATA_DIR / "ravdess_features.npz"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}. "
            "Run prepare_dataset.py first."
        )

    with np.load(dataset_path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(str)
        groups = data["groups"].astype(int)

    if X.shape != (1440, 80):
        raise ValueError(f"Unexpected feature matrix shape: {X.shape}")

    if len(y) != len(X) or len(groups) != len(X):
        raise ValueError("Feature, label, and actor counts do not match.")

    if len(np.unique(y)) != 8:
        raise ValueError("Expected exactly 8 emotion classes.")

    if len(np.unique(groups)) != 24:
        raise ValueError("Expected exactly 24 actors.")

    if not np.isfinite(X).all():
        raise ValueError("Features contain non-finite values.")

    classes = np.array(sorted(np.unique(y)))
    cv = GroupKFold(n_splits=N_SPLITS)

    # Each actor appears in exactly one test fold.
    # The same actor can never appear in both train and test for a fold.
    for train_idx, test_idx in cv.split(X, y, groups):
        if set(groups[train_idx]) & set(groups[test_idx]):
            raise RuntimeError("Actor leakage detected in cross-validation.")

        if set(y[test_idx]) != set(classes):
            raise RuntimeError(
                "A validation fold is missing an emotion class. "
                "Review the group split before training."
            )

    models = {
        "SVM_RBF": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced"),
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra_Trees": ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    }

    cv_results = {}
    print(f"Dataset: {X.shape[0]} recordings, {X.shape[1]} features")
    print(f"Evaluation: {N_SPLITS}-fold actor-independent cross-validation")
    print("Comparing classifiers...\n")

    with warnings.catch_warnings():
        warnings.simplefilter("error", FitFailedWarning)

        for name, estimator in models.items():
            print(f"Evaluating {name}...")

            scores = cross_validate(
                estimator,
                X,
                y,
                groups=groups,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
                return_train_score=False,
            )

            cv_results[name] = {
                metric: float(np.mean(scores[f"test_{metric}"]))
                for metric in scoring
            }
            cv_results[name].update(
                {
                    f"{metric}_std": float(np.std(scores[f"test_{metric}"]))
                    for metric in scoring
                }
            )

            print(
                f"  Accuracy:          "
                f"{cv_results[name]['accuracy']:.4f}"
            )
            print(
                f"  Balanced accuracy: "
                f"{cv_results[name]['balanced_accuracy']:.4f}"
            )
            print(
                f"  Macro F1:           "
                f"{cv_results[name]['macro_f1']:.4f}\n"
            )

    # Select the model with the highest mean actor-independent macro F1.
    best_name = max(cv_results, key=lambda name: cv_results[name]["macro_f1"])
    best_cv_score = cv_results[best_name]["macro_f1"]

    print(f"Best cross-validation model: {best_name}")
    print(f"Mean cross-validation macro F1: {best_cv_score:.4f}")

    # Produce out-of-fold predictions: every prediction is made by a model
    # that did not train on recordings from that actor.
    best_model = clone(models[best_name])
    oof_predictions = np.empty_like(y)

    for train_idx, test_idx in cv.split(X, y, groups):
        fold_model = clone(best_model)
        fold_model.fit(X[train_idx], y[train_idx])
        oof_predictions[test_idx] = fold_model.predict(X[test_idx])

    report = classification_report(
        y,
        oof_predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y, oof_predictions, labels=classes)

    # Save detailed cross-validation scores and out-of-fold evaluation.
    summary = {
        "evaluation": "6-fold GroupKFold grouped by actor",
        "random_state": RANDOM_STATE,
        "best_model": best_name,
        "best_mean_cv_macro_f1": float(best_cv_score),
        "cross_validation_results": cv_results,
        "out_of_fold_accuracy": float(accuracy_score(y, oof_predictions)),
        "out_of_fold_balanced_accuracy": float(
            balanced_accuracy_score(y, oof_predictions)
        ),
        "out_of_fold_macro_f1": float(
            f1_score(y, oof_predictions, average="macro")
        ),
        "classification_report": report,
        "class_labels": classes.tolist(),
        "confusion_matrix": matrix.tolist(),
    }

    (RESULTS_DIR / "evaluation.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    np.savez_compressed(
        RESULTS_DIR / "out_of_fold_predictions.npz",
        y_true=y,
        y_pred=oof_predictions,
        actor_ids=groups,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )
    display.plot(xticks_rotation=45, values_format="d")
    plt.title(f"Actor-independent OOF confusion matrix: {best_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # Train the selected model on all recordings for later inference.
    # Its reported performance remains the cross-validation estimate above.
    final_model = clone(models[best_name])
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_DIR / "emotion_classifier.joblib")

    print("\nOut-of-fold evaluation:")
    print(classification_report(y, oof_predictions, labels=classes, zero_division=0))
    print(f"Out-of-fold accuracy: {accuracy_score(y, oof_predictions):.4f}")
    print(
        "Out-of-fold balanced accuracy: "
        f"{balanced_accuracy_score(y, oof_predictions):.4f}"
    )
    print(
        "Out-of-fold macro F1: "
        f"{f1_score(y, oof_predictions, average='macro'):.4f}"
    )

    print("\nSaved files:")
    print(f"  Model: {MODEL_DIR / 'emotion_classifier.joblib'}")
    print(f"  Metrics: {RESULTS_DIR / 'evaluation.json'}")
    print(f"  Confusion matrix: {RESULTS_DIR / 'confusion_matrix.png'}")
    print(f"  OOF predictions: {RESULTS_DIR / 'out_of_fold_predictions.npz'}")


if __name__ == "__main__":
    main()
