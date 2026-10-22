
from pathlib import Path
import json

import numpy as np

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
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

    if len(np.unique(groups)) != 24:
        raise ValueError("Expected recordings from 24 actors.")

    cv = GroupKFold(n_splits=6)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf")),
    ])

    param_grid = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.001, 0.01, 0.1],
        "svc__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )

    print("Starting actor-independent SVM hyperparameter search...")
    print(f"Candidate configurations: {len(search.param_grid['svc__C']) * len(search.param_grid['svc__gamma']) * len(search.param_grid['svc__class_weight'])}")
    search.fit(X, y, groups=groups)

    results = []
    for index in search.cv_results_["rank_test_score"].argsort():
        results.append({
            "rank": int(search.cv_results_["rank_test_score"][index]),
            "mean_macro_f1": float(search.cv_results_["mean_test_score"][index]),
            "std_macro_f1": float(search.cv_results_["std_test_score"][index]),
            "params": {
                key: (
                    value.item() if isinstance(value, np.generic) else value
                )
                for key, value in search.cv_results_["params"][index].items()
            },
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "evaluation": "6-fold GroupKFold grouped by actor",
        "scoring": "macro F1",
        "candidate_configurations": len(search.cv_results_["params"]),
        "best_params": search.best_params_,
        "best_mean_cv_macro_f1": float(search.best_score_),
        "ranked_results": results,
    }

    output_path = RESULTS_DIR / "svm_tuning_results.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nBest parameters:")
    print(search.best_params_)
    print(f"Best mean CV macro F1: {search.best_score_:.4f}")
    print("\nTop 5 configurations:")
    for result in results[:5]:
        print(
            f"Rank {result['rank']}: "
            f"macro F1={result['mean_macro_f1']:.4f} "
            f"+/- {result['std_macro_f1']:.4f}; "
            f"{result['params']}"
        )

    print(f"\nFull results saved to: {output_path}")
    print(
        "\nNote: This search score is for model selection, not an "
        "independent final performance estimate."
    )


if __name__ == "__main__":
    main()
