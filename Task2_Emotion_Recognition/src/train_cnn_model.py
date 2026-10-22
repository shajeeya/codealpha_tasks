
"""Train and evaluate a CNN on RAVDESS MFCC sequences."""

from pathlib import Path
import json
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder


SEED = 42
EPOCHS = 35
BATCH_SIZE = 32
N_SPLITS = 5

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "ravdess_mfcc_sequences.npz"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_cnn(input_shape, num_classes):
    """Build a compact CNN for MFCC time-frequency features."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(
            16, (3, 3), padding="same", activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(
            32, (3, 3), padding="same", activation="relu"
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(
            64, (3, 3), padding="same", activation="relu"
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    set_seed()

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_PATH}\n"
            "Run prepare_cnn_dataset.py first."
        )

    with np.load(DATA_PATH, allow_pickle=False) as dataset:
        X = dataset["X"].astype(np.float32)
        labels = dataset["y"].astype(str)
        groups = dataset["groups"].astype(np.int16)

    if X.ndim != 3 or len(X) != len(labels) or len(X) != len(groups):
        raise ValueError("Dataset arrays have inconsistent shapes.")
    if not np.isfinite(X).all():
        raise ValueError("Dataset contains NaN or infinite feature values.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    class_names = encoder.classes_.tolist()

    if len(class_names) != 8:
        raise ValueError(
            f"Expected 8 emotion classes; found {len(class_names)}."
        )

    # CNN input: samples x MFCC coefficients x time frames x channels.
    X = X[..., np.newaxis]
    n_classes = len(class_names)
    splitter = GroupKFold(n_splits=N_SPLITS)

    oof_probabilities = np.zeros(
        (len(X), n_classes), dtype=np.float32
    )
    fold_results = []

    print(f"Samples: {len(X)}")
    print(f"Actors: {len(np.unique(groups))}")
    print(f"Classes: {class_names}")
    print(f"Input shape: {X.shape[1:]}")
    print(f"Cross-validation folds: {N_SPLITS}")

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        set_seed(SEED + fold)

        train_actors = set(groups[train_idx].tolist())
        val_actors = set(groups[val_idx].tolist())
        if train_actors & val_actors:
            raise RuntimeError("Actor leakage detected between train and validation.")

        # Fit normalization using training recordings only.
        mean = X[train_idx].mean(axis=(0, 1, 2), keepdims=True)
        std = X[train_idx].std(axis=(0, 1, 2), keepdims=True)
        std = np.maximum(std, 1e-6)

        X_train = (X[train_idx] - mean) / std
        X_val = (X[val_idx] - mean) / std

        tf.keras.backend.clear_session()
        model = build_cnn(X.shape[1:], n_classes)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
                verbose=0,
            ),
        ]

        print(
            f"\nFold {fold}/{N_SPLITS}: "
            f"train={len(train_idx)}, validation={len(val_idx)}, "
            f"held-out actors={sorted(val_actors)}"
        )

        model.fit(
            X_train,
            y[train_idx],
            validation_data=(X_val, y[val_idx]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,
        )

        probabilities = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
        oof_probabilities[val_idx] = probabilities
        predictions = probabilities.argmax(axis=1)

        metrics = {
            "fold": fold,
            "validation_samples": int(len(val_idx)),
            "held_out_actors": sorted(val_actors),
            "accuracy": float(accuracy_score(y[val_idx], predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y[val_idx], predictions)
            ),
            "macro_f1": float(
                f1_score(
                    y[val_idx],
                    predictions,
                    labels=np.arange(n_classes),
                    average="macro",
                    zero_division=0,
                )
            ),
            "best_validation_loss": float(
                min(model.history.history["val_loss"])
            ),
        }
        fold_results.append(metrics)

        print(
            f"Fold {fold} metrics: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

        del model, X_train, X_val

    # Every sample has one prediction from a model that did not train on its actor.
    oof_predictions = oof_probabilities.argmax(axis=1)
    overall = {
        "accuracy": float(accuracy_score(y, oof_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, oof_predictions)),
        "macro_f1": float(
            f1_score(
                y,
                oof_predictions,
                labels=np.arange(n_classes),
                average="macro",
                zero_division=0,
            )
        ),
        "fold_accuracy_mean": float(
            np.mean([item["accuracy"] for item in fold_results])
        ),
        "fold_accuracy_std": float(
            np.std([item["accuracy"] for item in fold_results])
        ),
        "fold_macro_f1_mean": float(
            np.mean([item["macro_f1"] for item in fold_results])
        ),
        "fold_macro_f1_std": float(
            np.std([item["macro_f1"] for item in fold_results])
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        RESULTS_DIR / "cnn_out_of_fold_predictions.npz",
        y_true=y,
        y_pred=oof_predictions,
        probabilities=oof_probabilities,
        groups=groups,
        class_names=np.asarray(class_names, dtype=str),
    )

    report = classification_report(
        y,
        oof_predictions,
        labels=np.arange(n_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    result_data = {
        "model": "CNN",
        "dataset": "RAVDESS",
        "samples": int(len(X)),
        "actors": int(len(np.unique(groups))),
        "classes": class_names,
        "feature_shape": list(X.shape[1:]),
        "normalization": "Training-fold mean/std only",
        "cross_validation": "5-fold GroupKFold by actor",
        "seed": SEED,
        "epochs_max": EPOCHS,
        "batch_size": BATCH_SIZE,
        "folds": fold_results,
        "overall_out_of_fold_metrics": overall,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            y, oof_predictions, labels=np.arange(n_classes)
        ).tolist(),
    }

    with (RESULTS_DIR / "cnn_evaluation.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result_data, file, indent=2)

    # Train the final deployable model on all recordings.
    # Its performance is NOT reported as held-out evaluation.
    final_mean = X.mean(axis=(0, 1, 2), keepdims=True)
    final_std = np.maximum(
        X.std(axis=(0, 1, 2), keepdims=True), 1e-6
    )
    X_all = (X - final_mean) / final_std

    tf.keras.backend.clear_session()
    set_seed(SEED)
    final_model = build_cnn(X.shape[1:], n_classes)
    final_model.fit(
        X_all,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=5, restore_best_weights=True
            )
        ],
        verbose=2,
    )

    final_model.save(MODELS_DIR / "emotion_cnn.keras")
    np.savez(
        MODELS_DIR / "emotion_cnn_normalization.npz",
        mean=final_mean,
        std=final_std,
        class_names=np.asarray(class_names, dtype=str),
    )

    print("\nActor-independent out-of-fold results:")
    print(json.dumps(overall, indent=2))
    print(f"\nSaved evaluation: {RESULTS_DIR / 'cnn_evaluation.json'}")
    print(f"Saved CNN model: {MODELS_DIR / 'emotion_cnn.keras'}")
    print(f"Saved normalization: {MODELS_DIR / 'emotion_cnn_normalization.npz'}")


if __name__ == "__main__":
    main()
