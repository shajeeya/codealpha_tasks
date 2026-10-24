import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load MNIST: 60,000 training images and 10,000 test images
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values and add the channel dimension
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# Build the CNN
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Conv2D(64, kernel_size=3, activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Train the model
history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=8,
    batch_size=128,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ],
    verbose=1,
)

# Evaluate on the held-out MNIST test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# Save the trained model
model_path = MODEL_DIR / "handwritten_digit_cnn.keras"
model.save(model_path)

# Save evaluation metrics
metrics = {
    "dataset": "MNIST",
    "training_images": int(len(x_train)),
    "test_images": int(len(x_test)),
    "classes": list(range(10)),
    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),
    "epochs_completed": len(history.history["loss"]),
}

metrics_path = RESULTS_DIR / "evaluation.json"
metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# Save a training-history plot
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MNIST CNN Training History")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_history.png", dpi=150)
plt.close()

print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Model saved to: {model_path}")
print(f"Metrics saved to: {metrics_path}")
