
"""Predict speech emotion using the trained resampled CNN."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "emotion_cnn_resampled.keras"
NORMALIZATION_PATH = (
    PROJECT_DIR / "models" / "emotion_cnn_resampled_normalization.npz"
)

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 174


def extract_mfcc_sequence(audio_path: Path) -> np.ndarray:
    """Extract MFCCs using the same settings as the training pipeline."""
    audio, _ = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=512,
    )

    original_frames = np.linspace(0, mfcc.shape[1] - 1, mfcc.shape[1])
    target_frames = np.linspace(0, mfcc.shape[1] - 1, MAX_FRAMES)

    resampled = np.empty((N_MFCC, MAX_FRAMES), dtype=np.float32)

    for coefficient in range(N_MFCC):
        resampled[coefficient] = np.interp(
            target_frames,
            original_frames,
            mfcc[coefficient],
        )

    return resampled


def predict_emotion(audio_path: str | Path) -> tuple[str, float]:
    """Return the predicted emotion and its model probability."""
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() != ".wav":
        raise ValueError("Please provide a .wav audio file.")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"CNN model not found: {MODEL_PATH}")

    if not NORMALIZATION_PATH.is_file():
        raise FileNotFoundError(
            f"Normalization file not found: {NORMALIZATION_PATH}"
        )

    features = extract_mfcc_sequence(audio_path)

    with np.load(NORMALIZATION_PATH, allow_pickle=False) as saved:
        mean = saved["mean"].astype(np.float32)
        std = saved["std"].astype(np.float32)
        class_names = saved["class_names"].astype(str).tolist()

    if mean.shape != (1, N_MFCC, 1, 1) or std.shape != mean.shape:
        raise ValueError(
            f"Unexpected normalization shapes: mean={mean.shape}, std={std.shape}"
        )

    if len(class_names) != 8:
        raise ValueError(f"Expected 8 class labels; found {len(class_names)}.")

    if not np.isfinite(features).all():
        raise ValueError("Audio produced invalid MFCC values.")

    # Match the final training model's normalization and input dimensions.
    features = features[np.newaxis, ..., np.newaxis]
    features = (features - mean) / std

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    probabilities = model.predict(features, verbose=0)[0]

    if len(probabilities) != len(class_names):
        raise ValueError("Model output does not match the saved class labels.")

    index = int(np.argmax(probabilities))
    return class_names[index], float(probabilities[index])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict emotion from a WAV audio recording using a CNN."
    )
    parser.add_argument("audio_path", help="Path to the input .wav file")
    args = parser.parse_args()

    try:
        emotion, probability = predict_emotion(args.audio_path)
        print(f"Predicted emotion: {emotion.title()}")
        print(f"Model probability: {probability:.2%}")
        print("Note: This probability is not a calibrated confidence score.")
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
