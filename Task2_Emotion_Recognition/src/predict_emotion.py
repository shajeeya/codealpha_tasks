
import argparse
from pathlib import Path

import joblib
import numpy as np

from extract_features import extract_mfcc


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "emotion_classifier.joblib"

EMOTION_LABELS = {
    "angry": "Angry",
    "calm": "Calm",
    "disgust": "Disgust",
    "fearful": "Fearful",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprised": "Surprised",
}


def predict_emotion(audio_path: str | Path) -> str:
    """Predict the emotion expressed in a WAV audio file."""
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() != ".wav":
        raise ValueError("Please provide a .wav audio file.")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    features = extract_mfcc(audio_path)

    if features.shape != (80,):
        raise ValueError(
            f"Expected 80 features, but received {features.shape}."
        )

    if not np.isfinite(features).all():
        raise ValueError("The audio produced invalid feature values.")

    model = joblib.load(MODEL_PATH)
    prediction = str(model.predict(features.reshape(1, -1))[0])

    return EMOTION_LABELS.get(prediction, prediction)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict emotion from a WAV audio recording."
    )
    parser.add_argument("audio_path", help="Path to the input .wav file")
    args = parser.parse_args()

    try:
        emotion = predict_emotion(args.audio_path)
        print(f"Predicted emotion: {emotion}")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
