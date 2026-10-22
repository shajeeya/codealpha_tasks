
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

sys.path.insert(0, str(SRC_DIR))
from extract_audio_features import extract_audio_features


def main():
    audio_files = sorted(
        path
        for actor_dir in PROJECT_ROOT.glob("Actor_*")
        if actor_dir.is_dir()
        and actor_dir.name[6:].isdigit()
        and 1 <= int(actor_dir.name[6:]) <= 24
        for path in actor_dir.glob("*.wav")
    )

    if len(audio_files) != 1440:
        raise RuntimeError(
            f"Expected 1440 audio files, found {len(audio_files)}."
        )

    features = []
    labels = []
    actor_ids = []
    filenames = []

    emotion_labels = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    for index, audio_path in enumerate(audio_files, start=1):
        parts = audio_path.stem.split("-")

        if len(parts) != 7 or not all(p.isdigit() for p in parts):
            raise ValueError(f"Unexpected filename: {audio_path.name}")

        emotion_code = parts[2]
        actor_id = int(parts[6])

        if emotion_code not in emotion_labels or not 1 <= actor_id <= 24:
            raise ValueError(f"Invalid emotion or actor: {audio_path.name}")

        vector = extract_audio_features(audio_path)

        if vector.shape != (174,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid feature vector: {audio_path.name}")

        features.append(vector)
        labels.append(emotion_labels[emotion_code])
        actor_ids.append(actor_id)
        filenames.append(audio_path.name)

        if index % 100 == 0:
            print(f"Processed {index}/{len(audio_files)} audio files")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=str)
    groups = np.asarray(actor_ids, dtype=np.int16)

    if X.shape != (1440, 174):
        raise RuntimeError(f"Unexpected dataset shape: {X.shape}")

    if len(np.unique(groups)) != 24 or len(np.unique(y)) != 8:
        raise RuntimeError("Unexpected actor or emotion class count.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "ravdess_enhanced_features.npz"

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        groups=groups,
        filenames=np.asarray(filenames, dtype=str),
    )

    print("\nEnhanced dataset created successfully.")
    print("Feature matrix shape:", X.shape)
    print("Recording count:", len(y))
    print("Actor count:", len(np.unique(groups)))
    print("Emotion classes:", len(np.unique(y)))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
