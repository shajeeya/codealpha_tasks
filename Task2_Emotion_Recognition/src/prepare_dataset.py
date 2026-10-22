
from pathlib import Path
import csv
import sys

import numpy as np

# Make the src directory importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

sys.path.insert(0, str(SRC_DIR))
from extract_features import extract_mfcc


EMOTION_LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def parse_ravdess_filename(path: Path):
    """Return the emotion label and actor ID from a RAVDESS filename."""
    parts = path.stem.split("-")

    if len(parts) != 7 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Unexpected RAVDESS filename: {path.name}")

    emotion_code = parts[2]
    actor_id = parts[6]

    if emotion_code not in EMOTION_LABELS:
        raise ValueError(f"Unknown emotion code in: {path.name}")

    if not 1 <= int(actor_id) <= 24:
        raise ValueError(f"Invalid actor ID in: {path.name}")

    return EMOTION_LABELS[emotion_code], int(actor_id)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        path
        for actor_dir in DATASET_DIR.glob("Actor_*")
        if actor_dir.is_dir()
        and actor_dir.name[6:].isdigit()
        and 1 <= int(actor_dir.name[6:]) <= 24
        for path in actor_dir.glob("*.wav")
    )

    if len(audio_files) != 1440:
        raise RuntimeError(
            f"Expected 1440 RAVDESS speech audio files, "
            f"but found {len(audio_files)}. Check the dataset before proceeding."
        )

    features = []
    labels = []
    actor_ids = []
    filenames = []
    errors = []

    for index, audio_path in enumerate(audio_files, start=1):
        try:
            emotion, actor_id = parse_ravdess_filename(audio_path)
            feature_vector = extract_mfcc(audio_path)

            if feature_vector.shape != (80,):
                raise ValueError(
                    f"Expected 80 features, got {feature_vector.shape}"
                )

            if not np.isfinite(feature_vector).all():
                raise ValueError("Feature vector contains non-finite values")

            features.append(feature_vector)
            labels.append(emotion)
            actor_ids.append(actor_id)
            filenames.append(audio_path.name)

        except Exception as exc:
            errors.append(f"{audio_path}: {type(exc).__name__}: {exc}")

        if index % 100 == 0:
            print(f"Processed {index}/{len(audio_files)} audio files")

    if errors:
        print(f"\nERROR: {len(errors)} file(s) failed processing.")
        for error in errors[:20]:
            print(error)
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more error(s).")
        raise RuntimeError("Dataset preparation failed; outputs were not saved.")

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=str)
    groups = np.asarray(actor_ids, dtype=np.int16)

    if X.shape != (1440, 80):
        raise RuntimeError(f"Unexpected feature matrix shape: {X.shape}")

    if len(y) != 1440 or len(groups) != 1440:
        raise RuntimeError("Feature, label, and actor counts do not match.")

    np.savez_compressed(
        OUTPUT_DIR / "ravdess_features.npz",
        X=X,
        y=y,
        groups=groups,
        filenames=np.asarray(filenames, dtype=str),
    )

    with (OUTPUT_DIR / "ravdess_metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "actor_id", "emotion"])
        writer.writerows(zip(filenames, groups.tolist(), labels))

    print("\nDataset preparation completed successfully.")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of recordings: {len(y)}")
    print(f"Number of actors: {len(np.unique(groups))}")
    print(f"Number of emotion classes: {len(np.unique(y))}")
    print(f"Features saved to: {OUTPUT_DIR / 'ravdess_features.npz'}")
    print(f"Metadata saved to: {OUTPUT_DIR / 'ravdess_metadata.csv'}")
    print("\nEmotion distribution:")
    for emotion in EMOTION_LABELS.values():
        print(f"  {emotion}: {int(np.sum(y == emotion))}")


if __name__ == "__main__":
    main()
