
"""Prepare fixed-length MFCC sequences for CNN-based speech emotion recognition."""

from pathlib import Path
import argparse

import librosa
import numpy as np


EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

SAMPLE_RATE = 22050
N_MFCC = 40
MAX_FRAMES = 174


def extract_mfcc_sequence(audio_path: Path) -> np.ndarray:
    """Return a (n_mfcc, max_frames) MFCC sequence."""
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

    # Truncate long recordings and zero-pad shorter recordings.
    if mfcc.shape[1] >= MAX_FRAMES:
        mfcc = mfcc[:, :MAX_FRAMES]
    else:
        padding = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode="constant")

    return mfcc.astype(np.float32)


def parse_filename(audio_path: Path) -> tuple[str, str]:
    """Extract the emotion label and actor ID from a RAVDESS filename."""
    parts = audio_path.stem.split("-")

    if len(parts) != 7:
        raise ValueError(f"Unexpected RAVDESS filename: {audio_path.name}")

    emotion_code = parts[2]
    actor_code = parts[6]

    if emotion_code not in EMOTION_CODES:
        raise ValueError(f"Unknown emotion code in {audio_path.name}")

    return EMOTION_CODES[emotion_code], actor_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MFCC sequences from RAVDESS speech audio."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Task2_Emotion_Recognition project directory.",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    actor_dirs = sorted(project_dir.glob("Actor_*"))
    audio_files = sorted(
        audio_path
        for actor_dir in actor_dirs
        for audio_path in actor_dir.glob("*.wav")
    )

    if not audio_files:
        raise FileNotFoundError(
            f"No RAVDESS WAV files found under {project_dir}"
        )

    sequences = []
    labels = []
    actors = []
    filenames = []
    failures = []

    for index, audio_path in enumerate(audio_files, start=1):
        try:
            label, actor = parse_filename(audio_path)
            sequence = extract_mfcc_sequence(audio_path)

            sequences.append(sequence)
            labels.append(label)
            actors.append(int(actor))
            filenames.append(str(audio_path.relative_to(project_dir)))

        except (ValueError, OSError, RuntimeError) as exc:
            failures.append((str(audio_path), str(exc)))
            continue

        if index % 100 == 0 or index == len(audio_files):
            print(f"Processed {index}/{len(audio_files)} files")

    if failures:
        print(f"\nFailed files: {len(failures)}")
        for filename, error in failures[:10]:
            print(f"  {filename}: {error}")

    if not sequences:
        raise RuntimeError("No audio files could be processed.")

    X = np.stack(sequences)
    y = np.asarray(labels, dtype=str)
    groups = np.asarray(actors, dtype=np.int16)
    filenames_array = np.asarray(filenames, dtype=str)

    output_dir = project_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ravdess_mfcc_sequences.npz"

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        groups=groups,
        filenames=filenames_array,
    )

    print("\nDataset preparation complete.")
    print(f"Successful recordings: {len(X)}")
    print(f"Feature tensor shape: {X.shape}")
    print(f"Unique actors: {len(np.unique(groups))}")
    print(f"Emotion classes: {sorted(np.unique(y).tolist())}")
    print(f"Saved to: {output_path}")

    if failures:
        raise RuntimeError(
            f"{len(failures)} recordings failed; inspect the errors above."
        )


if __name__ == "__main__":
    main()
