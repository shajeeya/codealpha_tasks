
from pathlib import Path

import librosa
import numpy as np


def extract_mfcc(
    audio_path: str | Path,
    n_mfcc: int = 40,
    duration: float | None = None,
) -> np.ndarray:
    """
    Extract mean and standard deviation of MFCCs from an audio file.

    Returns a 1D feature vector containing 80 values by default:
    40 MFCC means and 40 MFCC standard deviations.
    """
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if n_mfcc < 1:
        raise ValueError("n_mfcc must be at least 1.")

    y, sample_rate = librosa.load(
        audio_path,
        sr=22_050,
        mono=True,
        duration=duration,
    )

    if y.size == 0:
        raise ValueError(f"No audio samples found in: {audio_path}")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=2048,
        hop_length=512,
    )

    features = np.concatenate(
        [np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]
    )

    return features.astype(np.float32)
