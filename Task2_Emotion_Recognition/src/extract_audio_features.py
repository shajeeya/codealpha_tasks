
from pathlib import Path

import librosa
import numpy as np


def extract_audio_features(
    audio_path: str | Path,
    n_mfcc: int = 40,
    duration: float | None = None,
) -> np.ndarray:
    """
    Extract MFCC, delta-MFCC, pitch, energy, and spectral features.

    Returns a fixed-length, finite float32 feature vector.
    """
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if n_mfcc < 1:
        raise ValueError("n_mfcc must be at least 1.")

    y, sr = librosa.load(
        audio_path,
        sr=22_050,
        mono=True,
        duration=duration,
    )

    if y.size == 0:
        raise ValueError(f"No audio samples found in: {audio_path}")

    # MFCCs and their first-order temporal changes.
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512
    )
    delta_mfcc = librosa.feature.delta(mfcc, order=1)

    # Frame-level RMS energy.
    rms = librosa.feature.rms(
        y=y, frame_length=2048, hop_length=512
    )

    # Spectral descriptors.
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=2048, hop_length=512
    )
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=2048, hop_length=512
    )
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=2048, hop_length=512
    )
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y, frame_length=2048, hop_length=512
    )

    # Estimate fundamental frequency (pitch).
    f0 = librosa.yin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )

    # Exclude unvoiced / invalid pitch estimates from pitch statistics.
    voiced_f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced_f0.size:
        pitch_stats = np.array([
            np.mean(voiced_f0),
            np.std(voiced_f0),
            np.min(voiced_f0),
            np.max(voiced_f0),
        ])
    else:
        pitch_stats = np.zeros(4, dtype=np.float64)

    def mean_std(values):
        return np.concatenate([
            np.mean(values, axis=1),
            np.std(values, axis=1),
        ])

    features = np.concatenate([
        mean_std(mfcc),
        mean_std(delta_mfcc),
        mean_std(rms),
        mean_std(spectral_centroid),
        mean_std(spectral_bandwidth),
        mean_std(spectral_rolloff),
        mean_std(zero_crossing_rate),
        pitch_stats,
    ])

    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features extracted from: {audio_path}")

    return features.astype(np.float32)
