
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from extract_features import extract_mfcc


class TestExtractMFCC(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.audio_path = Path(self.temp_dir.name) / "test_audio.wav"

        # Create a short, valid audio signal for testing.
        sample_rate = 22050
        duration = 2.0
        time = np.arange(int(sample_rate * duration)) / sample_rate
        audio = 0.5 * np.sin(2 * np.pi * 440 * time)

        sf.write(self.audio_path, audio, sample_rate)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_feature_shape_and_values(self):
        features = extract_mfcc(self.audio_path)

        self.assertEqual(features.shape, (80,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_custom_mfcc_count(self):
        features = extract_mfcc(self.audio_path, n_mfcc=20)

        self.assertEqual(features.shape, (40,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_missing_audio_file(self):
        missing_path = self.audio_path.parent / "missing.wav"

        with self.assertRaises(FileNotFoundError):
            extract_mfcc(missing_path)

    def test_invalid_mfcc_count(self):
        with self.assertRaises(ValueError):
            extract_mfcc(self.audio_path, n_mfcc=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
