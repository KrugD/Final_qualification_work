"""Audio and text augmentations for training data."""

import random
import re

import numpy as np
import torch


class AudioAugmentor:
    """On-the-fly audio augmentations using audiomentations.

    Falls back gracefully if audiomentations is not installed.
    """

    def __init__(self, sample_rate: int = 16000, p: float = 0.5):
        self.sample_rate = sample_rate
        self.p = p
        self.transform = None

        try:
            import audiomentations as A

            self.transform = A.Compose(
                [
                    A.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
                    A.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
                    A.PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
                    A.Gain(min_gain_db=-6, max_gain_db=6, p=0.4),
                    A.GainTransition(
                        min_gain_db=-3, max_gain_db=3, p=0.2
                    ),
                ]
            )
        except ImportError:
            pass

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        if self.transform is None or random.random() > self.p:
            return audio

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        return self.transform(samples=audio, sample_rate=self.sample_rate)


class TextAugmentor:
    """Simple text augmentations for protocol target text."""

    def __init__(self, p: float = 0.3):
        self.p = p

    def shuffle_speaker_ids(self, text: str) -> str:
        """Randomly remap speaker IDs to prevent the model
        from relying on a fixed speaker ordering."""
        speaker_ids = list(set(re.findall(r'<speaker id="(\d+)">', text)))
        if len(speaker_ids) < 2:
            return text

        shuffled = speaker_ids.copy()
        random.shuffle(shuffled)
        mapping = dict(zip(speaker_ids, shuffled))

        temp_map = {}
        for old_id, new_id in mapping.items():
            placeholder = f"__SPK_PLACEHOLDER_{old_id}__"
            text = text.replace(f'<speaker id="{old_id}">', placeholder)
            temp_map[placeholder] = f'<speaker id="{new_id}">'

        for placeholder, replacement in temp_map.items():
            text = text.replace(placeholder, replacement)

        return text

    def __call__(self, text: str) -> str:
        if random.random() < self.p:
            text = self.shuffle_speaker_ids(text)
        return text


class SpecAugment:
    """SpecAugment-style masking applied to Whisper mel features."""

    def __init__(
        self,
        freq_masks: int = 2,
        freq_width: int = 10,
        time_masks: int = 2,
        time_width: int = 50,
        p: float = 0.5,
    ):
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.time_masks = time_masks
        self.time_width = time_width
        self.p = p

    def __call__(self, mel_features: torch.Tensor) -> torch.Tensor:
        """Apply frequency and time masking.

        Args:
            mel_features: (n_mels, T) or (batch, n_mels, T).

        Returns:
            Masked mel features.
        """
        if random.random() > self.p:
            return mel_features

        features = mel_features.clone()
        squeeze = False
        if features.dim() == 2:
            features = features.unsqueeze(0)
            squeeze = True

        _, n_mels, time_len = features.shape

        for _ in range(self.freq_masks):
            f = random.randint(0, min(self.freq_width, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            features[:, f0 : f0 + f, :] = 0

        for _ in range(self.time_masks):
            t = random.randint(0, min(self.time_width, time_len - 1))
            t0 = random.randint(0, time_len - t)
            features[:, :, t0 : t0 + t] = 0

        if squeeze:
            features = features.squeeze(0)

        return features
