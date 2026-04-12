"""Datasets for all training stages.

Stage 1   – ASR alignment:  (audio, transcription)
Stage 1.5 – Summarization:  (text, summary)       [text-only, no audio]
Stage 2   – Protocol gen:   (audio, protocol_text)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor

from training.augmentations import AudioAugmentor, TextAugmentor, SpecAugment


class ASRDataset(Dataset):
    """Stage 1: Audio-text alignment using Golos / Common Voice.

    Loads audio and its transcription for training the fusion adapter
    to project Whisper features into the LLM embedding space.
    """

    def __init__(
        self,
        hf_dataset,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer,
        max_audio_sec: int = 30,
        sample_rate: int = 16000,
        augment: bool = True,
    ):
        self.dataset = hf_dataset
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_audio_sec = max_audio_sec
        self.sample_rate = sample_rate
        self.audio_aug = AudioAugmentor(sample_rate=sample_rate) if augment else None
        self.spec_aug = SpecAugment() if augment else None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        max_samples = self.max_audio_sec * self.sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        audio = audio.astype(np.float32)

        if self.audio_aug is not None:
            audio = self.audio_aug(audio)

        mel = self.feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features.squeeze(0)

        if self.spec_aug is not None:
            mel = self.spec_aug(mel)

        waveform = torch.from_numpy(audio).float()

        text = item.get("sentence") or item.get("transcription") or item.get("text") or ""
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )

        return {
            "input_features": mel,
            "waveform": waveform,
            "labels": tokens.input_ids.squeeze(0),
            "label_attention_mask": tokens.attention_mask.squeeze(0),
        }


class SummarizationDataset(Dataset):
    """Stage 1.5: Text-only summarization (no audio).

    Uses RussianNLP/Mixed-Summarization-Dataset to teach
    the LoRA-adapted Qwen decoder to summarize Russian text.
    """

    def __init__(self, hf_dataset, tokenizer, max_input_len: int = 1024, max_target_len: int = 256):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        source = item.get("text", "")
        target = item.get("summary", "")

        prompt = f"Суммаризируй следующий текст:\n{source}\n\nСуммаризация:"

        input_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_input_len,
        )

        target_tokens = self.tokenizer(
            target,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_target_len,
        )

        return {
            "input_ids": input_tokens.input_ids.squeeze(0),
            "attention_mask": input_tokens.attention_mask.squeeze(0),
            "labels": target_tokens.input_ids.squeeze(0),
        }


class ProtocolDataset(Dataset):
    """Stage 2: Audio -> protocol pairs.

    Expects a directory structure:
        data/protocols/
            sample_001/
                audio.wav
                protocol.txt
            sample_002/
                audio.wav
                protocol.txt
            ...

    Or a JSONL file where each line has:
        {"audio_path": "...", "protocol": "..."}
    """

    def __init__(
        self,
        data_path: str,
        feature_extractor: WhisperFeatureExtractor,
        tokenizer,
        max_audio_sec: int = 30,
        sample_rate: int = 16000,
        max_target_len: int = 1024,
        augment: bool = True,
    ):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_audio_sec = max_audio_sec
        self.sample_rate = sample_rate
        self.max_target_len = max_target_len
        self.audio_aug = AudioAugmentor(sample_rate=sample_rate) if augment else None
        self.text_aug = TextAugmentor() if augment else None
        self.spec_aug = SpecAugment() if augment else None

        self.samples = self._load_samples(data_path)

    def _load_samples(self, data_path: str) -> list[dict]:
        path = Path(data_path)
        samples = []

        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    samples.append(item)
        elif path.is_dir():
            for sample_dir in sorted(path.iterdir()):
                if not sample_dir.is_dir():
                    continue
                audio_file = sample_dir / "audio.wav"
                protocol_file = sample_dir / "protocol.txt"
                if audio_file.exists() and protocol_file.exists():
                    protocol_text = protocol_file.read_text(encoding="utf-8").strip()
                    samples.append(
                        {"audio_path": str(audio_file), "protocol": protocol_text}
                    )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]

        audio, _ = librosa.load(item["audio_path"], sr=self.sample_rate)
        audio = audio.astype(np.float32)

        max_samples = self.max_audio_sec * self.sample_rate
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        if self.audio_aug is not None:
            audio = self.audio_aug(audio)

        mel = self.feature_extractor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        ).input_features.squeeze(0)

        if self.spec_aug is not None:
            mel = self.spec_aug(mel)

        waveform = torch.from_numpy(audio).float()

        protocol = item["protocol"]
        if self.text_aug is not None:
            protocol = self.text_aug(protocol)

        tokens = self.tokenizer(
            protocol,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_target_len,
        )

        return {
            "input_features": mel,
            "waveform": waveform,
            "labels": tokens.input_ids.squeeze(0),
            "label_attention_mask": tokens.attention_mask.squeeze(0),
        }
