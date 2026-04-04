"""Data collators for variable-length audio and text batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class AudioTextCollator:
    """Collates batches for Stage 1 (ASR) and Stage 2 (Protocol).

    Pads mel features, waveforms, labels, and attention masks
    to the longest element in the batch.
    """

    pad_token_id: int = 0

    def __call__(self, batch: list[dict]) -> dict:
        input_features = torch.stack([item["input_features"] for item in batch])

        waveforms = [item["waveform"] for item in batch]

        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )

        if "label_attention_mask" in batch[0]:
            label_attention_mask = pad_sequence(
                [item["label_attention_mask"] for item in batch],
                batch_first=True,
                padding_value=0,
            )
        else:
            label_attention_mask = (labels != -100).long()

        return {
            "input_features": input_features,
            "waveforms": waveforms,
            "labels": labels,
            "label_attention_mask": label_attention_mask,
        }


@dataclass
class TextOnlyCollator:
    """Collates batches for Stage 1.5 (text-only summarization)."""

    pad_token_id: int = 0

    def __call__(self, batch: list[dict]) -> dict:
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        attention_mask = pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )

        labels = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
