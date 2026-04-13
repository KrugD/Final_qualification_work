"""Evaluation metrics for SpeechProtocol model.

Metrics:
    - ROUGE-1/2/L: n-gram overlap with reference protocol
    - BERTScore: semantic similarity using Russian BERT
    - Speaker Attribution Accuracy: correctness of speaker assignment
    - Inference time comparison with pipeline
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EvalResult:
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    bert_score_f1: float = 0.0
    speaker_accuracy: float = 0.0
    inference_time_sec: float = 0.0
    num_samples: int = 0

    def to_dict(self) -> dict:
        return {
            "rouge1": self.rouge1,
            "rouge2": self.rouge2,
            "rougeL": self.rougeL,
            "bert_score_f1": self.bert_score_f1,
            "speaker_accuracy": self.speaker_accuracy,
            "inference_time_sec": self.inference_time_sec,
            "num_samples": self.num_samples,
        }


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1/2/L scores."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {k: np.mean(v) for k, v in scores.items()}


def compute_bert_score(
    predictions: list[str],
    references: list[str],
) -> float:
    """Compute BERTScore F1 using multilingual BERT."""
    from bert_score import score

    _, _, f1 = score(
        predictions,
        references,
        lang="ru",
        verbose=False,
    )
    return f1.mean().item()


def extract_speaker_texts(protocol: str) -> dict[str, str]:
    """Parse protocol text and extract per-speaker content.

    Expected format:
        <speaker id="1">text</speaker>
        <speaker id="2">text</speaker>
    """
    pattern = r'<speaker id="(\d+)">(.*?)</speaker>'
    matches = re.findall(pattern, protocol, re.DOTALL)
    return {speaker_id: text.strip() for speaker_id, text in matches}


def compute_speaker_accuracy(
    predictions: list[str], references: list[str]
) -> float:
    """Measure how well speaker attribution matches the reference.

    Compares the number of speakers detected and checks if
    speaker texts are assigned to the correct speaker IDs.
    """
    correct = 0
    total = 0

    for pred, ref in zip(predictions, references):
        pred_speakers = extract_speaker_texts(pred)
        ref_speakers = extract_speaker_texts(ref)

        if not ref_speakers:
            continue

        num_correct = len(set(pred_speakers.keys()) & set(ref_speakers.keys()))
        num_ref = len(ref_speakers)
        total += num_ref
        correct += num_correct

    return correct / max(total, 1)


def evaluate_model(
    model,
    test_samples: list[dict],
    device: str = "cuda",
) -> EvalResult:
    """Run full evaluation on test samples.

    Args:
        model: SpeechProtocolModel instance.
        test_samples: list of dicts with 'audio_path' and 'protocol' keys.
        device: computation device.

    Returns:
        EvalResult with all metrics.
    """
    from inference.generate import generate_protocol

    predictions = []
    references = []
    total_time = 0.0

    for i, sample in enumerate(test_samples):
        start = time.time()
        pred = generate_protocol(
            sample["audio_path"], model, device=device
        )
        elapsed = time.time() - start
        total_time += elapsed

        predictions.append(pred)
        references.append(sample["protocol"])

        if (i + 1) % 5 == 0:
            print(f"Evaluated {i+1}/{len(test_samples)} samples")

    rouge = compute_rouge(predictions, references)
    bert_f1 = compute_bert_score(predictions, references)
    speaker_acc = compute_speaker_accuracy(predictions, references)

    return EvalResult(
        rouge1=rouge["rouge1"],
        rouge2=rouge["rouge2"],
        rougeL=rouge["rougeL"],
        bert_score_f1=bert_f1,
        speaker_accuracy=speaker_acc,
        inference_time_sec=total_time / max(len(test_samples), 1),
        num_samples=len(test_samples),
    )
