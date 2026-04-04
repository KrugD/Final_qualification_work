"""Run evaluation on a test set and optionally log to CometML.

Usage:
    python -m evaluation.evaluate \
        --checkpoint checkpoints/stage2/best \
        --test-data data/protocols_test \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def load_test_samples(data_path: str) -> list[dict]:
    """Load test samples from directory or JSONL."""
    path = Path(data_path)
    samples = []

    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line.strip()))
    elif path.is_dir():
        for sample_dir in sorted(path.iterdir()):
            if not sample_dir.is_dir():
                continue
            audio_file = sample_dir / "audio.wav"
            protocol_file = sample_dir / "protocol.txt"
            if audio_file.exists() and protocol_file.exists():
                samples.append({
                    "audio_path": str(audio_file),
                    "protocol": protocol_file.read_text(encoding="utf-8").strip(),
                })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate SpeechProtocol model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    from inference.generate import load_model
    from evaluation.metrics import evaluate_model

    print("Loading model...")
    model = load_model(args.checkpoint, args.device)

    print("Loading test data...")
    samples = load_test_samples(args.test_data)
    print(f"Found {len(samples)} test samples")

    if not samples:
        print("No test samples found. Check the data path.")
        return

    print("Running evaluation...")
    result = evaluate_model(model, samples, device=args.device)

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"ROUGE-1:             {result.rouge1:.4f}")
    print(f"ROUGE-2:             {result.rouge2:.4f}")
    print(f"ROUGE-L:             {result.rougeL:.4f}")
    print(f"BERTScore F1:        {result.bert_score_f1:.4f}")
    print(f"Speaker Accuracy:    {result.speaker_accuracy:.4f}")
    print(f"Avg Inference Time:  {result.inference_time_sec:.2f}s")
    print(f"Num Samples:         {result.num_samples}")

    results_dict = result.to_dict()
    Path(args.output).write_text(
        json.dumps(results_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to {args.output}")

    try:
        import os
        from comet_ml import Experiment

        experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=os.getenv("COMET_PROJECT_NAME", "speech-protocol"),
        )
        experiment.add_tag("evaluation")
        for key, value in results_dict.items():
            experiment.log_metric(f"eval/{key}", value)
        experiment.end()
        print("Results logged to CometML")
    except Exception:
        pass


if __name__ == "__main__":
    main()
