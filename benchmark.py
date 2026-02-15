"""
Benchmark: Autoregressive ruT5 vs Diffusion Model.

Compares generation quality (ROUGE, BERTScore) and speed (tokens/sec, time/sample).

Usage:
    # After fine-tuning baseline:
    python benchmark.py \
        --ar_model checkpoints/baseline_rut5/best_model \
        --diffusion_weights checkpoints/best_model/weights \
        --output benchmark_results.txt

    # With HuggingFace model:
    python benchmark.py \
        --ar_model ai-forever/ruT5-base \
        --diffusion_weights checkpoints/best_model/weights \
        --num_samples 50 --output results.txt

    # Multi-step comparison:
    python benchmark.py \
        --ar_model checkpoints/baseline_rut5/best_model \
        --diffusion_weights checkpoints/best_model/weights \
        --multi_step --output results.txt
"""

import argparse
import gc
import time
import sys
import os
import numpy as np
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datasets import load_dataset

from src.model import MaskedDiffusionSummarizer
from src.utils.metrics import compute_rouge, compute_compression_ratio, compute_bertscore


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class Tee:
    """Write to both stdout and a file."""
    def __init__(self, filepath, stream):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return self.stream.isatty() if hasattr(self.stream, "isatty") else False

    def close(self):
        self.file.close()


def gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_header(title: str, char="=", width=80):
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_ar_model(model_name_or_path: str, device: str):
    """Load an autoregressive T5 model (HuggingFace or local checkpoint)."""
    print(f"Loading autoregressive model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,} total, {trainable:,} trainable")
    return model, tokenizer


def load_diffusion_model(weights_path: str, device: str):
    """Load our diffusion-based model."""
    print(f"Loading diffusion model: {weights_path}")
    model = MaskedDiffusionSummarizer.from_pretrained(weights_path, device=device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,} total, {trainable:,} trainable")
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruT5-base")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_ar(model, tokenizer, texts, device, max_source=512, max_target=128,
                num_beams=1, do_sample=False):
    """Generate summaries with autoregressive T5 (one sample at a time for fair timing)."""
    predictions = []
    times = []

    for text in texts:
        inputs = tokenizer(
            text, max_length=max_source, truncation=True,
            padding="max_length", return_tensors="pt"
        ).to(device)

        gpu_sync()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_target,
                num_beams=num_beams,
                do_sample=do_sample,
                early_stopping=True,
            )

        gpu_sync()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        predictions.append(pred)

    return predictions, times


def generate_diff(model, tokenizer, texts, device, max_source=512, max_target=128,
                  num_steps=50, temperature=1.0, strategy="cosine", sample=False,
                  rep_penalty=2.0, no_repeat_ngram=2):
    """Generate summaries with diffusion model (one sample at a time for fair timing)."""
    predictions = []
    times = []

    for text in texts:
        inputs = tokenizer(
            text, max_length=max_source, truncation=True,
            padding="max_length", return_tensors="pt"
        ).to(device)

        gpu_sync()
        t0 = time.perf_counter()

        with torch.no_grad():
            generated_ids, _ = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_target,
                num_inference_steps=num_steps,
                temperature=temperature,
                strategy=strategy,
                sample=sample,
                repetition_penalty=rep_penalty,
                no_repeat_ngram_size=no_repeat_ngram,
            )

        gpu_sync()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        predictions.append(pred)

    return predictions, times


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(predictions, references, sources, device):
    """Compute ROUGE, BERTScore, compression ratio."""
    rouge = compute_rouge(predictions, references)
    comp = compute_compression_ratio(predictions, sources)

    bert = compute_bertscore(
        predictions, references,
        model_type="bert-base-multilingual-cased",
        device=device,
    )

    return {**rouge, **comp, **bert}


# ──────────────────────────────────────────────────────────────────────────────
# Warm-up
# ──────────────────────────────────────────────────────────────────────────────

def warmup_ar(model, tokenizer, device, max_source=512, max_target=128, n=3):
    """GPU warm-up for AR model."""
    dummy = "Привет мир " * 50
    for _ in range(n):
        inputs = tokenizer(dummy, max_length=max_source, truncation=True,
                           padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(input_ids=inputs["input_ids"],
                           attention_mask=inputs["attention_mask"],
                           max_length=max_target, num_beams=1)
    gpu_sync()


def warmup_diff(model, tokenizer, device, max_source=512, max_target=128, n=3, steps=10):
    """GPU warm-up for diffusion model."""
    dummy = "Привет мир " * 50
    for _ in range(n):
        inputs = tokenizer(dummy, max_length=max_source, truncation=True,
                           padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            model.generate(input_ids=inputs["input_ids"],
                           attention_mask=inputs["attention_mask"],
                           max_length=max_target, num_inference_steps=steps,
                           strategy="linear", sample=False)
    gpu_sync()


# ──────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(args, sources, references):
    """Run the full benchmark."""
    device = args.device
    results = {}

    # ==================================================================
    # 1. Autoregressive baseline
    # ==================================================================
    print_header("AUTOREGRESSIVE MODEL (ruT5)")

    ar_model, ar_tok = load_ar_model(args.ar_model, device)

    # Warm-up
    print("Warming up (3 forward passes)...")
    warmup_ar(ar_model, ar_tok, device,
              args.max_source, args.max_target)

    print(f"Generating {len(sources)} summaries (greedy, max_target={args.max_target})...")
    ar_preds, ar_times = generate_ar(
        ar_model, ar_tok, sources, device,
        args.max_source, args.max_target,
    )

    ar_total = sum(ar_times)
    ar_per_sample = np.mean(ar_times)
    ar_std = np.std(ar_times)
    ar_tokens = sum(len(p.split()) for p in ar_preds)
    ar_tps = ar_tokens / ar_total if ar_total > 0 else 0

    print(f"  Total time: {ar_total:.2f}s")
    print(f"  Time/sample: {ar_per_sample:.4f}s (std={ar_std:.4f})")
    print(f"  Words generated: {ar_tokens}")
    print(f"  Words/sec: {ar_tps:.1f}")

    print("Computing metrics...")
    ar_metrics = compute_all_metrics(ar_preds, references, sources, device)

    results["ar"] = {
        "predictions": ar_preds,
        "times": ar_times,
        "total_time": ar_total,
        "per_sample": ar_per_sample,
        "per_sample_std": ar_std,
        "tokens": ar_tokens,
        "tps": ar_tps,
        "metrics": ar_metrics,
    }

    # Free memory
    del ar_model
    free_gpu()

    # ==================================================================
    # 2. Diffusion model (possibly multiple step configs)
    # ==================================================================
    diff_model, diff_tok = load_diffusion_model(args.diffusion_weights, device)

    step_configs = [args.diffusion_steps]
    if args.multi_step:
        step_configs = [5, 10, 25, 50]

    for steps in step_configs:
        label = f"diff_{steps}"
        print_header(f"DIFFUSION MODEL  (steps={steps})")

        # Warm-up
        print("Warming up (3 forward passes)...")
        warmup_diff(diff_model, diff_tok, device,
                    args.max_source, args.max_target, steps=min(steps, 10))

        print(f"Generating {len(sources)} summaries "
              f"(steps={steps}, rep_penalty={args.rep_penalty}, "
              f"no_repeat_ngram={args.no_repeat_ngram})...")

        diff_preds, diff_times = generate_diff(
            diff_model, diff_tok, sources, device,
            args.max_source, args.max_target,
            num_steps=steps,
            rep_penalty=args.rep_penalty,
            no_repeat_ngram=args.no_repeat_ngram,
        )

        d_total = sum(diff_times)
        d_per = np.mean(diff_times)
        d_std = np.std(diff_times)
        d_tokens = sum(len(p.split()) for p in diff_preds)
        d_tps = d_tokens / d_total if d_total > 0 else 0

        print(f"  Total time: {d_total:.2f}s")
        print(f"  Time/sample: {d_per:.4f}s (std={d_std:.4f})")
        print(f"  Words generated: {d_tokens}")
        print(f"  Words/sec: {d_tps:.1f}")

        print("Computing metrics...")
        d_metrics = compute_all_metrics(diff_preds, references, sources, device)

        results[label] = {
            "steps": steps,
            "predictions": diff_preds,
            "times": diff_times,
            "total_time": d_total,
            "per_sample": d_per,
            "per_sample_std": d_std,
            "tokens": d_tokens,
            "tps": d_tps,
            "metrics": d_metrics,
        }

    del diff_model
    free_gpu()

    return results


def print_results(results, sources, references):
    """Pretty-print comparison tables."""

    # ── Quality table ─────────────────────────────────────────────────
    print_header("QUALITY COMPARISON")

    labels_order = ["ar"] + sorted(
        [k for k in results if k.startswith("diff_")],
        key=lambda k: results[k].get("steps", 0),
    )

    quality_keys = [
        ("rouge1", "ROUGE-1"),
        ("rouge2", "ROUGE-2"),
        ("rougeL", "ROUGE-L"),
        ("bertscore_f1", "BERTScore F1"),
        ("bertscore_precision", "BERTScore Prec"),
        ("bertscore_recall", "BERTScore Rec"),
        ("compression_ratio_mean", "Compression"),
    ]

    # Column names
    col_names = []
    for lbl in labels_order:
        if lbl == "ar":
            col_names.append("AR (ruT5)")
        else:
            col_names.append(f"Diff s={results[lbl]['steps']}")

    header = f"{'Metric':<22s}"
    for cn in col_names:
        header += f"  {cn:>14s}"
    print(header)
    print("-" * len(header))

    for key, display in quality_keys:
        row = f"{display:<22s}"
        for lbl in labels_order:
            val = results[lbl]["metrics"].get(key, 0)
            row += f"  {val:14.4f}"
        print(row)

    # ── Speed table ───────────────────────────────────────────────────
    print_header("SPEED COMPARISON")

    header = f"{'Metric':<22s}"
    for cn in col_names:
        header += f"  {cn:>14s}"
    print(header)
    print("-" * len(header))

    speed_keys = [
        ("per_sample", "Time/sample (s)", ".4f"),
        ("per_sample_std", "  std (s)", ".4f"),
        ("total_time", "Total time (s)", ".2f"),
        ("tps", "Words/sec", ".1f"),
        ("tokens", "Total words", "d"),
    ]

    for key, display, fmt in speed_keys:
        row = f"{display:<22s}"
        for lbl in labels_order:
            val = results[lbl][key]
            row += f"  {val:>14{fmt}}"
        print(row)

    # Speedup row
    ar_per = results["ar"]["per_sample"]
    row = f"{'Speedup vs AR':<22s}  {'1.00x':>14s}"
    for lbl in labels_order[1:]:
        d_per = results[lbl]["per_sample"]
        speedup = ar_per / d_per if d_per > 0 else 0
        row += f"  {speedup:>13.2f}x"
    print(row)

    # ── Output statistics ─────────────────────────────────────────────
    print_header("OUTPUT STATISTICS")

    header = f"{'Metric':<22s}"
    for cn in col_names:
        header += f"  {cn:>14s}"
    print(header)
    print("-" * len(header))

    for lbl in labels_order:
        pass  # will print below

    # Avg words
    row = f"{'Avg output words':<22s}"
    for lbl in labels_order:
        lens = [len(p.split()) for p in results[lbl]["predictions"]]
        row += f"  {np.mean(lens):>14.1f}"
    print(row)

    # Empty outputs
    row = f"{'Empty outputs':<22s}"
    for lbl in labels_order:
        emp = sum(1 for p in results[lbl]["predictions"] if not p.strip())
        row += f"  {emp:>14d}"
    print(row)

    # Avg reference words
    ref_lens = [len(r.split()) for r in references]
    print(f"{'Avg reference words':<22s}  {np.mean(ref_lens):>14.1f}")

    # ── Sample outputs ─────────────────────────────────────────────────
    print_header("SAMPLE OUTPUTS (first 5)")

    n = min(5, len(sources))
    for i in range(n):
        print(f"\n--- Sample {i+1} ---")
        print(f"  Source:     {sources[i][:150]}...")
        print(f"  Reference:  {references[i][:200]}")
        for lbl in labels_order:
            name = "AR" if lbl == "ar" else f"Diff(s={results[lbl]['steps']})"
            print(f"  {name:<12s}: {results[lbl]['predictions'][i][:200]}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark: AR T5 vs Diffusion")
    parser.add_argument("--ar_model", type=str, default="RussianNLP/FRED-T5-Summarizer",
                        help="Autoregressive model (HF name or local path)")
    parser.add_argument("--diffusion_weights", type=str, required=True,
                        help="Path to diffusion model weights")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_source", type=int, default=512)
    parser.add_argument("--max_target", type=int, default=128)

    # Diffusion settings
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--multi_step", action="store_true",
                        help="Test diffusion with steps=[5,10,25,50]")
    parser.add_argument("--rep_penalty", type=float, default=2.0)
    parser.add_argument("--no_repeat_ngram", type=int, default=2)

    # Dataset
    parser.add_argument("--dataset", type=str,
                        default="RussianNLP/Mixed-Summarization-Dataset")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Save output to file")

    args = parser.parse_args()

    # Output redirection
    tee = None
    if args.output:
        tee = Tee(args.output, sys.stdout)
        sys.stdout = tee

    # Device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {args.device}")
    if args.device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Load dataset
    print(f"\nDataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="test")
    n = min(args.num_samples, len(dataset))
    print(f"Samples: {n} / {len(dataset)}")

    sources, references = [], []
    for i in range(n):
        ex = dataset[i]
        sources.append(ex["text"])
        references.append(ex["summary"])

    # Run benchmark
    results = run_benchmark(args, sources, references)

    # Print comparison
    print_results(results, sources, references)

    print_header("BENCHMARK COMPLETE")

    # Close file
    if tee:
        sys.stdout = tee.stream
        tee.close()
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
