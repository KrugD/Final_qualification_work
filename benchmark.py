"""
Benchmark: Autoregressive models vs Diffusion Model.

Compares generation quality (ROUGE, BERTScore) and speed (tokens/sec, time/sample).

Usage:
    python benchmark.py \
        --diffusion_weights checkpoints/best_model/weights \
        --multi_step --output benchmark_results.txt

    # Custom AR models:
    python benchmark.py \
        --diffusion_weights checkpoints/best_model/weights \
        --ar_model RussianNLP/FRED-T5-Summarizer \
        --ar_model2 IlyaGusev/rut5_base_sum_gazeta \
        --multi_step --output results.txt
"""

import argparse
import gc
import time
import sys
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
    print(f"  Parameters: {params:,}")
    return model, tokenizer, params


def load_diffusion_model(weights_path: str, device: str):
    """Load our diffusion-based model."""
    print(f"Loading diffusion model: {weights_path}")
    model = MaskedDiffusionSummarizer.from_pretrained(weights_path, device=device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,} total, {trainable:,} trainable")
    # Use the model's own tokenizer (matches encoder: FRED-T5 or ruT5)
    tokenizer = model.tokenizer
    return model, tokenizer, params


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_ar(model, tokenizer, texts, device, max_source=512, max_target=128,
                num_beams=1, do_sample=False):
    """Generate summaries with autoregressive T5."""
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
    """Generate summaries with diffusion model."""
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
# Benchmark one AR model
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_ar(label, model_name, sources, references, device, max_source, max_target):
    """Benchmark a single autoregressive model. Returns results dict."""
    print_header(f"AUTOREGRESSIVE: {label}")

    ar_model, ar_tok, params = load_ar_model(model_name, device)

    print("Warming up (3 forward passes)...")
    warmup_ar(ar_model, ar_tok, device, max_source, max_target)

    print(f"Generating {len(sources)} summaries (greedy, max_target={max_target})...")
    preds, times = generate_ar(ar_model, ar_tok, sources, device, max_source, max_target)

    total = sum(times)
    per = np.mean(times)
    std = np.std(times)
    words = sum(len(p.split()) for p in preds)
    wps = words / total if total > 0 else 0

    print(f"  Total time: {total:.2f}s")
    print(f"  Time/sample: {per:.4f}s (std={std:.4f})")
    print(f"  Words generated: {words}")
    print(f"  Words/sec: {wps:.1f}")

    print("Computing metrics...")
    metrics = compute_all_metrics(preds, references, sources, device)

    del ar_model
    free_gpu()

    return {
        "model_name": model_name,
        "params": params,
        "predictions": preds,
        "times": times,
        "total_time": total,
        "per_sample": per,
        "per_sample_std": std,
        "tokens": words,
        "tps": wps,
        "metrics": metrics,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(args, sources, references):
    """Run the full benchmark."""
    device = args.device
    results = {}

    # ==================================================================
    # 1. AR baselines
    # ==================================================================
    ar_models = [(args.ar_model, "ar1")]
    if args.ar_model2:
        ar_models.append((args.ar_model2, "ar2"))

    for model_name, key in ar_models:
        short = model_name.split("/")[-1] if "/" in model_name else model_name
        results[key] = benchmark_ar(
            short, model_name, sources, references,
            device, args.max_source, args.max_target,
        )

    # ==================================================================
    # 2. Diffusion model
    # ==================================================================
    diff_model, diff_tok, diff_params = load_diffusion_model(args.diffusion_weights, device)

    step_configs = [args.diffusion_steps]
    if args.multi_step:
        step_configs = [5, 10, 25, 50]

    for steps in step_configs:
        label = f"diff_{steps}"
        print_header(f"DIFFUSION MODEL  (steps={steps})")

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
            "model_name": "Diffusion (ours)",
            "params": diff_params,
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

    # Build column order: ar1, ar2 (if exists), then diff_*
    labels_order = []
    if "ar1" in results:
        labels_order.append("ar1")
    if "ar2" in results:
        labels_order.append("ar2")
    labels_order += sorted(
        [k for k in results if k.startswith("diff_")],
        key=lambda k: results[k].get("steps", 0),
    )

    # Column display names
    def col_name(lbl):
        r = results[lbl]
        if lbl.startswith("ar"):
            short = r["model_name"].split("/")[-1]
            # Truncate long names
            if len(short) > 16:
                short = short[:14] + ".."
            return short
        else:
            return f"Diff s={r['steps']}"

    col_names = [col_name(lbl) for lbl in labels_order]

    # ── Model info ────────────────────────────────────────────────────
    print_header("MODEL INFO")
    header = f"{'':>22s}"
    for cn in col_names:
        header += f"  {cn:>16s}"
    print(header)
    print("-" * len(header))

    row = f"{'Parameters':<22s}"
    for lbl in labels_order:
        p = results[lbl]["params"]
        if p >= 1e9:
            row += f"  {p/1e9:>14.2f}B "
        else:
            row += f"  {p/1e6:>14.0f}M "
    print(row)

    # ── Quality table ─────────────────────────────────────────────────
    print_header("QUALITY COMPARISON")

    quality_keys = [
        ("rouge1", "ROUGE-1"),
        ("rouge2", "ROUGE-2"),
        ("rougeL", "ROUGE-L"),
        ("bertscore_f1", "BERTScore F1"),
        ("bertscore_precision", "BERTScore Prec"),
        ("bertscore_recall", "BERTScore Rec"),
        ("compression_ratio_mean", "Compression"),
    ]

    header = f"{'Metric':<22s}"
    for cn in col_names:
        header += f"  {cn:>16s}"
    print(header)
    print("-" * len(header))

    for key, display in quality_keys:
        row = f"{display:<22s}"
        for lbl in labels_order:
            val = results[lbl]["metrics"].get(key, 0)
            row += f"  {val:16.4f}"
        print(row)

    # ── Speed table ───────────────────────────────────────────────────
    print_header("SPEED COMPARISON")

    header = f"{'Metric':<22s}"
    for cn in col_names:
        header += f"  {cn:>16s}"
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
            row += f"  {val:>16{fmt}}"
        print(row)

    # Speedup rows (vs each AR model)
    for ar_key in [k for k in labels_order if k.startswith("ar")]:
        ar_per = results[ar_key]["per_sample"]
        ar_name = col_name(ar_key)
        row = f"{'Speedup vs ' + ar_name:<22s}"
        for lbl in labels_order:
            d_per = results[lbl]["per_sample"]
            speedup = ar_per / d_per if d_per > 0 else 0
            row += f"  {speedup:>15.2f}x"
        print(row)

    # ── Output statistics ─────────────────────────────────────────────
    print_header("OUTPUT STATISTICS")

    header = f"{'Metric':<22s}"
    for cn in col_names:
        header += f"  {cn:>16s}"
    print(header)
    print("-" * len(header))

    row = f"{'Avg output words':<22s}"
    for lbl in labels_order:
        lens = [len(p.split()) for p in results[lbl]["predictions"]]
        row += f"  {np.mean(lens):>16.1f}"
    print(row)

    row = f"{'Empty outputs':<22s}"
    for lbl in labels_order:
        emp = sum(1 for p in results[lbl]["predictions"] if not p.strip())
        row += f"  {emp:>16d}"
    print(row)

    ref_lens = [len(r.split()) for r in references]
    print(f"{'Avg reference words':<22s}  {np.mean(ref_lens):>16.1f}")

    # ── Sample outputs ─────────────────────────────────────────────────
    print_header("SAMPLE OUTPUTS (first 5)")

    n = min(5, len(sources))
    for i in range(n):
        print(f"\n--- Sample {i+1} ---")
        print(f"  Source:     {sources[i][:150]}...")
        print(f"  Reference:  {references[i][:200]}")
        for lbl in labels_order:
            name = col_name(lbl)
            print(f"  {name:<16s}: {results[lbl]['predictions'][i][:200]}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark: AR T5 vs Diffusion")
    parser.add_argument("--ar_model", type=str, default="RussianNLP/FRED-T5-Summarizer",
                        help="First AR baseline (SOTA)")
    parser.add_argument("--ar_model2", type=str, default="IlyaGusev/rut5_base_sum_gazeta",
                        help="Second AR baseline (fair size comparison)")
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
    parser.add_argument("--output", type=str, default=None)

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
