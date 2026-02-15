"""
Inference script for testing generation with different parameters.

Finds the best model weights automatically and runs parameter sweep.

Usage:
    # Auto-find best model and run sweep:
    python inference.py --sweep

    # Evaluate on test set with ROUGE/BERTScore:
    python inference.py --eval --num_samples 50

    # Specify weights explicitly:
    python inference.py --weights checkpoints/best_model/weights --sweep

    # Single config:
    python inference.py --steps 10 --temperature 0.9 --strategy linear --sample

    # Custom text:
    python inference.py --text "Кратко суммаризируй текст: ваш текст здесь"

Weights priority (first found is used):
    1. --weights argument
    2. checkpoints/best_model/weights/
    3. checkpoints/best_model/
    4. Any checkpoint_*/  directory
"""

import argparse
import os
import sys
import glob
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from src.model import MaskedDiffusionSummarizer


def find_weights(args_weights: str = None) -> str:
    """Find the best available model weights."""
    candidates = []
    
    if args_weights:
        candidates.append(args_weights)
    
    # Common locations
    candidates.extend([
        "model_for_inference",
        "checkpoints/best_model/weights",
        "checkpoints/best_model",
        "output/best_model/weights",
        "output/best_model",
    ])
    
    # Search for any checkpoint directories
    for pattern in ["checkpoints/checkpoint_*", "output/checkpoint_*"]:
        found = sorted(glob.glob(pattern))
        if found:
            candidates.extend(reversed(found))  # Latest first
    
    for path in candidates:
        config_path = os.path.join(path, "config.pt")
        model_path = os.path.join(path, "model.pt")
        if os.path.isfile(config_path) and os.path.isfile(model_path):
            return path
    
    print("ERROR: No model weights found!")
    print("Searched in:")
    for c in candidates:
        exists = os.path.isdir(c)
        print(f"  {'[exists]' if exists else '[  --  ]'} {c}")
    print()
    print("Make sure you have weights saved. Typical structure:")
    print("  checkpoints/best_model/weights/config.pt")
    print("  checkpoints/best_model/weights/model.pt")
    sys.exit(1)


def load_model(weights_path: str, device: str = "cpu"):
    """Load model from weights directory."""
    print(f"Loading model from: {weights_path}")
    model = MaskedDiffusionSummarizer.from_pretrained(weights_path, device=device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")
    print(f"Device: {device}")
    print(f"Diffusion steps (model): {model.num_diffusion_steps}")
    print(f"Max target length: {model.max_target_length}")
    print(f"Mask token ID: {model.mask_token_id}")
    print(f"Decoder type: {model.decoder_type}")
    return model


def generate_one(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
    max_source_length: int = 512,
    max_target_length: int = 128,
    num_inference_steps: int = 20,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    strategy: str = "linear",
    sample: bool = False,
    temperature_annealing: bool = False,
):
    """Generate a summary for the given text."""
    inputs = tokenizer(
        text,
        max_length=max_source_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    
    generated_ids, confidence = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_target_length,
        num_inference_steps=num_inference_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        strategy=strategy,
        sample=sample,
        temperature_annealing=temperature_annealing,
    )
    
    # Decode
    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Debug info
    raw_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
    mask_tok = tokenizer.convert_ids_to_tokens([model.mask_token_id])[0]
    num_masks = sum(1 for t in raw_tokens if t == mask_tok)
    pad_tokens = sum(1 for t in raw_tokens if t in ("</s>", "<pad>"))
    eos_count = sum(1 for t in raw_tokens if t == "</s>")
    
    # Count unique tokens (diversity)
    real_tokens = [t for t in raw_tokens if t not in (mask_tok, "</s>", "<pad>")]
    unique_ratio = len(set(real_tokens)) / max(len(real_tokens), 1)
    
    debug = {
        "total_tokens": len(raw_tokens),
        "remaining_masks": num_masks,
        "pad_tokens": pad_tokens,
        "eos_tokens": eos_count,
        "real_tokens": len(real_tokens),
        "unique_token_ratio": unique_ratio,
        "confidence_mean": confidence[0].mean().item(),
        "confidence_min": confidence[0].min().item(),
        "output_words": len(summary.split()) if summary.strip() else 0,
        "output_chars": len(summary),
    }
    
    return summary, debug, raw_tokens[:30]  # First 30 tokens for inspection


# Test texts in Russian
TEST_TEXTS = [
    """Кратко суммаризируй текст: Российские учёные из Института ядерной физики СО РАН разработали новый метод диагностики материалов с помощью синхротронного излучения. Метод позволяет исследовать внутреннюю структуру объектов без их разрушения. Технология может применяться в медицине, промышленности и археологии. Результаты исследования опубликованы в журнале Nature Materials.""",
    
    """Кратко суммаризируй текст: Правительство Российской Федерации утвердило новую программу поддержки малого и среднего бизнеса на 2025-2030 годы. Программа предусматривает снижение налоговой нагрузки для предприятий с оборотом до 500 миллионов рублей, упрощение процедуры регистрации юридических лиц и расширение доступа к льготному кредитованию. По оценкам экспертов, реализация программы позволит создать более 2 миллионов новых рабочих мест.""",

    """Кратко суммаризируй текст: Международная группа астрономов обнаружила экзопланету, на которой условия максимально близки к земным. Планета находится в обитаемой зоне звезды, расположенной на расстоянии 40 световых лет от Земли. Температура на поверхности планеты составляет от минус 10 до плюс 30 градусов Цельсия, а атмосфера содержит следы кислорода и водяного пара.""",
]


def run_single(model, tokenizer, args, texts):
    """Run generation with a single configuration."""
    print(f"\nConfig: steps={args.steps}, temp={args.temperature}, "
          f"top_k={args.top_k}, top_p={args.top_p}, "
          f"strategy={args.strategy}, sample={args.sample}, "
          f"anneal={args.anneal}")
    print("=" * 90)
    
    for i, text in enumerate(texts):
        summary, debug, first_tokens = generate_one(
            model, tokenizer, text,
            device=args.device,
            max_target_length=args.max_target_length,
            num_inference_steps=args.steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            strategy=args.strategy,
            sample=args.sample,
            temperature_annealing=args.anneal,
        )
        
        print(f"\n--- Example {i+1} ---")
        print(f"Source ({len(text)} chars): {text[:150]}...")
        print(f"Summary ({debug['output_words']} words, {debug['output_chars']} chars):")
        print(f"  >>> {summary if summary.strip() else '[EMPTY]'}")
        print(f"Debug: real_tok={debug['real_tokens']}, masks_left={debug['remaining_masks']}, "
              f"eos={debug['eos_tokens']}, unique={debug['unique_token_ratio']:.2f}, "
              f"conf={debug['confidence_mean']:.4f}")
        print(f"First tokens: {first_tokens}")


def run_sweep(model, tokenizer, args, texts):
    """Run comprehensive parameter sweep."""
    configs = [
        # --- Strategy comparison (same steps/temp) ---
        {"label": "baseline-argmax",
         "steps": 20, "temperature": 1.0, "strategy": "linear", "sample": False, "anneal": False,
         "top_k": None, "top_p": None},
        {"label": "cosine-argmax",
         "steps": 20, "temperature": 1.0, "strategy": "cosine", "sample": False, "anneal": False,
         "top_k": None, "top_p": None},
        {"label": "confidence-argmax",
         "steps": 20, "temperature": 1.0, "strategy": "confidence", "sample": False, "anneal": False,
         "top_k": None, "top_p": None},
        
        # --- Sampling vs argmax ---
        {"label": "linear-sample-t1.0",
         "steps": 20, "temperature": 1.0, "strategy": "linear", "sample": True, "anneal": False,
         "top_k": None, "top_p": None},
        {"label": "linear-sample-t0.7",
         "steps": 20, "temperature": 0.7, "strategy": "linear", "sample": True, "anneal": False,
         "top_k": None, "top_p": None},
        {"label": "linear-sample-t0.5",
         "steps": 20, "temperature": 0.5, "strategy": "linear", "sample": True, "anneal": False,
         "top_k": None, "top_p": None},
        
        # --- More steps ---
        {"label": "linear-50steps",
         "steps": 50, "temperature": 1.0, "strategy": "linear", "sample": False, "anneal": False,
         "top_k": None, "top_p": None},
        {"label": "linear-100steps",
         "steps": 100, "temperature": 1.0, "strategy": "linear", "sample": False, "anneal": False,
         "top_k": None, "top_p": None},
        {"label": "cosine-50steps",
         "steps": 50, "temperature": 1.0, "strategy": "cosine", "sample": False, "anneal": False,
         "top_k": None, "top_p": None},
        
        # --- Temperature annealing ---
        {"label": "linear-anneal-t1.0",
         "steps": 50, "temperature": 1.0, "strategy": "linear", "sample": True, "anneal": True,
         "top_k": None, "top_p": None},
        {"label": "cosine-anneal-t0.8",
         "steps": 50, "temperature": 0.8, "strategy": "cosine", "sample": True, "anneal": True,
         "top_k": None, "top_p": None},
        
        # --- top-k / top-p ---
        {"label": "sample-topk50",
         "steps": 50, "temperature": 0.8, "strategy": "linear", "sample": True, "anneal": False,
         "top_k": 50, "top_p": None},
        {"label": "sample-topp0.9",
         "steps": 50, "temperature": 0.8, "strategy": "linear", "sample": True, "anneal": False,
         "top_k": None, "top_p": 0.9},
        {"label": "sample-topk50-topp0.9",
         "steps": 50, "temperature": 0.7, "strategy": "cosine", "sample": True, "anneal": True,
         "top_k": 50, "top_p": 0.9},
        
        # --- Low temperature + many steps ---
        {"label": "greedy-200steps",
         "steps": 200, "temperature": 0.5, "strategy": "linear", "sample": False, "anneal": False,
         "top_k": None, "top_p": None},
        {"label": "confidence-100steps-t0.7",
         "steps": 100, "temperature": 0.7, "strategy": "confidence", "sample": True, "anneal": True,
         "top_k": 50, "top_p": 0.9},
    ]
    
    text = texts[0]
    print("=" * 90)
    print("PARAMETER SWEEP")
    print(f"Source: {text[:120]}...")
    print("=" * 90)
    print()
    
    results = []
    for i, cfg in enumerate(configs):
        summary, debug, first_tokens = generate_one(
            model, tokenizer, text,
            device=args.device,
            max_target_length=args.max_target_length,
            num_inference_steps=cfg["steps"],
            temperature=cfg["temperature"],
            top_k=cfg["top_k"],
            top_p=cfg["top_p"],
            strategy=cfg["strategy"],
            sample=cfg["sample"],
            temperature_annealing=cfg["anneal"],
        )
        
        results.append({"label": cfg["label"], "summary": summary, "debug": debug})
        
        is_empty = not summary.strip()
        status = "EMPTY!" if is_empty else f"{debug['output_words']:3d} words"
        
        print(f"[{i+1:2d}/{len(configs)}] {cfg['label']:<30s}  {status}, "
              f"real={debug['real_tokens']:3d}, uniq={debug['unique_token_ratio']:.2f}, "
              f"conf={debug['confidence_mean']:.3f}")
        if not is_empty:
            print(f"         >>> {summary[:150]}")
        else:
            print(f"         tokens: {first_tokens[:15]}")
        print()
    
    # Summary table
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Config':<30s}  {'Words':>5s}  {'Chars':>5s}  {'Real':>5s}  "
          f"{'Uniq':>5s}  {'Conf':>5s}  {'Empty':>5s}")
    print("-" * 90)
    
    for r in results:
        d = r["debug"]
        is_empty = "YES" if not r["summary"].strip() else "no"
        print(f"{r['label']:<30s}  {d['output_words']:5d}  {d['output_chars']:5d}  "
              f"{d['real_tokens']:5d}  {d['unique_token_ratio']:5.2f}  "
              f"{d['confidence_mean']:5.3f}  {is_empty:>5s}")
    
    # Show best results
    non_empty = [r for r in results if r["summary"].strip()]
    if non_empty:
        best = max(non_empty, key=lambda r: r["debug"]["output_words"])
        print(f"\nBest by word count: [{best['label']}]")
        print(f"  >>> {best['summary'][:300]}")
        
        best_diverse = max(non_empty, key=lambda r: r["debug"]["unique_token_ratio"])
        print(f"\nMost diverse tokens: [{best_diverse['label']}]")
        print(f"  >>> {best_diverse['summary'][:300]}")
    else:
        print("\nAll configs produced empty output. The model may need more training.")
        print("Try:")
        print("  1. Use the checkpoint with the LOWEST eval_loss")
        print("  2. Increase num_epochs and continue training")
        print("  3. Check that mask_token_id matches the training config")


def run_eval(model, tokenizer, args):
    """Evaluate on the test dataset with ROUGE, BERTScore, and compression ratio."""
    from datasets import load_dataset
    from src.utils.metrics import compute_rouge, compute_compression_ratio
    
    print("=" * 90)
    print("EVALUATION ON TEST SET")
    print(f"Steps={args.steps}, temp={args.temperature}, top_k={args.top_k}, "
          f"sample={args.sample}, strategy={args.strategy}")
    print("=" * 90)
    
    # Load test data
    print("\nLoading test dataset...")
    dataset = load_dataset(
        args.eval_dataset, split="test",
    )
    print(f"Test set size: {len(dataset)}")
    
    num_samples = min(args.num_samples, len(dataset))
    print(f"Evaluating on {num_samples} samples...")
    
    predictions = []
    references = []
    sources = []
    
    t_start = time.time()
    
    for i in range(num_samples):
        example = dataset[i]
        source = example.get("text", example.get("article", ""))
        reference = example.get("summary", example.get("highlights", ""))
        
        # Add instruction prefix (same as training)
        prefixed_source = f"Кратко суммаризируй текст: {source}"
        
        # Tokenize
        inputs = tokenizer(
            prefixed_source,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(args.device)
        
        # Generate
        with torch.no_grad():
            generated_ids, confidence = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=args.max_target_length,
                num_inference_steps=args.steps,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                strategy=args.strategy,
                sample=args.sample,
                temperature_annealing=args.anneal,
            )
        
        prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        predictions.append(prediction)
        references.append(reference)
        sources.append(prefixed_source)
        
        # Show progress + first 5 samples
        if i < 5:
            print(f"\n--- Sample {i+1}/{num_samples} ---")
            print(f"  Source:     {prefixed_source[:150]}...")
            print(f"  Reference:  {reference[:150]}...")
            print(f"  Prediction: {prediction[:200]}")
        elif (i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            speed = (i + 1) / elapsed
            eta = (num_samples - i - 1) / speed
            print(f"  [{i+1}/{num_samples}] {speed:.1f} samples/s, ETA: {eta:.0f}s")
    
    elapsed = time.time() - t_start
    print(f"\nGeneration done: {num_samples} samples in {elapsed:.1f}s "
          f"({num_samples/elapsed:.1f} samples/s)")
    
    # --- Compute ROUGE ---
    print("\n" + "=" * 90)
    print("METRICS")
    print("=" * 90)
    
    rouge_scores = compute_rouge(predictions, references)
    print(f"\n  ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
    
    # Per-sample ROUGE for first 10
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
        
        print("\n  Per-sample ROUGE (first 10):")
        per_sample_r1 = []
        for i in range(min(10, num_samples)):
            pred = predictions[i] if predictions[i].strip() else "empty"
            ref = references[i] if references[i].strip() else "empty"
            result = scorer.score(ref, pred)
            r1 = result["rouge1"].fmeasure
            rL = result["rougeL"].fmeasure
            per_sample_r1.append(r1)
            print(f"    [{i+1}] R1={r1:.4f}, RL={rL:.4f} | "
                  f"pred='{predictions[i][:80]}...' | "
                  f"ref='{references[i][:80]}...'")
        
        nonzero = sum(1 for s in per_sample_r1 if s > 0)
        print(f"\n  Non-zero ROUGE-1: {nonzero}/{len(per_sample_r1)} samples")
    except ImportError:
        pass
    
    # --- Compression ratio ---
    comp_scores = compute_compression_ratio(predictions, sources)
    print(f"\n  Compression ratio: {comp_scores['compression_ratio_mean']:.4f} "
          f"(std={comp_scores['compression_ratio_std']:.4f})")
    
    # --- BERTScore ---
    try:
        from src.utils.metrics import compute_bertscore
        print("\n  Computing BERTScore (may take a moment)...")
        bert_scores = compute_bertscore(
            predictions, references,
            model_type="bert-base-multilingual-cased",
            device=args.device,
        )
        print(f"  BERTScore F1:        {bert_scores['bertscore_f1']:.4f}")
        print(f"  BERTScore Precision: {bert_scores['bertscore_precision']:.4f}")
        print(f"  BERTScore Recall:    {bert_scores['bertscore_recall']:.4f}")
    except Exception as e:
        print(f"  BERTScore skipped: {e}")
    
    # --- Output stats ---
    pred_lens = [len(p.split()) for p in predictions]
    ref_lens = [len(r.split()) for r in references]
    empty_count = sum(1 for p in predictions if not p.strip())
    
    print(f"\n  Avg prediction length: {np.mean(pred_lens):.1f} words")
    print(f"  Avg reference length:  {np.mean(ref_lens):.1f} words")
    print(f"  Empty predictions:     {empty_count}/{num_samples}")
    
    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Inference with Masked Diffusion Summarizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (auto-detected if not set)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detect if not set)")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of inference steps (default: 10, paper uses 10)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k filtering (default: None)")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p nucleus filtering (default: None)")
    parser.add_argument("--strategy", type=str, default="linear",
                        choices=["linear", "cosine", "confidence"],
                        help="Unmasking strategy (default: linear)")
    parser.add_argument("--sample", action="store_true",
                        help="Sample from distribution instead of argmax")
    parser.add_argument("--anneal", action="store_true",
                        help="Use temperature annealing")
    parser.add_argument("--max_target_length", type=int, default=128,
                        help="Maximum target length (default: 128)")
    parser.add_argument("--sweep", action="store_true",
                        help="Run comprehensive parameter sweep")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate on test dataset with ROUGE/BERTScore")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of test samples for --eval (default: 50)")
    parser.add_argument("--eval_dataset", type=str,
                        default="RussianNLP/Mixed-Summarization-Dataset",
                        help="Dataset for --eval")
    parser.add_argument("--text", type=str, default=None,
                        help="Custom text to summarize (overrides test texts)")
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {gpu} ({vram:.1f} GB)")
        else:
            args.device = "cpu"
            print("Using CPU")
    
    # Find and load model
    weights_path = find_weights(args.weights)
    model = load_model(weights_path, args.device)
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruT5-base")
    
    texts = [args.text] if args.text else TEST_TEXTS
    
    print()
    
    if args.eval:
        run_eval(model, tokenizer, args)
    elif args.sweep:
        run_sweep(model, tokenizer, args, texts)
    else:
        run_single(model, tokenizer, args, texts)


if __name__ == "__main__":
    main()
