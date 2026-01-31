"""
Generation script for Masked Diffusion Summarization Model.

Generate summaries for custom text inputs.

Usage:
    python generate.py --model_path checkpoints/best_model --text "Your text here..."
    python generate.py --model_path checkpoints/best_model --input_file texts.txt
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch

from src.model import MaskedDiffusionSummarizer
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def load_model(
    model_path: str,
    device: str = "cuda",
) -> MaskedDiffusionSummarizer:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    model = MaskedDiffusionSummarizer.from_pretrained(model_path, device=device)
    model.eval()
    
    return model


def generate_summary(
    model: MaskedDiffusionSummarizer,
    text: str,
    max_source_length: int = 512,
    max_target_length: int = 128,
    num_inference_steps: int = 20,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.9,
) -> str:
    """Generate summary for a single text."""
    device = next(model.parameters()).device
    tokenizer = model.tokenizer
    
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=max_source_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids, confidence = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_target_length,
            num_inference_steps=num_inference_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    
    # Decode
    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return summary


def generate_summaries_batch(
    model: MaskedDiffusionSummarizer,
    texts: List[str],
    max_source_length: int = 512,
    max_target_length: int = 128,
    num_inference_steps: int = 20,
    temperature: float = 1.0,
    top_p: Optional[float] = 0.9,
    batch_size: int = 8,
) -> List[str]:
    """Generate summaries for a batch of texts."""
    device = next(model.parameters()).device
    tokenizer = model.tokenizer
    
    all_summaries = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            max_length=max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids, _ = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_target_length,
                num_inference_steps=num_inference_steps,
                temperature=temperature,
                top_p=top_p,
            )
        
        # Decode
        summaries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_summaries.extend(summaries)
    
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Generate summaries with Diffusion Model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to summarize",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="File with texts to summarize (one per line)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save summaries",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=20,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    setup_logging("INFO")
    
    if args.text is None and args.input_file is None:
        parser.error("Either --text or --input_file must be provided")
    
    # Load model
    model = load_model(args.model_path, args.device)
    
    if args.text:
        # Single text
        summary = generate_summary(
            model=model,
            text=args.text,
            num_inference_steps=args.num_steps,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        print("\n" + "=" * 50)
        print("INPUT TEXT:")
        print(args.text[:500] + "..." if len(args.text) > 500 else args.text)
        print("\n" + "-" * 50)
        print("GENERATED SUMMARY:")
        print(summary)
        print("=" * 50 + "\n")
        
    elif args.input_file:
        # Multiple texts from file
        with open(args.input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Generating summaries for {len(texts)} texts")
        
        summaries = generate_summaries_batch(
            model=model,
            texts=texts,
            num_inference_steps=args.num_steps,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        # Save or print
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                for summary in summaries:
                    f.write(summary + "\n")
            logger.info(f"Summaries saved to {args.output_file}")
        else:
            for i, (text, summary) in enumerate(zip(texts, summaries)):
                print(f"\n{'='*50}")
                print(f"TEXT {i+1}:")
                print(text[:300] + "..." if len(text) > 300 else text)
                print(f"\nSUMMARY {i+1}:")
                print(summary)


if __name__ == "__main__":
    main()
