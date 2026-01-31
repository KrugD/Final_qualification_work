"""
Evaluation script for Masked Diffusion Summarization Model.

Computes ROUGE and BERTScore metrics on the test set.

Usage:
    python evaluate.py --model_path checkpoints/best_model --output results.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.model import MaskedDiffusionSummarizer
from src.data import SummarizationDataset, SummarizationCollator
from src.utils import compute_all_metrics, setup_logging
from torch.utils.data import DataLoader

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


def generate_summaries(
    model: MaskedDiffusionSummarizer,
    dataloader: DataLoader,
    device: str = "cuda",
    num_inference_steps: int = 20,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = 0.9,
) -> tuple:
    """Generate summaries for all examples in dataloader."""
    all_predictions = []
    all_references = []
    all_sources = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating summaries"):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            # Generate
            generated_ids, confidence = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_inference_steps=num_inference_steps,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            # Decode
            predictions = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            references = model.tokenizer.batch_decode(labels, skip_special_tokens=True)
            sources = model.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
            all_sources.extend(sources)
    
    return all_predictions, all_references, all_sources


def evaluate_model(
    model_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 8,
    max_source_length: int = 512,
    max_target_length: int = 128,
    num_inference_steps: int = 20,
    temperature: float = 1.0,
    top_p: float = 0.9,
    compute_bertscore: bool = True,
    device: str = "cuda",
    save_predictions: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model_path: Path to trained model
        output_path: Path to save results JSON
        batch_size: Batch size for generation
        max_source_length: Maximum source sequence length
        max_target_length: Maximum target sequence length
        num_inference_steps: Number of diffusion steps for generation
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        compute_bertscore: Whether to compute BERTScore
        device: Device to use
        save_predictions: Whether to save predictions to file
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load model
    model = load_model(model_path, device)
    tokenizer = model.tokenizer
    
    # Create test dataloader
    test_dataset = SummarizationDataset(
        tokenizer=tokenizer,
        split="test",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    
    collator = SummarizationCollator(pad_token_id=tokenizer.pad_token_id)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    
    logger.info(f"Evaluating on {len(test_dataset)} test examples")
    
    # Generate summaries
    predictions, references, sources = generate_summaries(
        model=model,
        dataloader=test_dataloader,
        device=device,
        num_inference_steps=num_inference_steps,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(
        predictions=predictions,
        references=references,
        compute_bertscore_flag=compute_bertscore,
        device=device,
    )
    
    # Log results
    logger.info("Evaluation Results:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")
    
    # Prepare output
    results = {
        "metrics": metrics,
        "num_examples": len(predictions),
        "model_path": model_path,
        "generation_config": {
            "num_inference_steps": num_inference_steps,
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    
    if save_predictions:
        results["examples"] = [
            {
                "source": src[:500],  # Truncate for readability
                "reference": ref,
                "prediction": pred,
            }
            for src, ref, pred in zip(sources[:50], references[:50], predictions[:50])
        ]
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Summarization Model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation",
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
        "--no_bertscore",
        action="store_true",
        help="Skip BERTScore computation (faster)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    setup_logging("INFO")
    
    evaluate_model(
        model_path=args.model_path,
        output_path=args.output,
        batch_size=args.batch_size,
        num_inference_steps=args.num_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        compute_bertscore=not args.no_bertscore,
        device=args.device,
    )


if __name__ == "__main__":
    main()
