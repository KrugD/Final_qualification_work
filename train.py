import os
import argparse
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import yaml

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer

from src.model import MaskedDiffusionSummarizer
from src.data import create_dataloaders
from src.utils import (
    setup_comet_logging,
    log_metrics,
    log_samples,
    log_hyperparameters,
    log_model,
    setup_logging,
    compute_rouge,
    compute_bertscore,
    compute_compression_ratio,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_optimizer_and_scheduler(
    model: nn.Module,
    config: Dict[str, Any],
    num_training_steps: int,
) -> tuple:
    """Create optimizer and learning rate scheduler."""
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.01),
        betas=(0.9, 0.999),
    )
    
    # Warmup + Cosine annealing scheduler
    warmup_steps = config["training"].get("warmup_steps", 1000)
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    
    # Ensure T_max is at least 1 to avoid division by zero
    t_max = max(1, num_training_steps - warmup_steps)
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=config["training"]["learning_rate"] * 0.1,
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    
    return optimizer, scheduler


def evaluate(
    model: MaskedDiffusionSummarizer,
    dataloader,
    accelerator: Accelerator,
    tokenizer: AutoTokenizer,
    max_samples: int = 100,
) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    
    all_predictions = []
    all_references = []
    all_sources = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            # Compute loss
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                labels_attention_mask=batch["labels_attention_mask"],
            )
            total_loss += outputs["loss"].item()
            num_batches += 1
            
            # Generate predictions (only for a subset)
            if len(all_predictions) < max_samples:
                generated_ids, _ = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=batch["labels"].shape[1],
                )
                
                # Decode predictions, references, and sources
                predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                sources = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                
                all_predictions.extend(predictions)
                all_references.extend(references)
                all_sources.extend(sources)
    
    # Gather predictions from all processes
    all_predictions = accelerator.gather_for_metrics(all_predictions)
    all_references = accelerator.gather_for_metrics(all_references)
    all_sources = accelerator.gather_for_metrics(all_sources)
    
    # Compute metrics on main process
    metrics = {"eval_loss": total_loss / max(num_batches, 1)}
    
    if accelerator.is_main_process and all_predictions:
        preds = all_predictions[:max_samples]
        refs = all_references[:max_samples]
        srcs = all_sources[:max_samples]
        
        # ROUGE scores
        rouge_scores = compute_rouge(preds, refs)
        metrics.update({f"eval_{k}": v for k, v in rouge_scores.items()})
        
        # BERTScore
        try:
            bertscore_scores = compute_bertscore(preds, refs)
            metrics.update({f"eval_{k}": v for k, v in bertscore_scores.items()})
        except Exception as e:
            logger.warning(f"BERTScore computation failed: {e}")
        
        # Compression ratio
        compression_scores = compute_compression_ratio(preds, srcs)
        metrics.update({f"eval_{k}": v for k, v in compression_scores.items()})
    
    model.train()
    return metrics, all_predictions[:5], all_references[:5]


def train(config: Dict[str, Any], resume_from: Optional[str] = None):
    """Main training function."""
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        mixed_precision=config["training"].get("mixed_precision", "no"),
        log_with="all",
    )
    
    # Setup logging
    setup_logging(config.get("log_level", "INFO"))
    
    if accelerator.is_main_process:
        logger.info(f"Starting training with config: {config}")
    
    # Set seed for reproducibility
    set_seed(config.get("seed", 42))
    
    # Setup CometML
    experiment = None
    if accelerator.is_main_process:
        experiment = setup_comet_logging(
            project_name=config["logging"].get("comet_project", "diffusion-summarization"),
            experiment_name=config["logging"].get("experiment_name"),
            tags=config["logging"].get("tags", ["diffusion", "summarization", "russian"]),
            disabled=config["logging"].get("disable_comet", False),
        )
        if experiment:
            log_hyperparameters(experiment, config)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder"])
    
    # Create dataloaders
    train_dataloader, test_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=config["training"]["batch_size"],
        max_source_length=config["data"]["max_source_length"],
        max_target_length=config["data"]["max_target_length"],
        num_workers=config["data"].get("num_workers", 4),
        dataset_name=config["data"]["dataset"],
        cache_dir=config["data"].get("cache_dir"),
        train_subset_size=config["data"].get("train_subset_size"),
    )
    
    # Create model
    model = MaskedDiffusionSummarizer(
        encoder_name=config["model"]["encoder"],
        num_decoder_layers=config["model"].get("num_decoder_layers", 6),
        num_diffusion_steps=config["model"].get("num_diffusion_steps", 20),
        max_target_length=config["data"]["max_target_length"],
        dropout=config["model"].get("dropout", 0.1),
        schedule_type=config["model"].get("schedule_type", "cosine"),
        use_semantic_noise=config["model"].get("use_semantic_noise", True),
        similarity_loss_weight=config["model"].get("similarity_loss_weight", 0.1),
        decoder_type=config["model"].get("decoder_type", "mamba"),
        mamba_state_size=config["model"].get("mamba_state_size", 16),
        mamba_conv_kernel=config["model"].get("mamba_conv_kernel", 4),
        mamba_expand_factor=config["model"].get("mamba_expand_factor", 2),
    )
    
    # Calculate training steps
    num_epochs = config["training"]["num_epochs"]
    gradient_accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
    num_training_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        accelerator.load_state(resume_from)
        # Try to extract epoch and step from checkpoint name
        try:
            checkpoint_name = Path(resume_from).name
            if "epoch" in checkpoint_name:
                start_epoch = int(checkpoint_name.split("epoch")[1].split("_")[0])
            if "step" in checkpoint_name:
                global_step = int(checkpoint_name.split("step")[1].split("_")[0])
        except:
            pass
    
    # Training loop
    save_dir = Path(config["training"].get("output_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    log_every = config["logging"].get("log_every_n_steps", 100)
    save_every = config["logging"].get("save_every_n_steps", 5000)
    eval_every = config["logging"].get("eval_every_n_steps", 1000)
    
    best_eval_loss = float("inf")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )
        
        for batch in progress_bar:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    labels_attention_mask=batch["labels_attention_mask"],
                )
                
                loss = outputs["loss"]
                accelerator.backward(loss)
                
                # Gradient clipping
                if config["training"].get("max_grad_norm"):
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        config["training"]["max_grad_norm"],
                    )
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
            
            # Log metrics
            if global_step % log_every == 0 and accelerator.is_main_process:
                # Compute gradient norm
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
                
                metrics = {
                    "train_loss": loss.item(),
                    "diffusion_loss": outputs.get("diffusion_loss", loss).item(),
                    "similarity_loss": outputs.get("similarity_loss", torch.tensor(0.0)).item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    "epoch": epoch + 1,
                }
                log_metrics(experiment, metrics, step=global_step)
                logger.info(
                    f"Step {global_step}: loss={loss.item():.4f}, "
                    f"diff_loss={outputs.get('diffusion_loss', loss).item():.4f}, "
                    f"sim_loss={outputs.get('similarity_loss', torch.tensor(0.0)).item():.4f}, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
            
            # Evaluate
            if global_step % eval_every == 0:
                eval_metrics, pred_samples, ref_samples = evaluate(
                    model, test_dataloader, accelerator, tokenizer
                )
                
                if accelerator.is_main_process:
                    log_metrics(experiment, eval_metrics, step=global_step)
                    logger.info(f"Step {global_step} evaluation: {eval_metrics}")
                    
                    # Log samples
                    if pred_samples and ref_samples:
                        sources = tokenizer.batch_decode(
                            next(iter(test_dataloader))["input_ids"][:5],
                            skip_special_tokens=True,
                        )
                        samples = [
                            {"source": s, "target": t, "prediction": p}
                            for s, t, p in zip(sources, ref_samples, pred_samples)
                        ]
                        log_samples(experiment, samples, step=global_step)
                    
                    # Save best model
                    if eval_metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        best_model_dir = save_dir / "best_model"
                        accelerator.save_state(str(best_model_dir))
                        logger.info(f"New best model saved with eval_loss={best_eval_loss:.4f}")
            
            # Save checkpoint
            if global_step % save_every == 0:
                checkpoint_dir = save_dir / f"checkpoint_epoch{epoch+1}_step{global_step}"
                accelerator.save_state(str(checkpoint_dir))
                if accelerator.is_main_process:
                    logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            log_metrics(experiment, {"epoch_loss": avg_epoch_loss}, epoch=epoch + 1)
    
    # Save final model
    final_model_dir = save_dir / "final_model"
    accelerator.save_state(str(final_model_dir))
    
    # Save model in custom format for inference
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(str(save_dir / "model_for_inference"))
        logger.info(f"Training completed. Final model saved to {final_model_dir}")
        
        if experiment:
            log_model(experiment, str(save_dir / "model_for_inference"))
            experiment.end()


def main():
    parser = argparse.ArgumentParser(description="Train Masked Diffusion Summarization Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
