import os
import shutil
import argparse
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import yaml

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, T5ForConditionalGeneration

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
    warmup_steps_override: Optional[int] = None,
) -> tuple:
    """Create optimizer and learning rate scheduler with per-group LRs."""
    num_total = sum(p.numel() for p in model.parameters())

    # Separate encoder vs non-encoder trainable params for different LRs
    encoder_lr = config["training"].get("encoder_learning_rate", None)
    decoder_lr = config["training"]["learning_rate"]

    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder.encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    param_groups = [{"params": decoder_params, "lr": decoder_lr}]
    if encoder_params and encoder_lr is not None:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})

    num_trainable = sum(p.numel() for group in param_groups for p in group["params"])
    logger.info(f"Parameters: {num_total:,} total, {num_trainable:,} trainable")
    if encoder_params:
        enc_count = sum(p.numel() for p in encoder_params)
        logger.info(f"  Encoder trainable: {enc_count:,} (lr={encoder_lr})")
        logger.info(f"  Decoder trainable: {num_trainable - enc_count:,} (lr={decoder_lr})")

    optimizer = AdamW(
        param_groups,
        lr=decoder_lr,
        weight_decay=config["training"].get("weight_decay", 0.01),
        betas=(0.9, 0.999),
    )

    warmup_steps = warmup_steps_override or config["training"].get("warmup_steps", 1000)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    t_max = max(1, num_training_steps - warmup_steps)

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=decoder_lr * 0.1,
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
                    num_inference_steps=10,
                    temperature=0.9,
                    top_k=50,
                    sample=True,
                )
                
                # Decode predictions, references, and sources
                predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                sources = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                
                # Log first example for debugging
                if len(all_predictions) == 0 and accelerator.is_main_process:
                    raw_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
                    num_mask = sum(1 for t in raw_tokens if t == "▁" or "extra_id" in t)
                    num_pad = sum(1 for t in raw_tokens if t == "</s>" or t == "<pad>")
                    logger.info(
                        f"Generation debug: total_tokens={len(raw_tokens)}, "
                        f"mask/special={num_mask}, pad={num_pad}, "
                        f"prediction_len={len(predictions[0].split())}, "
                        f"prediction='{predictions[0][:200]}'"
                    )
                
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
        
        # BERTScore (skip if bert-score not installed or model unavailable)
        try:
            bertscore_scores = compute_bertscore(
                preds, refs,
                model_type="bert-base-multilingual-cased",
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            metrics.update({f"eval_{k}": v for k, v in bertscore_scores.items()})
        except Exception as e:
            logger.warning(f"BERTScore computation skipped: {e}")
        
        # Compression ratio
        compression_scores = compute_compression_ratio(preds, srcs)
        metrics.update({f"eval_{k}": v for k, v in compression_scores.items()})
    
    model.train()
    return metrics, all_predictions[:5], all_references[:5]


def train(config: Dict[str, Any], resume_from: Optional[str] = None):
    """Main training function with distillation, self-conditioning, and curriculum unfreezing."""

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 1),
        mixed_precision=config["training"].get("mixed_precision", "no"),
        log_with="all",
    )

    setup_logging(config.get("log_level", "INFO"))

    if accelerator.is_main_process:
        logger.info(f"Starting training with config: {config}")

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
        similarity_loss_weight=config["model"].get("similarity_loss_weight", 1.0),
        decoder_type=config["model"].get("decoder_type", "mamba"),
        mamba_state_size=config["model"].get("mamba_state_size", 16),
        mamba_conv_kernel=config["model"].get("mamba_conv_kernel", 4),
        mamba_expand_factor=config["model"].get("mamba_expand_factor", 2),
        freeze_encoder=config["model"].get("freeze_encoder", False),
        use_self_conditioning=config["model"].get("use_self_conditioning", False),
        self_cond_prob=config["model"].get("self_cond_prob", 0.5),
        unfreeze_encoder_layers=0,
        gradient_checkpointing=config["model"].get("gradient_checkpointing", False),
    )

    # ── Teacher model for knowledge distillation ──────────────────────
    distill_cfg = config.get("distillation", {})
    distill_enabled = distill_cfg.get("enabled", False)
    teacher_model = None

    if distill_enabled:
        teacher_name = distill_cfg["teacher_model"]
        logger.info(f"Loading teacher model for distillation: {teacher_name}")
        teacher_model = T5ForConditionalGeneration.from_pretrained(teacher_name)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model = teacher_model.to(accelerator.device)
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        logger.info(f"Teacher model loaded: {teacher_params:,} params (frozen)")

    distill_weight = distill_cfg.get("weight", 1.0)
    distill_temp = distill_cfg.get("temperature", 2.0)

    # ── Curriculum unfreezing config ──────────────────────────────────
    unfreeze_after_steps = config["model"].get("unfreeze_after_steps", 0)
    unfreeze_n_layers = config["model"].get("unfreeze_encoder_layers", 0)
    encoder_unfrozen = False

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
        try:
            checkpoint_name = Path(resume_from).name
            if "epoch" in checkpoint_name:
                start_epoch = int(checkpoint_name.split("epoch")[1].split("_")[0])
            if "step" in checkpoint_name:
                global_step = int(checkpoint_name.split("step")[1].split("_")[0])
        except Exception:
            pass

    # Training loop
    save_dir = Path(config["training"].get("output_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    log_every = config["logging"].get("log_every_n_steps", 100)
    save_every = config["logging"].get("save_every_n_steps", 5000)
    eval_every = config["logging"].get("eval_every_n_steps", 2000)

    best_eval_loss = float("inf")
    patience = config["training"].get("early_stopping_patience", 0)
    epochs_without_improvement = 0
    best_epoch = 0

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
            # ── Curriculum: unfreeze encoder layers at the right step ─
            if (
                unfreeze_after_steps > 0
                and unfreeze_n_layers > 0
                and not encoder_unfrozen
                and global_step >= unfreeze_after_steps
            ):
                logger.info(
                    f"Step {global_step}: curriculum unfreezing "
                    f"top {unfreeze_n_layers} encoder layers"
                )
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.unfreeze_encoder_top_layers(unfreeze_n_layers)
                encoder_unfrozen = True

                remaining_steps = num_training_steps - global_step
                optimizer, scheduler = create_optimizer_and_scheduler(
                    unwrapped, config, remaining_steps,
                    warmup_steps_override=min(1000, remaining_steps // 4),
                )
                optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
                logger.info("Optimizer rebuilt with encoder param group")

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    labels_attention_mask=batch["labels_attention_mask"],
                )

                loss = outputs["loss"]

                # ── Knowledge distillation loss ───────────────────────
                kd_loss_val = 0.0
                if teacher_model is not None:
                    with torch.no_grad():
                        decoder_input_ids = teacher_model._shift_right(batch["labels"])
                        teacher_out = teacher_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            decoder_input_ids=decoder_input_ids,
                        )
                        teacher_logits = teacher_out.logits

                    noise_masks = outputs["noise_masks"]
                    student_logits = outputs["logits"]

                    student_log_probs = F.log_softmax(
                        student_logits / distill_temp, dim=-1
                    )
                    teacher_probs = F.softmax(
                        teacher_logits / distill_temp, dim=-1
                    )

                    kd_raw = F.kl_div(
                        student_log_probs, teacher_probs, reduction="none"
                    )
                    kd_per_token = kd_raw.sum(dim=-1)
                    kd_loss = (kd_per_token * noise_masks.float()).sum() / (
                        noise_masks.float().sum() + 1e-8
                    )
                    kd_loss = kd_loss * (distill_temp ** 2)

                    loss = loss + distill_weight * kd_loss
                    kd_loss_val = kd_loss.item()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if config["training"].get("max_grad_norm"):
                        _grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(),
                            config["training"]["max_grad_norm"],
                        )
                        if isinstance(_grad_norm, torch.Tensor):
                            _grad_norm = _grad_norm.item()
                    else:
                        _grad_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                _grad_norm += p.grad.data.norm(2).item() ** 2
                        _grad_norm = _grad_norm ** 0.5

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            progress_bar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

            # Log metrics
            if global_step % log_every == 0 and accelerator.is_main_process:
                grad_norm = _grad_norm

                metrics = {
                    "train_loss": loss.item(),
                    "diffusion_loss": outputs.get("diffusion_loss", loss).item(),
                    "reconstruction_loss": outputs.get("reconstruction_loss", torch.tensor(0.0)).item(),
                    "similarity_loss": outputs.get("similarity_loss", torch.tensor(0.0)).item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                    "epoch": epoch + 1,
                }
                if distill_enabled:
                    metrics["distillation_loss"] = kd_loss_val
                log_metrics(experiment, metrics, step=global_step)
                logger.info(
                    f"Step {global_step}: loss={loss.item():.4f}, "
                    f"diff={outputs.get('diffusion_loss', loss).item():.4f}, "
                    f"recon={outputs.get('reconstruction_loss', torch.tensor(0.0)).item():.4f}, "
                    f"sim={outputs.get('similarity_loss', torch.tensor(0.0)).item():.4f}, "
                    f"kd={kd_loss_val:.4f}, "
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

                    if eval_metrics["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_metrics["eval_loss"]
                        best_epoch = epoch + 1
                        best_model_dir = save_dir / "best_model"

                        accelerator.save_state(str(best_model_dir))

                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(str(best_model_dir / "weights"))

                        best_metrics = {
                            "eval_loss": best_eval_loss,
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            **{k: v for k, v in eval_metrics.items() if k != "eval_loss"},
                        }
                        torch.save(best_metrics, str(best_model_dir / "best_metrics.pt"))

                        logger.info(
                            f"New best model at step {global_step} "
                            f"eval_loss={best_eval_loss:.4f}, "
                            f"rouge1={eval_metrics.get('eval_rouge1', 0):.4f}, "
                            f"rougeL={eval_metrics.get('eval_rougeL', 0):.4f}"
                        )

            # Save checkpoint (keep only last 2 to save disk space)
            if global_step % save_every == 0:
                checkpoint_dir = save_dir / f"checkpoint_epoch{epoch+1}_step{global_step}"
                accelerator.save_state(str(checkpoint_dir))
                if accelerator.is_main_process:
                    logger.info(f"Checkpoint saved to {checkpoint_dir}")
                    existing = sorted(
                        [d for d in save_dir.iterdir()
                         if d.is_dir() and d.name.startswith("checkpoint_epoch")],
                        key=lambda d: d.stat().st_mtime,
                    )
                    while len(existing) > 2:
                        old = existing.pop(0)
                        shutil.rmtree(old)
                        logger.info(f"Removed old checkpoint: {old.name}")

        # End of epoch
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            log_metrics(experiment, {"epoch_loss": avg_epoch_loss}, epoch=epoch + 1)

        if patience > 0:
            if best_epoch == epoch + 1:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"No improvement for {epochs_without_improvement}/{patience} epochs "
                    f"(best was epoch {best_epoch}, eval_loss={best_eval_loss:.4f})"
                )
                if epochs_without_improvement >= patience:
                    logger.info(
                        f"Early stopping triggered after {patience} epochs. "
                        f"Best eval_loss: {best_eval_loss:.4f} at epoch {best_epoch}."
                    )
                    break

    # Save final model
    final_model_dir = save_dir / "final_model"
    accelerator.save_state(str(final_model_dir))

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        inference_dir = save_dir / "model_for_inference"
        unwrapped_model.save_pretrained(str(inference_dir))

        training_summary = {
            "total_steps": global_step,
            "total_epochs": num_epochs,
            "best_eval_loss": best_eval_loss,
            "final_train_loss": avg_epoch_loss,
        }
        torch.save(training_summary, str(save_dir / "training_summary.pt"))

        logger.info(
            f"Training completed. Final model saved to {inference_dir}\n"
            f"  Best eval_loss: {best_eval_loss:.4f}\n"
            f"  Total steps: {global_step}\n"
            f"  Best model weights: {save_dir / 'best_model' / 'weights'}"
        )

        if experiment:
            log_model(experiment, str(inference_dir))
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
