"""Main training script for all stages of SpeechProtocol model.

Usage:
    python -m training.train --stage 1      # Audio-text alignment
    python -m training.train --stage 1.5    # Text summarization
    python -m training.train --stage 2      # Protocol generation
"""

from __future__ import annotations

import argparse
import os
import math
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, WhisperFeatureExtractor

load_dotenv()


def get_experiment():
    """Create CometML experiment."""
    try:
        from comet_ml import Experiment

        return Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=os.getenv("COMET_PROJECT_NAME", "speech-protocol"),
        )
    except Exception:
        print("CometML not available, logging to stdout only")
        return None


def load_config(path: str = "training/train_config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(config: dict, stage: str):
    from model.config import ModelConfig
    from model.speech_protocol_model import SpeechProtocolModel

    mc = config["model"]
    model_config = ModelConfig(
        whisper_model=mc["whisper_model"],
        speaker_model=mc["speaker_model"],
        llm_model=mc["llm_model"],
        num_query_tokens=mc["num_query_tokens"],
        adapter_num_layers=mc["adapter_num_layers"],
        lora_r=mc["lora_r"],
        lora_alpha=mc["lora_alpha"],
    )

    model = SpeechProtocolModel(model_config)

    if stage == "1":
        sc = config["stage1"]
        if sc.get("train_adapter_only", False):
            for param in model.llm.parameters():
                param.requires_grad = False
            for param in model.fusion_adapter.parameters():
                param.requires_grad = True

    if config["checkpointing"].get("gradient_checkpointing", False):
        if hasattr(model.llm, "gradient_checkpointing_enable"):
            model.llm.gradient_checkpointing_enable()

    info = model.get_trainable_params_info()
    print(f"Total params: {info['total_params']:,}")
    print(f"Trainable params: {info['trainable_params']:,} ({info['trainable_pct']:.2f}%)")

    return model


def build_asr_dataset(config: dict, tokenizer, feature_extractor):
    from datasets import load_dataset
    from training.dataset import ASRDataset

    sc = config["stage1"]
    ds_name = sc["dataset"]
    subset = sc.get("dataset_subset")

    print(f"Loading ASR dataset: {ds_name} / {subset}")

    if subset:
        raw = load_dataset(ds_name, subset, split="train", trust_remote_code=True)
    else:
        raw = load_dataset(ds_name, split="train", trust_remote_code=True)

    max_samples = sc.get("max_samples")
    if max_samples and len(raw) > max_samples:
        raw = raw.select(range(max_samples))

    dataset = ASRDataset(
        hf_dataset=raw,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_audio_sec=sc["max_audio_sec"],
        augment=sc.get("augment", True),
    )

    val_size = min(1000, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    return train_ds, val_ds


def build_summarization_dataset(config: dict, tokenizer):
    from datasets import load_dataset
    from training.dataset import SummarizationDataset

    sc = config["stage1_5"]
    ds_name = sc["dataset"]

    print(f"Loading summarization dataset: {ds_name}")
    raw = load_dataset(ds_name, split="train", trust_remote_code=True)

    max_samples = sc.get("max_samples")
    if max_samples and len(raw) > max_samples:
        raw = raw.shuffle(seed=42).select(range(max_samples))

    dataset = SummarizationDataset(
        hf_dataset=raw,
        tokenizer=tokenizer,
        max_input_len=sc["max_input_len"],
        max_target_len=sc["max_target_len"],
    )

    val_size = min(500, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    return train_ds, val_ds


def build_protocol_dataset(config: dict, tokenizer, feature_extractor):
    from training.dataset import ProtocolDataset

    sc = config["stage2"]
    data_path = sc["data_path"]

    print(f"Loading protocol dataset from: {data_path}")

    dataset = ProtocolDataset(
        data_path=data_path,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_audio_sec=sc["max_audio_sec"],
        max_target_len=sc["max_target_len"],
        augment=sc.get("augment", True),
    )

    val_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    return train_ds, val_ds


def train_audio_stage(
    model,
    train_loader,
    val_loader,
    config: dict,
    stage_config: dict,
    experiment,
    stage_name: str,
):
    """Training loop for stages with audio input (Stage 1 and Stage 2)."""
    device = torch.device(config["hardware"]["device"])
    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=stage_config["lr"],
        weight_decay=stage_config["weight_decay"],
    )

    total_steps = len(train_loader) * stage_config["epochs"]
    warmup_steps = int(total_steps * stage_config["warmup_ratio"])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    grad_accum = stage_config["gradient_accumulation_steps"]
    use_fp16 = config["hardware"].get("fp16", True)
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    output_dir = Path(config["checkpointing"]["output_dir"]) / stage_name
    output_dir.mkdir(parents=True, exist_ok=True)

    save_every = config["checkpointing"].get("save_every_n_steps", 500)
    eval_every = config["checkpointing"].get("eval_every_n_steps", 250)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(stage_config["epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(device)
            waveforms = [w.to(device) for w in batch["waveforms"]]
            labels = batch["labels"].to(device)
            label_mask = batch["label_attention_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=use_fp16):
                outputs = model(
                    input_features=input_features,
                    waveforms=waveforms,
                    labels=labels,
                    label_attention_mask=label_mask,
                )
                loss = outputs["loss"] / grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()

                if global_step >= warmup_steps:
                    scheduler.step()
                else:
                    warmup_lr = stage_config["lr"] * (global_step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr

                optimizer.zero_grad()
                global_step += 1

                actual_loss = loss.item() * grad_accum
                epoch_loss += actual_loss
                num_batches += 1

                if experiment:
                    experiment.log_metric(f"{stage_name}/train_loss", actual_loss, step=global_step)
                    experiment.log_metric(f"{stage_name}/lr", optimizer.param_groups[0]["lr"], step=global_step)

                if global_step % 50 == 0:
                    print(
                        f"[{stage_name}] Epoch {epoch+1}, Step {global_step}, "
                        f"Loss: {actual_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}"
                    )

                if global_step % eval_every == 0:
                    val_loss = evaluate_audio(model, val_loader, device, use_fp16)
                    print(f"[{stage_name}] Step {global_step}, Val Loss: {val_loss:.4f}")
                    if experiment:
                        experiment.log_metric(f"{stage_name}/val_loss", val_loss, step=global_step)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(model, optimizer, global_step, output_dir / "best")
                    model.train()

                if global_step % save_every == 0:
                    save_checkpoint(model, optimizer, global_step, output_dir / f"step_{global_step}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[{stage_name}] Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
        if experiment:
            experiment.log_metric(f"{stage_name}/epoch_loss", avg_loss, epoch=epoch + 1)

    save_checkpoint(model, optimizer, global_step, output_dir / "final")
    return model


def train_text_stage(
    model,
    train_loader,
    val_loader,
    config: dict,
    stage_config: dict,
    experiment,
):
    """Training loop for Stage 1.5 (text-only summarization with LoRA)."""
    device = torch.device(config["hardware"]["device"])

    llm = model.llm.to(device)
    trainable_params = [p for p in llm.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=stage_config["lr"],
        weight_decay=stage_config["weight_decay"],
    )

    total_steps = len(train_loader) * stage_config["epochs"]
    warmup_steps = int(total_steps * stage_config["warmup_ratio"])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    grad_accum = stage_config["gradient_accumulation_steps"]
    use_fp16 = config["hardware"].get("fp16", True)
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    output_dir = Path(config["checkpointing"]["output_dir"]) / "stage1_5"
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(stage_config["epochs"]):
        llm.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            combined_ids = torch.cat([input_ids, labels], dim=1)
            ignore_input = torch.full_like(input_ids, -100)
            combined_labels = torch.cat([ignore_input, labels], dim=1)
            combined_mask = torch.cat(
                [attention_mask, torch.ones_like(labels)], dim=1
            )

            with torch.amp.autocast("cuda", enabled=use_fp16):
                outputs = llm(
                    input_ids=combined_ids,
                    attention_mask=combined_mask,
                    labels=combined_labels,
                )
                loss = outputs.loss / grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()

                if global_step >= warmup_steps:
                    scheduler.step()
                else:
                    warmup_lr = stage_config["lr"] * (global_step + 1) / warmup_steps
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr

                optimizer.zero_grad()
                global_step += 1

                actual_loss = loss.item() * grad_accum
                epoch_loss += actual_loss
                num_batches += 1

                if experiment:
                    experiment.log_metric("stage1_5/train_loss", actual_loss, step=global_step)

                if global_step % 50 == 0:
                    print(
                        f"[Stage 1.5] Epoch {epoch+1}, Step {global_step}, "
                        f"Loss: {actual_loss:.4f}"
                    )

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[Stage 1.5] Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")

        val_loss = evaluate_text(llm, val_loader, device, use_fp16)
        print(f"[Stage 1.5] Val Loss: {val_loss:.4f}")
        if experiment:
            experiment.log_metric("stage1_5/val_loss", val_loss, epoch=epoch + 1)
            experiment.log_metric("stage1_5/epoch_loss", avg_loss, epoch=epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            llm.save_pretrained(str(output_dir / "best_lora"))

    llm.save_pretrained(str(output_dir / "final_lora"))
    return model


@torch.no_grad()
def evaluate_audio(model, val_loader, device, use_fp16: bool) -> float:
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in val_loader:
        input_features = batch["input_features"].to(device)
        waveforms = [w.to(device) for w in batch["waveforms"]]
        labels = batch["labels"].to(device)
        label_mask = batch["label_attention_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=use_fp16):
            outputs = model(
                input_features=input_features,
                waveforms=waveforms,
                labels=labels,
                label_attention_mask=label_mask,
            )
        total_loss += outputs["loss"].item()
        count += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate_text(llm, val_loader, device, use_fp16: bool) -> float:
    llm.eval()
    total_loss = 0.0
    count = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        combined_ids = torch.cat([input_ids, labels], dim=1)
        ignore_input = torch.full_like(input_ids, -100)
        combined_labels = torch.cat([ignore_input, labels], dim=1)
        combined_mask = torch.cat(
            [attention_mask, torch.ones_like(labels)], dim=1
        )

        with torch.amp.autocast("cuda", enabled=use_fp16):
            outputs = llm(
                input_ids=combined_ids,
                attention_mask=combined_mask,
                labels=combined_labels,
            )
        total_loss += outputs.loss.item()
        count += 1

    return total_loss / max(count, 1)


def save_checkpoint(model, optimizer, step, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    adapter_state = model.fusion_adapter.state_dict()
    torch.save(adapter_state, path / "adapter.pt")

    model.llm.save_pretrained(str(path / "lora"))

    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step},
        path / "optimizer.pt",
    )
    print(f"Checkpoint saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Train SpeechProtocol model")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["1", "1.5", "2"],
        help="Training stage: 1 (ASR), 1.5 (summarization), 2 (protocol)",
    )
    parser.add_argument("--config", type=str, default="training/train_config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = get_experiment()

    if experiment:
        experiment.log_parameters(config)
        experiment.add_tag(f"stage_{args.stage}")

    if args.stage in ("1", "2"):
        model = build_model(config, args.stage)

        if args.resume:
            ckpt_path = Path(args.resume)
            adapter_path = ckpt_path / "adapter.pt"
            if adapter_path.exists():
                model.fusion_adapter.load_state_dict(torch.load(adapter_path, weights_only=True))
                print(f"Loaded adapter from {adapter_path}")
            lora_path = ckpt_path / "lora"
            if lora_path.exists():
                from peft import PeftModel
                model.llm = PeftModel.from_pretrained(
                    model.llm.base_model.model, str(lora_path)
                )
                print(f"Loaded LoRA from {lora_path}")

        tokenizer = model.tokenizer
        feature_extractor = model.audio_encoder.feature_extractor

        if args.stage == "1":
            sc = config["stage1"]
            train_ds, val_ds = build_asr_dataset(config, tokenizer, feature_extractor)
        else:
            sc = config["stage2"]
            train_ds, val_ds = build_protocol_dataset(config, tokenizer, feature_extractor)

        from training.collator import AudioTextCollator

        collator = AudioTextCollator(pad_token_id=tokenizer.pad_token_id)
        hw = config["hardware"]

        train_loader = DataLoader(
            train_ds,
            batch_size=sc["batch_size"],
            shuffle=True,
            num_workers=hw.get("num_workers", 4),
            collate_fn=collator,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=sc["batch_size"],
            shuffle=False,
            num_workers=hw.get("num_workers", 4),
            collate_fn=collator,
            pin_memory=True,
        )

        stage_name = "stage1" if args.stage == "1" else "stage2"
        train_audio_stage(model, train_loader, val_loader, config, sc, experiment, stage_name)

    elif args.stage == "1.5":
        model = build_model(config, "1.5")

        if args.resume:
            lora_path = Path(args.resume) / "lora"
            if lora_path.exists():
                from peft import PeftModel
                model.llm = PeftModel.from_pretrained(
                    model.llm.base_model.model, str(lora_path)
                )

        tokenizer = model.tokenizer
        sc = config["stage1_5"]

        train_ds, val_ds = build_summarization_dataset(config, tokenizer)

        from training.collator import TextOnlyCollator

        collator = TextOnlyCollator(pad_token_id=tokenizer.pad_token_id)
        hw = config["hardware"]

        train_loader = DataLoader(
            train_ds,
            batch_size=sc["batch_size"],
            shuffle=True,
            num_workers=hw.get("num_workers", 4),
            collate_fn=collator,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=sc["batch_size"],
            shuffle=False,
            num_workers=hw.get("num_workers", 4),
            collate_fn=collator,
            pin_memory=True,
        )

        train_text_stage(model, train_loader, val_loader, config, sc, experiment)

    if experiment:
        experiment.end()


if __name__ == "__main__":
    main()
