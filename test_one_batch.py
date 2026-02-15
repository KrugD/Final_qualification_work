"""
Quick sanity check: one batch forward + backward + generation.
Verifies the architecture works before launching full training.

Usage:
    python test_one_batch.py
"""

import torch
import time
from transformers import AutoTokenizer
from src.model import MaskedDiffusionSummarizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # --- 1. Create model ---
    print("\n" + "="*60)
    print("1. Creating model...")
    t0 = time.time()
    
    model = MaskedDiffusionSummarizer(
        encoder_name="ai-forever/ruT5-base",
        num_decoder_layers=6,
        num_diffusion_steps=50,
        max_target_length=128,
        dropout=0.1,
        schedule_type="cosine",
        use_semantic_noise=True,
        similarity_loss_weight=1.0,
        decoder_type="mamba",
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable:    {trainable_params:,}")
    print(f"   Time: {time.time() - t0:.1f}s")
    
    # --- 2. Create a fake batch ---
    print("\n" + "="*60)
    print("2. Creating fake batch...")
    
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruT5-base")
    
    sources = [
        "Кратко суммаризируй текст: Москва — столица России, крупнейший город страны. "
        "Население Москвы составляет более 12 миллионов человек. Город является важнейшим "
        "экономическим, политическим и культурным центром страны.",
        "Расскажи основной смысл: Искусственный интеллект развивается стремительными темпами. "
        "Нейронные сети используются в медицине, финансах и образовании. Многие эксперты "
        "считают, что ИИ изменит рынок труда в ближайшие десятилетия.",
    ]
    targets = [
        "Москва — столица и крупнейший город России с населением более 12 млн человек.",
        "ИИ быстро развивается и применяется во многих сферах, что изменит рынок труда.",
    ]
    
    src_enc = tokenizer(sources, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    tgt_enc = tokenizer(targets, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    
    batch = {
        "input_ids": src_enc["input_ids"].to(device),
        "attention_mask": src_enc["attention_mask"].to(device),
        "labels": tgt_enc["input_ids"].to(device),
        "labels_attention_mask": tgt_enc["attention_mask"].to(device),
    }
    
    print(f"   Source shape:  {batch['input_ids'].shape}")
    print(f"   Target shape:  {batch['labels'].shape}")
    
    # --- 3. Forward pass ---
    print("\n" + "="*60)
    print("3. Forward pass (training)...")
    t0 = time.time()
    
    model.train()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        labels_attention_mask=batch["labels_attention_mask"],
    )
    
    print(f"   total_loss:          {outputs['loss'].item():.4f}")
    print(f"   diffusion_loss:      {outputs['diffusion_loss'].item():.4f}")
    print(f"   reconstruction_loss: {outputs['reconstruction_loss'].item():.4f}")
    print(f"   similarity_loss:     {outputs['similarity_loss'].item():.4f}")
    print(f"   logits shape:        {outputs['logits'].shape}")
    print(f"   noise_masks shape:   {outputs['noise_masks'].shape}")
    print(f"   masked tokens:       {outputs['noise_masks'].sum().item():.0f} / {outputs['noise_masks'].numel()}")
    print(f"   Time: {time.time() - t0:.1f}s")
    
    # --- 4. Backward pass ---
    print("\n" + "="*60)
    print("4. Backward pass...")
    t0 = time.time()
    
    outputs["loss"].backward()
    
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    
    print(f"   Gradient norm: {grad_norm:.2f}")
    print(f"   Time: {time.time() - t0:.1f}s")
    
    # Check for NaN/Inf
    has_nan = any(p.grad is not None and torch.isnan(p.grad).any() for p in model.parameters())
    has_inf = any(p.grad is not None and torch.isinf(p.grad).any() for p in model.parameters())
    print(f"   NaN in gradients: {has_nan}")
    print(f"   Inf in gradients: {has_inf}")
    
    model.zero_grad()
    
    # --- 5. Generation ---
    print("\n" + "="*60)
    print("5. Generation (10 steps, sample=True, temp=0.9, top_k=50)...")
    t0 = time.time()
    
    model.eval()
    with torch.no_grad():
        generated_ids, confidence = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_length=128,
            num_inference_steps=10,
            temperature=0.9,
            top_k=50,
            sample=True,
        )
    
    print(f"   Generated shape: {generated_ids.shape}")
    print(f"   Avg confidence:  {confidence.mean().item():.4f}")
    print(f"   Time: {time.time() - t0:.1f}s")
    
    # Decode and show
    predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    for i in range(len(sources)):
        print(f"\n--- Sample {i+1} ---")
        print(f"  Source:     {sources[i][:120]}...")
        print(f"  Target:     {targets[i]}")
        print(f"  Prediction: {predictions[i][:200]}")
        
        # Token stats
        raw_tokens = tokenizer.convert_ids_to_tokens(generated_ids[i])
        unique = len(set(raw_tokens))
        total = len([t for t in raw_tokens if t not in ["<pad>", "</s>"]])
        print(f"  Tokens: {total} total, {unique} unique, confidence={confidence[i].mean().item():.3f}")
    
    # --- 6. Summary ---
    print("\n" + "="*60)
    print("SANITY CHECK SUMMARY:")
    print("="*60)
    
    checks = [
        ("Forward pass",     not torch.isnan(outputs["loss"])),
        ("Loss is finite",   torch.isfinite(outputs["loss"])),
        ("Backward pass",    not has_nan and not has_inf),
        ("Grad norm > 0",    grad_norm > 0),
        ("Generation runs",  generated_ids.shape == (2, 128)),
        ("Output not empty",  any(len(p) > 0 for p in predictions)),
    ]
    
    all_ok = True
    for name, passed in checks:
        status = "OK" if passed else "FAIL"
        if not passed:
            all_ok = False
        print(f"  [{status}] {name}")
    
    print()
    if all_ok:
        print("All checks PASSED! Model is ready for training.")
        print("Note: predictions are random (untrained model) — this is expected.")
    else:
        print("Some checks FAILED. Fix issues before training.")


if __name__ == "__main__":
    main()
