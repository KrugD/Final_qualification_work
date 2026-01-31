import torch
import time
from transformers import AutoTokenizer

# Force CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Check if mamba_ssm is available
try:
    from mamba_ssm import Mamba
    print("mamba_ssm is available")
except ImportError:
    print("mamba_ssm not available - using fallback implementation")

print("\n" + "="*60)
print("Step 1: Loading tokenizer and sample batch...")
print("="*60)

from src.data import get_sample_batch

tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruT5-base")
batch = get_sample_batch(
    tokenizer=tokenizer,
    batch_size=2,  # Small batch for CPU
    max_source_length=256,  # Reduced for faster testing
    max_target_length=64,   # Reduced for faster testing
)

print(f"Batch loaded successfully!")
print(f"  - input_ids shape: {batch['input_ids'].shape}")
print(f"  - attention_mask shape: {batch['attention_mask'].shape}")
print(f"  - labels shape: {batch['labels'].shape}")
print(f"  - labels_attention_mask shape: {batch['labels_attention_mask'].shape}")

# Move batch to device
batch = {k: v.to(device) for k, v in batch.items()}

print("\n" + "="*60)
print("Step 2: Initializing model...")
print("="*60)

from src.model import MaskedDiffusionSummarizer

# Use smaller model for CPU testing
model = MaskedDiffusionSummarizer(
    encoder_name="ai-forever/ruT5-base",
    num_decoder_layers=2,  # Reduced for faster testing
    num_diffusion_steps=5,  # Reduced for faster testing  
    max_target_length=64,
    dropout=0.1,
    schedule_type="cosine",
    use_semantic_noise=True,
    similarity_loss_weight=0.1,
    decoder_type="mamba",  # Will use fallback if mamba_ssm not available
    mamba_state_size=8,    # Reduced for faster testing
    mamba_conv_kernel=4,
    mamba_expand_factor=2,
)

model = model.to(device)
model.train()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model initialized successfully!")
print(f"  - Total parameters: {total_params:,}")
print(f"  - Trainable parameters: {trainable_params:,}")
print(f"  - Decoder type: {model.decoder_type}")

print("\n" + "="*60)
print("Step 3: Running forward pass...")
print("="*60)

start_time = time.time()

outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
    labels_attention_mask=batch["labels_attention_mask"],
)

forward_time = time.time() - start_time

print(f"Forward pass completed in {forward_time:.2f}s")
print(f"  - Total loss: {outputs['loss'].item():.4f}")
print(f"  - Diffusion loss: {outputs['diffusion_loss'].item():.4f}")
print(f"  - Similarity loss: {outputs['similarity_loss'].item():.4f}")
print(f"  - Logits shape: {outputs['logits'].shape}")
print(f"  - Noise masks shape: {outputs['noise_masks'].shape}")

print("\n" + "="*60)
print("Step 4: Running backward pass...")
print("="*60)

start_time = time.time()

loss = outputs["loss"]
loss.backward()

backward_time = time.time() - start_time

print(f"Backward pass completed in {backward_time:.2f}s")

# Check gradients
grad_norms = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norms.append((name, param.grad.norm().item()))

print(f"  - Parameters with gradients: {len(grad_norms)}")
if grad_norms:
    max_grad = max(grad_norms, key=lambda x: x[1])
    min_grad = min(grad_norms, key=lambda x: x[1])
    print(f"  - Max gradient norm: {max_grad[1]:.6f} ({max_grad[0]})")
    print(f"  - Min gradient norm: {min_grad[1]:.6f} ({min_grad[0]})")

print("\n" + "="*60)
print("Step 5: Testing optimizer step...")
print("="*60)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

start_time = time.time()
optimizer.step()
optimizer.zero_grad()
optimizer_time = time.time() - start_time

print(f"Optimizer step completed in {optimizer_time:.4f}s")

print("\n" + "="*60)
print("Step 6: Testing generation (inference)...")
print("="*60)

model.eval()

start_time = time.time()

with torch.no_grad():
    generated_ids, confidence = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        max_length=32,  # Short for testing
        num_inference_steps=3,  # Reduced for testing
    )

generation_time = time.time() - start_time

print(f"Generation completed in {generation_time:.2f}s")
print(f"  - Generated shape: {generated_ids.shape}")
print(f"  - Confidence shape: {confidence.shape}")

# Decode and show example
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
source_text = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
target_text = tokenizer.decode(batch["labels"][0], skip_special_tokens=True)

print("\nExample output:")
print(f"  Source (truncated): {source_text[:200]}...")
print(f"  Target: {target_text}")
print(f"  Generated: {generated_text}")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print(f"""
Summary:
  - Forward pass: {forward_time:.2f}s
  - Backward pass: {backward_time:.2f}s
  - Optimizer step: {optimizer_time:.4f}s
  - Generation: {generation_time:.2f}s
  - Total time: {forward_time + backward_time + optimizer_time + generation_time:.2f}s

The model is ready for training!

For full training on GPU, run:
  python train.py --config config/train_config.yaml

For quick debug run (small subset), add to config/train_config.yaml:
  data:
    train_subset_size: 100  # Use only 100 samples
""")
