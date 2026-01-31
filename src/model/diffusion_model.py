import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel, AutoTokenizer
from typing import Dict, Optional, Tuple
import math

from .noise_scheduler import NoiseScheduler, SemanticAwareNoiseScheduler
from .mamba_decoder import CrossMambaDecoder, is_mamba_available


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TimestepEmbedding(nn.Module):
    """Embedding for diffusion timesteps."""
    
    def __init__(self, hidden_size: int, max_timesteps: int = 1000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        
        half_dim = hidden_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer("emb", emb)
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = timesteps.float()[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


class DiffusionDecoderLayer(nn.Module):
    """Transformer decoder layer with timestep conditioning (fallback)."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_attention_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_attention_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        
        self.timestep_proj = nn.Linear(hidden_size, hidden_size * 2)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        timestep_cond = self.timestep_proj(timestep_emb).unsqueeze(1)
        scale, shift = timestep_cond.chunk(2, dim=-1)
        
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states, _ = self.self_attn(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attention_mask,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states, _ = self.cross_attn(
            hidden_states, encoder_hidden_states, encoder_hidden_states,
            key_padding_mask=encoder_attention_mask,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class TransformerDiffusionDecoder(nn.Module):
    """Transformer decoder for masked diffusion (fallback when Mamba unavailable)."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, max_seq_len, dropout)
        
        self.timestep_embedding = TimestepEmbedding(hidden_size)
        
        self.layers = nn.ModuleList([
            DiffusionDecoderLayer(
                hidden_size, num_attention_heads, intermediate_size, dropout
            )
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.positional_encoding(hidden_states)
        
        timestep_emb = self.timestep_embedding(timesteps)
        
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()
        if encoder_attention_mask is not None:
            encoder_attention_mask = ~encoder_attention_mask.bool()
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states,
                timestep_emb,
                attention_mask,
                encoder_attention_mask,
            )
        
        hidden_states = self.output_norm(hidden_states)
        logits = self.output_projection(hidden_states)
        
        return logits


class SemanticEncoder(nn.Module):
    """
    Encoder that computes semantic importance scores for each token.
    Uses [CLS] token attention to determine token importance.
    """
    
    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.cls_projection = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input and compute token importance scores.
        
        Returns:
            hidden_states: Encoder outputs [batch_size, seq_len, hidden_size]
            cls_embedding: [CLS] token embedding [batch_size, hidden_size]
            attention_scores: Token importance scores [batch_size, seq_len]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        
        hidden_states = outputs.last_hidden_state
        
        # [CLS] embedding (first token)
        cls_embedding = hidden_states[:, 0, :]
        cls_embedding = self.cls_projection(cls_embedding)
        
        # Attention scores from last layer
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            last_layer_attention = outputs.attentions[-1]
            cls_attention = last_layer_attention[:, :, 0, :].mean(dim=1)
            attention_scores = cls_attention
        else:
            attention_scores = torch.ones(
                hidden_states.shape[0], hidden_states.shape[1],
                device=hidden_states.device
            ) / hidden_states.shape[1]
        
        # Normalize
        attention_scores = attention_scores / (attention_scores.sum(dim=-1, keepdim=True) + 1e-8)
        
        return hidden_states, cls_embedding, attention_scores


class MaskedDiffusionSummarizer(nn.Module):
    """
    Masked Diffusion Language Model for Summarization.
    
    Key features from NAACL 2025 paper:
    1. Semantic-Aware Noising: Uses attention scores to mask important tokens later
    2. Similarity Loss: Ensures semantic alignment between source and target
    3. CrossMamba Decoder: Mamba-based decoder with O(n) complexity
    4. Encoder-Decoder with timestep conditioning
    """
    
    def __init__(
        self,
        encoder_name: str = "ai-forever/ruT5-base",
        num_decoder_layers: int = 6,
        num_diffusion_steps: int = 20,
        max_target_length: int = 128,
        dropout: float = 0.1,
        schedule_type: str = "cosine",
        use_semantic_noise: bool = True,
        similarity_loss_weight: float = 0.1,
        decoder_type: str = "mamba",  # "mamba" or "transformer"
        mamba_state_size: int = 16,
        mamba_conv_kernel: int = 4,
        mamba_expand_factor: int = 2,
    ):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.num_diffusion_steps = num_diffusion_steps
        self.max_target_length = max_target_length
        self.use_semantic_noise = use_semantic_noise
        self.similarity_loss_weight = similarity_loss_weight
        self.decoder_type = decoder_type
        
        # Load pretrained encoder
        base_encoder = T5EncoderModel.from_pretrained(encoder_name)
        encoder_config = base_encoder.config
        
        # Wrap with semantic encoder
        self.encoder = SemanticEncoder(base_encoder, encoder_config.d_model)
        
        # Get tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        vocab_size = len(self.tokenizer)
        
        # Mask token
        if self.tokenizer.mask_token_id is not None:
            self.mask_token_id = self.tokenizer.mask_token_id
        else:
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # Initialize decoder based on type
        if decoder_type == "mamba":
            print(f"Using CrossMamba decoder (Mamba available: {is_mamba_available()})")
            self.decoder = CrossMambaDecoder(
                vocab_size=vocab_size,
                hidden_size=encoder_config.d_model,
                num_layers=num_decoder_layers,
                num_heads=encoder_config.num_heads,
                intermediate_size=encoder_config.d_ff,
                max_seq_len=max_target_length,
                state_size=mamba_state_size,
                conv_kernel=mamba_conv_kernel,
                expand_factor=mamba_expand_factor,
                dropout=dropout,
            )
        else:
            print("Using Transformer decoder (fallback)")
            self.decoder = TransformerDiffusionDecoder(
                vocab_size=vocab_size,
                hidden_size=encoder_config.d_model,
                num_layers=num_decoder_layers,
                num_attention_heads=encoder_config.num_heads,
                intermediate_size=encoder_config.d_ff,
                max_seq_len=max_target_length,
                dropout=dropout,
            )
        
        # Target CLS projection for similarity loss
        self.target_cls_projection = nn.Linear(encoder_config.d_model, encoder_config.d_model)
        
        # Initialize noise scheduler
        if use_semantic_noise:
            self.noise_scheduler = SemanticAwareNoiseScheduler(
                num_diffusion_steps=num_diffusion_steps,
                mask_token_id=self.mask_token_id,
                schedule_type=schedule_type,
            )
        else:
            self.noise_scheduler = NoiseScheduler(
                num_diffusion_steps=num_diffusion_steps,
                mask_token_id=self.mask_token_id,
                schedule_type=schedule_type,
            )
    
    def compute_target_cls(
        self,
        labels: torch.Tensor,
        labels_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute [CLS] embedding for target sequence using mean pooling."""
        target_embeds = self.decoder.token_embedding(labels)
        
        mask = labels_attention_mask.unsqueeze(-1).float()
        pooled = (target_embeds * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        cls_embedding = self.target_cls_projection(pooled)
        
        return cls_embedding
    
    def compute_similarity_loss(
        self,
        source_cls: torch.Tensor,
        target_cls: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity loss between source and target [CLS] tokens.
        
        From NAACL 2025 paper Eq. 4:
        Lcls = 1 - cos(Cs, Ct)
        """
        source_cls = F.normalize(source_cls, p=2, dim=-1)
        target_cls = F.normalize(target_cls.detach(), p=2, dim=-1)
        
        similarity = (source_cls * target_cls).sum(dim=-1)
        loss = 1 - similarity.mean()
        
        return loss
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        labels_attention_mask: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Uses SOURCE attention for semantic-aware noising (per NAACL 2025).
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Encode source with attention scores
        encoder_hidden_states, source_cls, source_attention = self.encoder(
            input_ids, attention_mask
        )
        
        # Compute target [CLS] for similarity loss
        target_cls = self.compute_target_cls(labels, labels_attention_mask)
        
        # Sample timesteps if not provided
        if timesteps is None:
            timesteps = self.noise_scheduler.sample_timesteps(batch_size, device)
        
        # Apply noise to labels using SOURCE attention for importance
        if self.use_semantic_noise:
            # Use SOURCE attention scores to determine which TARGET tokens to mask
            # This aligns target generation with source importance
            # Interpolate source attention to target length if needed
            source_len = source_attention.shape[1]
            target_len = labels.shape[1]
            
            if source_len != target_len:
                # Use average importance from source for all target positions
                avg_importance = source_attention.mean(dim=1, keepdim=True)
                target_importance = avg_importance.expand(-1, target_len)
            else:
                target_importance = source_attention
            
            noisy_labels_list = []
            noise_masks_list = []
            
            for i in range(batch_size):
                t_ratio = (timesteps[i].item() + 1) / self.num_diffusion_steps
                noisy, mask = self.noise_scheduler.add_semantic_noise(
                    labels[i:i+1],
                    target_importance[i:i+1],
                    t_ratio,
                    labels_attention_mask[i:i+1],
                )
                noisy_labels_list.append(noisy)
                noise_masks_list.append(mask)
            
            noisy_labels = torch.cat(noisy_labels_list, dim=0)
            noise_masks = torch.cat(noise_masks_list, dim=0)
        else:
            mask_ratios = self.noise_scheduler.get_mask_ratio_for_training(timesteps)
            
            noisy_labels_list = []
            noise_masks_list = []
            
            for i in range(batch_size):
                noisy, mask = self.noise_scheduler.add_noise(
                    labels[i:i+1],
                    mask_ratios[i].item(),
                    labels_attention_mask[i:i+1],
                )
                noisy_labels_list.append(noisy)
                noise_masks_list.append(mask)
            
            noisy_labels = torch.cat(noisy_labels_list, dim=0)
            noise_masks = torch.cat(noise_masks_list, dim=0)
        
        # Decode
        logits = self.decoder(
            input_ids=noisy_labels,
            encoder_hidden_states=encoder_hidden_states,
            timesteps=timesteps,
            attention_mask=labels_attention_mask,
            encoder_attention_mask=attention_mask,
        )
        
        # Diffusion loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.pad_token_id,
            reduction="none",
        )
        ce_loss = ce_loss.view(batch_size, -1)
        diffusion_loss = (ce_loss * noise_masks.float()).sum() / (noise_masks.float().sum() + 1e-8)
        
        # Similarity loss
        similarity_loss = self.compute_similarity_loss(source_cls, target_cls)
        
        # Total loss
        total_loss = diffusion_loss + self.similarity_loss_weight * similarity_loss
        
        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "similarity_loss": similarity_loss,
            "logits": logits,
            "noise_masks": noise_masks,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate summaries using iterative denoising."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        max_length = max_length or self.max_target_length
        num_inference_steps = num_inference_steps or self.num_diffusion_steps
        
        # Encode source
        encoder_hidden_states, _, attention_scores = self.encoder(input_ids, attention_mask)
        
        # Initialize with fully masked sequence
        generated_ids = torch.full(
            (batch_size, max_length),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        
        target_attention_mask = torch.ones(
            batch_size, max_length, dtype=torch.long, device=device
        )
        
        # Iterative denoising
        for step in range(num_inference_steps):
            t = num_inference_steps - step - 1
            timesteps = torch.full((batch_size,), t, dtype=torch.long, device=device)
            
            logits = self.decoder(
                input_ids=generated_ids,
                encoder_hidden_states=encoder_hidden_states,
                timesteps=timesteps,
                attention_mask=target_attention_mask,
                encoder_attention_mask=attention_mask,
            )
            
            logits = logits / temperature
            
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            predicted_ids = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values
            
            mask_ratio = self.noise_scheduler.get_mask_ratio_for_timestep(t)
            is_masked = generated_ids == self.mask_token_id
            
            for b in range(batch_size):
                if not is_masked[b].any():
                    continue
                
                masked_positions = is_masked[b].nonzero().squeeze(-1)
                masked_confidence = confidence[b, masked_positions]
                
                num_unmask = max(1, int((1 - mask_ratio) * max_length) - (max_length - is_masked[b].sum().item()))
                num_unmask = min(num_unmask, len(masked_positions))
                
                if num_unmask > 0:
                    _, top_indices = masked_confidence.topk(num_unmask)
                    positions_to_unmask = masked_positions[top_indices]
                    generated_ids[b, positions_to_unmask] = predicted_ids[b, positions_to_unmask]
        
        # Final pass
        is_masked = generated_ids == self.mask_token_id
        if is_masked.any():
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
            logits = self.decoder(
                input_ids=generated_ids,
                encoder_hidden_states=encoder_hidden_states,
                timesteps=timesteps,
                attention_mask=target_attention_mask,
                encoder_attention_mask=attention_mask,
            )
            predicted_ids = logits.argmax(dim=-1)
            generated_ids[is_masked] = predicted_ids[is_masked]
        
        confidence_scores = F.softmax(logits, dim=-1).max(dim=-1).values
        
        return generated_ids, confidence_scores
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        torch.save(self.state_dict(), os.path.join(save_directory, "model.pt"))
        
        config = {
            "encoder_name": self.encoder_name,
            "num_decoder_layers": len(self.decoder.layers),
            "num_diffusion_steps": self.num_diffusion_steps,
            "max_target_length": self.max_target_length,
            "use_semantic_noise": self.use_semantic_noise,
            "similarity_loss_weight": self.similarity_loss_weight,
            "decoder_type": self.decoder_type,
        }
        torch.save(config, os.path.join(save_directory, "config.pt"))
        self.tokenizer.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, save_directory: str, device: str = "cpu"):
        """Load model from directory."""
        import os
        
        config = torch.load(os.path.join(save_directory, "config.pt"))
        model = cls(**config)
        
        state_dict = torch.load(
            os.path.join(save_directory, "model.pt"),
            map_location=device,
        )
        model.load_state_dict(state_dict)
        
        return model.to(device)
