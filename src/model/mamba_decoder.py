import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)

# Try to import Mamba, fallback to a simple implementation if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    logger.info("mamba-ssm not available. Using fallback implementation.")


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP projection."""
    
    def __init__(self, hidden_size: int):
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


class MambaBlock(nn.Module):
    """
    Mamba block with residual connection.
    
    Mamba is a selective state space model that provides:
    - O(n) complexity (vs O(n²) for attention)
    - Hardware-efficient implementation
    - Strong performance on long sequences
    """
    
    def __init__(
        self,
        hidden_size: int,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(hidden_size)
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=hidden_size,
                d_state=state_size,
                d_conv=conv_kernel,
                expand=expand_factor,
            )
        else:
            # Fallback: simple gated convolution + linear
            self.mamba = FallbackMamba(
                hidden_size=hidden_size,
                state_size=state_size,
                conv_kernel=conv_kernel,
                expand_factor=expand_factor,
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class FallbackMamba(nn.Module):
    """
    Fallback implementation when mamba-ssm is not available.
    Uses gated convolution to approximate Mamba behavior.
    """
    
    def __init__(
        self,
        hidden_size: int,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand_factor: int = 2,
    ):
        super().__init__()
        
        inner_size = hidden_size * expand_factor
        
        # Input projection
        self.in_proj = nn.Linear(hidden_size, inner_size * 2)
        
        # Causal convolution
        self.conv = nn.Conv1d(
            inner_size,
            inner_size,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=inner_size,
        )
        
        # SSM-like linear recurrence (simplified)
        self.ssm_proj = nn.Linear(inner_size, state_size * 2)
        self.ssm_out = nn.Linear(state_size, inner_size)
        
        # Output projection
        self.out_proj = nn.Linear(inner_size, hidden_size)
        
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # Input projection with gate
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        # Causal convolution
        x = x.transpose(1, 2)  # [B, D, L]
        x = self.conv(x)[:, :, :seq_len]  # Causal: remove future
        x = x.transpose(1, 2)  # [B, L, D]
        
        x = self.act(x)
        
        # Simplified SSM
        ssm = self.ssm_proj(x)
        ssm_state, ssm_gate = ssm.chunk(2, dim=-1)
        ssm_out = self.ssm_out(ssm_state * torch.sigmoid(ssm_gate))
        x = x + ssm_out
        
        # Gated output
        x = x * self.act(z)
        
        # Output projection
        x = self.out_proj(x)
        
        return x


class CrossAttention(nn.Module):
    """Cross-attention to encoder outputs."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Convert mask format
        key_padding_mask = None
        if encoder_attention_mask is not None:
            key_padding_mask = ~encoder_attention_mask.bool()
        
        hidden_states, _ = self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            key_padding_mask=key_padding_mask,
        )
        
        hidden_states = self.dropout(hidden_states)
        return residual + hidden_states


class CrossMambaLayer(nn.Module):
    """
    CrossMamba layer combining:
    1. Mamba block for sequence modeling
    2. Cross-attention to encoder
    3. Timestep conditioning (AdaLN-style)
    4. Feed-forward network
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        intermediate_size: int = 3072,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Mamba block
        self.mamba = MambaBlock(
            hidden_size=hidden_size,
            state_size=state_size,
            conv_kernel=conv_kernel,
            expand_factor=expand_factor,
            dropout=dropout,
        )
        
        # Cross-attention
        self.cross_attn = CrossAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Feed-forward
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )
        
        # Timestep conditioning (AdaLN)
        self.timestep_proj = nn.Linear(hidden_size, hidden_size * 6)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_emb: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Timestep conditioning: 6 parameters (scale, shift for each of 3 blocks)
        timestep_cond = self.timestep_proj(timestep_emb).unsqueeze(1)
        scale_mamba, shift_mamba, scale_cross, shift_cross, scale_ffn, shift_ffn = \
            timestep_cond.chunk(6, dim=-1)
        
        # Mamba block with timestep modulation
        hidden_states = hidden_states * (1 + scale_mamba) + shift_mamba
        hidden_states = self.mamba(hidden_states)
        
        # Cross-attention with timestep modulation
        hidden_states = hidden_states * (1 + scale_cross) + shift_cross
        hidden_states = self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        
        # FFN with timestep modulation
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = hidden_states * (1 + scale_ffn) + shift_ffn
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class CrossMambaDecoder(nn.Module):
    """
    CrossMamba Decoder for Masked Diffusion.
    
    From NAACL 2025 paper:
    - Uses Mamba blocks for O(n) sequence modeling
    - Cross-attention to encoder for conditioning on source
    - Timestep embedding for diffusion process
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        max_seq_len: int = 128,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Learnable positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        # Timestep embedding
        self.timestep_embedding = TimestepEmbedding(hidden_size)
        
        # CrossMamba layers
        self.layers = nn.ModuleList([
            CrossMambaLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                state_size=state_size,
                conv_kernel=conv_kernel,
                expand_factor=expand_factor,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights
        self.output_projection.weight = self.token_embedding.weight
        
        # Initialize position ids
        self.register_buffer(
            "position_ids",
            torch.arange(max_seq_len).unsqueeze(0),
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        hidden_states = self.token_embedding(input_ids)
        
        # Add positional embeddings
        position_ids = self.position_ids[:, :seq_len].expand(batch_size, -1)
        hidden_states = hidden_states + self.position_embedding(position_ids)
        
        # Get timestep embeddings
        timestep_emb = self.timestep_embedding(timesteps)
        
        # Apply CrossMamba layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states,
                timestep_emb,
                encoder_attention_mask,
            )
        
        # Project to vocabulary
        hidden_states = self.output_norm(hidden_states)
        logits = self.output_projection(hidden_states)
        
        return logits


def is_mamba_available() -> bool:
    """Check if Mamba is available."""
    return MAMBA_AVAILABLE
