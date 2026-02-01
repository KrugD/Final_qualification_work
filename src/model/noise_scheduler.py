import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class NoiseScheduler:
    """
    Basic noise scheduler for masked diffusion models.
    
    Implements a cosine schedule for masking tokens during training
    and a linear unmasking schedule for inference.
    """
    
    def __init__(
        self,
        num_diffusion_steps: int = 20,
        mask_token_id: int = 0,
        schedule_type: str = "cosine",
    ):
        """
        Initialize the noise scheduler.
        
        Args:
            num_diffusion_steps: Number of diffusion steps for inference
            mask_token_id: Token ID to use for masking
            schedule_type: Type of schedule ("cosine", "linear", "sqrt")
        """
        self.num_diffusion_steps = num_diffusion_steps
        self.mask_token_id = mask_token_id
        self.schedule_type = schedule_type
        
        # Precompute the mask ratios for each timestep
        self.mask_ratios = self._compute_mask_schedule()
    
    def _compute_mask_schedule(self) -> torch.Tensor:
        """Compute mask ratios for each timestep."""
        t = torch.linspace(0, 1, self.num_diffusion_steps + 1)
        
        if self.schedule_type == "cosine":
            # Cosine schedule: smooth transition
            ratios = torch.cos(t * math.pi / 2)
        elif self.schedule_type == "linear":
            # Linear schedule
            ratios = 1 - t
        elif self.schedule_type == "sqrt":
            # Square root schedule: faster at the beginning
            ratios = torch.sqrt(1 - t)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return ratios
    
    def add_noise(
        self,
        token_ids: torch.Tensor,
        mask_ratio: float,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise (masking) to token IDs for training.
        
        Args:
            token_ids: Input token IDs [batch_size, seq_len]
            mask_ratio: Ratio of tokens to mask (0.0 to 1.0)
            attention_mask: Optional attention mask [batch_size, seq_len]
        
        Returns:
            noisy_tokens: Token IDs with some tokens masked
            noise_mask: Boolean mask indicating which tokens were masked
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Create random mask
        rand_mask = torch.rand(batch_size, seq_len, device=device)
        
        # Determine which tokens to mask
        noise_mask = rand_mask < mask_ratio
        
        # Don't mask padding tokens if attention_mask is provided
        if attention_mask is not None:
            noise_mask = noise_mask & (attention_mask.bool())
        
        # Apply masking
        noisy_tokens = token_ids.clone()
        noisy_tokens[noise_mask] = self.mask_token_id
        
        return noisy_tokens, noise_mask
    
    def get_mask_ratio_for_timestep(self, timestep: int) -> float:
        """Get the mask ratio for a specific timestep during inference."""
        return self.mask_ratios[timestep].item()
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(
            0, self.num_diffusion_steps, (batch_size,), device=device
        )
    
    def get_mask_ratio_for_training(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get mask ratios for given timesteps during training.
        
        Args:
            timesteps: Tensor of timestep indices [batch_size]
        
        Returns:
            mask_ratios: Tensor of mask ratios [batch_size]
        """
        return self.mask_ratios[timesteps].to(timesteps.device)


class SemanticAwareNoiseScheduler(NoiseScheduler):
    """
    Semantic-Aware Noise Scheduler from NAACL 2025 paper.
    
    Uses attention scores to determine token importance.
    Important tokens (high attention) are masked with lower probability,
    meaning they are generated first during inference.
    
    From paper: Pt = t/T - (1 - t/T) * attention_score[i]
    """
    
    def __init__(
        self,
        num_diffusion_steps: int = 20,
        mask_token_id: int = 0,
        schedule_type: str = "cosine",
        importance_weight: float = 1.0,
    ):
        """
        Initialize the semantic-aware noise scheduler.
        
        Args:
            num_diffusion_steps: Number of diffusion steps
            mask_token_id: Token ID for masking
            schedule_type: Base schedule type
            importance_weight: Weight for importance scores (higher = more influence)
        """
        super().__init__(num_diffusion_steps, mask_token_id, schedule_type)
        self.importance_weight = importance_weight
    
    def compute_semantic_mask_probability(
        self,
        timestep_ratio: float,
        attention_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token mask probability based on semantic importance.
        
        From NAACL 2025 paper Eq. 3:
        Pt = t/T - (1 - t/T) * attention_score[i]
        
        Args:
            timestep_ratio: t/T ratio (0 to 1)
            attention_scores: Normalized attention scores [batch_size, seq_len]
        
        Returns:
            mask_probs: Per-token mask probabilities [batch_size, seq_len]
        """
        # Normalize attention scores to [0, 1]
        attention_scores = attention_scores.clamp(0, 1)
        
        # Compute mask probability: important tokens have lower mask probability
        # Pt = t/T - (1 - t/T) * importance * attention_score
        mask_probs = timestep_ratio - (1 - timestep_ratio) * self.importance_weight * attention_scores
        
        # Clamp to valid probability range
        mask_probs = mask_probs.clamp(0, 1)
        
        return mask_probs
    
    def add_semantic_noise(
        self,
        token_ids: torch.Tensor,
        attention_scores: torch.Tensor,
        timestep_ratio: float,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise based on semantic importance (attention scores).
        
        Tokens with higher attention scores (more important) have
        lower probability of being masked.
        
        Args:
            token_ids: Input token IDs [batch_size, seq_len]
            attention_scores: Attention scores from encoder [batch_size, seq_len]
            timestep_ratio: Current timestep ratio t/T
            attention_mask: Optional attention mask [batch_size, seq_len]
        
        Returns:
            noisy_tokens: Token IDs with semantic-aware masking
            noise_mask: Boolean mask indicating masked positions
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device
        
        # Compute per-token mask probabilities
        mask_probs = self.compute_semantic_mask_probability(
            timestep_ratio, attention_scores
        )
        
        # Sample mask based on probabilities
        rand_vals = torch.rand(batch_size, seq_len, device=device)
        noise_mask = rand_vals < mask_probs
        
        # Don't mask padding tokens
        if attention_mask is not None:
            noise_mask = noise_mask & attention_mask.bool()
        
        # Apply masking
        noisy_tokens = token_ids.clone()
        noisy_tokens[noise_mask] = self.mask_token_id
        
        return noisy_tokens, noise_mask


