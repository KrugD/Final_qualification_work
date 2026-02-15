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
        x = x + self.pe[:, : x.size(1)]
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
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            last_layer_attention = outputs.attentions[-1]
            cls_attention = last_layer_attention[:, :, 0, :].mean(dim=1)
            attention_scores = cls_attention
        else:
            attention_scores = (
                torch.ones(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    device=hidden_states.device,
                )
                / hidden_states.shape[1]
            )

        # Normalize
        attention_scores = attention_scores / (
            attention_scores.sum(dim=-1, keepdim=True) + 1e-8
        )

        return hidden_states, cls_embedding, attention_scores


class MaskedDiffusionSummarizer(nn.Module):
    """
    Masked Diffusion Language Model for Summarization.

    Key features from NAACL 2025 paper:
    1. Semantic-Aware Noising: Uses attention scores to mask important tokens later
    2. Similarity Loss: Ensures semantic alignment between source and target
    3. CrossMamba Decoder: Mamba-based decoder with O(n) complexity
    4. Encoder-Decoder with timestep conditioning

    This version uses FRED-T5-1.7B encoder (frozen) for high-quality
    source representations, with a trainable CrossMamba diffusion decoder.
    """

    def __init__(
        self,
        encoder_name: str = "ai-forever/FRED-T5-1.7B",
        num_decoder_layers: int = 6,
        num_diffusion_steps: int = 50,
        max_target_length: int = 128,
        dropout: float = 0.1,
        schedule_type: str = "cosine",
        use_semantic_noise: bool = True,
        similarity_loss_weight: float = 1.0,
        decoder_type: str = "mamba",
        mamba_state_size: int = 16,
        mamba_conv_kernel: int = 4,
        mamba_expand_factor: int = 2,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.num_diffusion_steps = num_diffusion_steps
        self.max_target_length = max_target_length
        self.use_semantic_noise = use_semantic_noise
        self.similarity_loss_weight = similarity_loss_weight
        self.decoder_type = decoder_type
        self.freeze_encoder = freeze_encoder

        # Load pretrained encoder
        base_encoder = T5EncoderModel.from_pretrained(encoder_name)
        encoder_config = base_encoder.config

        # Wrap with semantic encoder
        self.encoder = SemanticEncoder(base_encoder, encoder_config.d_model)

        # Freeze encoder if requested (saves ~756M params from optimizer)
        if freeze_encoder:
            for param in self.encoder.encoder.parameters():
                param.requires_grad = False
            # cls_projection stays trainable (small, ~2.4M)
            print(f"Encoder frozen ({sum(p.numel() for p in self.encoder.encoder.parameters()):,} params)")

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

        # Initialize decoder
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
            raise ValueError(
                f"Unknown decoder_type: {decoder_type}. Only 'mamba' is supported."
            )

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

    def get_trainable_parameters(self):
        """Return only trainable parameters (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def compute_similarity_loss(
        self,
        source_cls: torch.Tensor,
        target_cls: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity loss between source and target [CLS] tokens.
        From NAACL 2025 paper Eq. 4: Lcls = 1 - cos(Cs, Ct)
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

        Per NAACL 2025 paper (Section 3.4, Figure 2):
        1. Encode SOURCE → hidden states for decoder cross-attention + source [CLS]
        2. Encode TARGET → attention scores for semantic-aware noising + target [CLS]
        3. Apply semantic-aware noise: important target tokens masked later
        4. Decode noisy target conditioned on source encoder output
        5. Losses: CE on masked positions + similarity(source_CLS, target_CLS.detach())
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 1. Encode SOURCE (encoder frozen → no_grad for memory savings)
        if self.freeze_encoder:
            with torch.no_grad():
                encoder_hidden_states, source_cls_raw, _ = self.encoder(
                    input_ids, attention_mask
                )
            # Detach so backward doesn't try to go through frozen encoder
            encoder_hidden_states = encoder_hidden_states.detach()
            # CLS projection is trainable, re-apply it
            source_cls = self.encoder.cls_projection(encoder_hidden_states[:, 0, :])
        else:
            encoder_hidden_states, source_cls, _ = self.encoder(
                input_ids, attention_mask
            )

        # 2. Encode TARGET → attention scores for noising + target [CLS]
        with torch.no_grad():
            _, target_cls, target_attention = self.encoder(
                labels, labels_attention_mask
            )

        # Sample timesteps
        if timesteps is None:
            timesteps = self.noise_scheduler.sample_timesteps(batch_size, device)

        # 3. Apply noise to labels
        if self.use_semantic_noise:
            noisy_labels_list = []
            noise_masks_list = []

            for i in range(batch_size):
                t_ratio = (timesteps[i].item() + 1) / self.num_diffusion_steps
                noisy, mask = self.noise_scheduler.add_semantic_noise(
                    labels[i : i + 1],
                    target_attention[i : i + 1],
                    t_ratio,
                    labels_attention_mask[i : i + 1],
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
                    labels[i : i + 1],
                    mask_ratios[i].item(),
                    labels_attention_mask[i : i + 1],
                )
                noisy_labels_list.append(noisy)
                noise_masks_list.append(mask)

            noisy_labels = torch.cat(noisy_labels_list, dim=0)
            noise_masks = torch.cat(noise_masks_list, dim=0)

        # 4. Decode noisy target conditioned on source encoder output
        logits = self.decoder(
            input_ids=noisy_labels,
            encoder_hidden_states=encoder_hidden_states,
            timesteps=timesteps,
            attention_mask=labels_attention_mask,
            encoder_attention_mask=attention_mask,
        )

        # 5. Compute losses (paper Eq. 5: L = Lvb + Lcls + CE_reconstruction)
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=self.pad_token_id,
            reduction="none",
        )
        ce_loss = ce_loss.view(batch_size, -1)

        # Lvb: CE on masked positions only
        diffusion_loss = (ce_loss * noise_masks.float()).sum() / (
            noise_masks.float().sum() + 1e-8
        )

        # CE reconstruction: CE on ALL non-padding positions
        non_pad_mask = (labels != self.pad_token_id).float()
        reconstruction_loss = (ce_loss * non_pad_mask).sum() / (
            non_pad_mask.sum() + 1e-8
        )

        # Lcls: similarity loss (paper Eq. 4)
        similarity_loss = self.compute_similarity_loss(source_cls, target_cls)

        # Total loss (paper Eq. 5)
        total_loss = (
            diffusion_loss
            + self.similarity_loss_weight * similarity_loss
            + reconstruction_loss
        )

        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "reconstruction_loss": reconstruction_loss,
            "similarity_loss": similarity_loss,
            "logits": logits,
            "noise_masks": noise_masks,
        }

    def _apply_logit_filtering(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply temperature, top-k and top-p filtering to logits."""
        logits = logits / max(temperature, 1e-8)

        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        return logits

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
        strategy: str = "linear",
        sample: bool = False,
        temperature_annealing: bool = False,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate summaries using iterative denoising.

        Args:
            input_ids: Source token IDs [batch_size, seq_len]
            attention_mask: Source attention mask [batch_size, seq_len]
            max_length: Maximum target sequence length
            num_inference_steps: Number of denoising steps
            temperature: Sampling temperature
            top_k: Top-k filtering (None to disable)
            top_p: Nucleus sampling threshold (None to disable)
            strategy: Unmasking strategy: "linear", "cosine", "confidence"
            sample: If True, sample from distribution; if False, use argmax
            temperature_annealing: If True, decrease temperature over steps
            repetition_penalty: Penalize already-generated tokens (>1.0)
            no_repeat_ngram_size: Block repeated n-grams (0 to disable)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        max_length = max_length or self.max_target_length
        num_inference_steps = num_inference_steps or self.num_diffusion_steps

        # Encode source
        encoder_hidden_states, _, attention_scores = self.encoder(
            input_ids, attention_mask
        )

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

        # Precompute unmask schedule
        tokens_to_unmask_per_step = self._compute_unmask_schedule(
            max_length, num_inference_steps, strategy
        )

        # Iterative denoising
        for step in range(num_inference_steps):
            t = int(
                round(
                    (num_inference_steps - step - 1)
                    / num_inference_steps
                    * (self.num_diffusion_steps - 1)
                )
            )
            t = max(0, min(self.num_diffusion_steps - 1, t))

            timesteps = torch.full(
                (batch_size,), t, dtype=torch.long, device=device
            )

            logits = self.decoder(
                input_ids=generated_ids,
                encoder_hidden_states=encoder_hidden_states,
                timesteps=timesteps,
                attention_mask=target_attention_mask,
                encoder_attention_mask=attention_mask,
            )

            # Temperature (optionally with annealing)
            current_temp = temperature
            if temperature_annealing:
                progress = step / max(num_inference_steps - 1, 1)
                current_temp = temperature * (1.0 - 0.5 * progress)

            logits = self._apply_logit_filtering(logits, current_temp, top_k, top_p)

            # Repetition penalty
            if repetition_penalty > 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated_ids, repetition_penalty
                )

            # No-repeat n-gram blocking
            if no_repeat_ngram_size > 0:
                logits = self._block_repeated_ngrams(
                    logits, generated_ids, no_repeat_ngram_size
                )

            probs = F.softmax(logits, dim=-1)

            if sample:
                predicted_ids = torch.multinomial(
                    probs.view(-1, probs.size(-1)), num_samples=1
                ).view(batch_size, max_length)
            else:
                predicted_ids = probs.argmax(dim=-1)

            confidence = probs.max(dim=-1).values

            is_masked = generated_ids == self.mask_token_id
            num_to_unmask = tokens_to_unmask_per_step[step]

            for b in range(batch_size):
                if not is_masked[b].any():
                    continue

                masked_positions = is_masked[b].nonzero().squeeze(-1)
                if masked_positions.dim() == 0:
                    masked_positions = masked_positions.unsqueeze(0)

                masked_confidence = confidence[b, masked_positions]

                actual_unmask = min(num_to_unmask, len(masked_positions))
                actual_unmask = max(1, actual_unmask)

                if actual_unmask > 0 and len(masked_positions) > 0:
                    _, top_indices = masked_confidence.topk(
                        min(actual_unmask, len(masked_positions))
                    )
                    positions_to_unmask = masked_positions[top_indices]
                    generated_ids[b, positions_to_unmask] = predicted_ids[
                        b, positions_to_unmask
                    ]

        # Final pass: unmask any remaining tokens
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
            logits = self._apply_logit_filtering(logits, temperature, top_k, top_p)
            if repetition_penalty > 1.0:
                logits = self._apply_repetition_penalty(
                    logits, generated_ids, repetition_penalty
                )
            if no_repeat_ngram_size > 0:
                logits = self._block_repeated_ngrams(
                    logits, generated_ids, no_repeat_ngram_size
                )
            if sample:
                predicted_ids = torch.multinomial(
                    F.softmax(logits, dim=-1).view(-1, logits.size(-1)),
                    num_samples=1,
                ).view(batch_size, max_length)
            else:
                predicted_ids = logits.argmax(dim=-1)
            generated_ids[is_masked] = predicted_ids[is_masked]

        confidence_scores = F.softmax(logits, dim=-1).max(dim=-1).values

        return generated_ids, confidence_scores

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty (vectorized)."""
        for b in range(logits.shape[0]):
            existing = generated_ids[b][generated_ids[b] != self.mask_token_id]
            if len(existing) == 0:
                continue
            existing_unique = existing.unique()
            scores = logits[b, :, existing_unique]
            scores = torch.where(scores > 0, scores / penalty, scores * penalty)
            logits[b, :, existing_unique] = scores
        return logits

    def _block_repeated_ngrams(
        self,
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        ngram_size: int,
    ) -> torch.Tensor:
        """Block n-grams that already appear in the unmasked sequence."""
        batch_size, seq_len, _ = logits.shape
        mask_id = self.mask_token_id

        for b in range(batch_size):
            ids = generated_ids[b].tolist()

            banned = {}
            for i in range(seq_len - ngram_size + 1):
                ngram = ids[i : i + ngram_size]
                if mask_id in ngram:
                    continue
                prefix = tuple(ngram[:-1])
                if prefix not in banned:
                    banned[prefix] = set()
                banned[prefix].add(ngram[-1])

            if not banned:
                continue

            for pos in range(ngram_size - 1, seq_len):
                prefix = tuple(ids[pos - ngram_size + 1 : pos])
                if mask_id in prefix:
                    continue
                if prefix in banned:
                    for token_id in banned[prefix]:
                        logits[b, pos, token_id] = float("-inf")

        return logits

    def _compute_unmask_schedule(
        self,
        total_tokens: int,
        num_steps: int,
        strategy: str = "linear",
    ) -> list:
        """Compute how many tokens to unmask at each step."""
        if strategy == "linear":
            per_step = total_tokens / num_steps
            schedule = []
            unmasked_so_far = 0
            for i in range(num_steps):
                target = int(round((i + 1) * per_step))
                to_unmask = target - unmasked_so_far
                schedule.append(max(1, to_unmask))
                unmasked_so_far += max(1, to_unmask)
            return schedule

        elif strategy == "cosine":
            schedule = []
            unmasked_so_far = 0
            for i in range(num_steps):
                progress = (i + 1) / num_steps
                target = int(
                    round(total_tokens * (1 - math.cos(progress * math.pi / 2)))
                )
                to_unmask = target - unmasked_so_far
                schedule.append(max(1, to_unmask))
                unmasked_so_far += max(1, to_unmask)
            return schedule

        elif strategy == "confidence":
            per_step = max(1, total_tokens // (num_steps * 2))
            schedule = [per_step] * num_steps
            remaining = total_tokens - sum(schedule)
            if remaining > 0:
                schedule[-1] += remaining
            return schedule

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

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
            "freeze_encoder": self.freeze_encoder,
        }
        torch.save(config, os.path.join(save_directory, "config.pt"))
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, save_directory: str, device: str = "cpu"):
        """Load model from directory."""
        import os

        config = torch.load(
            os.path.join(save_directory, "config.pt"), map_location="cpu"
        )
        model = cls(**config)

        state_dict = torch.load(
            os.path.join(save_directory, "model.pt"),
            map_location=device,
        )
        model.load_state_dict(state_dict)

        return model.to(device)
