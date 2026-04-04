import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerContentFusionAdapter(nn.Module):
    """Fuses Whisper content features with speaker identity embeddings
    and compresses them via learnable Q-Former queries into a fixed-length
    sequence suitable for the LLM decoder.

    Architecture:
        1. Project speaker embeddings to Whisper dimension & temporally align
        2. Cross-attention: content features attend to speaker information
        3. Q-Former: learnable queries compress the fused sequence
        4. Project to LLM embedding dimension
    """

    def __init__(
        self,
        whisper_dim: int = 768,
        speaker_dim: int = 192,
        llm_dim: int = 2048,
        num_query_tokens: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.whisper_dim = whisper_dim
        self.num_query_tokens = num_query_tokens

        self.speaker_proj = nn.Sequential(
            nn.Linear(speaker_dim, whisper_dim),
            nn.LayerNorm(whisper_dim),
            nn.GELU(),
        )

        self.fusion_cross_attn = nn.MultiheadAttention(
            embed_dim=whisper_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(whisper_dim)

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, whisper_dim) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=whisper_dim,
            nhead=num_heads,
            dim_feedforward=whisper_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.qformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.qformer_norm = nn.LayerNorm(whisper_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(whisper_dim, llm_dim),
            nn.LayerNorm(llm_dim),
        )

    def _align_speaker_to_content(
        self,
        speaker_embs: torch.Tensor,
        content_len: int,
    ) -> torch.Tensor:
        """Temporally upsample speaker embeddings to match content frames.

        Args:
            speaker_embs: (batch, n_windows, whisper_dim) projected speaker embeddings.
            content_len: number of Whisper encoder frames T.

        Returns:
            (batch, T, whisper_dim) aligned speaker features.
        """
        spk = speaker_embs.permute(0, 2, 1)
        spk = F.interpolate(spk, size=content_len, mode="linear", align_corners=False)
        return spk.permute(0, 2, 1)

    def forward(
        self,
        content_features: torch.Tensor,
        speaker_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse content and speaker features, compress via Q-Former.

        Args:
            content_features: (batch, T, whisper_dim) from Whisper encoder.
            speaker_embeddings: (batch, n_windows, speaker_dim) from speaker encoder.

        Returns:
            (batch, num_query_tokens, llm_dim) compressed speaker-attributed tokens.
        """
        batch_size = content_features.shape[0]
        T = content_features.shape[1]

        spk_proj = self.speaker_proj(speaker_embeddings)
        spk_aligned = self._align_speaker_to_content(spk_proj, T)

        fused, _ = self.fusion_cross_attn(
            query=content_features,
            key=spk_aligned,
            value=spk_aligned,
        )
        fused = self.fusion_norm(fused + content_features)

        queries = self.query_tokens.expand(batch_size, -1, -1)
        compressed = self.qformer(tgt=queries, memory=fused)
        compressed = self.qformer_norm(compressed)

        return self.output_proj(compressed)
