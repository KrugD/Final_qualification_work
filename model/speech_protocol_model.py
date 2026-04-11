from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from model.config import ModelConfig
from model.audio_encoder import WhisperAudioEncoder
from model.speaker_encoder import SpeakerEncoder
from model.fusion_adapter import SpeakerContentFusionAdapter


class SpeechProtocolModel(nn.Module):
    """End-to-end multimodal model: audio -> meeting protocol.

    Components:
        1. WhisperAudioEncoder  (frozen)  — content features
        2. SpeakerEncoder       (frozen)  — speaker identity
        3. FusionAdapter        (trainable) — fuse & compress
        4. Qwen2.5-3B + LoRA   (partially trainable) — generate protocol text
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()

        self.audio_encoder = WhisperAudioEncoder(
            model_name=self.config.whisper_model,
            freeze=self.config.freeze_whisper,
        )

        self.speaker_encoder = SpeakerEncoder(
            model_name=self.config.speaker_model,
            window_sec=self.config.speaker_window_sec,
            hop_sec=self.config.speaker_hop_sec,
            sample_rate=self.config.sample_rate,
            freeze=self.config.freeze_speaker,
        )

        self.fusion_adapter = SpeakerContentFusionAdapter(
            whisper_dim=self.config.whisper_dim,
            speaker_dim=self.config.speaker_dim,
            llm_dim=self.config.llm_dim,
            num_query_tokens=self.config.num_query_tokens,
            num_layers=self.config.adapter_num_layers,
            num_heads=self.config.adapter_num_heads,
            dropout=self.config.adapter_dropout,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        self.llm.resize_token_embeddings(len(self.tokenizer))

        if self.config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
        )
        self.llm = get_peft_model(self.llm, lora_config)

    def get_trainable_params_info(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100.0 * trainable / total,
        }

    def encode_audio(
        self,
        input_features: torch.Tensor,
        waveforms: list[torch.Tensor],
    ) -> torch.Tensor:
        """Encode audio through both encoders and fusion adapter.

        Args:
            input_features: (batch, n_mels, T_mel) Whisper mel features.
            waveforms: list of (samples,) tensors for speaker encoding.

        Returns:
            (batch, num_query_tokens, llm_dim) audio tokens for the LLM.
        """
        content = self.audio_encoder(input_features)

        speaker_embs = []
        for waveform in waveforms:
            emb = self.speaker_encoder(waveform)
            speaker_embs.append(emb)

        max_windows = max(e.shape[0] for e in speaker_embs)
        padded = []
        for emb in speaker_embs:
            if emb.shape[0] < max_windows:
                pad = torch.zeros(
                    max_windows - emb.shape[0],
                    emb.shape[1],
                    device=emb.device,
                    dtype=emb.dtype,
                )
                emb = torch.cat([emb, pad], dim=0)
            padded.append(emb)
        speaker_batch = torch.stack(padded)

        return self.fusion_adapter(content, speaker_batch)

    def forward(
        self,
        input_features: torch.Tensor,
        waveforms: list[torch.Tensor],
        labels: torch.Tensor | None = None,
        label_attention_mask: torch.Tensor | None = None,
    ) -> dict:
        """Full forward pass: audio -> loss / logits.

        Args:
            input_features: (batch, n_mels, T_mel) Whisper mel features.
            waveforms: list of (samples,) raw audio tensors.
            labels: (batch, seq_len) token IDs of target protocol text.
            label_attention_mask: (batch, seq_len) attention mask for labels.

        Returns:
            dict with 'loss' and/or 'logits'.
        """
        audio_tokens = self.encode_audio(input_features, waveforms)

        audio_embeds = audio_tokens.to(self.llm.dtype)

        if labels is not None:
            label_embeds = self.llm.get_input_embeddings()(labels)
            inputs_embeds = torch.cat([audio_embeds, label_embeds], dim=1)

            audio_len = audio_embeds.shape[1]
            ignore_labels = torch.full(
                (labels.shape[0], audio_len),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels = torch.cat([ignore_labels, labels], dim=1)

            audio_mask = torch.ones(
                labels.shape[0],
                audio_len,
                dtype=torch.long,
                device=labels.device,
            )
            if label_attention_mask is not None:
                full_attention_mask = torch.cat(
                    [audio_mask, label_attention_mask], dim=1
                )
            else:
                full_attention_mask = torch.cat(
                    [audio_mask, torch.ones_like(labels)], dim=1
                )

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=full_labels,
            )
            return {"loss": outputs.loss, "logits": outputs.logits}

        outputs = self.llm(inputs_embeds=audio_embeds)
        return {"logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        waveforms: list[torch.Tensor],
        max_new_tokens: int = 1024,
        **generate_kwargs,
    ) -> list[str]:
        """Generate protocol text from audio.

        Args:
            input_features: (batch, n_mels, T_mel) mel features.
            waveforms: list of raw audio tensors.
            max_new_tokens: maximum tokens to generate.

        Returns:
            list of generated protocol strings.
        """
        audio_tokens = self.encode_audio(input_features, waveforms)
        audio_embeds = audio_tokens.to(self.llm.dtype)

        generated_ids = self.llm.generate(
            inputs_embeds=audio_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )

        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return texts
