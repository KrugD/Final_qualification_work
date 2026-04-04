from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the SpeechProtocol multimodal model."""

    # Audio encoder (Whisper)
    whisper_model: str = "openai/whisper-small"
    whisper_dim: int = 768
    freeze_whisper: bool = True

    # Speaker encoder (ECAPA-TDNN)
    speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    speaker_dim: int = 192
    speaker_window_sec: float = 1.5
    speaker_hop_sec: float = 0.75
    freeze_speaker: bool = True

    # Fusion adapter
    num_query_tokens: int = 64
    adapter_num_layers: int = 4
    adapter_num_heads: int = 8
    adapter_dropout: float = 0.1

    # LLM decoder (Qwen2.5-3B)
    llm_model: str = "Qwen/Qwen2.5-3B"
    llm_dim: int = 2048
    freeze_llm: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Audio processing
    sample_rate: int = 16000
    max_audio_sec: int = 30

    # Protocol format
    protocol_start_token: str = "<protocol>"
    protocol_end_token: str = "</protocol>"
    speaker_start_token: str = "<speaker id=\"{id}\">"
    speaker_end_token: str = "</speaker>"
