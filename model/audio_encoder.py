import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor


class WhisperAudioEncoder(nn.Module):
    """Wraps the Whisper encoder to extract content features from audio.

    Produces a sequence of hidden states (T, whisper_dim) capturing
    *what* is being said, without decoding into text.
    """

    def __init__(self, model_name: str = "openai/whisper-small", freeze: bool = True):
        super().__init__()
        whisper = WhisperModel.from_pretrained(model_name)
        self.encoder = whisper.encoder
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.encoder.config.d_model

    @torch.no_grad()
    def extract_features(
        self, audio_array, sampling_rate: int = 16000
    ) -> torch.Tensor:
        """Preprocess raw waveform into log-mel spectrogram features.

        Args:
            audio_array: numpy array or list of numpy arrays.
            sampling_rate: audio sample rate (Whisper expects 16kHz).

        Returns:
            Tensor of shape (batch, n_mels, T_mel).
        """
        features = self.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        return features.input_features

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Run the Whisper encoder.

        Args:
            input_features: (batch, n_mels, T_mel) from extract_features.

        Returns:
            (batch, T, whisper_dim) encoder hidden states.
        """
        return self.encoder(input_features).last_hidden_state
