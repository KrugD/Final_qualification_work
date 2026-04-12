import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    """Extracts per-window speaker identity embeddings using ECAPA-TDNN.

    Slides a window over the waveform and produces a speaker embedding
    for each window, capturing *who* is speaking at each moment.
    """

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        window_sec: float = 1.5,
        hop_sec: float = 0.75,
        sample_rate: int = 16000,
        freeze: bool = True,
    ):
        super().__init__()
        from speechbrain.inference.speaker import EncoderClassifier

        self.classifier = EncoderClassifier.from_hparams(
            source=model_name,
            run_opts={"device": "cpu"},
        )
        self.window_samples = int(window_sec * sample_rate)
        self.hop_samples = int(hop_sec * sample_rate)
        self.sample_rate = sample_rate

        if freeze:
            for param in self.classifier.mods.parameters():
                param.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return 192

    def _extract_windows(self, waveform: torch.Tensor) -> torch.Tensor:
        """Slice waveform into overlapping windows.

        Args:
            waveform: (samples,) single-channel audio.

        Returns:
            (n_windows, window_samples) tensor.
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        length = waveform.shape[0]
        if length < self.window_samples:
            waveform = F.pad(waveform, (0, self.window_samples - length))
            return waveform.unsqueeze(0)

        windows = []
        start = 0
        while start + self.window_samples <= length:
            windows.append(waveform[start : start + self.window_samples])
            start += self.hop_samples

        if start < length:
            last = waveform[-self.window_samples :]
            windows.append(last)

        return torch.stack(windows)

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute speaker embeddings for each window.

        Args:
            waveform: (samples,) or (1, samples) raw audio at self.sample_rate.

        Returns:
            (n_windows, 192) speaker embeddings.
        """
        model_device = next(self.classifier.mods.parameters()).device
        windows = self._extract_windows(waveform).to(model_device).float()
        self.classifier.device = model_device
        embeddings = self.classifier.encode_batch(windows)
        return embeddings.squeeze(1)
