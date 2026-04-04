"""Inference script: generate a meeting protocol from an audio file.

Usage:
    python -m inference.generate --audio path/to/audio.wav --checkpoint checkpoints/stage2/best
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from peft import PeftModel

from model.config import ModelConfig
from model.speech_protocol_model import SpeechProtocolModel


def load_model(checkpoint_path: str, device: str = "cuda") -> SpeechProtocolModel:
    """Load a trained SpeechProtocol model from checkpoint."""
    ckpt = Path(checkpoint_path)
    config = ModelConfig()
    model = SpeechProtocolModel(config)

    adapter_path = ckpt / "adapter.pt"
    if adapter_path.exists():
        state_dict = torch.load(adapter_path, map_location="cpu", weights_only=True)
        model.fusion_adapter.load_state_dict(state_dict)
        print(f"Loaded adapter from {adapter_path}")

    lora_path = ckpt / "lora"
    if lora_path.exists():
        model.llm = PeftModel.from_pretrained(
            model.llm.base_model.model, str(lora_path)
        )
        print(f"Loaded LoRA from {lora_path}")

    model = model.to(device)
    model.eval()
    return model


def process_long_audio(
    audio: np.ndarray,
    model: SpeechProtocolModel,
    sample_rate: int = 16000,
    chunk_sec: int = 30,
    overlap_sec: int = 5,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Split long audio into overlapping chunks and encode each.

    Returns a list of audio token tensors, one per chunk.
    """
    chunk_samples = chunk_sec * sample_rate
    hop_samples = (chunk_sec - overlap_sec) * sample_rate
    total_samples = len(audio)

    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk = audio[start:end]
        if len(chunk) < sample_rate:
            break
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)
        start += hop_samples

    all_audio_tokens = []
    for chunk in chunks:
        mel = model.audio_encoder.feature_extractor(
            chunk, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(device)

        waveform = torch.from_numpy(chunk).float().to(device)
        audio_tokens = model.encode_audio(mel, [waveform])
        all_audio_tokens.append(audio_tokens)

    return all_audio_tokens


def generate_protocol(
    audio_path: str,
    model: SpeechProtocolModel,
    sample_rate: int = 16000,
    max_new_tokens: int = 2048,
    device: str = "cuda",
) -> str:
    """Generate a meeting protocol from an audio file."""
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    audio = audio.astype(np.float32)

    duration_sec = len(audio) / sample_rate
    print(f"Audio duration: {duration_sec:.1f}s ({duration_sec/60:.1f} min)")

    if duration_sec <= 30:
        mel = model.audio_encoder.feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(device)

        waveform = torch.from_numpy(audio).float().to(device)
        texts = model.generate(
            mel, [waveform], max_new_tokens=max_new_tokens
        )
        return texts[0]

    audio_token_list = process_long_audio(audio, model, sample_rate, device=device)
    combined_tokens = torch.cat(audio_token_list, dim=1)

    audio_embeds = combined_tokens.to(model.llm.dtype)
    generated_ids = model.llm.generate(
        inputs_embeds=audio_embeds,
        max_new_tokens=max_new_tokens,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    return model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate protocol from audio")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output", type=str, default=None, help="Save protocol to file")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.checkpoint, args.device)

    print("Generating protocol...")
    start_time = time.time()
    protocol = generate_protocol(
        args.audio, model, max_new_tokens=args.max_tokens, device=args.device
    )
    elapsed = time.time() - start_time

    print(f"\nGeneration time: {elapsed:.1f}s")
    print(f"\n{'='*60}")
    print("GENERATED PROTOCOL:")
    print(f"{'='*60}")
    print(protocol)

    if args.output:
        Path(args.output).write_text(protocol, encoding="utf-8")
        print(f"\nProtocol saved to {args.output}")


if __name__ == "__main__":
    main()
