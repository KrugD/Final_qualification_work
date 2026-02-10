"""
Pipeline modules for offline meeting transcription.

This package contains modules for:
- Speaker diarization
- Automatic speech recognition (ASR)
- Text correction (ASR error correction)
- Text summarization
- Speaker clustering for long audio files

Pipeline order: Diarization -> ASR -> Correction -> Summarization
"""

from .diarization import perform_diarization
from .asr import perform_speech_recognition
from .correction import perform_correction
from .summarization import perform_summarization
from .speaker_clustering import SpeakerClustering

__all__ = [
    'perform_diarization',
    'perform_speech_recognition',
    'perform_correction',
    'perform_summarization',
    'SpeakerClustering',
]
