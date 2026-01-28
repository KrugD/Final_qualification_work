"""
Pipeline modules for offline meeting transcription.

This package contains modules for:
- Speaker diarization
- Automatic speech recognition (ASR)
- Text summarization
- Text correction
- Speaker clustering for long audio files
"""

from .diarization import perform_diarization
from .asr import perform_speech_recognition
from .summarization import perform_summarization
from .correction import perform_correction
from .speaker_clustering import SpeakerClustering

__all__ = [
    'perform_diarization',
    'perform_speech_recognition',
    'perform_summarization',
    'perform_correction',
    'SpeakerClustering',
]
