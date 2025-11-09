import torch
import os
from dotenv import load_dotenv

load_dotenv()

class ModelConfig:
    """Configuration class for all models."""
    
    # Diarization model configuration
    DIARIZATION_MODEL_NAME = "pyannote/speaker-diarization-3.1"
    DIARIZATION_TOKEN = os.getenv("HF_TOKEN")
    
    # Automatic Speech Recognition model configuration
    ASR_MODEL_NAME = "openai/whisper-small"
    ASR_LANGUAGE = "russian"
    ASR_TASK = "transcribe"
    
    # Text correction model configuration
    CORRECTION_MODEL_NAME = "ai-forever/sage-m2m100-1.2B"
    CORRECTION_SRC_LANG = "ru"
    CORRECTION_TGT_LANG = "ru"
    
    # Summarization model configuration
    SUMMARIZATION_MODEL_NAME = "RussianNLP/FRED-T5-Summarizer"
    
    # Processing parameters
    MIN_SEGMENT_DURATION = 0.5  # seconds
    MAX_SUMMARY_INPUT_LENGTH = 2000  # characters

    # Speaker clustering parameters
    CLUSTERING_DISTANCE_THRESHOLD = 0.4
    TARGET_CHUNK_DURATION_MIN = 25
    MAX_CHUNK_DURATION_MIN = 30
    MIN_SILENCE_LEN_MS = 2000  # 2 seconds minimum silence
    SILENCE_THRESH_DB = -40    # Silence threshold in dB

def get_device():
    """Get available device (CUDA or CPU)."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"