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
    
    # Speaker clustering models configuration
    EMBEDDING_MODEL_NAME = "pyannote/embedding"
    CLUSTERING_MODEL_NAME = "pyannote/speaker-diarization-3.1"
    
    # Processing parameters
    MIN_SEGMENT_DURATION = 0.5  # seconds
    MAX_SUMMARY_INPUT_TOKENS = 512  # max tokens for summarization input (FRED-T5 limit)
    MERGE_GAP_SECONDS = 2.0  # max gap to merge consecutive same-speaker segments

    # Speaker clustering parameters
    CLUSTERING_DISTANCE_THRESHOLD = 0.4
    TARGET_CHUNK_DURATION_MIN = 20
    TARGET_CHUNK_DURATION_MAX = 25
    MAX_CHUNK_DURATION = 50   # Also used as threshold for long/short audio classification
    MIN_SILENCE_LEN_MS = 2000  # 2 seconds minimum silence
    SILENCE_THRESH_DB = -40    # Silence threshold in dB


class BotConfig:
    """Configuration class for Telegram bot."""
    
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
    TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
    
    # Telegram Bot API Local Server (for files up to 2 GB)
    USE_LOCAL_API = os.getenv("USE_LOCAL_API", "true").lower() == "true"
    LOCAL_API_BASE_URL = os.getenv("LOCAL_API_BASE_URL", "http://localhost:8081")
    
    # Redis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Processing limits
    MAX_FILE_SIZE_MB = 2000  # 2 GB with Local API Server
    MAX_HISTORY_PER_USER = 20
    
    # Supported audio formats
    SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".opus", ".wma", ".aac"}


def get_device():
    """Get available device (CUDA or CPU)."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"