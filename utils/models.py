from transformers import (
    pipeline, 
    GPT2Tokenizer, 
    T5ForConditionalGeneration, 
    M2M100ForConditionalGeneration, 
    M2M100Tokenizer
)
from pyannote.audio import Pipeline

from .config import ModelConfig, get_device

# Lazy singleton cache — each model is loaded once and reused
_model_cache = {}


def load_diarization_model():
    """Load speaker diarization model (cached).
    
    Returns:
        Pipeline: Pyannote diarization pipeline
    """
    if 'diarization' not in _model_cache:
        _model_cache['diarization'] = Pipeline.from_pretrained(
            ModelConfig.DIARIZATION_MODEL_NAME,
            token=ModelConfig.DIARIZATION_TOKEN
        )
    return _model_cache['diarization']


def load_asr_model():
    """Load automatic speech recognition model (cached).
    
    Returns:
        pipeline: HuggingFace ASR pipeline
    """
    if 'asr' not in _model_cache:
        _model_cache['asr'] = pipeline(
            "automatic-speech-recognition",
            model=ModelConfig.ASR_MODEL_NAME,
            tokenizer=ModelConfig.ASR_MODEL_NAME,
            feature_extractor=ModelConfig.ASR_MODEL_NAME,
            device=get_device(),
            generate_kwargs={
                "language": ModelConfig.ASR_LANGUAGE,
                "task": ModelConfig.ASR_TASK
            }
        )
    return _model_cache['asr']


def load_correction_model():
    """Load text correction model (cached, moved to GPU).
    
    Returns:
        tuple: (model, tokenizer) for text correction
    """
    if 'correction' not in _model_cache:
        device = get_device()
        model = M2M100ForConditionalGeneration.from_pretrained(
            ModelConfig.CORRECTION_MODEL_NAME
        )
        model.to(device)
        tokenizer = M2M100Tokenizer.from_pretrained(
            ModelConfig.CORRECTION_MODEL_NAME,
            src_lang=ModelConfig.CORRECTION_SRC_LANG,
            tgt_lang=ModelConfig.CORRECTION_TGT_LANG
        )
        _model_cache['correction'] = (model, tokenizer)
    return _model_cache['correction']


def load_summarization_model():
    """Load text summarization model (cached).
    
    Returns:
        tuple: (model, tokenizer) for text summarization
    """
    if 'summarization' not in _model_cache:
        tokenizer = GPT2Tokenizer.from_pretrained(
            ModelConfig.SUMMARIZATION_MODEL_NAME, 
            eos_token="</s>"
        )
        model = T5ForConditionalGeneration.from_pretrained(
            ModelConfig.SUMMARIZATION_MODEL_NAME
        )
        model.to(get_device())
        _model_cache['summarization'] = (model, tokenizer)
    return _model_cache['summarization']


def clear_model_cache():
    """Clear all cached models to free memory."""
    _model_cache.clear()
    print("Model cache cleared")
