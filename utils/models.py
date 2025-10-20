from transformers import (
    pipeline, 
    GPT2Tokenizer, 
    T5ForConditionalGeneration, 
    M2M100ForConditionalGeneration, 
    M2M100Tokenizer
)
from pyannote.audio import Pipeline

from .config import ModelConfig, get_device


def load_diarization_model():
    """Load speaker diarization model.
    
    Returns:
        Pipeline: Pyannote diarization pipeline
    """
    return Pipeline.from_pretrained(
        ModelConfig.DIARIZATION_MODEL_NAME,
        token=ModelConfig.DIARIZATION_TOKEN
    )


def load_asr_model():
    """Load automatic speech recognition model.
    
    Returns:
        pipeline: HuggingFace ASR pipeline
    """
    return pipeline(
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


def load_correction_model():
    """Load text correction model.
    
    Returns:
        tuple: (model, tokenizer) for text correction
    """
    model = M2M100ForConditionalGeneration.from_pretrained(
        ModelConfig.CORRECTION_MODEL_NAME
    )
    tokenizer = M2M100Tokenizer.from_pretrained(
        ModelConfig.CORRECTION_MODEL_NAME,
        src_lang=ModelConfig.CORRECTION_SRC_LANG,
        tgt_lang=ModelConfig.CORRECTION_TGT_LANG
    )
    return model, tokenizer


def load_summarization_model():
    """Load text summarization model.
    
    Returns:
        tuple: (model, tokenizer) for text summarization
    """
    tokenizer = GPT2Tokenizer.from_pretrained(
        ModelConfig.SUMMARIZATION_MODEL_NAME, 
        eos_token="</s>"
    )
    model = T5ForConditionalGeneration.from_pretrained(
        ModelConfig.SUMMARIZATION_MODEL_NAME
    )
    model.to(get_device())
    return model, tokenizer