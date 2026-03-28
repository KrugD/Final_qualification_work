import re
import time
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_device
from utils.models import load_correction_model


MAX_CHUNK_CHARS = 500


def save_correction_to_txt(dataframe, output_txt_path):
    """Save corrected ASR results to text file.
    
    Args:
        dataframe: DataFrame with corrected ASR results (must have 'text' and 'corrected_text')
        output_txt_path: Path for output text file
    """
    with open(output_txt_path, 'w', encoding='utf-8') as file:
        file.write("TEXT CORRECTION RESULTS\n")
        file.write("=" * 60 + "\n\n")
        
        for _, row in dataframe.iterrows():
            file.write(f"Speaker: {row['speaker']}\n")
            file.write(f"Start Time: {row.get('start_time', 0):.2f}s\n")
            file.write(f"End Time: {row.get('end_time', 0):.2f}s\n")
            file.write(f"Original Text: {row['text']}\n")
            file.write(f"Corrected Text: {row['corrected_text']}\n")
            file.write("=" * 60 + "\n\n")


def correct_text(input_text, correction_model, correction_tokenizer):
    """Correct text using the correction model.
    
    Handles long texts by splitting into sentence-level chunks to avoid
    truncation by the M2M100 model.
    
    Args:
        input_text: Text to correct
        correction_model: Loaded correction model
        correction_tokenizer: Loaded correction tokenizer
        
    Returns:
        tuple: (corrected_text, success_status)
    """
    try:
        device = get_device()
        
        if len(input_text) <= MAX_CHUNK_CHARS:
            return _correct_chunk(input_text, correction_model, correction_tokenizer, device)
        
        sentences = re.split(r'(?<=[.!?])\s+', input_text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if current_chunk and len(current_chunk) + len(sentence) + 1 > MAX_CHUNK_CHARS:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        if not chunks:
            return input_text, False
        
        corrected_parts = []
        for chunk in chunks:
            corrected_chunk, _ = _correct_chunk(chunk, correction_model, correction_tokenizer, device)
            corrected_parts.append(corrected_chunk)
        
        return " ".join(corrected_parts), True
        
    except Exception as error:
        print(f"Error correcting text: {error}")
        return input_text, False


def _correct_chunk(text, model, tokenizer, device):
    """Correct a single chunk of text."""
    encodings = tokenizer(text, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}
    
    generated_tokens = model.generate(
        **encodings,
        forced_bos_token_id=tokenizer.get_lang_id("ru"),
        max_new_tokens=512
    )
    corrected = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return corrected, True


def perform_correction(input_txt_path=None, output_txt_path=None, asr_df=None,
                       correction_model=None, correction_tokenizer=None,
                       progress_callback=None):
    """Perform text correction on ASR transcription results.
    
    Args:
        input_txt_path: Path to input text file with ASR results (CLI mode)
        output_txt_path: Path for output text file with corrected texts (CLI mode)
        asr_df: DataFrame with ASR results (bot mode, skips TXT parsing)
        correction_model: Optional pre-loaded model (for bot mode)
        correction_tokenizer: Optional pre-loaded tokenizer (for bot mode)
        progress_callback: Optional callback function for progress updates
        
    Returns:
        DataFrame: DataFrame with corrected texts (adds 'corrected_text' column)
    """
    start_time = time.time()
    
    if correction_model is None or correction_tokenizer is None:
        print("Loading correction model...")
        correction_model, correction_tokenizer = load_correction_model()
    
    if asr_df is not None:
        input_dataframe = asr_df.copy()
        print("Using provided ASR DataFrame...")
    elif input_txt_path:
        from pipeline.asr import parse_asr_from_txt
        input_dataframe = parse_asr_from_txt(input_txt_path)
    else:
        print("No ASR data provided")
        return pd.DataFrame()
    
    if input_dataframe.empty:
        print("No ASR data found to correct")
        return pd.DataFrame()
    
    text_column = 'text'
    if text_column not in input_dataframe.columns:
        if 'corrected_text' in input_dataframe.columns:
            text_column = 'corrected_text'
        else:
            print(f"No text column found in input data")
            return pd.DataFrame()
    
    print(f"Correcting {len(input_dataframe)} ASR segments...")
    
    corrected_texts = []
    
    for index, row in input_dataframe.iterrows():
        speaker = row['speaker']
        start_t = row.get('start_time', None)
        if start_t is not None:
            print(f"Correcting segment for {speaker} [{start_t:.1f}s]...")
        else:
            print(f"Correcting segment for {speaker}...")
        
        original_text = row[text_column]
        corrected_text, success = correct_text(
            original_text,
            correction_model,
            correction_tokenizer
        )
        
        corrected_texts.append(corrected_text)
        
        print(f"Original:  {original_text[:100]}...")
        print(f"Corrected: {corrected_text[:100]}...")
        print("---")
    
    input_dataframe["corrected_text"] = corrected_texts
    
    if output_txt_path:
        save_correction_to_txt(input_dataframe, output_txt_path)
    
    total_execution_time = time.time() - start_time
    print(f"Text correction completed in {total_execution_time:.2f} seconds")
    
    return input_dataframe


if __name__ == "__main__":
    result_dataframe = perform_correction("asr_output.txt", "correction_output.txt")
