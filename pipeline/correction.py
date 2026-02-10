import time
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_device
from utils.models import load_correction_model


def save_correction_to_txt(dataframe, output_txt_path):
    """Save corrected ASR results to text file.
    
    Args:
        dataframe: DataFrame with corrected ASR results
        output_txt_path: Path for output text file
    """
    with open(output_txt_path, 'w', encoding='utf-8') as file:
        file.write("CORRECTED ASR RESULTS\n")
        file.write("=" * 60 + "\n\n")
        
        for _, row in dataframe.iterrows():
            file.write(f"Speaker: {row['speaker']}\n")
            file.write(f"Start Time: {row['start_time']:.2f}s\n")
            file.write(f"End Time: {row['end_time']:.2f}s\n")
            file.write(f"Duration: {row['duration']:.2f}s\n")
            file.write(f"Original Text: {row['text']}\n")
            file.write(f"Corrected Text: {row['corrected_text']}\n")
            file.write(f"Word Count: {row['word_count']}\n")
            file.write("-" * 60 + "\n")


def correct_text(input_text, correction_model, correction_tokenizer):
    """Correct text using the SAGE M2M100 correction model.
    
    Args:
        input_text: Text to correct
        correction_model: Loaded correction model
        correction_tokenizer: Loaded correction tokenizer
        
    Returns:
        tuple: (corrected_text, success_status)
    """
    try:
        device = next(correction_model.parameters()).device
        
        # Split long texts into sentences and correct each part
        # M2M100 has a limited generation length (~256 tokens)
        MAX_CHUNK_CHARS = 500
        
        if len(input_text) <= MAX_CHUNK_CHARS:
            # Short text — correct in one pass
            encodings = correction_tokenizer(input_text, return_tensors="pt").to(device)
            generated_tokens = correction_model.generate(
                **encodings, 
                forced_bos_token_id=correction_tokenizer.get_lang_id("ru"),
                max_new_tokens=512
            )
            corrected_text = correction_tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0]
            return corrected_text, True
        else:
            # Long text — split into sentence-level chunks, correct each, rejoin
            import re
            sentences = re.split(r'(?<=[.!?])\s+', input_text)
            
            corrected_parts = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 <= MAX_CHUNK_CHARS:
                    current_chunk = (current_chunk + " " + sentence).strip()
                else:
                    # Correct accumulated chunk
                    if current_chunk:
                        encodings = correction_tokenizer(current_chunk, return_tensors="pt").to(device)
                        generated_tokens = correction_model.generate(
                            **encodings, 
                            forced_bos_token_id=correction_tokenizer.get_lang_id("ru"),
                            max_new_tokens=512
                        )
                        corrected = correction_tokenizer.batch_decode(
                            generated_tokens, skip_special_tokens=True
                        )[0]
                        corrected_parts.append(corrected)
                    current_chunk = sentence
            
            # Correct the last chunk
            if current_chunk:
                encodings = correction_tokenizer(current_chunk, return_tensors="pt").to(device)
                generated_tokens = correction_model.generate(
                    **encodings, 
                    forced_bos_token_id=correction_tokenizer.get_lang_id("ru"),
                    max_new_tokens=512
                )
                corrected = correction_tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )[0]
                corrected_parts.append(corrected)
            
            corrected_text = " ".join(corrected_parts)
            return corrected_text, True
            
    except Exception as error:
        print(f"Error correcting text: {error}")
        return input_text, False


def perform_correction(asr_data, output_txt_path=None):
    """Perform text correction on ASR results (before summarization).
    
    Corrects recognition errors in transcribed text segments.
    Accepts ASR results either as a DataFrame (pipeline mode)
    or as a path to ASR text file (standalone mode).
    
    Args:
        asr_data: DataFrame with ASR results OR path to ASR text file
        output_txt_path: Optional path for output text file (None = don't save)
        
    Returns:
        DataFrame: DataFrame with original text and corrected_text columns
    """
    start_time = time.time()
    
    print("Loading correction model...")
    correction_model, correction_tokenizer = load_correction_model()
    
    # Accept both DataFrame and file path
    if isinstance(asr_data, pd.DataFrame):
        input_dataframe = asr_data.copy()
    else:
        print("Loading ASR results from file...")
        try:
            from pipeline.asr import parse_asr_from_txt
        except ImportError:
            from asr import parse_asr_from_txt
        input_dataframe = parse_asr_from_txt(asr_data)
    
    if input_dataframe.empty:
        print("No ASR data found to correct")
        return pd.DataFrame()
    
    print(f"Correcting {len(input_dataframe)} segments...")
    
    corrected_texts = []
    changes_count = 0
    
    for _, row in input_dataframe.iterrows():
        original_text = row["text"]
        corrected_text, success = correct_text(
            original_text, 
            correction_model, 
            correction_tokenizer
        )
        corrected_texts.append(corrected_text)
        
        if corrected_text != original_text:
            changes_count += 1
            print(f"  {row['speaker']} [{row['start_time']:.1f}s]:")
            print(f"    Original:  {original_text}")
            print(f"    Corrected: {corrected_text}")
        else:
            print(f"  {row['speaker']} [{row['start_time']:.1f}s]: OK (no changes)")
    
    input_dataframe["corrected_text"] = corrected_texts
    
    # Save to text file if path provided
    if output_txt_path:
        save_correction_to_txt(input_dataframe, output_txt_path)
    
    total_time = time.time() - start_time
    print(f"Correction completed in {total_time:.2f} seconds")
    print(f"Segments corrected: {changes_count}/{len(input_dataframe)}")
    
    return input_dataframe


if __name__ == "__main__":
    result_dataframe = perform_correction("asr_output.txt", "correction_output.txt")
