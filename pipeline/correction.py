import time
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import load_correction_model


def save_correction_to_txt(dataframe, output_txt_path):
    """Save corrected summary results to text file.
    
    Args:
        dataframe: DataFrame with corrected summary results
        output_txt_path: Path for output text file
    """
    with open(output_txt_path, 'w', encoding='utf-8') as file:
        file.write("CORRECTED SUMMARIZATION RESULTS\n")
        file.write("=" * 60 + "\n\n")
        
        for _, row in dataframe.iterrows():
            file.write(f"Speaker: {row['speaker']}\n")
            file.write(f"Original Summary Length: {row['summary_length']} chars\n")
            file.write(f"Corrected Summary Length: {len(row['corrected_summary'])} chars\n")
            file.write(f"Compression Ratio: {row['compression_ratio']:.2f}\n")
            file.write(f"Original Summary: {row['summary']}\n")
            file.write(f"Corrected Summary: {row['corrected_summary']}\n")
            file.write("=" * 60 + "\n\n")


def parse_summarization_from_txt(txt_file_path):
    """Parse summarization results from text file.
    
    Args:
        txt_file_path: Path to summarization text file
        
    Returns:
        DataFrame: Parsed summarization data
    """
    data = []
    
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        current_speaker = None
        current_summary = None
        current_original_length = None
        current_summary_length = None
        current_compression_ratio = None
        capture_summary = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Speaker:"):
                current_speaker = line.replace("Speaker:", "").strip()
            elif line.startswith("Original Text Length:"):
                length_str = line.replace("Original Text Length:", "").replace("chars", "").strip()
                current_original_length = int(length_str)
            elif line.startswith("Summary Length:"):
                length_str = line.replace("Summary Length:", "").replace("chars", "").strip()
                current_summary_length = int(length_str)
            elif line.startswith("Compression Ratio:"):
                ratio_str = line.replace("Compression Ratio:", "").strip()
                current_compression_ratio = float(ratio_str)
            elif line.startswith("Summary:"):
                # We begin capturing the summarization text
                current_summary = line.replace("Summary:", "").strip()
                capture_summary = True
            elif capture_summary:
                # If this is a continuation of the summary text (not the next section)
                if line and not line.startswith("=") and not line.startswith("Speaker:"):
                    current_summary += " " + line
                else:
                    # Completed the summation capture
                    capture_summary = False
                    
                    # When all the data was collected
                    if current_speaker and current_summary:
                        data.append({
                            "speaker": current_speaker,
                            "original_text_length": current_original_length,
                            "summary_length": current_summary_length,
                            "compression_ratio": current_compression_ratio,
                            "summary": current_summary.strip()
                        })
                        
                        # Reset for next segment
                        current_speaker = None
                        current_summary = None
                        current_original_length = None
                        current_summary_length = None
                        current_compression_ratio = None
                    
                    # If you meet a new speaker, we start over.
                    if line.startswith("Speaker:"):
                        current_speaker = line.replace("Speaker:", "").strip()
    
    return pd.DataFrame(data)


def correct_text(input_text, correction_model, correction_tokenizer):
    """Correct text using the correction model.
    
    Args:
        input_text: Text to correct
        correction_model: Loaded correction model
        correction_tokenizer: Loaded correction tokenizer
        
    Returns:
        tuple: (corrected_text, success_status)
    """
    try:
        # For Russian text correction
        encodings = correction_tokenizer(input_text, return_tensors="pt")
        generated_tokens = correction_model.generate(
            **encodings, 
            forced_bos_token_id=correction_tokenizer.get_lang_id("ru")
        )
        corrected_text = correction_tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )[0]
        return corrected_text, True
    except Exception as error:
        print(f"Error correcting text: {error}")
        return input_text, False


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
    
    # Use provided DataFrame or parse from TXT file
    if asr_df is not None:
        input_dataframe = asr_df.copy()
        print("Using provided ASR DataFrame...")
    elif input_txt_path:
        input_dataframe = parse_summarization_from_txt(input_txt_path)
    else:
        print("No ASR data provided")
        return pd.DataFrame()
    
    if input_dataframe.empty:
        print("No ASR data found to correct")
        return pd.DataFrame()
    
    print(f"Correcting {len(input_dataframe)} ASR segments...")
    
    corrected_texts = []
    
    for index, row in input_dataframe.iterrows():
        print(f"Correcting text for {row['speaker']} [{row.get('start_time', ''):.1f}s]...")
        
        original_text = row["text"]
        corrected_text, success_status = correct_text(
            original_text, 
            correction_model, 
            correction_tokenizer
        )
        
        corrected_texts.append(corrected_text)
        
        print(f"Original:  {original_text}")
        print(f"Corrected: {corrected_text}")
        print("---")
    
    input_dataframe["corrected_text"] = corrected_texts
    
    # Save to text file only if path provided (CLI mode)
    if output_txt_path:
        save_correction_to_txt(input_dataframe, output_txt_path)
    
    total_execution_time = time.time() - start_time
    print(f"ASR text correction completed in {total_execution_time:.2f} seconds")
    
    return input_dataframe


if __name__ == "__main__":
    result_dataframe = perform_correction("summarization.txt", "correction_output.txt")
