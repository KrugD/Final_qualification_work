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


def test_correction(input_txt_path, output_txt_path):
    """Test text correction model on summarization results.
    
    Args:
        input_txt_path: Path to input text file with summarization results
        output_txt_path: Path for output text file with corrected summaries
        
    Returns:
        DataFrame: DataFrame with corrected summaries
    """
    start_time = time.time()
    
    print("Loading correction model...")
    correction_model, correction_tokenizer = load_correction_model()
    
    # Parse summarization data from text file
    input_dataframe = parse_summarization_from_txt(input_txt_path)
    
    if input_dataframe.empty:
        print("No summarization data found to correct")
        return pd.DataFrame()
    
    print(f"Correcting {len(input_dataframe)} speaker summaries...")
    
    corrected_summaries = []
    
    for index, row in input_dataframe.iterrows():
        print(f"Correcting summary for {row['speaker']}...")
        
        original_summary = row["summary"]
        corrected_summary, success_status = correct_text(
            original_summary, 
            correction_model, 
            correction_tokenizer
        )
        
        corrected_summaries.append(corrected_summary)
        
        print(f"Original: {original_summary}")
        print(f"Corrected: {corrected_summary}")
        print("---")
    
    input_dataframe["corrected_summary"] = corrected_summaries
    
    # Save to text file
    save_correction_to_txt(input_dataframe, output_txt_path)
    
    total_execution_time = time.time() - start_time
    print(f"Summarization correction completed in {total_execution_time:.2f} seconds")
    
    return input_dataframe


if __name__ == "__main__":
    result_dataframe = test_correction("summarization.txt", "test_correction.txt")