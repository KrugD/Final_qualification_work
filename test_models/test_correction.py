import time
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import load_correction_model


def save_correction_to_txt(dataframe, output_txt_path):
    """Save correction results to text file.
    
    Args:
        dataframe: DataFrame with correction results
        output_txt_path: Path for output text file
    """
    with open(output_txt_path, 'w', encoding='utf-8') as file:
        file.write("TEXT CORRECTION RESULTS\n")
        file.write("=" * 50 + "\n\n")
        
        for _, row in dataframe.iterrows():
            file.write(f"Speaker: {row['speaker']}\n")
            file.write(f"Start Time: {row['start_time']:.2f}s\n")
            file.write(f"End Time: {row['end_time']:.2f}s\n")
            file.write(f"Original: {row['text']}\n")
            file.write(f"Corrected: {row['corrected_text']}\n")
            file.write("-" * 50 + "\n")


def parse_asr_from_txt(txt_file_path):
    """Parse ASR results from text file.
    
    Args:
        txt_file_path: Path to ASR text file
        
    Returns:
        DataFrame: Parsed ASR data
    """
    data = []
    
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        current_speaker = None
        current_start = None
        current_end = None
        current_text = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Speaker:"):
                current_speaker = line.replace("Speaker:", "").strip()
            elif line.startswith("Start Time:"):
                start_str = line.replace("Start Time:", "").replace("s", "").strip()
                current_start = float(start_str)
            elif line.startswith("End Time:"):
                end_str = line.replace("End Time:", "").replace("s", "").strip()
                current_end = float(end_str)
            elif line.startswith("Text:"):
                current_text = line.replace("Text:", "").strip()
                
                # When we have all data, add to results
                if current_speaker and current_start is not None and current_end is not None and current_text:
                    data.append({
                        "speaker": current_speaker,
                        "start_time": current_start,
                        "end_time": current_end,
                        "text": current_text
                    })
                    
                    # Reset for next segment
                    current_speaker = None
                    current_start = None
                    current_end = None
                    current_text = None
    
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
    """Test text correction model on ASR results.
    
    Args:
        input_txt_path: Path to input text file with ASR results
        output_txt_path: Path for output text file with corrected texts
        
    Returns:
        DataFrame: DataFrame with corrected texts
    """
    start_time = time.time()
    
    print("Loading correction model...")
    correction_model, correction_tokenizer = load_correction_model()
    
    # Parse ASR data from text file
    input_dataframe = parse_asr_from_txt(input_txt_path)
    
    if input_dataframe.empty:
        print("No ASR data found to correct")
        return pd.DataFrame()
    
    print(f"Correcting {len(input_dataframe)} segments...")
    
    corrected_texts = []
    
    for index, row in input_dataframe.iterrows():
        print(f"Correcting segment {index + 1}/{len(input_dataframe)}...")
        
        original_text = row["text"]
        corrected_text, success_status = correct_text(
            original_text, 
            correction_model, 
            correction_tokenizer
        )
        
        corrected_texts.append(corrected_text)
        
        print(f"Original: {original_text}")
        print(f"Corrected: {corrected_text}")
        print("---")
    
    input_dataframe["corrected_text"] = corrected_texts
    
    # Save to text file
    save_correction_to_txt(input_dataframe, output_txt_path)
    
    total_execution_time = time.time() - start_time
    print(f"Correction completed in {total_execution_time:.2f} seconds")
    
    return input_dataframe


if __name__ == "__main__":
    result_dataframe = test_correction("asr.txt", "test_correction.txt")