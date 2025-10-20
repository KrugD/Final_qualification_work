import time
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import load_correction_model


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


def test_correction(input_csv_path="diarization_results.csv", output_csv_path="correction_results.csv"):
    """Test text correction model on diarization results.
    
    Args:
        input_csv_path: Path to input CSV with diarization results
        output_csv_path: Path for output CSV with corrected texts
        
    Returns:
        tuple: (DataFrame with corrected texts, dictionary with metrics)
    """
    start_time = time.time()
    
    print("Loading correction model...")
    correction_model, correction_tokenizer = load_correction_model()
    
    input_dataframe = pd.read_csv(input_csv_path, encoding="utf-8-sig")
    
    print(f"Correcting {len(input_dataframe)} segments...")
    
    corrected_texts = []
    successful_corrections = 0
    processing_times = []
    
    for index, text in enumerate(input_dataframe["text"]):
        print(f"Correcting segment {index + 1}/{len(input_dataframe)}...")
        
        segment_start_time = time.time()
        corrected_text, success_status = correct_text(
            text, 
            correction_model, 
            correction_tokenizer
        )
        segment_processing_time = time.time() - segment_start_time
        processing_times.append(segment_processing_time)
        
        corrected_texts.append(corrected_text)
        if success_status:
            successful_corrections += 1
        
        print(f"Original: {text}")
        print(f"Corrected: {corrected_text}")
        print(f"Processing time: {segment_processing_time:.2f} seconds")
        print("---")
    
    input_dataframe["corrected_text"] = corrected_texts
    input_dataframe.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    
    total_execution_time = time.time() - start_time
    metrics = {
        "execution_time": float(total_execution_time),
        "segments_count": int(len(input_dataframe)),
        "successful_corrections": int(successful_corrections),
        "success_rate": float(successful_corrections / len(input_dataframe) if len(input_dataframe) > 0 else 0),
        "avg_processing_time_per_segment": float(
            sum(processing_times) / len(processing_times) if processing_times else 0
        ),
        "total_characters_processed": int(sum(len(text) for text in input_dataframe["text"]))
    }
    
    print(f"Correction completed in {total_execution_time:.2f} seconds")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Average time per segment: {metrics['avg_processing_time_per_segment']:.2f} seconds")
    
    return input_dataframe, metrics


if __name__ == "__main__":
    result_dataframe, performance_metrics = test_correction(
        "diarization_results.csv", 
        "test_correction_results.csv"
    )