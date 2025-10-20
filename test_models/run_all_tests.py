import json
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_diarization import perform_diarization
from test_asr import perform_speech_recognition
from test_correction import test_correction
from test_summarization import test_summarization, create_meeting_minutes

def convert_metrics_for_json(metrics):
    """Convert metrics with numpy/pandas types to JSON-serializable types.
    
    Args:
        metrics: Dictionary with metrics
        
    Returns:
        dict: JSON-serializable dictionary
    """
    def convert_value(value):
        if isinstance(value, (np.integer, np.int64)):
            return int(value)
        elif isinstance(value, (np.floating, np.float64)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return value
    
    if isinstance(metrics, dict):
        return {key: convert_metrics_for_json(value) for key, value in metrics.items()}
    else:
        return convert_value(metrics)

def run_complete_pipeline(audio_file_path, output_prefix="test"):
    """Run complete pipeline with metrics collection.
    
    Args:
        audio_file_path: Path to input audio file
        output_prefix: Prefix for output files
        
    Returns:
        dict: Dictionary with all collected metrics
    """
    all_metrics = {}
    
    print("=" * 60)
    print("STARTING COMPLETE PIPELINE TEST")
    print("=" * 60)
    
    # 1. Diarization
    print("\n1. DIARIZATION STAGE")
    diarization_output_path = f"{output_prefix}_diarization.csv"
    diarization_dataframe, diarization_metrics = perform_diarization(
        audio_file_path, 
        diarization_output_path
    )
    all_metrics["diarization"] = diarization_metrics
    
    if diarization_dataframe.empty:
        print("Diarization failed - stopping pipeline")
        return all_metrics
    
    # 2. Speech Recognition
    print("\n2. SPEECH RECOGNITION STAGE")
    asr_output_path = f"{output_prefix}_asr.csv"
    asr_dataframe, asr_metrics = perform_speech_recognition(
        audio_file_path,
        diarization_output_path,
        asr_output_path
    )
    all_metrics["asr"] = asr_metrics
    
    if asr_dataframe.empty:
        print("Speech recognition failed - stopping pipeline")
        return all_metrics
    
    # 3. Correction
    print("\n3. CORRECTION STAGE")
    correction_output_path = f"{output_prefix}_correction.csv"
    correction_dataframe, correction_metrics = test_correction(
        asr_output_path, 
        correction_output_path
    )
    all_metrics["correction"] = correction_metrics
    
    # 4. Summarization
    print("\n4. SUMMARIZATION STAGE")
    summarization_output_path = f"{output_prefix}_summarization.csv"
    summarization_dataframe, summarization_metrics = test_summarization(
        correction_output_path, 
        summarization_output_path
    )
    all_metrics["summarization"] = summarization_metrics
    
    # 5. Create Meeting Minutes
    minutes_output_path = f"{output_prefix}_meeting_minutes.txt"
    create_meeting_minutes(summarization_output_path, minutes_output_path)
    
    # Save all metrics to JSON
    metrics_output_path = f"{output_prefix}_metrics.json"
    
    # Convert metrics to JSON-serializable format
    json_metrics = convert_metrics_for_json(all_metrics)
    
    with open(metrics_output_path, "w", encoding="utf-8") as metrics_file:
        json.dump(json_metrics, metrics_file, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    # Counting total execution time
    total_pipeline_time = (
        all_metrics["diarization"]["execution_time"] + 
        all_metrics["asr"]["execution_time"] + 
        all_metrics["correction"]["execution_time"] + 
        all_metrics["summarization"]["execution_time"]
    )
    
    print(f"Total pipeline execution time: {total_pipeline_time:.2f} seconds")
    print(f"Final metrics saved to: {metrics_output_path}")
    
    # Output key metrics
    print("\nKEY METRICS:")
    print(f"- Speakers identified: {all_metrics['diarization']['speakers_count']}")
    print(f"- Speech segments: {all_metrics['asr']['segments_with_speech']}")
    print(f"- Total words: {all_metrics['asr']['total_words']}")
    print(f"- Correction success rate: {all_metrics['correction']['success_rate']:.2%}")
    print(f"- Summarization success rate: {all_metrics['summarization']['success_rate']:.2%}")
    
    return all_metrics


if __name__ == "__main__":

    input_audio_file = "audio_test/1.wav"
    
    if not os.path.exists(input_audio_file):
        print(f"Error: Audio file {input_audio_file} not found!")
        print("Please make sure the file exists in the audio_test directory.")
    else:
        print(f"Processing audio file: {input_audio_file}")
        pipeline_metrics = run_complete_pipeline(input_audio_file, "full_test")