import time
import sys
import os
import pandas as pd
from pyannote.audio import Pipeline
from pydub import AudioSegment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ModelConfig
from utils.models import load_diarization_model


def perform_diarization(audio_file_path, output_csv_path="diarization_results.csv"):
    """Perform speaker diarization on audio file.
    
    Args:
        audio_file_path: Path to input audio file
        output_csv_path: Path for output CSV file
        
    Returns:
        tuple: (DataFrame with diarization results, dictionary with metrics)
    """
    start_time = time.time()
    
    print("Loading diarization model...")
    diarization_pipeline = load_diarization_model()
    
    print("Performing diarization...")
    diarization = diarization_pipeline(audio_file_path)
    audio = AudioSegment.from_file(audio_file_path)
    
    segments = []
    
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        if turn.end - turn.start < ModelConfig.MIN_SEGMENT_DURATION:
            continue
            
        segments.append({
            "speaker": speaker,
            "start_time": turn.start,
            "end_time": turn.end,
            "duration": turn.end - turn.start
        })
        print(f"{speaker} [{turn.start:.1f}s - {turn.end:.1f}s]")
    
    results_dataframe = pd.DataFrame(segments)
    execution_time = time.time() - start_time
    
    if not results_dataframe.empty:
        results_dataframe = results_dataframe.sort_values("start_time").reset_index(drop=True)
        results_dataframe.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        
        metrics = {
            "execution_time": float(execution_time),
            "segments_count": int(len(results_dataframe)),
            "speakers_count": int(results_dataframe["speaker"].nunique()),
            "audio_duration": float(results_dataframe["end_time"].max())
        }
        
        print(f"Diarization completed in {execution_time:.2f} seconds")
        print(f"Segments found: {metrics['segments_count']}")
        print(f"Unique speakers: {metrics['speakers_count']}")
        
        return results_dataframe, metrics
    
    print("No speech segments found")
    return pd.DataFrame(), {"execution_time": execution_time, "segments_count": 0}


if __name__ == "__main__":
    audio_file = "1.wav"
    dataframe, performance_metrics = perform_diarization(audio_file, "test_diarization_results.csv")