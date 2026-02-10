import time
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ModelConfig
from utils.models import load_diarization_model


def save_diarization_to_txt(dataframe, output_txt_path):
    """Save diarization results to text file.
    
    Args:
        dataframe: DataFrame with diarization results
        output_txt_path: Path for output text file
    """
    with open(output_txt_path, 'w', encoding='utf-8') as file:
        file.write("SPEAKER DIARIZATION RESULTS\n")
        file.write("=" * 40 + "\n\n")
        
        for _, row in dataframe.iterrows():
            file.write(f"Speaker: {row['speaker']}\n")
            file.write(f"Start: {row['start_time']:.2f}s\n")
            file.write(f"End: {row['end_time']:.2f}s\n")
            file.write(f"Duration: {row['duration']:.2f}s\n")
            file.write("-" * 30 + "\n")


def merge_consecutive_segments(dataframe, max_gap=None):
    """Merge consecutive segments from the same speaker.
    
    Reduces the number of segments by combining adjacent speech from the same
    speaker, giving Whisper more context and reducing ASR overhead.
    
    Args:
        dataframe: DataFrame with diarization results (sorted by start_time)
        max_gap: Maximum gap in seconds to merge segments (default from config)
        
    Returns:
        DataFrame: Merged segments
    """
    if max_gap is None:
        max_gap = ModelConfig.MERGE_GAP_SECONDS
    
    if dataframe.empty or len(dataframe) < 2:
        return dataframe
    
    merged = []
    current = dataframe.iloc[0].to_dict()
    
    for i in range(1, len(dataframe)):
        row = dataframe.iloc[i]
        
        # Merge if same speaker and gap is small enough
        if (row['speaker'] == current['speaker'] and 
            row['start_time'] - current['end_time'] <= max_gap):
            current['end_time'] = row['end_time']
            current['duration'] = current['end_time'] - current['start_time']
        else:
            merged.append(current)
            current = row.to_dict()
    
    merged.append(current)
    result = pd.DataFrame(merged).reset_index(drop=True)
    
    print(f"Merged {len(dataframe)} segments into {len(result)} segments")
    return result


def perform_diarization(audio_file_path, output_txt_path=None):
    """Perform speaker diarization on audio file.
    
    Args:
        audio_file_path: Path to input audio file
        output_txt_path: Optional path for output text file (None = don't save)
        
    Returns:
        DataFrame: Diarization results with columns: speaker, start_time, end_time, duration
    """
    start_time = time.time()
    
    print("Loading diarization model...")
    diarization_pipeline = load_diarization_model()
    
    print("Performing diarization...")
    diarization = diarization_pipeline(audio_file_path)
    
    segments = []
    
    # Access diarization results via speaker_diarization attribute (pyannote 3.x)
    for segment, track, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        if segment.end - segment.start < ModelConfig.MIN_SEGMENT_DURATION:
            continue
            
        segments.append({
            "speaker": speaker,
            "start_time": segment.start,
            "end_time": segment.end,
            "duration": segment.end - segment.start
        })
        print(f"{speaker} [{segment.start:.1f}s - {segment.end:.1f}s]")
    
    results_dataframe = pd.DataFrame(segments)
    
    if not results_dataframe.empty:
        results_dataframe = results_dataframe.sort_values("start_time").reset_index(drop=True)
        
        # Merge consecutive segments from same speaker to reduce ASR overhead
        results_dataframe = merge_consecutive_segments(results_dataframe)
        
        # Save to text file if path provided
        if output_txt_path:
            save_diarization_to_txt(results_dataframe, output_txt_path)
        
        execution_time = time.time() - start_time
        print(f"Diarization completed in {execution_time:.2f} seconds")
        print(f"Segments found: {len(results_dataframe)}")
        print(f"Unique speakers: {results_dataframe['speaker'].nunique()}")
        
        return results_dataframe
    
    print("No speech segments found")
    return pd.DataFrame()


if __name__ == "__main__":
    audio_file = "1.wav"
    dataframe = perform_diarization(audio_file, "diarization_output.txt")
