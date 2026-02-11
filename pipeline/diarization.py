import time
import sys
import os
import pandas as pd
from pydub import AudioSegment

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


def perform_diarization(audio_file_path, output_txt_path=None, diarization_model=None,
                        progress_callback=None):
    """Perform speaker diarization on audio file.
    
    Args:
        audio_file_path: Path to input audio file
        output_txt_path: Optional path for output text file (for CLI compatibility)
        diarization_model: Optional pre-loaded diarization model (for bot mode)
        progress_callback: Optional callback function for progress updates
        
    Returns:
        DataFrame: DataFrame with diarization results
    """
    start_time = time.time()
    
    if diarization_model is None:
        print("Loading diarization model...")
        diarization_model = load_diarization_model()
    
    print("Performing diarization...")
    diarization = diarization_model(audio_file_path)
    audio = AudioSegment.from_file(audio_file_path)
    
    segments = []
    
    # Correct way to access diarization results in new pyannote
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
        
        # Save to text file only if path provided (CLI mode)
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
