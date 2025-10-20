import os
import sys
import time
import pandas as pd
from pydub import AudioSegment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import load_asr_model


def extract_audio_segments(audio_file_path, diarization_dataframe):
    """Extract audio segments based on diarization results.
    
    Args:
        audio_file_path: Path to original audio file
        diarization_dataframe: DataFrame with diarization segments
        
    Returns:
        list: List of dictionaries with segment information and audio data
    """
    audio = AudioSegment.from_file(audio_file_path)
    segments = []
    
    for index, row in diarization_dataframe.iterrows():
        start_ms = int(row["start_time"] * 1000)
        end_ms = int(row["end_time"] * 1000)
        segment_audio = audio[start_ms:end_ms]
        
        segments.append({
            "speaker": row["speaker"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "duration": row["duration"],
            "audio_segment": segment_audio,
            "segment_index": index
        })
    
    return segments


def transcribe_audio_segments(segments, asr_pipeline):
    """Transcribe audio segments using ASR model.
    
    Args:
        segments: List of audio segments
        asr_pipeline: Loaded ASR pipeline
        
    Returns:
        list: List of transcription results
    """
    results = []
    
    for segment in segments:
        temp_file = f"temp_{segment['speaker']}_{segment['segment_index']}.wav"
        
        try:
            segment["audio_segment"].export(temp_file, format="wav")
            asr_result = asr_pipeline(temp_file)
            text = asr_result["text"].strip()
            
            if text and text not in ["", ".", "..."]:
                segment_result = {
                    "speaker": segment["speaker"],
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "duration": segment["duration"],
                    "text": text,
                    "word_count": len(text.split())
                }
                results.append(segment_result)
                print(f"{segment['speaker']} [{segment['start_time']:.1f}s]: {text}")
            else:
                print(f"{segment['speaker']} [{segment['start_time']:.1f}s]: No speech detected")
                
        except Exception as error:
            print(f"Error processing segment {segment['segment_index']}: {error}")
        
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    return results


def perform_speech_recognition(audio_file_path, diarization_csv_path, output_csv_path="asr_results.csv"):
    """Perform speech recognition on diarized segments.
    
    Args:
        audio_file_path: Path to input audio file
        diarization_csv_path: Path to CSV with diarization results
        output_csv_path: Path for output CSV file
        
    Returns:
        tuple: (DataFrame with transcription results, dictionary with metrics)
    """
    start_time = time.time()
    
    print("Loading ASR model...")
    asr_pipeline = load_asr_model()
    
    print("Loading diarization results...")
    diarization_dataframe = pd.read_csv(diarization_csv_path)
    
    print("Extracting audio segments...")
    audio_segments = extract_audio_segments(audio_file_path, diarization_dataframe)
    
    print(f"Transcribing {len(audio_segments)} segments...")
    transcription_results = transcribe_audio_segments(audio_segments, asr_pipeline)
    
    results_dataframe = pd.DataFrame(transcription_results)
    execution_time = time.time() - start_time
    
    if not results_dataframe.empty:
        results_dataframe = results_dataframe.sort_values("start_time").reset_index(drop=True)
        results_dataframe.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        
        metrics = {
            "execution_time": float(execution_time),
            "segments_processed": int(len(audio_segments)),
            "segments_with_speech": int(len(results_dataframe)),
            "success_rate": float(len(results_dataframe) / len(audio_segments) if len(audio_segments) > 0 else 0),
            "total_words": int(results_dataframe["word_count"].sum()),
            "speakers_count": int(results_dataframe["speaker"].nunique())
        }
        
        print(f"Speech recognition completed in {execution_time:.2f} seconds")
        print(f"Segments processed: {metrics['segments_processed']}")
        print(f"Segments with speech: {metrics['segments_with_speech']}")
        print(f"Success rate: {metrics['success_rate']:.2%}")
        print(f"Total words recognized: {metrics['total_words']}")
        
        return results_dataframe, metrics
    
    print("No speech recognized in any segments")
    return pd.DataFrame(), {"execution_time": execution_time, "segments_processed": len(audio_segments)}


if __name__ == "__main__":
    audio_file = "1.wav"
    dataframe, performance_metrics = perform_speech_recognition(
        audio_file, 
        "diarization_results.csv", 
        "test_asr_results.csv"
    )