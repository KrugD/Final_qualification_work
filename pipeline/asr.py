import os
import sys
import time
import tempfile
import pandas as pd
from pydub import AudioSegment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.models import load_asr_model


def save_asr_to_txt(dataframe, output_txt_path):
    """Save ASR results to text file.
    
    Args:
        dataframe: DataFrame with ASR results
        output_txt_path: Path for output text file
    """
    with open(output_txt_path, 'w', encoding='utf-8') as file:
        file.write("SPEECH RECOGNITION RESULTS\n")
        file.write("=" * 50 + "\n\n")
        
        for _, row in dataframe.iterrows():
            file.write(f"Speaker: {row['speaker']}\n")
            file.write(f"Start Time: {row['start_time']:.2f}s\n")
            file.write(f"End Time: {row['end_time']:.2f}s\n")
            file.write(f"Duration: {row['duration']:.2f}s\n")
            file.write(f"Text: {row['text']}\n")
            file.write(f"Word Count: {row['word_count']}\n")
            file.write("-" * 50 + "\n")


def parse_diarization_from_txt(txt_file_path):
    """Parse diarization results from text file.
    
    Args:
        txt_file_path: Path to diarization text file
        
    Returns:
        DataFrame: Parsed diarization data
    """
    data = []
    
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        current_speaker = None
        current_start = None
        current_end = None
        current_duration = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Speaker:"):
                current_speaker = line.replace("Speaker:", "").strip()
            elif line.startswith("Start:"):
                start_str = line.replace("Start:", "").replace("s", "").strip()
                current_start = float(start_str)
            elif line.startswith("End:"):
                end_str = line.replace("End:", "").replace("s", "").strip()
                current_end = float(end_str)
            elif line.startswith("Duration:"):
                duration_str = line.replace("Duration:", "").replace("s", "").strip()
                current_duration = float(duration_str)
                
                if current_speaker and current_start is not None and current_end is not None:
                    data.append({
                        "speaker": current_speaker,
                        "start_time": current_start,
                        "end_time": current_end,
                        "duration": current_duration
                    })
                    
                    current_speaker = None
                    current_start = None
                    current_end = None
                    current_duration = None
    
    return pd.DataFrame(data)


def parse_asr_from_txt(txt_file_path):
    """Parse ASR results from text file.
    
    Args:
        txt_file_path: Path to ASR text file
        
    Returns:
        DataFrame: Parsed ASR data with columns: speaker, start_time, end_time, duration, text, word_count
    """
    data = []
    
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        current_speaker = None
        current_start = None
        current_end = None
        current_duration = None
        current_text = None
        current_word_count = None
        
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
            elif line.startswith("Duration:"):
                duration_str = line.replace("Duration:", "").replace("s", "").strip()
                current_duration = float(duration_str)
            elif line.startswith("Text:"):
                current_text = line.replace("Text:", "").strip()
            elif line.startswith("Word Count:"):
                wc_str = line.replace("Word Count:", "").strip()
                current_word_count = int(wc_str)
                
                if current_speaker and current_text:
                    data.append({
                        "speaker": current_speaker,
                        "start_time": current_start,
                        "end_time": current_end,
                        "duration": current_duration,
                        "text": current_text,
                        "word_count": current_word_count
                    })
                    
                    current_speaker = None
                    current_start = None
                    current_end = None
                    current_duration = None
                    current_text = None
                    current_word_count = None
    
    return pd.DataFrame(data)


def extract_audio_segments(audio_file_path, diarization_dataframe):
    """Extract audio segments based on diarization results.
    
    Args:
        audio_file_path: Path to original audio file
        diarization_dataframe: DataFrame with diarization segments
        
    Returns:
        list: List of dictionaries with segment information and audio data
    """
    try:
        audio = AudioSegment.from_file(audio_file_path)
        segments = []
        
        for index, row in diarization_dataframe.iterrows():
            start_ms = int(row["start_time"] * 1000)
            end_ms = int(row["end_time"] * 1000)
            
            if end_ms > len(audio):
                end_ms = len(audio)
            
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
        
    except Exception as error:
        print(f"Error loading audio file: {error}")
        return []


def transcribe_audio_segments(segments, asr_pipeline):
    """Transcribe audio segments using ASR model.
    
    Uses tempfile for secure temporary file handling.
    
    Args:
        segments: List of audio segments
        asr_pipeline: Loaded ASR pipeline
        
    Returns:
        list: List of transcription results
    """
    results = []
    
    for segment in segments:
        temp_file = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_file = tmp.name
                segment["audio_segment"].export(temp_file, format="wav")
            
            asr_result = asr_pipeline(temp_file, return_timestamps=True)
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
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
    
    return results


def perform_speech_recognition(audio_file_path, diarization_txt_path=None,
                                output_txt_path=None, diarization_df=None,
                                asr_model=None, progress_callback=None):
    """Perform speech recognition on diarized segments.
    
    Args:
        audio_file_path: Path to input audio file
        diarization_txt_path: Path to text file with diarization results (CLI mode)
        output_txt_path: Path for output text file (CLI mode)
        diarization_df: DataFrame with diarization results (bot mode, skips TXT parsing)
        asr_model: Optional pre-loaded ASR model (for bot mode)
        progress_callback: Optional callback function for progress updates
        
    Returns:
        DataFrame: DataFrame with transcription results
    """
    start_time = time.time()
    
    if asr_model is None:
        print("Loading ASR model...")
        asr_model = load_asr_model()
    
    if diarization_df is not None:
        diarization_dataframe = diarization_df
        print("Using provided diarization DataFrame...")
    elif diarization_txt_path:
        print("Loading diarization results from file...")
        diarization_dataframe = parse_diarization_from_txt(diarization_txt_path)
    else:
        print("No diarization data provided")
        return pd.DataFrame()
    
    if diarization_dataframe.empty:
        print("No diarization data found")
        return pd.DataFrame()
    
    print(f"Found {len(diarization_dataframe)} diarization segments")
    
    print("Extracting audio segments...")
    audio_segments = extract_audio_segments(audio_file_path, diarization_dataframe)
    
    print(f"Transcribing {len(audio_segments)} segments...")
    transcription_results = transcribe_audio_segments(audio_segments, asr_model)
    
    results_dataframe = pd.DataFrame(transcription_results)
    
    if not results_dataframe.empty:
        results_dataframe = results_dataframe.sort_values("start_time").reset_index(drop=True)
        
        if output_txt_path:
            save_asr_to_txt(results_dataframe, output_txt_path)
        
        execution_time = time.time() - start_time
        print(f"Speech recognition completed in {execution_time:.2f} seconds")
        print(f"Segments processed: {len(audio_segments)}")
        print(f"Segments with speech: {len(results_dataframe)}")
        print(f"Total words recognized: {results_dataframe['word_count'].sum()}")
        
        return results_dataframe
    
    print("No speech recognized in any segments")
    return pd.DataFrame()


if __name__ == "__main__":
    audio_file = "1.wav"
    dataframe = perform_speech_recognition(audio_file, "diarization.txt", "asr_output.txt")
