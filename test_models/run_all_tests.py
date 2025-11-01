import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_diarization import perform_diarization
from test_asr import perform_speech_recognition
from test_correction import test_correction
from test_summarization import test_summarization, create_meeting_minutes


def ensure_directory(directory_path):
    """Create directory if it doesn't exist."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def run_complete_pipeline(audio_file_path, base_output_dir="pipeline_output"):
    """Run complete pipeline with organized output structure.
    
    Args:
        audio_file_path: Path to input audio file
        base_output_dir: Base directory for all outputs
        
    Returns:
        bool: True if pipeline completed successfully
    """
    # Get audio file name without extension
    audio_filename = Path(audio_file_path).stem
    
    # Create output directories
    dirs = {
        'diarization': os.path.join(base_output_dir, "diarization"),
        'asr': os.path.join(base_output_dir, "asr"), 
        'correction': os.path.join(base_output_dir, "correction"),
        'summarization': os.path.join(base_output_dir, "summarization")
    }
    
    for dir_path in dirs.values():
        ensure_directory(dir_path)
    
    print("=" * 60)
    print("STARTING COMPLETE PIPELINE TEST")
    print("=" * 60)
    
    # 1. Diarization
    print("\n1. DIARIZATION STAGE")
    diarization_output_path = os.path.join(dirs['diarization'], f"{audio_filename}_diarization.txt")
    diarization_dataframe = perform_diarization(audio_file_path, diarization_output_path)
    
    if diarization_dataframe.empty:
        print("Diarization failed - stopping pipeline")
        return False
    
    # 2. Speech Recognition
    print("\n2. SPEECH RECOGNITION STAGE")
    asr_output_path = os.path.join(dirs['asr'], f"{audio_filename}_asr.txt")
    asr_dataframe = perform_speech_recognition(
        audio_file_path,
        diarization_output_path,
        asr_output_path
    )
    
    if asr_dataframe.empty:
        print("Speech recognition failed - stopping pipeline")
        return False
    
    # 3. Correction
    print("\n3. CORRECTION STAGE")
    correction_output_path = os.path.join(dirs['correction'], f"{audio_filename}_correction.txt")
    correction_dataframe = test_correction(asr_output_path, correction_output_path)
    
    if correction_dataframe.empty:
        print("Correction failed - stopping pipeline")
        return False
    
    # 4. Summarization
    print("\n4. SUMMARIZATION STAGE")
    summarization_output_path = os.path.join(dirs['summarization'], f"{audio_filename}_summarization.txt")
    summarization_dataframe = test_summarization(correction_output_path, summarization_output_path)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    print(f"Output files created:")
    print(f"- Diarization: {diarization_output_path}")
    print(f"- ASR: {asr_output_path}")
    print(f"- Correction: {correction_output_path}")
    print(f"- Summarization: {summarization_output_path}")
    
    return True


if __name__ == "__main__":
    # Process all audio files in audio_test directory
    audio_test_dir = "audio_test"
    base_output_dir = "pipeline_output"
    
    if not os.path.exists(audio_test_dir):
        print(f"Error: Audio test directory {audio_test_dir} not found!")
        sys.exit(1)
    
    audio_files = [f for f in os.listdir(audio_test_dir) if f.endswith('.wav')]
    
    if not audio_files:
        print(f"No audio files found in {audio_test_dir}")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio files to process")
    
    for audio_file in audio_files:
        input_audio_file = os.path.join(audio_test_dir, audio_file)
        print(f"\nProcessing audio file: {input_audio_file}")
        
        success = run_complete_pipeline(input_audio_file, base_output_dir)
        
        if success:
            print(f"✓ Successfully processed {audio_file}")
        else:
            print(f"✗ Failed to process {audio_file}")