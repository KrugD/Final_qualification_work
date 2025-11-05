import argparse
import os
import sys
import shutil
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
        'summarization': os.path.join(base_output_dir, "summarization"),
        'correction': os.path.join(base_output_dir, "correction")
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
    
    # 3. Summarization
    print("\n3. SUMMARIZATION STAGE")
    summarization_output_path = os.path.join(dirs['summarization'], f"{audio_filename}_summarization.txt")
    summarization_dataframe = test_summarization(asr_output_path, summarization_output_path)
    
    if summarization_dataframe.empty:
        print("Summarization failed - stopping pipeline")
        return False
    
    # 4. Correction
    print("\n4. CORRECTION STAGE (Summarization Correction)")
    correction_output_path = os.path.join(dirs['correction'], f"{audio_filename}_correction.txt")
    correction_dataframe = test_correction(summarization_output_path, correction_output_path)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    print(f"Output files created:")
    print(f"- Diarization: {diarization_output_path}")
    print(f"- ASR: {asr_output_path}")
    print(f"- Summarization: {summarization_output_path}")
    print(f"- Correction: {correction_output_path}")
    
    return True


def split_audio_file(audio_file_path, chunk_duration_minutes=30):
    """Split audio file into chunks.
    
    Args:
        audio_file_path: Path to input audio file
        chunk_duration_minutes: Duration of each chunk in minutes
        
    Returns:
        list: List of paths to chunk files
    """
    from pydub import AudioSegment
    import os
    
    print(f"🔄 Splitting {audio_file_path} into {chunk_duration_minutes}-minute chunks...")
    
    audio = AudioSegment.from_file(audio_file_path)
    total_duration_minutes = len(audio) / (60 * 1000)
    chunk_duration_ms = chunk_duration_minutes * 60 * 1000
    
    audio_filename = Path(audio_file_path).stem
    chunks_dir = f"audio_chunks_{audio_filename}"
    os.makedirs(chunks_dir, exist_ok=True)
    
    chunk_paths = []
    num_chunks = (len(audio) + chunk_duration_ms - 1) // chunk_duration_ms
    
    for i in range(num_chunks):
        start_ms = i * chunk_duration_ms
        end_ms = min((i + 1) * chunk_duration_ms, len(audio))
        
        chunk = audio[start_ms:end_ms]
        chunk_path = os.path.join(chunks_dir, f"chunk_{i+1:02d}_{audio_filename}.wav")
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
        
        chunk_duration_min = len(chunk) / (60 * 1000)
        print(f"   Chunk {i+1}/{num_chunks}: {chunk_path} ({chunk_duration_min:.1f} min)")
    
    print(f"✅ Created {len(chunk_paths)} chunks")
    return chunk_paths, chunks_dir


def combine_txt_files(chunk_dirs, combined_dir, file_type, output_filename):
    """Combine text files from chunks.
    
    Args:
        chunk_dirs: List of chunk directory paths
        combined_dir: Directory for combined results
        file_type: Type of files to combine ('diarization', 'asr', etc.)
        output_filename: Name for combined output file
    """
    all_content = []
    
    for chunk_dir in chunk_dirs:
        file_type_dir = os.path.join(chunk_dir, file_type)
        if os.path.exists(file_type_dir):
            files = os.listdir(file_type_dir)
            if files:
                file_path = os.path.join(file_type_dir, files[0])
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        all_content.append(content)
    
    if all_content:
        output_path = os.path.join(combined_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(all_content))
        print(f"   📄 Combined {file_type} files")


def combine_summaries(chunk_dirs, combined_dir, audio_filename):
    """Combine summaries and create overall summary.
    
    Args:
        chunk_dirs: List of chunk directory paths
        combined_dir: Directory for combined results
        audio_filename: Original audio filename
    """
    # Collect all ASR texts for overall summarization
    all_asr_texts = []
    
    for chunk_dir in chunk_dirs:
        asr_dir = os.path.join(chunk_dir, "asr")
        if os.path.exists(asr_dir):
            files = os.listdir(asr_dir)
            if files:
                asr_file_path = os.path.join(asr_dir, files[0])
                if os.path.exists(asr_file_path):
                    with open(asr_file_path, 'r', encoding='utf-8') as f:
                        all_asr_texts.append(f.read())
    
    if all_asr_texts:
        # Create temporary file with all text
        combined_asr_path = os.path.join(combined_dir, f"{audio_filename}_all_asr.txt")
        with open(combined_asr_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(all_asr_texts))
        
        # Create overall summarization from all text
        summarization_output_path = os.path.join(combined_dir, f"{audio_filename}_summarization.txt")
        test_summarization(combined_asr_path, summarization_output_path)
        
        # Create corrected summarization
        correction_output_path = os.path.join(combined_dir, f"{audio_filename}_correction.txt")
        test_correction(summarization_output_path, correction_output_path)
        
        print(f"   📊 Created overall summarization")


def process_long_audio(audio_file_path, base_output_dir="pipeline_output", chunk_duration_minutes=30):
    """Process long audio file by splitting into chunks.
    
    Args:
        audio_file_path: Path to input audio file
        base_output_dir: Base directory for all outputs
        chunk_duration_minutes: Duration of each chunk in minutes
        
    Returns:
        bool: True if processing completed successfully
    """
    # Split audio into chunks
    chunk_paths, chunks_dir = split_audio_file(audio_file_path, chunk_duration_minutes)
    audio_filename = Path(audio_file_path).stem
    
    chunk_dirs = []
    
    try:
        # Process each chunk
        for i, chunk_path in enumerate(chunk_paths):
            print(f"\n{'='*60}")
            print(f"🎯 PROCESSING CHUNK {i+1}/{len(chunk_paths)}")
            print(f"{'='*60}")
            
            # Create separate directory for each chunk
            chunk_output_dir = os.path.join(base_output_dir, f"{audio_filename}_chunk_{i+1:02d}")
            success = run_complete_pipeline(chunk_path, chunk_output_dir)
            
            if success:
                print(f"✅ Chunk {i+1} processed successfully")
                chunk_dirs.append(chunk_output_dir)
            else:
                print(f"❌ Failed to process chunk {i+1}")
                return False
        
        # Create final output directory (same as for short files)
        final_output_dir = os.path.join(base_output_dir, audio_filename)
        ensure_directory(final_output_dir)
        
        # Combine results into final directory
        print(f"\n🔄 Combining results into final directory...")
        
        # Combine diarization
        combine_txt_files(chunk_dirs, final_output_dir, "diarization", f"{audio_filename}_diarization.txt")
        
        # Combine ASR
        combine_txt_files(chunk_dirs, final_output_dir, "asr", f"{audio_filename}_asr.txt")
        
        # Combine summaries and create overall summary
        combine_summaries(chunk_dirs, final_output_dir, audio_filename)
        
        print(f"✅ Final results saved to {final_output_dir}")
        
        return True
        
    finally:
        # Cleanup: remove temporary chunk files and directories
        print(f"\n🧹 Cleaning up temporary files...")
        
        # Remove chunk audio files
        if os.path.exists(chunks_dir):
            shutil.rmtree(chunks_dir)
            print(f"   Removed temporary audio chunks: {chunks_dir}")
        
        # Remove chunk processing directories
        for chunk_dir in chunk_dirs:
            if os.path.exists(chunk_dir):
                shutil.rmtree(chunk_dir)
                print(f"   Removed temporary directory: {chunk_dir}")


def get_audio_duration(audio_file_path):
    """Get duration of audio file in minutes.
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        float: Duration in minutes
    """
    from pydub import AudioSegment
    audio = AudioSegment.from_file(audio_file_path)
    return len(audio) / (60 * 1000)


def process_audio_file(audio_file_path, base_output_dir="pipeline_output"):
    """Process single audio file with automatic chunking if needed.
    
    Args:
        audio_file_path: Path to audio file
        base_output_dir: Base directory for outputs
        
    Returns:
        bool: True if processing completed successfully
    """
    duration_minutes = get_audio_duration(audio_file_path)
    audio_filename = Path(audio_file_path).stem
    
    print(f"Duration: {duration_minutes:.1f} minutes")
    
    # Create output directory with audio filename
    output_dir = os.path.join(base_output_dir, audio_filename)
    
    if duration_minutes > 30:
        print("File is long - splitting into chunks...")
        return process_long_audio(audio_file_path, base_output_dir)
    else:
        print("File is short - processing as single file...")
        # For short files, use the named directory directly
        return run_complete_pipeline(audio_file_path, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run audio processing pipeline')
    parser.add_argument('--audio-file', type=str, help='Process specific audio file from audio_test folder')
    
    args = parser.parse_args()
    
    audio_test_dir = "audio_test"
    base_output_dir = "pipeline_output"
    
    if args.audio_file:
        # Process specific file
        input_audio_file = os.path.join(audio_test_dir, args.audio_file)
        
        if not os.path.exists(input_audio_file):
            print(f"File {input_audio_file} not found!")
            sys.exit(1)
            
        result = process_audio_file(input_audio_file, base_output_dir)
        
        if result:
            print(f"✓ Successfully processed {args.audio_file}")
        else:
            print(f"✗ Failed to process {args.audio_file}")
    else:
        # Process all files in directory
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
            print(f"\n{'='*60}")
            print(f"PROCESSING: {input_audio_file}")
            print(f"{'='*60}")
            
            result = process_audio_file(input_audio_file, base_output_dir)
            
            if result:
                print(f"✓ Successfully processed {audio_file}")
            else:
                print(f"✗ Failed to process {audio_file}")