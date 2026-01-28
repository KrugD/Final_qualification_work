import argparse
import os
import sys
import shutil
from pathlib import Path

# Support both module execution (python -m pipeline.pipeline) and direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # When running as module: python -m pipeline.pipeline
    from pipeline.diarization import perform_diarization
    from pipeline.asr import perform_speech_recognition
    from pipeline.correction import perform_correction
    from pipeline.summarization import perform_summarization
    from pipeline.speaker_clustering import SpeakerClustering
except ImportError:
    # When running directly: python pipeline.py
    from diarization import perform_diarization
    from asr import perform_speech_recognition
    from correction import perform_correction
    from summarization import perform_summarization
    from speaker_clustering import SpeakerClustering

from utils.config import ModelConfig


def ensure_directory(directory_path):
    """Create directory if it doesn't exist."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def run_complete_pipeline(audio_file_path, output_dir):
    """Run complete pipeline with unified output structure.
    
    Args:
        audio_file_path: Path to input audio file
        output_dir: Output directory for this specific audio file
        
    Returns:
        bool: True if pipeline completed successfully
    """
    # Get audio file name without extension
    audio_filename = Path(audio_file_path).stem
    
    # Create output directory for this audio file
    ensure_directory(output_dir)
    
    print("=" * 60)
    print("STARTING MEETING TRANSCRIPTION PIPELINE")
    print("=" * 60)
    
    # Define output file paths
    diarization_output_path = os.path.join(output_dir, f"{audio_filename}_diarization.txt")
    asr_output_path = os.path.join(output_dir, f"{audio_filename}_asr.txt")
    summarization_output_path = os.path.join(output_dir, f"{audio_filename}_summarization.txt")
    correction_output_path = os.path.join(output_dir, f"{audio_filename}_correction.txt")
    
    # 1. Diarization
    print("\n1. DIARIZATION STAGE")
    diarization_dataframe = perform_diarization(audio_file_path, diarization_output_path)
    
    if diarization_dataframe.empty:
        print("Diarization failed - stopping pipeline")
        return False
    
    # 2. Speech Recognition
    print("\n2. SPEECH RECOGNITION STAGE")
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
    summarization_dataframe = perform_summarization(asr_output_path, summarization_output_path)
    
    if summarization_dataframe.empty:
        print("Summarization failed - stopping pipeline")
        return False
    
    # 4. Correction
    print("\n4. CORRECTION STAGE (Summarization Correction)")
    correction_dataframe = perform_correction(summarization_output_path, correction_output_path)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    print(f"Output files created in {output_dir}:")
    print(f"- Diarization: {diarization_output_path}")
    print(f"- ASR: {asr_output_path}")
    print(f"- Summarization: {summarization_output_path}")
    print(f"- Correction: {correction_output_path}")
    
    return True


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


def process_audio_file(audio_file_path, base_output_dir="pipeline_output", force_clustering=False):
    """Process single audio file with automatic chunking if needed.
    
    Args:
        audio_file_path: Path to audio file
        base_output_dir: Base directory for outputs
        force_clustering: Force speaker clustering even for short files
        
    Returns:
        bool: True if processing completed successfully
    """
    duration_minutes = get_audio_duration(audio_file_path)
    audio_filename = Path(audio_file_path).stem
    
    print(f"Duration: {duration_minutes:.1f} minutes")
    
    # Create output directory with audio filename
    output_dir = os.path.join(base_output_dir, audio_filename)
    
    if duration_minutes > ModelConfig.MAX_CHUNK_DURATION:
        print("File is long - processing with speaker clustering...")
        return process_long_audio_with_clustering(audio_file_path, base_output_dir)
    elif force_clustering:
        print("Force clustering enabled - processing with speaker analysis...")
        return process_with_speaker_analysis(audio_file_path, base_output_dir)
    else:
        print("File is short - processing as single file...")
        return run_complete_pipeline(audio_file_path, output_dir)


def process_with_speaker_analysis(audio_file_path, base_output_dir="pipeline_output"):
    """Process audio file with speaker embedding analysis and visualization.
    
    This function processes the audio normally and additionally extracts
    speaker embeddings to generate clustering visualization and quality metrics.
    
    Args:
        audio_file_path: Path to input audio file
        base_output_dir: Base directory for all outputs
        
    Returns:
        bool: True if processing completed successfully
    """
    audio_filename = Path(audio_file_path).stem
    output_dir = os.path.join(base_output_dir, audio_filename)
    ensure_directory(output_dir)
    
    # First, run the normal pipeline
    success = run_complete_pipeline(audio_file_path, output_dir)
    
    if not success:
        return False
    
    # Then extract speaker embeddings and generate visualization
    print("\n" + "=" * 60)
    print("SPEAKER ANALYSIS STAGE")
    print("=" * 60)
    
    try:
        speaker_clustering = SpeakerClustering(ModelConfig.DIARIZATION_TOKEN)
        
        # Extract per-segment embeddings for proper metrics calculation
        print("Extracting segment embeddings for analysis...")
        embeddings, labels, label_to_speaker = speaker_clustering.extract_segment_embeddings(audio_file_path)
        
        if embeddings is None or len(embeddings) == 0:
            print("No embeddings extracted - skipping visualization")
            return True
        
        n_speakers = len(set(labels))
        n_segments = len(labels)
        
        print(f"Found {n_segments} segments from {n_speakers} speakers")
        
        if n_speakers < 2:
            print("Only one speaker detected - skipping clustering analysis")
            return True
        
        # Calculate clustering quality metrics using segment-level embeddings
        # This measures how well segments from the same speaker cluster together
        speaker_clustering.calculate_clustering_metrics(embeddings, labels, output_dir)
        
        # For visualization, also get averaged embeddings per speaker
        speaker_embeddings = speaker_clustering.extract_speaker_embeddings(audio_file_path)
        
        if speaker_embeddings and len(speaker_embeddings) >= 3:
            # Create speaker data structure for visualization
            all_speaker_data = {audio_file_path: speaker_embeddings}
            
            # Generate speaker mapping
            speaker_mapping = {}
            for i, speaker in enumerate(speaker_embeddings.keys()):
                speaker_mapping[(audio_file_path, speaker)] = f"speaker_{i:02d}"
            
            speaker_clustering.visualize_clusters(all_speaker_data, speaker_mapping, output_dir)
        else:
            print(f"Only {n_speakers} speakers detected - need at least 3 for visualization")
        
        print("Speaker analysis completed")
        return True
        
    except Exception as e:
        print(f"Error in speaker analysis: {e}")
        import traceback
        traceback.print_exc()
        # Return True because the main pipeline succeeded
        return True


def process_long_audio_with_clustering(audio_file_path, base_output_dir="pipeline_output"):
    """Process long audio file with speaker clustering.
    
    Args:
        audio_file_path: Path to input audio file
        base_output_dir: Base directory for all outputs
        
    Returns:
        bool: True if processing completed successfully
    """
    audio_filename = Path(audio_file_path).stem
    final_output_dir = os.path.join(base_output_dir, audio_filename)
    ensure_directory(final_output_dir)
    
    # Initialize speaker clustering
    from utils.config import ModelConfig
    speaker_clustering = SpeakerClustering(ModelConfig.DIARIZATION_TOKEN)
    
    try:
        # Process long audio with clustering
        print("Processing long audio with speaker clustering...")
        clustering_result = speaker_clustering.process_long_audio(audio_file_path, final_output_dir)
        
        speaker_mapping = clustering_result['speaker_mapping']
        chunk_paths = clustering_result['chunk_paths']
        temp_dir = clustering_result['temp_dir']
        
        # Process each chunk with global speaker IDs
        chunk_diarizations = {}
        chunk_asr_contents = []
        
        for i, chunk_path in enumerate(chunk_paths):
            print(f"\n{'='*60}")
            print(f"PROCESSING CHUNK {i+1}/{len(chunk_paths)} WITH GLOBAL SPEAKER IDs")
            print(f"{'='*60}")
            
            # Create temporary directory for this chunk
            chunk_output_dir = os.path.join(base_output_dir, f"{audio_filename}_chunk_{i+1:02d}")
            
            # Process chunk
            success = run_complete_pipeline(chunk_path, chunk_output_dir)
            
            if success:
                print(f"Chunk {i+1} processed successfully")
                
                # Update diarization with global speaker IDs
                diarization_file = os.path.join(chunk_output_dir, f"{Path(chunk_path).stem}_diarization.txt")
                if os.path.exists(diarization_file):
                    updated_content = speaker_clustering.update_diarization_with_global_speakers(
                        diarization_file, speaker_mapping, chunk_path
                    )
                    chunk_diarizations[chunk_path] = updated_content
                    
                    # Save updated diarization
                    with open(diarization_file, 'w', encoding='utf-8') as f:
                        f.write(updated_content)
                
                # Store ASR content for combining
                asr_file = os.path.join(chunk_output_dir, f"{Path(chunk_path).stem}_asr.txt")
                if os.path.exists(asr_file):
                    with open(asr_file, 'r', encoding='utf-8') as f:
                        chunk_asr_contents.append(f.read())
                        
            else:
                print(f"Failed to process chunk {i+1}")
                return False
        
        # Combine diarization results with global speaker IDs
        print("\nCombining diarization results with global speaker IDs...")
        combined_diarization_path = os.path.join(final_output_dir, f"{audio_filename}_diarization.txt")
        with open(combined_diarization_path, 'w', encoding='utf-8') as f:
            f.write("SPEAKER DIARIZATION RESULTS (GLOBAL SPEAKER IDs)\n")
            f.write("=" * 50 + "\n\n")
            for chunk_path, content in chunk_diarizations.items():
                # Remove headers from subsequent files
                lines = content.split('\n')
                content_without_header = '\n'.join([line for line in lines 
                                                  if not line.startswith('SPEAKER DIARIZATION') 
                                                  and not line.startswith('=')])
                f.write(content_without_header)
                f.write("\n")
        
        # Combine ASR results
        print("Combining ASR results...")
        combined_asr_path = os.path.join(final_output_dir, f"{audio_filename}_asr.txt")
        with open(combined_asr_path, 'w', encoding='utf-8') as f:
            f.write("SPEECH RECOGNITION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            for i, content in enumerate(chunk_asr_contents):
                if i > 0:  # Remove header from subsequent files
                    lines = content.split('\n')
                    content = '\n'.join([line for line in lines 
                                       if not line.startswith('SPEECH RECOGNITION') 
                                       and not line.startswith('=')])
                f.write(content)
                if i < len(chunk_asr_contents) - 1:
                    f.write("\n\n")
        
        # Create overall summarization and correction
        print("Creating overall summarization...")
        summarization_output_path = os.path.join(final_output_dir, f"{audio_filename}_summarization.txt")
        correction_output_path = os.path.join(final_output_dir, f"{audio_filename}_correction.txt")
        
        perform_summarization(combined_asr_path, summarization_output_path)
        perform_correction(summarization_output_path, correction_output_path)
        
        print(f"Final results with global speaker IDs saved to {final_output_dir}")
        return True
        
    except Exception as e:
        print(f"Error in long audio processing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup temporary files
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            print("Cleaning up temporary audio chunk files...")
            shutil.rmtree(temp_dir)
            
        # Cleanup chunk directories
        chunk_dirs = [os.path.join(base_output_dir, d) for d in os.listdir(base_output_dir) 
                     if d.startswith(f"{audio_filename}_chunk_")]
        for chunk_dir in chunk_dirs:
            if os.path.exists(chunk_dir):
                shutil.rmtree(chunk_dir)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Offline meeting transcription pipeline: diarization, ASR, summarization, and correction'
    )
    parser.add_argument('--audio-file', type=str, help='Process specific audio file from audio_test folder')
    parser.add_argument('--force-clustering', action='store_true', 
                        help='Force speaker clustering analysis even for short files (generates visualization and metrics)')
    
    args = parser.parse_args()
    
    audio_test_dir = "audio_test"
    base_output_dir = "pipeline_output"
    
    if args.force_clustering:
        print("Force clustering mode enabled - will generate speaker visualization and metrics")
    
    if args.audio_file:
        # Process specific file
        input_audio_file = os.path.join(audio_test_dir, args.audio_file)
        
        if not os.path.exists(input_audio_file):
            print(f"File {input_audio_file} not found!")
            sys.exit(1)
            
        result = process_audio_file(input_audio_file, base_output_dir, args.force_clustering)
        
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
            
            result = process_audio_file(input_audio_file, base_output_dir, args.force_clustering)
            
            if result:
                print(f"✓ Successfully processed {audio_file}")
            else:
                print(f"✗ Failed to process {audio_file}")


if __name__ == "__main__":
    main()