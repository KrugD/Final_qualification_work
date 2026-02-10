import argparse
import os
import sys
import shutil
import pandas as pd
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
from utils.models import clear_model_cache


def ensure_directory(directory_path):
    """Create directory if it doesn't exist."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def run_complete_pipeline(audio_file_path, output_dir):
    """Run complete pipeline: Diarization -> ASR -> Correction -> Summarization.
    
    Data flows directly between stages as DataFrames.
    Text files are saved at each stage for logging/debugging.
    
    Args:
        audio_file_path: Path to input audio file
        output_dir: Output directory for this specific audio file
        
    Returns:
        dict: Pipeline results with DataFrames, or None on failure
    """
    audio_filename = Path(audio_file_path).stem
    ensure_directory(output_dir)
    
    print("=" * 60)
    print("STARTING MEETING TRANSCRIPTION PIPELINE")
    print("=" * 60)
    
    # Define output file paths
    diarization_output_path = os.path.join(output_dir, f"{audio_filename}_diarization.txt")
    asr_output_path = os.path.join(output_dir, f"{audio_filename}_asr.txt")
    correction_output_path = os.path.join(output_dir, f"{audio_filename}_correction.txt")
    summarization_output_path = os.path.join(output_dir, f"{audio_filename}_summarization.txt")
    
    # 1. Diarization
    print("\n1. DIARIZATION STAGE")
    diarization_df = perform_diarization(audio_file_path, diarization_output_path)
    
    if diarization_df.empty:
        print("Diarization failed - stopping pipeline")
        return None
    
    # 2. Speech Recognition (receives DataFrame directly)
    print("\n2. SPEECH RECOGNITION STAGE")
    asr_df = perform_speech_recognition(audio_file_path, diarization_df, asr_output_path)
    
    if asr_df.empty:
        print("Speech recognition failed - stopping pipeline")
        return None
    
    # 3. Correction (corrects ASR errors before summarization)
    print("\n3. TEXT CORRECTION STAGE (ASR Error Correction)")
    corrected_df = perform_correction(asr_df, correction_output_path)
    
    if corrected_df.empty:
        print("Correction failed - stopping pipeline")
        return None
    
    # 4. Summarization (uses corrected text)
    print("\n4. SUMMARIZATION STAGE")
    summary_df = perform_summarization(corrected_df, summarization_output_path)
    
    if summary_df.empty:
        print("Summarization failed - stopping pipeline")
        return None
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    print(f"Output files created in {output_dir}:")
    print(f"- Diarization: {diarization_output_path}")
    print(f"- ASR: {asr_output_path}")
    print(f"- Correction: {correction_output_path}")
    print(f"- Summarization: {summarization_output_path}")
    
    return {
        'diarization': diarization_df,
        'asr': asr_df,
        'corrected': corrected_df,
        'summary': summary_df
    }


def run_chunk_pipeline(audio_file_path, output_dir):
    """Run partial pipeline for a chunk: Diarization -> ASR -> Correction.
    
    Used for long audio processing where summarization is deferred
    until all chunks are merged.
    
    Args:
        audio_file_path: Path to audio chunk
        output_dir: Output directory
        
    Returns:
        dict: Partial results {diarization, asr, corrected}, or None on failure
    """
    audio_filename = Path(audio_file_path).stem
    ensure_directory(output_dir)
    
    # 1. Diarization
    diarization_df = perform_diarization(audio_file_path)
    if diarization_df.empty:
        return None
    
    # 2. ASR
    asr_df = perform_speech_recognition(audio_file_path, diarization_df)
    if asr_df.empty:
        return None
    
    # 3. Correction
    corrected_df = perform_correction(asr_df)
    if corrected_df.empty:
        return None
    
    return {
        'diarization': diarization_df,
        'asr': asr_df,
        'corrected': corrected_df
    }


def get_audio_duration(audio_file_path):
    """Get duration of audio file in minutes."""
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
    
    output_dir = os.path.join(base_output_dir, audio_filename)
    
    if duration_minutes > ModelConfig.MAX_CHUNK_DURATION:
        print("File is long - processing with speaker clustering...")
        return process_long_audio_with_clustering(audio_file_path, base_output_dir)
    elif force_clustering:
        print("Force clustering enabled - processing with speaker analysis...")
        return process_with_speaker_analysis(audio_file_path, base_output_dir)
    else:
        print("File is short - processing as single file...")
        result = run_complete_pipeline(audio_file_path, output_dir)
        return result is not None


def process_with_speaker_analysis(audio_file_path, base_output_dir="pipeline_output"):
    """Process audio file with additional speaker embedding analysis.
    
    Runs normal pipeline + extracts speaker embeddings for
    clustering visualization and quality metrics.
    
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
    result = run_complete_pipeline(audio_file_path, output_dir)
    
    if result is None:
        return False
    
    # Then extract speaker embeddings and generate visualization
    print("\n" + "=" * 60)
    print("SPEAKER ANALYSIS STAGE")
    print("=" * 60)
    
    try:
        speaker_clustering = SpeakerClustering()
        
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
        speaker_clustering.calculate_clustering_metrics(embeddings, labels, output_dir)
        
        # For visualization, get averaged embeddings per speaker
        speaker_embeddings = speaker_clustering.extract_speaker_embeddings(audio_file_path)
        
        if speaker_embeddings and len(speaker_embeddings) >= 3:
            all_speaker_data = {audio_file_path: speaker_embeddings}
            
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
    
    Optimized flow:
    1. Split audio at silence points
    2. For each chunk: Diarization + ASR + Correction (no summarization)
    3. Extract speaker embeddings from diarization results (avoids double diarization)
    4. Cluster speakers across chunks
    5. Merge corrected results with global speaker IDs
    6. Summarize merged result
    
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
    speaker_clustering = SpeakerClustering()
    
    temp_dir = None
    
    try:
        # 1. Split audio at natural silence points
        print("Splitting audio at natural silence points...")
        chunk_info_list, temp_dir = speaker_clustering.split_audio_at_silence(audio_file_path)
        
        all_speaker_data = {}
        all_corrected_dfs = []
        chunk_paths_processed = []
        
        # 2. Process each chunk: diarization + ASR + correction
        for i, chunk_info in enumerate(chunk_info_list):
            chunk_path = chunk_info['path']
            offset_ms = chunk_info['offset_ms']
            offset_s = offset_ms / 1000
            
            print(f"\n{'='*60}")
            print(f"PROCESSING CHUNK {i+1}/{len(chunk_info_list)}")
            print(f"{'='*60}")
            
            # Run diarization + ASR + correction (no summarization)
            chunk_result = run_chunk_pipeline(chunk_path, final_output_dir)
            
            if chunk_result is None:
                print(f"Failed to process chunk {i+1}")
                return False
            
            # 3. Extract speaker embeddings from existing diarization (avoids re-diarization)
            print(f"Extracting speaker embeddings from chunk {i+1}...")
            embeddings = speaker_clustering.extract_speaker_embeddings_from_diarization(
                chunk_path, chunk_result['diarization']
            )
            if embeddings:
                all_speaker_data[chunk_path] = embeddings
            
            # Adjust timestamps to original audio timeline
            corrected_df = chunk_result['corrected'].copy()
            corrected_df['start_time'] += offset_s
            corrected_df['end_time'] += offset_s
            corrected_df['_chunk_path'] = chunk_path  # Track origin chunk
            
            all_corrected_dfs.append(corrected_df)
            chunk_paths_processed.append(chunk_path)
            
            print(f"Chunk {i+1} processed successfully")
        
        # 4. Cluster speakers across chunks
        print("\n" + "=" * 60)
        print("CLUSTERING SPEAKERS ACROSS CHUNKS")
        print("=" * 60)
        
        speaker_mapping = speaker_clustering.cluster_speakers(
            all_speaker_data,
            output_dir=final_output_dir
        )
        
        # 5. Merge corrected results with global speaker IDs
        print("\nMerging results with global speaker IDs...")
        merged_df = pd.concat(all_corrected_dfs, ignore_index=True)
        
        # Update speaker IDs using the clustering mapping
        for (chunk_path, local_speaker), global_speaker in speaker_mapping.items():
            mask = (merged_df['_chunk_path'] == chunk_path) & (merged_df['speaker'] == local_speaker)
            merged_df.loc[mask, 'speaker'] = global_speaker
        
        # Fallback: any speakers not in mapping get a normalized ID
        # (handles edge cases where embedding extraction failed for a chunk)
        unmapped_mask = merged_df['speaker'].str.isupper()  # pyannote uses SPEAKER_XX
        if unmapped_mask.any():
            unmapped_speakers = merged_df.loc[unmapped_mask, 'speaker'].unique()
            print(f"Warning: {len(unmapped_speakers)} unmapped speaker(s) found, assigning fallback IDs")
            # Map SPEAKER_XX -> speaker_xx (lowercase) to merge with existing global IDs
            for old_name in unmapped_speakers:
                new_name = old_name.lower()
                merged_df.loc[merged_df['speaker'] == old_name, 'speaker'] = new_name
        
        # Remove helper column and sort by time
        merged_df = merged_df.drop(columns=['_chunk_path'])
        merged_df = merged_df.sort_values('start_time').reset_index(drop=True)
        
        # Save merged corrected results
        from pipeline.correction import save_correction_to_txt
        correction_output_path = os.path.join(final_output_dir, f"{audio_filename}_correction.txt")
        save_correction_to_txt(merged_df, correction_output_path)
        
        # Save merged ASR results (with global speaker IDs)
        from pipeline.asr import save_asr_to_txt
        asr_output_path = os.path.join(final_output_dir, f"{audio_filename}_asr.txt")
        # Create ASR-like DataFrame (without corrected_text column)
        asr_columns = ['speaker', 'start_time', 'end_time', 'duration', 'text', 'word_count']
        asr_df = merged_df[asr_columns].copy()
        save_asr_to_txt(asr_df, asr_output_path)
        
        # Save merged diarization results (with global speaker IDs)
        from pipeline.diarization import save_diarization_to_txt
        diarization_output_path = os.path.join(final_output_dir, f"{audio_filename}_diarization.txt")
        diar_columns = ['speaker', 'start_time', 'end_time', 'duration']
        diar_df = merged_df[diar_columns].copy()
        save_diarization_to_txt(diar_df, diarization_output_path)
        
        # 6. Summarize combined result
        print("\n" + "=" * 60)
        print("SUMMARIZING COMBINED RESULTS")
        print("=" * 60)
        
        summarization_output_path = os.path.join(final_output_dir, f"{audio_filename}_summarization.txt")
        perform_summarization(merged_df, summarization_output_path)
        
        print(f"\nFinal results with global speaker IDs saved to {final_output_dir}")
        return True
        
    except Exception as e:
        print(f"Error in long audio processing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup temporary audio chunks
        if temp_dir and os.path.exists(temp_dir):
            print("Cleaning up temporary audio chunk files...")
            shutil.rmtree(temp_dir)


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Offline meeting transcription pipeline: diarization, ASR, correction, and summarization'
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
            print(f"Successfully processed {args.audio_file}")
        else:
            print(f"Failed to process {args.audio_file}")
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
                print(f"Successfully processed {audio_file}")
            else:
                print(f"Failed to process {audio_file}")
        
        # Free GPU memory after batch processing
        clear_model_cache()


if __name__ == "__main__":
    main()
