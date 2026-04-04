import argparse
import os
import sys
import shutil
import io
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pipeline.diarization import perform_diarization
    from pipeline.asr import perform_speech_recognition
    from pipeline.correction import perform_correction
    from pipeline.summarization import perform_summarization
    from pipeline.speaker_clustering import SpeakerClustering
except ImportError:
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


def run_pipeline_in_memory(audio_file_path, force_clustering=False,
                           progress_callback=None, preloaded_models=None):
    """Run the complete pipeline in-memory without intermediate TXT files.
    
    This is the main entry point for bot mode. All data flows through
    DataFrames without writing to disk.
    
    Args:
        audio_file_path: Path to input audio WAV file
        force_clustering: Force speaker clustering even for short files
        progress_callback: Async-compatible callback function(stage: str, percent: int)
        preloaded_models: Optional dict with pre-loaded models
            
    Returns:
        dict: Pipeline results with DataFrames and metadata
    """
    result = {
        'diarization_df': None,
        'asr_df': None,
        'summarization_df': None,
        'correction_df': None,
        'clustering_png': None,
        'audio_duration_min': 0,
        'num_speakers': 0,
        'success': False,
    }
    
    models = preloaded_models or {}
    
    def notify(stage, percent):
        if progress_callback:
            progress_callback(stage, percent)
    
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_file_path)
        result['audio_duration_min'] = len(audio) / (60 * 1000)
        
        # --- Stage 1: Diarization ---
        notify("diarization", 10)
        print("\n1. DIARIZATION STAGE")
        
        diarization_df = perform_diarization(
            audio_file_path,
            output_txt_path=None,
            diarization_model=models.get('diarization'),
        )
        
        if diarization_df.empty:
            print("Diarization failed - stopping pipeline")
            notify("error", 0)
            return result
        
        result['diarization_df'] = diarization_df
        result['num_speakers'] = diarization_df['speaker'].nunique()
        notify("asr", 30)
        
        # --- Stage 2: ASR ---
        print("\n2. SPEECH RECOGNITION STAGE")
        
        asr_df = perform_speech_recognition(
            audio_file_path,
            diarization_df=diarization_df,
            asr_model=models.get('asr'),
        )
        
        if asr_df.empty:
            print("Speech recognition failed - stopping pipeline")
            notify("error", 0)
            return result
        
        result['asr_df'] = asr_df
        notify("correction", 55)
        
        # --- Stage 3: Correction ---
        print("\n3. CORRECTION STAGE")
        
        corr_model, corr_tokenizer = models.get('correction', (None, None))
        
        correction_df = perform_correction(
            asr_df=asr_df,
            correction_model=corr_model,
            correction_tokenizer=corr_tokenizer,
        )
        
        if correction_df.empty:
            print("Correction failed - stopping pipeline")
            notify("error", 0)
            return result
        
        result['correction_df'] = correction_df
        notify("summarization", 75)
        
        # --- Stage 4: Summarization ---
        print("\n4. SUMMARIZATION STAGE")
        
        summ_model, summ_tokenizer = models.get('summarization', (None, None))
        
        summarization_df = perform_summarization(
            corrected_df=correction_df,
            summarization_model=summ_model,
            summarization_tokenizer=summ_tokenizer,
        )
        
        if summarization_df.empty:
            print("Summarization failed - stopping pipeline")
            notify("error", 0)
            return result
        
        result['summarization_df'] = summarization_df
        notify("clustering", 90)
        
        # --- Optional: Clustering ---
        duration_minutes = result['audio_duration_min']
        
        if duration_minutes > ModelConfig.MAX_CHUNK_DURATION or force_clustering:
            print("\n5. SPEAKER CLUSTERING STAGE")
            try:
                speaker_clustering = SpeakerClustering(ModelConfig.DIARIZATION_TOKEN)
                
                if duration_minutes > ModelConfig.MAX_CHUNK_DURATION:
                    clustering_result = speaker_clustering.process_long_audio(
                        audio_file_path, output_dir=None
                    )
                    all_speaker_data = clustering_result.get('all_speaker_data', {})
                    speaker_mapping = clustering_result.get('speaker_mapping', {})
                else:
                    speaker_embeddings = speaker_clustering.extract_speaker_embeddings(audio_file_path)
                    if speaker_embeddings and len(speaker_embeddings) >= 3:
                        all_speaker_data = {audio_file_path: speaker_embeddings}
                        speaker_mapping = {}
                        for i, speaker in enumerate(speaker_embeddings.keys()):
                            speaker_mapping[(audio_file_path, speaker)] = f"speaker_{i:02d}"
                    else:
                        all_speaker_data = {}
                        speaker_mapping = {}
                
                if all_speaker_data and len(all_speaker_data) > 0:
                    png_buffer = speaker_clustering.visualize_clusters_to_buffer(
                        all_speaker_data, speaker_mapping
                    )
                    result['clustering_png'] = png_buffer
                    
            except Exception as e:
                print(f"Clustering error (non-fatal): {e}")
                import traceback
                traceback.print_exc()
        
        notify("pdf", 95)
        result['success'] = True
        
        print("\nPIPELINE COMPLETED SUCCESSFULLY (in-memory)")
        return result
        
    except Exception as e:
        if "cancelled" in str(e).lower():
            raise
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        notify("error", 0)
        return result


# ============================================================
# CLI functions
# ============================================================

def run_complete_pipeline(audio_file_path, output_dir):
    """Run complete pipeline with DataFrame passing between stages.
    
    Args:
        audio_file_path: Path to input audio file
        output_dir: Output directory for this specific audio file
        
    Returns:
        tuple: (success: bool, diarization_df: DataFrame or None)
    """
    audio_filename = Path(audio_file_path).stem
    ensure_directory(output_dir)
    
    print("=" * 60)
    print("STARTING MEETING TRANSCRIPTION PIPELINE")
    print("=" * 60)
    
    diarization_output_path = os.path.join(output_dir, f"{audio_filename}_diarization.txt")
    asr_output_path = os.path.join(output_dir, f"{audio_filename}_asr.txt")
    correction_output_path = os.path.join(output_dir, f"{audio_filename}_correction.txt")
    summarization_output_path = os.path.join(output_dir, f"{audio_filename}_summarization.txt")
    
    # 1. Diarization
    print("\n1. DIARIZATION STAGE")
    diarization_df = perform_diarization(audio_file_path, diarization_output_path)
    
    if diarization_df.empty:
        print("Diarization failed - stopping pipeline")
        return False, None
    
    # 2. Speech Recognition (pass DataFrame directly)
    print("\n2. SPEECH RECOGNITION STAGE")
    asr_df = perform_speech_recognition(
        audio_file_path,
        output_txt_path=asr_output_path,
        diarization_df=diarization_df
    )
    
    if asr_df.empty:
        print("Speech recognition failed - stopping pipeline")
        return False, diarization_df
    
    # 3. Correction (pass DataFrame directly)
    print("\n3. CORRECTION STAGE")
    correction_df = perform_correction(
        output_txt_path=correction_output_path,
        asr_df=asr_df
    )
    
    if correction_df.empty:
        print("Correction failed - stopping pipeline")
        return False, diarization_df
    
    # 4. Summarization (pass DataFrame directly)
    print("\n4. SUMMARIZATION STAGE")
    summarization_df = perform_summarization(
        output_txt_path=summarization_output_path,
        corrected_df=correction_df
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    print(f"Output files created in {output_dir}:")
    print(f"- Diarization: {diarization_output_path}")
    print(f"- ASR: {asr_output_path}")
    print(f"- Correction: {correction_output_path}")
    print(f"- Summarization: {summarization_output_path}")
    
    return True, diarization_df


def run_chunk_pipeline(audio_file_path, output_dir):
    """Run pipeline on a single chunk (Diarization + ASR + Correction only, no summarization).
    
    Used by process_long_audio_with_clustering so that summarization happens
    once on the combined corrected text.
    
    Args:
        audio_file_path: Path to chunk audio file
        output_dir: Output directory for this chunk
        
    Returns:
        tuple: (success: bool, correction_df: DataFrame or None, diarization_df: DataFrame or None)
    """
    audio_filename = Path(audio_file_path).stem
    ensure_directory(output_dir)
    
    diarization_output_path = os.path.join(output_dir, f"{audio_filename}_diarization.txt")
    asr_output_path = os.path.join(output_dir, f"{audio_filename}_asr.txt")
    correction_output_path = os.path.join(output_dir, f"{audio_filename}_correction.txt")
    
    # 1. Diarization
    print("\n1. DIARIZATION STAGE")
    diarization_df = perform_diarization(audio_file_path, diarization_output_path)
    
    if diarization_df.empty:
        print("Diarization failed")
        return False, None, None
    
    # 2. ASR
    print("\n2. SPEECH RECOGNITION STAGE")
    asr_df = perform_speech_recognition(
        audio_file_path,
        output_txt_path=asr_output_path,
        diarization_df=diarization_df
    )
    
    if asr_df.empty:
        print("Speech recognition failed")
        return False, None, diarization_df
    
    # 3. Correction
    print("\n3. CORRECTION STAGE")
    correction_df = perform_correction(
        output_txt_path=correction_output_path,
        asr_df=asr_df
    )
    
    if correction_df.empty:
        print("Correction failed")
        return False, None, diarization_df
    
    return True, correction_df, diarization_df


def get_audio_duration(audio_file_path):
    """Get duration of audio file in minutes."""
    from pydub import AudioSegment
    audio = AudioSegment.from_file(audio_file_path)
    return len(audio) / (60 * 1000)


def _extract_embeddings_from_diarization(diarization_df, audio_file_path, speaker_clustering):
    """Extract speaker embeddings reusing existing diarization results.
    
    Avoids redundant re-diarization by using the DataFrame already produced
    by the pipeline.
    
    Args:
        diarization_df: DataFrame from perform_diarization
        audio_file_path: Path to the audio file
        speaker_clustering: SpeakerClustering instance (for embedding_inference)
        
    Returns:
        tuple: (segment_embeddings, segment_labels, label_to_speaker,
                speaker_avg_embeddings)
    """
    from pyannote.core import Segment
    
    all_embeddings = []
    all_labels = []
    speaker_to_label = {}
    speaker_raw = {}
    label_counter = 0
    
    for _, row in diarization_df.iterrows():
        speaker = row['speaker']
        seg = Segment(row['start_time'], row['end_time'])
        
        if seg.end - seg.start < 1.0:
            continue
        
        try:
            embedding = speaker_clustering.embedding_inference.crop(audio_file_path, seg)
            emb_np = np.array(embedding).flatten()
            
            if speaker not in speaker_to_label:
                speaker_to_label[speaker] = label_counter
                label_counter += 1
                speaker_raw[speaker] = []
            
            all_embeddings.append(emb_np)
            all_labels.append(speaker_to_label[speaker])
            speaker_raw[speaker].append(emb_np)
            
        except Exception as e:
            print(f"Error extracting embedding for {speaker}: {e}")
            continue
    
    if not all_embeddings:
        return None, None, None, {}
    
    embeddings_array = np.vstack(all_embeddings)
    labels_array = np.array(all_labels)
    label_to_speaker = {v: k for k, v in speaker_to_label.items()}
    
    avg_embeddings = {}
    for speaker, embs in speaker_raw.items():
        avg_emb = np.mean(embs, axis=0)
        avg_embeddings[speaker] = {
            'embedding': avg_emb,
            'total_duration': float(
                diarization_df[diarization_df['speaker'] == speaker]['duration'].sum()
            ),
            'num_segments': len(embs),
            'chunk_path': audio_file_path
        }
    
    print(f"Extracted {len(all_embeddings)} segment embeddings for {len(speaker_to_label)} speakers")
    return embeddings_array, labels_array, label_to_speaker, avg_embeddings


def process_audio_file(audio_file_path, base_output_dir="pipeline_output", force_clustering=False):
    """Process single audio file with automatic chunking if needed."""
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
        success, _ = run_complete_pipeline(audio_file_path, output_dir)
        return success


def process_with_speaker_analysis(audio_file_path, base_output_dir="pipeline_output"):
    """Process audio file with speaker embedding analysis and visualization.
    
    Reuses diarization results from run_complete_pipeline to avoid redundant
    diarization passes during embedding extraction.
    """
    audio_filename = Path(audio_file_path).stem
    output_dir = os.path.join(base_output_dir, audio_filename)
    ensure_directory(output_dir)
    
    success, diarization_df = run_complete_pipeline(audio_file_path, output_dir)
    
    if not success:
        return False
    
    print("\n" + "=" * 60)
    print("SPEAKER ANALYSIS STAGE")
    print("=" * 60)
    
    try:
        speaker_clustering = SpeakerClustering(ModelConfig.DIARIZATION_TOKEN)
        
        print("Extracting embeddings from existing diarization results...")
        embeddings, labels, label_to_speaker, avg_embeddings = \
            _extract_embeddings_from_diarization(diarization_df, audio_file_path, speaker_clustering)
        
        if embeddings is None or len(embeddings) == 0:
            print("No embeddings extracted - skipping visualization")
            return True
        
        n_speakers = len(set(labels))
        n_segments = len(labels)
        
        print(f"Found {n_segments} segments from {n_speakers} speakers")
        
        if n_speakers < 2:
            print("Only one speaker detected - skipping clustering analysis")
            return True
        
        speaker_clustering.calculate_clustering_metrics(embeddings, labels, output_dir)
        
        if avg_embeddings and len(avg_embeddings) >= 3:
            all_speaker_data = {audio_file_path: avg_embeddings}
            speaker_mapping = {}
            for i, speaker in enumerate(avg_embeddings.keys()):
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
        return True


def process_long_audio_with_clustering(audio_file_path, base_output_dir="pipeline_output"):
    """Process long audio file with speaker clustering.
    
    Uses run_chunk_pipeline (no per-chunk summarization), merges corrected
    DataFrames, applies global speaker mapping, and runs one final summarization.
    """
    audio_filename = Path(audio_file_path).stem
    final_output_dir = os.path.join(base_output_dir, audio_filename)
    ensure_directory(final_output_dir)
    
    speaker_clustering = SpeakerClustering(ModelConfig.DIARIZATION_TOKEN)
    
    try:
        print("Processing long audio with speaker clustering...")
        clustering_result = speaker_clustering.process_long_audio(audio_file_path, final_output_dir)
        
        speaker_mapping = clustering_result['speaker_mapping']
        chunk_paths = clustering_result['chunk_paths']
        temp_dir = clustering_result['temp_dir']
        
        all_correction_dfs = []
        all_diarization_dfs = []
        
        for i, chunk_path in enumerate(chunk_paths):
            print(f"\n{'='*60}")
            print(f"PROCESSING CHUNK {i+1}/{len(chunk_paths)} WITH GLOBAL SPEAKER IDs")
            print(f"{'='*60}")
            
            chunk_output_dir = os.path.join(base_output_dir, f"{audio_filename}_chunk_{i+1:02d}")
            
            success, correction_df, diarization_df = run_chunk_pipeline(chunk_path, chunk_output_dir)
            
            if success and correction_df is not None:
                print(f"Chunk {i+1} processed successfully")
                
                # Apply global speaker mapping to correction DataFrame
                for (map_chunk_path, local_speaker), global_speaker in speaker_mapping.items():
                    if map_chunk_path == chunk_path:
                        correction_df.loc[
                            correction_df['speaker'] == local_speaker, 'speaker'
                        ] = global_speaker
                
                if diarization_df is not None:
                    for (map_chunk_path, local_speaker), global_speaker in speaker_mapping.items():
                        if map_chunk_path == chunk_path:
                            diarization_df.loc[
                                diarization_df['speaker'] == local_speaker, 'speaker'
                            ] = global_speaker
                    all_diarization_dfs.append(diarization_df)
                
                all_correction_dfs.append(correction_df)
            else:
                print(f"Failed to process chunk {i+1}")
                return False
        
        # Merge all chunk DataFrames
        merged_correction_df = pd.concat(all_correction_dfs, ignore_index=True)
        
        # Fallback: convert any remaining uppercase SPEAKER_XX IDs to lowercase
        upper_mask = merged_correction_df['speaker'].str.match(r'^SPEAKER_\d+$', na=False)
        if upper_mask.any():
            print(f"Converting {upper_mask.sum()} remaining uppercase speaker IDs...")
            merged_correction_df.loc[upper_mask, 'speaker'] = \
                merged_correction_df.loc[upper_mask, 'speaker'].str.lower()
        
        # Save combined diarization
        if all_diarization_dfs:
            merged_diarization_df = pd.concat(all_diarization_dfs, ignore_index=True)
            upper_mask_d = merged_diarization_df['speaker'].str.match(r'^SPEAKER_\d+$', na=False)
            if upper_mask_d.any():
                merged_diarization_df.loc[upper_mask_d, 'speaker'] = \
                    merged_diarization_df.loc[upper_mask_d, 'speaker'].str.lower()
            
            from pipeline.diarization import save_diarization_to_txt
            diarization_path = os.path.join(final_output_dir, f"{audio_filename}_diarization.txt")
            save_diarization_to_txt(merged_diarization_df, diarization_path)
        
        # Save combined correction
        from pipeline.correction import save_correction_to_txt
        correction_path = os.path.join(final_output_dir, f"{audio_filename}_correction.txt")
        save_correction_to_txt(merged_correction_df, correction_path)
        
        # Final summarization on the merged corrected data
        print("\nCreating overall summarization...")
        summarization_output_path = os.path.join(final_output_dir, f"{audio_filename}_summarization.txt")
        perform_summarization(
            output_txt_path=summarization_output_path,
            corrected_df=merged_correction_df
        )
        
        print(f"Final results with global speaker IDs saved to {final_output_dir}")
        return True
        
    except Exception as e:
        print(f"Error in long audio processing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            print("Cleaning up temporary audio chunk files...")
            shutil.rmtree(temp_dir)
            
        chunk_dirs = [os.path.join(base_output_dir, d) for d in os.listdir(base_output_dir) 
                     if d.startswith(f"{audio_filename}_chunk_")]
        for chunk_dir in chunk_dirs:
            if os.path.exists(chunk_dir):
                shutil.rmtree(chunk_dir)


def main():
    """Main entry point for the pipeline (CLI mode)."""
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
    
    clear_model_cache()


if __name__ == "__main__":
    main()
