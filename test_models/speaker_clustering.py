import os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment
import tempfile
import shutil
from pydub import AudioSegment
import librosa
import sys
import pydub
import pydub.silence
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ModelConfig


class SpeakerClustering:
    def __init__(self, hf_token):
        """Initialize speaker clustering with pyannote models."""
        
        self.diarization_pipeline = Pipeline.from_pretrained(
            ModelConfig.CLUSTERING_MODEL_NAME,
            token=hf_token
        )
        self.embedding_model = Model.from_pretrained(
            ModelConfig.EMBEDDING_MODEL_NAME, 
            token=hf_token
        )
        self.embedding_inference = Inference(
            self.embedding_model, 
            window="whole"
        )
        
    def find_silence_periods(self, audio_file_path, min_silence_len=None, silence_thresh=None):
        """Find silence periods in audio for natural chunking."""
        
        if min_silence_len is None:
            min_silence_len = ModelConfig.MIN_SILENCE_LEN_MS
        if silence_thresh is None:
            silence_thresh = ModelConfig.SILENCE_THRESH_DB
        
        audio = AudioSegment.from_file(audio_file_path)
        
        # Convert to mono for silence detection
        if audio.channels > 1:
            audio = audio.set_channels(1)
            
        silence_periods = []
        
        # Use pydub's proper silence detection
        silent_ranges = pydub.silence.detect_silence(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        
        # Convert to list of (start, end) tuples
        for start, end in silent_ranges:
            silence_periods.append((start, end))
        
        print(f"Found {len(silence_periods)} silence periods for chunking")
        return silence_periods

    def split_audio_at_silence(self, audio_file_path, target_chunk_duration_min=None, 
                         max_chunk_duration_min=None, min_silence_len=None):
        """Split audio at natural silence points."""
        
        if target_chunk_duration_min is None:
            target_chunk_duration_min = ModelConfig.TARGET_CHUNK_DURATION_MIN
        if max_chunk_duration_min is None:
            max_chunk_duration_min = ModelConfig.TARGET_CHUNK_DURATION_MAX
        if min_silence_len is None:
            min_silence_len = ModelConfig.MIN_SILENCE_LEN_MS
        
        audio = AudioSegment.from_file(audio_file_path)
        total_duration_ms = len(audio)
        target_chunk_duration_ms = target_chunk_duration_min * 60 * 1000
        max_chunk_duration_ms = max_chunk_duration_min * 60 * 1000
        
        silence_periods = self.find_silence_periods(audio_file_path, min_silence_len)
        
        chunks = []
        current_start = 0
        audio_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
        
        # Create temporary directory for chunks
        temp_dir = tempfile.mkdtemp(prefix=f"audio_chunks_{audio_filename}_")
        
        chunk_num = 1
        while current_start < total_duration_ms:
            target_end = current_start + target_chunk_duration_ms
            
            # Find the best silence point near target end
            best_split_point = target_end
            min_distance = float('inf')
            
            for silence_start, silence_end in silence_periods:
                if (silence_start > current_start and 
                    silence_start <= current_start + max_chunk_duration_ms):
                    distance = abs(silence_start - target_end)
                    if distance < min_distance:
                        min_distance = distance
                        best_split_point = silence_start
            
            # Ensure we don't create too small chunks
            if best_split_point - current_start < 60 * 1000:  # At least 1 minute
                best_split_point = min(current_start + max_chunk_duration_ms, total_duration_ms)
            
            # Extract chunk
            chunk_end = min(best_split_point, total_duration_ms)
            chunk = audio[current_start:chunk_end]
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_num:03d}_{audio_filename}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
            
            print(f"  Chunk {chunk_num}: {current_start/1000/60:.1f}min - {chunk_end/1000/60:.1f}min "
                f"({(chunk_end-current_start)/1000/60:.1f}min)")
            
            current_start = chunk_end
            chunk_num += 1
            
            # Stop if we've reached the end
            if current_start >= total_duration_ms:
                break
        
        print(f"Split audio into {len(chunks)} chunks at natural silence points")
        return chunks, temp_dir

    def extract_speaker_embeddings(self, audio_file_path):
        """Extract speaker embeddings from audio file with correct attribute name."""
        try:
            print(f"Starting diarization for {os.path.basename(audio_file_path)}")
            diarization = self.diarization_pipeline(audio_file_path)
            
            # Debug: check what we got from diarization
            print(f"Diarization type: {type(diarization)}")
            print(f"Diarization attributes: {[attr for attr in dir(diarization) if not attr.startswith('_')]}")
            
            speaker_data = {}
            segment_count = 0
            
            # Use speaker_diarization attribute instead of annotation
            if hasattr(diarization, 'speaker_diarization'):
                print(f"Using speaker_diarization attribute")
                for segment, track, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                    segment_count += 1
                    if segment.end - segment.start < 1.0:
                        continue
                        
                    try:
                        print(f"Processing segment {segment_count}: {speaker} at {segment.start:.1f}-{segment.end:.1f}s")
                        
                        # Extract embedding for this speaker segment
                        embedding = self.embedding_inference.crop(audio_file_path, segment)
                        print(f"Embedding shape: {embedding.shape}")
                        
                        if speaker not in speaker_data:
                            speaker_data[speaker] = {
                                'embeddings': [],
                                'segments': [],
                                'durations': []
                            }
                        
                        speaker_data[speaker]['embeddings'].append(embedding)
                        speaker_data[speaker]['segments'].append(segment)
                        speaker_data[speaker]['durations'].append(segment.end - segment.start)
                        
                    except Exception as e:
                        print(f"Error extracting embedding: {e}")
                        continue
            else:
                print(f"No speaker_diarization attribute found in diarization result")
                return {}
            
            print(f"Processed {segment_count} segments total")
            
            # Average embeddings for each speaker
            avg_embeddings = {}
            for speaker, data in speaker_data.items():
                if data['embeddings']:
                    avg_embedding = np.mean(data['embeddings'], axis=0)
                    total_duration = sum(data['durations'])
                    avg_embeddings[speaker] = {
                        'embedding': avg_embedding,
                        'total_duration': total_duration,
                        'num_segments': len(data['segments']),
                        'chunk_path': audio_file_path
                    }
                    print(f"Speaker {speaker}: {len(data['embeddings'])} segments, avg embedding shape: {avg_embedding.shape}")
            
            print(f"Successfully processed {len(avg_embeddings)} speakers")
            return avg_embeddings
            
        except Exception as e:
            print(f"Error processing {audio_file_path}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def cluster_speakers(self, all_speaker_data, distance_threshold=None, output_dir=None):
        """Cluster speakers across all chunks."""

        if distance_threshold is None:
            distance_threshold = ModelConfig.CLUSTERING_DISTANCE_THRESHOLD

        if not all_speaker_data:
            return {}
        
        # Prepare data for clustering
        embeddings = []
        speaker_info = []
        
        for chunk_path, speakers in all_speaker_data.items():
            for local_speaker, data in speakers.items():
                embeddings.append(data['embedding'])
                speaker_info.append({
                    'chunk_path': chunk_path,
                    'local_speaker': local_speaker,
                    'duration': data['total_duration'],
                    'num_segments': data['num_segments']
                })
        
        if len(embeddings) < 2:
            # Not enough speakers to cluster
            global_id = 0
            mapping = {}
            for info in speaker_info:
                mapping[(info['chunk_path'], info['local_speaker'])] = f"speaker_{global_id:02d}"
                global_id += 1
            return mapping
        
        # Convert to numpy array
        X = np.vstack(embeddings)
        
        # Apply PCA for dimensionality reduction if needed
        if X.shape[1] > 50:
            pca = PCA(n_components=min(50, X.shape[0]))
            X = pca.fit_transform(X)
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='average',
            metric='cosine'
        )
        
        cluster_labels = clustering.fit_predict(X)
        
        # Create mapping from local to global speaker IDs
        speaker_mapping = {}
        for i, info in enumerate(speaker_info):
            global_speaker_id = f"speaker_{cluster_labels[i]:02d}"
            speaker_mapping[(info['chunk_path'], info['local_speaker'])] = global_speaker_id
        
        # Add visualization if output directory provided and enough data
        if output_dir and len(embeddings) >= 3:
            self.visualize_clusters(all_speaker_data, speaker_mapping, output_dir)

        print(f"Clustered {len(speaker_info)} local speakers into {len(set(cluster_labels))} global speakers")
        return speaker_mapping

    def visualize_clusters(self, all_speaker_data, speaker_mapping, output_dir):
        """Visualization of speaker clusters with correct labeling."""
        try:
            # Prepare data
            embeddings = []
            speaker_labels = []
            global_ids = []
            local_speakers = []
            chunk_names = []
            
            for chunk_path, speakers in all_speaker_data.items():
                chunk_name = os.path.basename(chunk_path)
                for local_speaker, data in speakers.items():
                    global_id = speaker_mapping.get((chunk_path, local_speaker), "unknown")
                    embeddings.append(data['embedding'])
                    speaker_labels.append(local_speaker)
                    global_ids.append(global_id)
                    local_speakers.append(f"{local_speaker}")
                    chunk_names.append(chunk_name)
            
            if len(embeddings) < 3:
                print("Not enough data for visualization")
                return
            
            X = np.vstack(embeddings)
            
            # Get unique global speakers that actually exist in the data
            unique_global = sorted(set(global_ids))
            print(f"Visualizing {len(unique_global)} global speakers: {unique_global}")
            
            # Create figure with multiple plots
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(f'Speaker Clustering Visualization - {len(unique_global)} Global Speakers', 
                        fontsize=16, fontweight='bold')
            
            # 1. t-SNE visualization
            print("Performing t-SNE...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
            X_tsne = tsne.fit_transform(X)
            
            # 2. PCA visualization
            print("Performing PCA...")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # Color scheme for global speakers - use consistent colors
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_global)))
            color_map = {spk: colors[i] for i, spk in enumerate(unique_global)}
            
            # Plot 1: t-SNE with global IDs
            ax1 = axes[0, 0]
            for i, global_id in enumerate(unique_global):
                mask = np.array(global_ids) == global_id
                if np.sum(mask) > 0:  # Only plot if there are points
                    ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                            c=[color_map[global_id]], label=global_id, alpha=0.7, s=60)
            ax1.set_title(f't-SNE: {len(unique_global)} Global Speaker Clusters')
            ax1.set_xlabel('t-SNE 1')
            ax1.set_ylabel('t-SNE 2')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: PCA with global IDs
            ax2 = axes[0, 1]
            for i, global_id in enumerate(unique_global):
                mask = np.array(global_ids) == global_id
                if np.sum(mask) > 0:  # Only plot if there are points
                    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                            c=[color_map[global_id]], label=global_id, alpha=0.7, s=60)
            ax2.set_title(f'PCA: {len(unique_global)} Global Speaker Clusters')
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Distance Matrix
            ax3 = axes[1, 0]
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(X)
            
            # Grouping by global speakers
            global_indices = {}
            for i, global_id in enumerate(global_ids):
                if global_id not in global_indices:
                    global_indices[global_id] = []
                global_indices[global_id].append(i)
            
            # Sorting indices for better visualization - use only existing speakers
            sorted_indices = []
            sorted_labels = []
            for global_id in unique_global:
                if global_id in global_indices:  # Check if speaker exists in current data
                    indices = global_indices[global_id]
                    sorted_indices.extend(indices)
                    sorted_labels.extend([global_id] * len(indices))
            
            if sorted_indices:  # Only proceed if we have data
                sorted_similarity = similarity_matrix[np.ix_(sorted_indices, sorted_indices)]
                
                im = ax3.imshow(sorted_similarity, cmap='RdYlBu', aspect='auto')
                ax3.set_title('Cosine Similarity Matrix')
                ax3.set_xlabel('Speaker Segments')
                ax3.set_ylabel('Speaker Segments')
                
                # Add separator lines between clusters
                current_pos = 0
                for global_id in unique_global:
                    if global_id in global_indices:
                        cluster_size = len(global_indices[global_id])
                        if current_pos > 0:
                            ax3.axhline(current_pos - 0.5, color='white', linewidth=2)
                            ax3.axvline(current_pos - 0.5, color='white', linewidth=2)
                        current_pos += cluster_size
                
                plt.colorbar(im, ax=ax3)
            else:
                ax3.text(0.5, 0.5, 'No data for similarity matrix', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Cosine Similarity Matrix (No Data)')
            
            # Plot 4: Cluster Statistics
            ax4 = axes[1, 1]
            cluster_sizes = {}
            for global_id in global_ids:
                cluster_sizes[global_id] = cluster_sizes.get(global_id, 0) + 1
            
            # Only include speakers that actually have segments
            valid_clusters = [spk for spk in unique_global if spk in cluster_sizes]
            valid_sizes = [cluster_sizes[spk] for spk in valid_clusters]
            
            if valid_clusters:
                bars = ax4.bar(valid_clusters, valid_sizes, 
                            color=[color_map[spk] for spk in valid_clusters])
                ax4.set_title(f'Cluster Sizes ({len(valid_clusters)} Speakers with Data)')
                ax4.set_xlabel('Global Speaker ID')
                ax4.set_ylabel('Number of Segments')
                ax4.tick_params(axis='x', rotation=45)
                
                # Add values on bars
                for bar, size in zip(bars, valid_sizes):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(size), ha='center', va='bottom')
            else:
                ax4.text(0.5, 0.5, 'No cluster data available', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Cluster Sizes (No Data)')
            
            plt.tight_layout()
            
            # Save plots
            output_path = os.path.join(output_dir, "speaker_clusters_visualization.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Cluster visualization saved: {output_path}")
            print(f"Visualized {len(valid_clusters)} speakers with data from {len(valid_clusters)} global IDs")
            
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback
            traceback.print_exc()

    def print_cluster_statistics(self, all_speaker_data, speaker_mapping):
        """Print clustering statistics."""
        print("\nCLUSTERING STATISTICS:")
        print("=" * 50)
        
        # Count statistics
        cluster_stats = {}
        local_to_global = {}
        
        for chunk_path, speakers in all_speaker_data.items():
            for local_speaker, data in speakers.items():
                global_id = speaker_mapping.get((chunk_path, local_speaker))
                if global_id not in cluster_stats:
                    cluster_stats[global_id] = {
                        'local_speakers': set(),
                        'total_segments': 0,
                        'total_duration': 0,
                        'chunks': set()
                    }
                
                cluster_stats[global_id]['local_speakers'].add(local_speaker)
                cluster_stats[global_id]['total_segments'] += data['num_segments']
                cluster_stats[global_id]['total_duration'] += data['total_duration']
                cluster_stats[global_id]['chunks'].add(os.path.basename(chunk_path))
                
                local_to_global[local_speaker] = global_id
        
        # Output statistics
        for global_id, stats in sorted(cluster_stats.items()):
            local_count = len(stats['local_speakers'])
            chunk_count = len(stats['chunks'])
            duration_min = stats['total_duration'] / 60
            
            print(f"Speaker {global_id}:")
            print(f"   Local IDs: {local_count}")
            print(f"   Segments: {stats['total_segments']}")
            print(f"   Duration: {duration_min:.1f} min")
            print(f"   Chunks: {chunk_count}")
            print(f"   Local IDs: {', '.join(sorted(stats['local_speakers']))}")
        
        print("=" * 50)
        print(f"TOTAL: {len(cluster_stats)} global speakers from {len(local_to_global)} local")

    def process_long_audio(self, audio_file_path, output_dir):
        """Process long audio with speaker clustering."""
        print("Splitting audio at natural silence points...")
        chunk_paths, temp_dir = self.split_audio_at_silence(audio_file_path)
        
        all_speaker_data = {}
        
        try:
            # Process each chunk and extract speaker embeddings
            for i, chunk_path in enumerate(chunk_paths):
                print(f"Processing chunk {i+1}/{len(chunk_paths)}...")
                
                speaker_embeddings = self.extract_speaker_embeddings(chunk_path)
                if speaker_embeddings:
                    all_speaker_data[chunk_path] = speaker_embeddings
                    print(f"Found {len(speaker_embeddings)} speakers in this chunk")
                else:
                    print(f"No speakers found in this chunk")
            
            # Cluster speakers across all chunks with visualization
            print("Clustering speakers across all chunks...")
            speaker_mapping = self.cluster_speakers(
                all_speaker_data, 
                distance_threshold=0.4,
                output_dir=output_dir
            )

            return {
                'speaker_mapping': speaker_mapping,
                'chunk_paths': chunk_paths,
                'temp_dir': temp_dir,
                'all_speaker_data': all_speaker_data
            }
            
        except Exception as e:
            print(f"Error in process_long_audio: {e}")
            # Cleanup on error
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise

    def update_diarization_with_global_speakers(self, diarization_txt_path, speaker_mapping, chunk_path):
        """Update diarization file with global speaker IDs."""
        with open(diarization_txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace local speaker IDs with global ones
        for (map_chunk_path, local_speaker), global_speaker in speaker_mapping.items():
            if map_chunk_path == chunk_path:
                content = content.replace(f"Speaker: {local_speaker}", f"Speaker: {global_speaker}")
        
        return content