import os
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from pyannote.audio import Pipeline, Model, Inference
from pyannote.core import Segment
import tempfile
import shutil
from pydub import AudioSegment
import pydub
import pydub.silence
import matplotlib.pyplot as plt
import matplotlib
import sys

# Use non-interactive backend for server/headless environments
matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ModelConfig
from utils.models import load_diarization_model


class SpeakerClustering:
    def __init__(self):
        """Initialize speaker clustering with shared diarization model and embedding model."""
        
        # Use cached diarization model (shared with perform_diarization)
        self.diarization_pipeline = load_diarization_model()
        self.embedding_model = Model.from_pretrained(
            ModelConfig.EMBEDDING_MODEL_NAME, 
            token=ModelConfig.DIARIZATION_TOKEN
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
            
        # Use pydub's silence detection
        silent_ranges = pydub.silence.detect_silence(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        
        silence_periods = [(start, end) for start, end in silent_ranges]
        
        print(f"Found {len(silence_periods)} silence periods for chunking")
        return silence_periods

    def split_audio_at_silence(self, audio_file_path, target_chunk_duration_min=None, 
                         max_chunk_duration_min=None, min_silence_len=None):
        """Split audio at natural silence points.
        
        Returns:
            tuple: (list of chunk info dicts with 'path' and 'offset_ms', temp_dir path)
        """
        
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
            
            chunks.append({
                'path': chunk_path,
                'offset_ms': current_start
            })
            
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
        """Extract speaker embeddings from audio file using internal diarization.
        
        Runs diarization and then extracts embeddings for each detected speaker.
        For use when diarization results are not yet available.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            dict: Speaker embeddings {speaker: {embedding, total_duration, num_segments, chunk_path}}
        """
        try:
            print(f"Starting diarization for {os.path.basename(audio_file_path)}")
            diarization = self.diarization_pipeline(audio_file_path)
            
            speaker_data = {}
            segment_count = 0
            
            if hasattr(diarization, 'speaker_diarization'):
                for segment, track, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                    segment_count += 1
                    if segment.end - segment.start < 1.0:
                        continue
                        
                    try:
                        embedding = self.embedding_inference.crop(audio_file_path, segment)
                        
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
                print("No speaker_diarization attribute found")
                return {}
            
            # Average embeddings for each speaker
            avg_embeddings = {}
            for speaker, data in speaker_data.items():
                if data['embeddings']:
                    avg_embeddings[speaker] = {
                        'embedding': np.mean(data['embeddings'], axis=0),
                        'total_duration': sum(data['durations']),
                        'num_segments': len(data['segments']),
                        'chunk_path': audio_file_path
                    }
            
            print(f"Processed {segment_count} segments, {len(avg_embeddings)} speakers")
            return avg_embeddings
            
        except Exception as e:
            print(f"Error processing {audio_file_path}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def extract_speaker_embeddings_from_diarization(self, audio_file_path, diarization_dataframe):
        """Extract speaker embeddings using pre-computed diarization results.
        
        Avoids running diarization twice by reusing existing results.
        
        Args:
            audio_file_path: Path to audio file
            diarization_dataframe: DataFrame with diarization results
            
        Returns:
            dict: Speaker embeddings {speaker: {embedding, total_duration, num_segments, chunk_path}}
        """
        speaker_data = {}
        
        for _, row in diarization_dataframe.iterrows():
            segment = Segment(row['start_time'], row['end_time'])
            speaker = row['speaker']
            
            if segment.end - segment.start < 1.0:
                continue
            
            try:
                embedding = self.embedding_inference.crop(audio_file_path, segment)
                
                if speaker not in speaker_data:
                    speaker_data[speaker] = {
                        'embeddings': [],
                        'durations': []
                    }
                
                speaker_data[speaker]['embeddings'].append(embedding)
                speaker_data[speaker]['durations'].append(segment.end - segment.start)
                
            except Exception as e:
                continue
        
        # Average embeddings for each speaker
        avg_embeddings = {}
        for speaker, data in speaker_data.items():
            if data['embeddings']:
                avg_embeddings[speaker] = {
                    'embedding': np.mean(data['embeddings'], axis=0),
                    'total_duration': sum(data['durations']),
                    'num_segments': len(data['embeddings']),
                    'chunk_path': audio_file_path
                }
        
        print(f"Extracted embeddings for {len(avg_embeddings)} speakers from diarization")
        return avg_embeddings

    def extract_segment_embeddings(self, audio_file_path):
        """Extract per-segment speaker embeddings for metrics calculation.
        
        Returns individual embeddings for each speech segment (not averaged).
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            tuple: (embeddings_array, speaker_labels, label_to_speaker_dict)
        """
        try:
            print(f"Extracting segment embeddings for {os.path.basename(audio_file_path)}")
            diarization = self.diarization_pipeline(audio_file_path)
            
            all_embeddings = []
            all_labels = []
            speaker_to_label = {}
            label_counter = 0
            
            if hasattr(diarization, 'speaker_diarization'):
                for segment, track, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                    if segment.end - segment.start < 1.0:
                        continue
                    
                    try:
                        embedding = self.embedding_inference.crop(audio_file_path, segment)
                        
                        if speaker not in speaker_to_label:
                            speaker_to_label[speaker] = label_counter
                            label_counter += 1
                        
                        all_embeddings.append(embedding)
                        all_labels.append(speaker_to_label[speaker])
                        
                    except Exception as e:
                        continue
            
            if not all_embeddings:
                return None, None, None
            
            embeddings_array = np.vstack(all_embeddings)
            labels_array = np.array(all_labels)
            label_to_speaker = {v: k for k, v in speaker_to_label.items()}
            
            print(f"Extracted {len(all_embeddings)} segment embeddings for {len(speaker_to_label)} speakers")
            return embeddings_array, labels_array, label_to_speaker
            
        except Exception as e:
            print(f"Error extracting segment embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def cluster_speakers(self, all_speaker_data, distance_threshold=None, output_dir=None):
        """Cluster speakers across all chunks using agglomerative clustering.
        
        Args:
            all_speaker_data: Dict {chunk_path: {speaker: {embedding, ...}}}
            distance_threshold: Cosine distance threshold for clustering
            output_dir: Optional directory for saving metrics report
            
        Returns:
            dict: Mapping {(chunk_path, local_speaker): global_speaker_id}
        """
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
            mapping = {}
            for i, info in enumerate(speaker_info):
                mapping[(info['chunk_path'], info['local_speaker'])] = f"speaker_{i:02d}"
            return mapping
        
        # Convert to numpy array
        X = np.vstack(embeddings)
        
        # Apply PCA for dimensionality reduction if needed
        if X.shape[1] > 50:
            pca = PCA(n_components=min(50, X.shape[0]))
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
        
        # Perform hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='average',
            metric='cosine'
        )
        
        cluster_labels = clustering.fit_predict(X_reduced)
        
        # Create mapping from local to global speaker IDs
        speaker_mapping = {}
        for i, info in enumerate(speaker_info):
            global_speaker_id = f"speaker_{cluster_labels[i]:02d}"
            speaker_mapping[(info['chunk_path'], info['local_speaker'])] = global_speaker_id
        
        # Calculate and save clustering quality metrics
        self.calculate_clustering_metrics(X_reduced, cluster_labels, output_dir)
        
        # Generate visualization if enough data
        if output_dir and len(embeddings) >= 3:
            self.visualize_clusters(all_speaker_data, speaker_mapping, output_dir)

        print(f"Clustered {len(speaker_info)} local speakers into {len(set(cluster_labels))} global speakers")
        return speaker_mapping

    def calculate_clustering_metrics(self, embeddings, cluster_labels, output_dir=None):
        """Calculate clustering quality metrics.
        
        Args:
            embeddings: Numpy array of speaker embeddings
            cluster_labels: Cluster labels from clustering algorithm
            output_dir: Optional output directory for saving metrics report
            
        Returns:
            dict: Dictionary with clustering quality metrics
        """
        metrics = {}
        n_clusters = len(set(cluster_labels))
        n_samples = len(cluster_labels)
        
        print("\n" + "=" * 50)
        print("CLUSTERING QUALITY METRICS")
        print("=" * 50)
        
        # Need at least 2 clusters and more samples than clusters for valid metrics
        if n_clusters >= 2 and n_samples > n_clusters:
            try:
                silhouette = silhouette_score(embeddings, cluster_labels, metric='cosine')
                metrics['silhouette_score'] = silhouette
                print(f"Silhouette Score: {silhouette:.4f}")
                print(f"  (Range: -1 to 1, higher is better)")
                if silhouette > 0.7:
                    print("  Interpretation: Excellent cluster separation")
                elif silhouette > 0.5:
                    print("  Interpretation: Good cluster separation")
                elif silhouette > 0.25:
                    print("  Interpretation: Moderate cluster separation")
                else:
                    print("  Interpretation: Weak cluster separation")
            except Exception as e:
                print(f"Could not calculate Silhouette Score: {e}")
                metrics['silhouette_score'] = None
            
            try:
                calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
                metrics['calinski_harabasz_index'] = calinski_harabasz
                print(f"\nCalinski-Harabasz Index: {calinski_harabasz:.4f}")
                print(f"  (Higher is better, no fixed range)")
            except Exception as e:
                print(f"Could not calculate Calinski-Harabasz Index: {e}")
                metrics['calinski_harabasz_index'] = None
            
            try:
                davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
                metrics['davies_bouldin_index'] = davies_bouldin
                print(f"\nDavies-Bouldin Index: {davies_bouldin:.4f}")
                print(f"  (Range: 0 to inf, lower is better)")
                if davies_bouldin < 0.5:
                    print("  Interpretation: Excellent cluster separation")
                elif davies_bouldin < 1.0:
                    print("  Interpretation: Good cluster separation")
                elif davies_bouldin < 1.5:
                    print("  Interpretation: Moderate cluster separation")
                else:
                    print("  Interpretation: Weak cluster separation")
            except Exception as e:
                print(f"Could not calculate Davies-Bouldin Index: {e}")
                metrics['davies_bouldin_index'] = None
                
        else:
            print(f"Insufficient data for metrics (clusters: {n_clusters}, samples: {n_samples})")
            metrics['silhouette_score'] = None
            metrics['calinski_harabasz_index'] = None
            metrics['davies_bouldin_index'] = None
        
        # Add basic statistics
        metrics['n_clusters'] = n_clusters
        metrics['n_samples'] = n_samples
        
        print(f"\nSummary: {n_samples} segments -> {n_clusters} global speakers")
        print("=" * 50)
        
        # Save metrics to file if output directory provided
        if output_dir:
            self._save_clustering_metrics(metrics, output_dir)
        
        return metrics

    def _save_clustering_metrics(self, metrics, output_dir):
        """Save clustering metrics to a text file."""
        metrics_path = os.path.join(output_dir, "clustering_metrics_report.txt")
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("ОТЧЁТ О КАЧЕСТВЕ КЛАСТЕРИЗАЦИИ СПИКЕРОВ\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("ОБЩАЯ ИНФОРМАЦИЯ\n")
            f.write("-" * 40 + "\n")
            f.write(f"Всего сегментов спикеров: {metrics.get('n_samples', 'N/A')}\n")
            f.write(f"Глобальных спикеров: {metrics.get('n_clusters', 'N/A')}\n\n")
            
            f.write("МЕТРИКИ КАЧЕСТВА\n")
            f.write("-" * 40 + "\n\n")
            
            silhouette = metrics.get('silhouette_score')
            f.write("1. Silhouette Score\n")
            f.write("   Описание: Мера схожести точек с их кластером\n")
            f.write("   Диапазон: от -1 до 1 (выше — лучше)\n")
            if silhouette is not None:
                f.write(f"   Значение: {silhouette:.4f}\n")
                if silhouette > 0.7:
                    f.write("   Оценка: Отличное разделение кластеров\n")
                elif silhouette > 0.5:
                    f.write("   Оценка: Хорошее разделение кластеров\n")
                elif silhouette > 0.25:
                    f.write("   Оценка: Умеренное разделение кластеров\n")
                else:
                    f.write("   Оценка: Слабое разделение кластеров\n")
            else:
                f.write("   Значение: Не удалось вычислить\n")
            f.write("\n")
            
            calinski = metrics.get('calinski_harabasz_index')
            f.write("2. Calinski-Harabasz Index\n")
            f.write("   Описание: Отношение межкластерной к внутрикластерной дисперсии\n")
            f.write("   Диапазон: от 0 до бесконечности (выше — лучше)\n")
            if calinski is not None:
                f.write(f"   Значение: {calinski:.4f}\n")
            else:
                f.write("   Значение: Не удалось вычислить\n")
            f.write("\n")
            
            davies = metrics.get('davies_bouldin_index')
            f.write("3. Davies-Bouldin Index\n")
            f.write("   Описание: Средняя схожесть между кластерами\n")
            f.write("   Диапазон: от 0 до бесконечности (ниже — лучше)\n")
            if davies is not None:
                f.write(f"   Значение: {davies:.4f}\n")
                if davies < 0.5:
                    f.write("   Оценка: Отличное разделение кластеров\n")
                elif davies < 1.0:
                    f.write("   Оценка: Хорошее разделение кластеров\n")
                elif davies < 1.5:
                    f.write("   Оценка: Умеренное разделение кластеров\n")
                else:
                    f.write("   Оценка: Слабое разделение кластеров\n")
            else:
                f.write("   Значение: Не удалось вычислить\n")
            f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("Примечание: Silhouette Score — основная метрика оценки\n")
            f.write("качества кластеризации. Значения выше 0.5 означают\n")
            f.write("хорошее разделение между спикерами.\n")
        
        print(f"Clustering metrics report saved: {metrics_path}")

    def visualize_clusters(self, all_speaker_data, speaker_mapping, output_dir):
        """Improved visualization of speaker clusters.
        
        Adapts to data size:
        - PCA projection always shown (reliable at any scale)
        - t-SNE shown only for >=20 points, otherwise PCA scree plot
        - Cosine similarity matrix with speaker labels
        - Cluster sizes bar chart
        
        Args:
            all_speaker_data: Dict {chunk_path: {speaker: {embedding, ...}}}
            speaker_mapping: Dict {(chunk_path, local_speaker): global_speaker_id}
            output_dir: Directory to save visualization
        """
        try:
            # Collect data for visualization
            embeddings = []
            global_ids = []
            point_labels = []  # Annotations for each point
            
            for chunk_path, speakers in all_speaker_data.items():
                chunk_name = os.path.splitext(os.path.basename(chunk_path))[0]
                # Shorten chunk name for readability
                if len(chunk_name) > 20:
                    chunk_name = chunk_name[:17] + "..."
                
                for local_speaker, data in speakers.items():
                    global_id = speaker_mapping.get((chunk_path, local_speaker), "unknown")
                    embeddings.append(data['embedding'])
                    global_ids.append(global_id)
                    point_labels.append(f"{chunk_name}\n{local_speaker}")
            
            if len(embeddings) < 3:
                print("Not enough data for visualization (need at least 3 points)")
                return
            
            X = np.vstack(embeddings)
            unique_global = sorted(set(global_ids))
            n_speakers = len(unique_global)
            n_points = len(embeddings)
            
            # Choose color palette based on number of speakers
            if n_speakers <= 10:
                cmap = plt.cm.tab10
            elif n_speakers <= 20:
                cmap = plt.cm.tab20
            else:
                cmap = plt.cm.gist_ncar
            colors = cmap(np.linspace(0, 0.95, n_speakers))
            color_map = {spk: colors[i] for i, spk in enumerate(unique_global)}
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(
                f'Кластеризация спикеров — {n_speakers} глобальных спикеров '
                f'({n_points} сегментов)', 
                fontsize=16, fontweight='bold'
            )
            
            # ── Panel 1: PCA projection (always reliable) ──────────────────
            ax1 = axes[0, 0]
            pca = PCA(n_components=min(2, n_points, X.shape[1]))
            X_pca = pca.fit_transform(X)
            explained = pca.explained_variance_ratio_
            total_explained = sum(explained)
            
            for global_id in unique_global:
                mask = np.array(global_ids) == global_id
                if np.sum(mask) > 0:
                    ax1.scatter(
                        X_pca[mask, 0], X_pca[mask, 1],
                        c=[color_map[global_id]], label=global_id,
                        alpha=0.8, s=120, edgecolors='black', linewidths=0.5
                    )
            
            # Add point annotations (show chunk+speaker info)
            if n_points <= 30:  # Only annotate if not too crowded
                for i, label in enumerate(point_labels):
                    ax1.annotate(
                        label, (X_pca[i, 0], X_pca[i, 1]),
                        fontsize=6, ha='center', va='bottom',
                        xytext=(0, 8), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray')
                    )
            
            ax1.set_title(
                f'PCA проекция (PC1: {explained[0]:.1%}, PC2: {explained[1]:.1%}, '
                f'суммарно: {total_explained:.1%})',
                fontsize=11
            )
            ax1.set_xlabel(f'PC1 ({explained[0]:.1%} дисперсии)')
            ax1.set_ylabel(f'PC2 ({explained[1]:.1%} дисперсии)')
            ax1.legend(
                loc='upper left', fontsize=7,
                ncol=max(1, n_speakers // 6),
                bbox_to_anchor=(0, 1), framealpha=0.8
            )
            ax1.grid(True, alpha=0.3)
            
            # ── Panel 2: t-SNE (if enough points) or PCA scree plot ───────
            ax2 = axes[0, 1]
            
            if n_points >= 20:
                # t-SNE is meaningful with enough data
                perplexity = min(30, n_points - 1)
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                X_tsne = tsne.fit_transform(X)
                
                for global_id in unique_global:
                    mask = np.array(global_ids) == global_id
                    if np.sum(mask) > 0:
                        ax2.scatter(
                            X_tsne[mask, 0], X_tsne[mask, 1],
                            c=[color_map[global_id]], label=global_id,
                            alpha=0.8, s=120, edgecolors='black', linewidths=0.5
                        )
                
                if n_points <= 30:
                    for i, label in enumerate(point_labels):
                        ax2.annotate(
                            label, (X_tsne[i, 0], X_tsne[i, 1]),
                            fontsize=6, ha='center', va='bottom',
                            xytext=(0, 8), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray')
                        )
                
                ax2.set_title(f't-SNE проекция (perplexity={perplexity})', fontsize=11)
                ax2.set_xlabel('t-SNE 1')
                ax2.set_ylabel('t-SNE 2')
                ax2.legend(
                    loc='upper left', fontsize=7,
                    ncol=max(1, n_speakers // 6),
                    bbox_to_anchor=(0, 1), framealpha=0.8
                )
                ax2.grid(True, alpha=0.3)
            else:
                # Too few points for t-SNE — show PCA explained variance
                n_components = min(n_points, X.shape[1])
                pca_full = PCA(n_components=n_components)
                pca_full.fit(X)
                
                individual = pca_full.explained_variance_ratio_
                cumulative = np.cumsum(individual)
                
                x_range = range(1, len(cumulative) + 1)
                bars = ax2.bar(x_range, individual, alpha=0.6, color='steelblue', label='Отдельная')
                ax2.step(x_range, cumulative, where='mid', color='red', linewidth=2, label='Кумулятивная')
                
                # Mark 80% threshold
                ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Порог 80%')
                
                # Add percentage labels on bars
                for bar, val in zip(bars, individual):
                    if val > 0.05:
                        ax2.text(
                            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{val:.0%}', ha='center', va='bottom', fontsize=8
                        )
                
                ax2.set_title('PCA — объяснённая дисперсия по компонентам', fontsize=11)
                ax2.set_xlabel('Компонента')
                ax2.set_ylabel('Доля дисперсии')
                ax2.legend(fontsize=9)
                ax2.set_ylim(0, 1.1)
                ax2.grid(True, alpha=0.3, axis='y')
            
            # ── Panel 3: Cosine Similarity Matrix ──────────────────────────
            ax3 = axes[1, 0]
            similarity_matrix = cosine_similarity(X)
            
            # Sort indices by global speaker ID for block-diagonal structure
            global_indices = {}
            for i, gid in enumerate(global_ids):
                if gid not in global_indices:
                    global_indices[gid] = []
                global_indices[gid].append(i)
            
            sorted_indices = []
            separator_positions = []
            tick_positions = []
            tick_labels_list = []
            current_pos = 0
            
            for gid in unique_global:
                if gid in global_indices:
                    indices = global_indices[gid]
                    sorted_indices.extend(indices)
                    
                    if current_pos > 0:
                        separator_positions.append(current_pos)
                    
                    # Place tick at center of cluster block
                    tick_positions.append(current_pos + len(indices) / 2 - 0.5)
                    tick_labels_list.append(gid)
                    current_pos += len(indices)
            
            if sorted_indices:
                sorted_sim = similarity_matrix[np.ix_(sorted_indices, sorted_indices)]
                
                im = ax3.imshow(sorted_sim, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
                ax3.set_title('Матрица косинусного сходства (по спикерам)', fontsize=11)
                
                # Add separator lines between clusters
                for pos in separator_positions:
                    ax3.axhline(pos - 0.5, color='white', linewidth=2)
                    ax3.axvline(pos - 0.5, color='white', linewidth=2)
                
                # Set speaker ID labels on axes
                ax3.set_xticks(tick_positions)
                ax3.set_xticklabels(tick_labels_list, rotation=45, ha='right', fontsize=7)
                ax3.set_yticks(tick_positions)
                ax3.set_yticklabels(tick_labels_list, fontsize=7)
                
                plt.colorbar(im, ax=ax3, label='Косинусное сходство')
            else:
                ax3.text(0.5, 0.5, 'Нет данных', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=14)
                ax3.set_title('Матрица косинусного сходства')
            
            # ── Panel 4: Cluster Sizes ─────────────────────────────────────
            ax4 = axes[1, 1]
            cluster_sizes = {}
            for gid in global_ids:
                cluster_sizes[gid] = cluster_sizes.get(gid, 0) + 1
            
            valid_clusters = [spk for spk in unique_global if spk in cluster_sizes]
            valid_sizes = [cluster_sizes[spk] for spk in valid_clusters]
            
            if valid_clusters:
                x_pos = range(len(valid_clusters))
                bars = ax4.bar(
                    x_pos, valid_sizes,
                    color=[color_map[spk] for spk in valid_clusters],
                    edgecolor='black', linewidth=0.5
                )
                ax4.set_title(f'Размеры кластеров ({n_speakers} спикеров)', fontsize=11)
                ax4.set_xlabel('Глобальный ID спикера')
                ax4.set_ylabel('Количество сегментов')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(valid_clusters, rotation=45, ha='right', fontsize=7)
                
                # Add count labels on bars
                for bar, size in zip(bars, valid_sizes):
                    ax4.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(size), ha='center', va='bottom', fontsize=9, fontweight='bold'
                    )
                
                # Add mean line
                mean_size = np.mean(valid_sizes)
                ax4.axhline(y=mean_size, color='red', linestyle='--', alpha=0.6, 
                           label=f'Среднее: {mean_size:.1f}')
                ax4.legend(fontsize=9)
            else:
                ax4.text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                        transform=ax4.transAxes, fontsize=14)
                ax4.set_title('Размеры кластеров')
            
            ax4.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Save
            output_path = os.path.join(output_dir, "speaker_clusters_visualization.png")
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"Cluster visualization saved: {output_path}")
            
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback
            traceback.print_exc()

    def update_diarization_with_global_speakers(self, diarization_txt_path, speaker_mapping, chunk_path):
        """Update diarization file with global speaker IDs."""
        with open(diarization_txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace local speaker IDs with global ones
        for (map_chunk_path, local_speaker), global_speaker in speaker_mapping.items():
            if map_chunk_path == chunk_path:
                content = content.replace(f"Speaker: {local_speaker}", f"Speaker: {global_speaker}")
        
        return content

    def update_dataframe_with_global_speakers(self, dataframe, speaker_mapping, chunk_path):
        """Update DataFrame speaker column with global speaker IDs.
        
        Args:
            dataframe: DataFrame with 'speaker' column
            speaker_mapping: Dict {(chunk_path, local_speaker): global_speaker_id}
            chunk_path: Path of the chunk this DataFrame came from
            
        Returns:
            DataFrame: Updated DataFrame with global speaker IDs
        """
        df = dataframe.copy()
        for (map_chunk_path, local_speaker), global_speaker in speaker_mapping.items():
            if map_chunk_path == chunk_path:
                df.loc[df['speaker'] == local_speaker, 'speaker'] = global_speaker
        return df
