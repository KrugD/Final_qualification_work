import io
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
import librosa
import sys
import pydub
import pydub.silence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        if audio.channels > 1:
            audio = audio.set_channels(1)
            
        silence_periods = []
        
        silent_ranges = pydub.silence.detect_silence(
            audio, 
            min_silence_len=min_silence_len, 
            silence_thresh=silence_thresh
        )
        
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
        
        temp_dir = tempfile.mkdtemp(prefix=f"audio_chunks_{audio_filename}_")
        
        chunk_num = 1
        while current_start < total_duration_ms:
            target_end = current_start + target_chunk_duration_ms
            
            best_split_point = target_end
            min_distance = float('inf')
            
            for silence_start, silence_end in silence_periods:
                if (silence_start > current_start and 
                    silence_start <= current_start + max_chunk_duration_ms):
                    distance = abs(silence_start - target_end)
                    if distance < min_distance:
                        min_distance = distance
                        best_split_point = silence_start
            
            if best_split_point - current_start < 60 * 1000:
                best_split_point = min(current_start + max_chunk_duration_ms, total_duration_ms)
            
            chunk_end = min(best_split_point, total_duration_ms)
            chunk = audio[current_start:chunk_end]
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_num:03d}_{audio_filename}.wav")
            chunk.export(chunk_path, format="wav")
            chunks.append(chunk_path)
            
            print(f"  Chunk {chunk_num}: {current_start/1000/60:.1f}min - {chunk_end/1000/60:.1f}min "
                f"({(chunk_end-current_start)/1000/60:.1f}min)")
            
            current_start = chunk_end
            chunk_num += 1
            
            if current_start >= total_duration_ms:
                break
        
        print(f"Split audio into {len(chunks)} chunks at natural silence points")
        return chunks, temp_dir

    def extract_speaker_embeddings(self, audio_file_path):
        """Extract speaker embeddings from audio file."""
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
                        emb_np = np.array(embedding).flatten()
                        
                        if speaker not in speaker_data:
                            speaker_data[speaker] = {
                                'embeddings': [],
                                'segments': [],
                                'durations': []
                            }
                        
                        speaker_data[speaker]['embeddings'].append(emb_np)
                        speaker_data[speaker]['segments'].append(segment)
                        speaker_data[speaker]['durations'].append(segment.end - segment.start)
                        
                    except Exception as e:
                        print(f"Error extracting embedding: {e}")
                        continue
            else:
                print(f"No speaker_diarization attribute found in diarization result")
                return {}
            
            print(f"Processed {segment_count} segments total")
            
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
                    print(f"Speaker {speaker}: {len(data['embeddings'])} segments, "
                          f"avg embedding shape: {avg_embedding.shape}")
            
            print(f"Successfully processed {len(avg_embeddings)} speakers")
            return avg_embeddings
            
        except Exception as e:
            print(f"Error processing {audio_file_path}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def extract_segment_embeddings(self, audio_file_path):
        """Extract per-segment speaker embeddings for metrics calculation.
        
        Returns:
            tuple: (embeddings_array, speaker_labels, speaker_names)
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
                        emb_np = np.array(embedding).flatten()
                        
                        if speaker not in speaker_to_label:
                            speaker_to_label[speaker] = label_counter
                            label_counter += 1
                        
                        all_embeddings.append(emb_np)
                        all_labels.append(speaker_to_label[speaker])
                        
                    except Exception as e:
                        continue
            
            if not all_embeddings:
                return None, None, None
            
            embeddings_array = np.vstack(all_embeddings)
            labels_array = np.array(all_labels)
            label_to_speaker = {v: k for k, v in speaker_to_label.items()}
            
            print(f"Extracted {len(all_embeddings)} segment embeddings for "
                  f"{len(speaker_to_label)} speakers")
            return embeddings_array, labels_array, label_to_speaker
            
        except Exception as e:
            print(f"Error extracting segment embeddings: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def cluster_speakers(self, all_speaker_data, distance_threshold=None, output_dir=None):
        """Cluster speakers across all chunks."""

        if distance_threshold is None:
            distance_threshold = ModelConfig.CLUSTERING_DISTANCE_THRESHOLD

        if not all_speaker_data:
            return {}
        
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
            global_id = 0
            mapping = {}
            for info in speaker_info:
                mapping[(info['chunk_path'], info['local_speaker'])] = f"speaker_{global_id:02d}"
                global_id += 1
            return mapping
        
        X = np.vstack(embeddings)
        
        if X.shape[1] > 50:
            pca = PCA(n_components=min(50, X.shape[0]))
            X = pca.fit_transform(X)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='average',
            metric='cosine'
        )
        
        cluster_labels = clustering.fit_predict(X)
        
        speaker_mapping = {}
        for i, info in enumerate(speaker_info):
            global_speaker_id = f"speaker_{cluster_labels[i]:02d}"
            speaker_mapping[(info['chunk_path'], info['local_speaker'])] = global_speaker_id
        
        clustering_metrics = self.calculate_clustering_metrics(X, cluster_labels, output_dir)
        
        if output_dir and len(embeddings) >= 3:
            self.visualize_clusters(all_speaker_data, speaker_mapping, output_dir)

        print(f"Clustered {len(speaker_info)} local speakers into "
              f"{len(set(cluster_labels))} global speakers")
        return speaker_mapping

    def calculate_clustering_metrics(self, embeddings, cluster_labels, output_dir=None):
        """Calculate clustering quality metrics."""
        metrics = {}
        n_clusters = len(set(cluster_labels))
        n_samples = len(cluster_labels)
        
        print("\n" + "=" * 50)
        print("МЕТРИКИ КАЧЕСТВА КЛАСТЕРИЗАЦИИ")
        print("=" * 50)
        
        if n_clusters >= 2 and n_samples > n_clusters:
            try:
                silhouette = silhouette_score(embeddings, cluster_labels, metric='cosine')
                metrics['silhouette_score'] = silhouette
                print(f"Silhouette Score: {silhouette:.4f}")
                print(f"  (Диапазон: от -1 до 1, чем выше — тем лучше)")
                print(f"  Интерпретация: ", end="")
                if silhouette > 0.7:
                    print("Отличное разделение кластеров")
                elif silhouette > 0.5:
                    print("Хорошее разделение кластеров")
                elif silhouette > 0.25:
                    print("Умеренное разделение кластеров")
                else:
                    print("Слабое разделение кластеров")
            except Exception as e:
                print(f"Не удалось рассчитать Silhouette Score: {e}")
                metrics['silhouette_score'] = None
            
            try:
                calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
                metrics['calinski_harabasz_index'] = calinski_harabasz
                print(f"\nCalinski-Harabasz Index: {calinski_harabasz:.4f}")
                print(f"  (Чем выше — тем лучше, нет фиксированного диапазона)")
            except Exception as e:
                print(f"Не удалось рассчитать Calinski-Harabasz Index: {e}")
                metrics['calinski_harabasz_index'] = None
            
            try:
                davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
                metrics['davies_bouldin_index'] = davies_bouldin
                print(f"\nDavies-Bouldin Index: {davies_bouldin:.4f}")
                print(f"  (Диапазон: от 0 до ∞, чем ниже — тем лучше)")
                print(f"  Интерпретация: ", end="")
                if davies_bouldin < 0.5:
                    print("Отличное разделение кластеров")
                elif davies_bouldin < 1.0:
                    print("Хорошее разделение кластеров")
                elif davies_bouldin < 1.5:
                    print("Умеренное разделение кластеров")
                else:
                    print("Слабое разделение кластеров")
            except Exception as e:
                print(f"Не удалось рассчитать Davies-Bouldin Index: {e}")
                metrics['davies_bouldin_index'] = None
        else:
            print(f"Недостаточно данных для расчёта метрик кластеризации")
            print(f"  Кластеров: {n_clusters}, Сегментов: {n_samples}")
            print(f"  Нужно минимум 2 кластера и больше сегментов, чем кластеров")
            metrics['silhouette_score'] = None
            metrics['calinski_harabasz_index'] = None
            metrics['davies_bouldin_index'] = None
        
        metrics['n_clusters'] = n_clusters
        metrics['n_samples'] = n_samples
        
        print(f"\nИтог кластеризации:")
        print(f"  Всего сегментов спикеров: {n_samples}")
        print(f"  Глобальных спикеров выявлено: {n_clusters}")
        print("=" * 50)
        
        if output_dir:
            self._save_clustering_metrics(metrics, output_dir)
        
        return metrics

    def _save_clustering_metrics(self, metrics, output_dir):
        """Save clustering metrics report in Russian."""
        metrics_path = os.path.join(output_dir, "clustering_metrics_report.txt")
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("ОТЧЁТ О МЕТРИКАХ КАЧЕСТВА КЛАСТЕРИЗАЦИИ СПИКЕРОВ\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("ИТОГ КЛАСТЕРИЗАЦИИ\n")
            f.write("-" * 40 + "\n")
            f.write(f"Всего сегментов спикеров проанализировано: {metrics.get('n_samples', 'Н/Д')}\n")
            f.write(f"Глобальных спикеров выявлено: {metrics.get('n_clusters', 'Н/Д')}\n\n")
            
            f.write("МЕТРИКИ КАЧЕСТВА\n")
            f.write("-" * 40 + "\n\n")
            
            silhouette = metrics.get('silhouette_score')
            f.write("1. Silhouette Score (Коэффициент силуэта)\n")
            f.write("   Описание: Измеряет, насколько объекты похожи на свой кластер\n")
            f.write("   Диапазон: от -1 до 1 (чем выше — тем лучше)\n")
            if silhouette is not None:
                f.write(f"   Значение: {silhouette:.4f}\n")
                if silhouette > 0.7:
                    f.write("   Интерпретация: Отличное разделение кластеров\n")
                elif silhouette > 0.5:
                    f.write("   Интерпретация: Хорошее разделение кластеров\n")
                elif silhouette > 0.25:
                    f.write("   Интерпретация: Умеренное разделение кластеров\n")
                else:
                    f.write("   Интерпретация: Слабое разделение кластеров\n")
            else:
                f.write("   Значение: Невозможно рассчитать\n")
            f.write("\n")
            
            calinski = metrics.get('calinski_harabasz_index')
            f.write("2. Calinski-Harabasz Index (Индекс Калински-Харабаса)\n")
            f.write("   Описание: Отношение межкластерной дисперсии к внутрикластерной\n")
            f.write("   Диапазон: от 0 до ∞ (чем выше — тем лучше)\n")
            if calinski is not None:
                f.write(f"   Значение: {calinski:.4f}\n")
            else:
                f.write("   Значение: Невозможно рассчитать\n")
            f.write("\n")
            
            davies = metrics.get('davies_bouldin_index')
            f.write("3. Davies-Bouldin Index (Индекс Дэвиса-Болдина)\n")
            f.write("   Описание: Средняя мера сходства между кластерами\n")
            f.write("   Диапазон: от 0 до ∞ (чем ниже — тем лучше)\n")
            if davies is not None:
                f.write(f"   Значение: {davies:.4f}\n")
                if davies < 0.5:
                    f.write("   Интерпретация: Отличное разделение кластеров\n")
                elif davies < 1.0:
                    f.write("   Интерпретация: Хорошее разделение кластеров\n")
                elif davies < 1.5:
                    f.write("   Интерпретация: Умеренное разделение кластеров\n")
                else:
                    f.write("   Интерпретация: Слабое разделение кластеров\n")
            else:
                f.write("   Значение: Невозможно рассчитать\n")
            f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("Примечание: Silhouette Score — основная метрика для оценки\n")
            f.write("качества кластеризации спикеров. Значения выше 0.5 указывают\n")
            f.write("на хорошее разделение между спикерами.\n")
        
        print(f"Отчёт о метриках кластеризации сохранён: {metrics_path}")

    # ----------------------------------------------------------------
    # Visualization helpers
    # ----------------------------------------------------------------

    def _get_color_palette(self, n_colors):
        """Get an adaptive color palette for n clusters."""
        if n_colors <= 10:
            base = plt.cm.tab10(np.linspace(0, 1, 10))
            return base[:n_colors]
        elif n_colors <= 20:
            base = plt.cm.tab20(np.linspace(0, 1, 20))
            return base[:n_colors]
        else:
            return plt.cm.gist_rainbow(np.linspace(0, 1, n_colors))

    def _build_visualization_data(self, all_speaker_data, speaker_mapping):
        """Prepare data arrays for visualization.
        
        Returns:
            dict with keys: X, global_ids, unique_global, color_map, n_points
            or None if not enough data.
        """
        embeddings = []
        global_ids = []
        
        for chunk_path, speakers in all_speaker_data.items():
            for local_speaker, data in speakers.items():
                global_id = speaker_mapping.get((chunk_path, local_speaker), "unknown")
                embeddings.append(data['embedding'])
                global_ids.append(global_id)
        
        if len(embeddings) < 3:
            print("Недостаточно данных для визуализации (нужно ≥ 3)")
            return None
        
        X = np.vstack(embeddings)
        unique_global = sorted(set(global_ids))
        
        colors = self._get_color_palette(len(unique_global))
        color_map = {spk: colors[i] for i, spk in enumerate(unique_global)}
        
        return {
            'X': X,
            'global_ids': global_ids,
            'unique_global': unique_global,
            'color_map': color_map,
            'n_points': len(embeddings),
        }

    def _draw_4panel_figure(self, vis, title_suffix=""):
        """Draw the 4-panel visualization figure.
        
        Args:
            vis: dict from _build_visualization_data
            title_suffix: optional suffix for the figure title
            
        Returns:
            matplotlib Figure
        """
        X = vis['X']
        global_ids = vis['global_ids']
        unique_global = vis['unique_global']
        color_map = vis['color_map']
        n_points = vis['n_points']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(
            f'Визуализация кластеризации спикеров — {len(unique_global)} глобальных спикеров'
            + (f' {title_suffix}' if title_suffix else ''),
            fontsize=16, fontweight='bold'
        )
        
        # ---- Panel 1: PCA ----
        print("Выполняется PCA...")
        pca_full = PCA()
        X_pca_full = pca_full.fit_transform(X)
        
        X_pca = X_pca_full[:, :2]
        
        ax1 = axes[0, 0]
        for global_id in unique_global:
            mask = np.array(global_ids) == global_id
            if np.sum(mask) > 0:
                ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                            c=[color_map[global_id]], label=global_id, alpha=0.7, s=60)
        
        if n_points <= 20:
            for i, gid in enumerate(global_ids):
                ax1.annotate(gid, (X_pca[i, 0], X_pca[i, 1]),
                             fontsize=7, alpha=0.8,
                             xytext=(5, 5), textcoords='offset points')
        
        var1 = pca_full.explained_variance_ratio_[0]
        var2 = pca_full.explained_variance_ratio_[1]
        ax1.set_title(f'PCA: {len(unique_global)} кластеров спикеров')
        ax1.set_xlabel(f'PC1 ({var1:.1%} дисперсии)')
        ax1.set_ylabel(f'PC2 ({var2:.1%} дисперсии)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # ---- Panel 2: t-SNE (if ≥ 5 points) or PCA scree plot ----
        ax2 = axes[0, 1]
        
        if n_points >= 5:
            print("Выполняется t-SNE...")
            perp = min(30, n_points - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
            X_tsne = tsne.fit_transform(X)
            
            for global_id in unique_global:
                mask = np.array(global_ids) == global_id
                if np.sum(mask) > 0:
                    ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                                c=[color_map[global_id]], label=global_id, alpha=0.7, s=60)
            
            ax2.set_title(f't-SNE: {len(unique_global)} кластеров спикеров')
            ax2.set_xlabel('t-SNE 1')
            ax2.set_ylabel('t-SNE 2')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)
        else:
            n_components = min(len(pca_full.explained_variance_ratio_), 10)
            explained = pca_full.explained_variance_ratio_[:n_components]
            cumulative = np.cumsum(explained)
            
            ax2.bar(range(1, n_components + 1), explained, alpha=0.6, label='Вклад компоненты')
            ax2.plot(range(1, n_components + 1), cumulative, 'ro-', label='Накопленный вклад')
            ax2.axhline(y=0.95, color='k', linestyle='--', alpha=0.5, label='95% дисперсии')
            ax2.set_title('Объяснённая дисперсия PCA (Scree Plot)')
            ax2.set_xlabel('Номер компоненты')
            ax2.set_ylabel('Доля объяснённой дисперсии')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
        
        # ---- Panel 3: Cosine Similarity Matrix ----
        ax3 = axes[1, 0]
        similarity_matrix = cosine_similarity(X)
        
        global_indices = {}
        for i, gid in enumerate(global_ids):
            global_indices.setdefault(gid, []).append(i)
        
        sorted_indices = []
        for gid in unique_global:
            if gid in global_indices:
                sorted_indices.extend(global_indices[gid])
        
        if sorted_indices:
            sorted_sim = similarity_matrix[np.ix_(sorted_indices, sorted_indices)]
            im = ax3.imshow(sorted_sim, cmap='RdYlBu', aspect='auto')
            ax3.set_title('Матрица косинусного сходства')
            ax3.set_xlabel('Сегменты спикеров')
            ax3.set_ylabel('Сегменты спикеров')
            
            current_pos = 0
            for gid in unique_global:
                if gid in global_indices:
                    cluster_size = len(global_indices[gid])
                    if current_pos > 0:
                        ax3.axhline(current_pos - 0.5, color='white', linewidth=2)
                        ax3.axvline(current_pos - 0.5, color='white', linewidth=2)
                    current_pos += cluster_size
            
            plt.colorbar(im, ax=ax3)
        else:
            ax3.text(0.5, 0.5, 'Нет данных для матрицы сходства',
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Матрица косинусного сходства (нет данных)')
        
        # ---- Panel 4: Cluster sizes histogram ----
        ax4 = axes[1, 1]
        cluster_sizes = {}
        for gid in global_ids:
            cluster_sizes[gid] = cluster_sizes.get(gid, 0) + 1
        
        valid_clusters = [spk for spk in unique_global if spk in cluster_sizes]
        valid_sizes = [cluster_sizes[spk] for spk in valid_clusters]
        
        if valid_clusters:
            bars = ax4.bar(valid_clusters, valid_sizes,
                           color=[color_map[spk] for spk in valid_clusters])
            
            mean_size = np.mean(valid_sizes)
            ax4.axhline(y=mean_size, color='red', linestyle='--', alpha=0.7,
                        label=f'Среднее: {mean_size:.1f}')
            
            ax4.set_title(f'Размеры кластеров ({len(valid_clusters)} спикеров)')
            ax4.set_xlabel('Глобальный ID спикера')
            ax4.set_ylabel('Количество сегментов')
            ax4.tick_params(axis='x', rotation=45)
            ax4.legend(fontsize=8)
            
            for bar, size in zip(bars, valid_sizes):
                ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         str(size), ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'Нет данных о кластерах',
                     ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Размеры кластеров (нет данных)')
        
        plt.tight_layout()
        return fig

    # ----------------------------------------------------------------
    # Public visualization methods
    # ----------------------------------------------------------------

    def visualize_clusters(self, all_speaker_data, speaker_mapping, output_dir):
        """Save cluster visualization to PNG file."""
        try:
            vis = self._build_visualization_data(all_speaker_data, speaker_mapping)
            if vis is None:
                return
            
            print(f"Визуализация {len(vis['unique_global'])} глобальных спикеров: "
                  f"{vis['unique_global']}")
            
            fig = self._draw_4panel_figure(vis)
            
            output_path = os.path.join(output_dir, "speaker_clusters_visualization.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Визуализация кластеров сохранена: {output_path}")
            
        except Exception as e:
            print(f"Ошибка визуализации: {e}")
            import traceback
            traceback.print_exc()

    def visualize_clusters_to_buffer(self, all_speaker_data, speaker_mapping):
        """Generate cluster visualization and return as BytesIO buffer (for bot mode).
        
        Returns:
            BytesIO: PNG image buffer, or None on failure
        """
        try:
            vis = self._build_visualization_data(all_speaker_data, speaker_mapping)
            if vis is None:
                return None
            
            fig = self._draw_4panel_figure(vis)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            
            print(f"Визуализация кластеров сгенерирована в буфер "
                  f"({buf.getbuffer().nbytes} байт)")
            return buf
            
        except Exception as e:
            print(f"Ошибка генерации визуализации в буфер: {e}")
            import traceback
            traceback.print_exc()
            return None

    def print_cluster_statistics(self, all_speaker_data, speaker_mapping):
        """Print clustering statistics."""
        print("\nСТАТИСТИКА КЛАСТЕРИЗАЦИИ:")
        print("=" * 50)
        
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
        
        for global_id, stats in sorted(cluster_stats.items()):
            local_count = len(stats['local_speakers'])
            chunk_count = len(stats['chunks'])
            duration_min = stats['total_duration'] / 60
            
            print(f"Спикер {global_id}:")
            print(f"   Локальных ID: {local_count}")
            print(f"   Сегментов: {stats['total_segments']}")
            print(f"   Длительность: {duration_min:.1f} мин")
            print(f"   Чанков: {chunk_count}")
            print(f"   Локальные ID: {', '.join(sorted(stats['local_speakers']))}")
        
        print("=" * 50)
        print(f"ИТОГО: {len(cluster_stats)} глобальных спикеров "
              f"из {len(local_to_global)} локальных")

    def process_long_audio(self, audio_file_path, output_dir):
        """Process long audio with speaker clustering."""
        print("Splitting audio at natural silence points...")
        chunk_paths, temp_dir = self.split_audio_at_silence(audio_file_path)
        
        all_speaker_data = {}
        
        try:
            for i, chunk_path in enumerate(chunk_paths):
                print(f"Processing chunk {i+1}/{len(chunk_paths)}...")
                
                speaker_embeddings = self.extract_speaker_embeddings(chunk_path)
                if speaker_embeddings:
                    all_speaker_data[chunk_path] = speaker_embeddings
                    print(f"Found {len(speaker_embeddings)} speakers in this chunk")
                else:
                    print(f"No speakers found in this chunk")
            
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
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise

    def update_diarization_with_global_speakers(self, diarization_txt_path, speaker_mapping, chunk_path):
        """Update diarization file with global speaker IDs."""
        with open(diarization_txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for (map_chunk_path, local_speaker), global_speaker in speaker_mapping.items():
            if map_chunk_path == chunk_path:
                content = content.replace(f"Speaker: {local_speaker}", f"Speaker: {global_speaker}")
        
        return content
