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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ModelConfig


class SpeakerClustering:
    def __init__(self, hf_token):
        """Initialize speaker clustering with pyannote models."""
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        self.embedding_model = Model.from_pretrained(
            "pyannote/embedding", 
            token=hf_token
        )
        self.embedding_inference = Inference(
            self.embedding_model, 
            window="whole"
        )
        
    def find_silence_periods(self, audio_file_path, min_silence_len=2000, silence_thresh=-40):
        """Find silence periods in audio for natural chunking."""
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

    def split_audio_at_silence(self, audio_file_path, target_chunk_duration_min=25, 
                             max_chunk_duration_min=30, min_silence_len=2000):
        """Split audio at natural silence points."""
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


    def cluster_speakers(self, all_speaker_data, distance_threshold=0.4):
        """Cluster speakers across all chunks."""
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
        
        print(f"Clustered {len(speaker_info)} local speakers into {len(set(cluster_labels))} global speakers")
        return speaker_mapping

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
            
            # Cluster speakers across all chunks
            print("Clustering speakers across all chunks...")
            speaker_mapping = self.cluster_speakers(all_speaker_data)
            
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