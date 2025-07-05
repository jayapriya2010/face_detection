import numpy as np
import faiss
import pickle
import os

class FaceMemory:
    def __init__(self, embedding_dim=512, similarity_threshold=0.7):  # Increased threshold
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.face_ids = []
        self.embeddings_history = []  # Store embeddings for each ID
        self.next_id = 0
        self.min_quality_score = 0.5  # Minimum quality threshold
    
    def _normalize_embedding(self, embedding):
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        return np.divide(embedding, norm, where=norm != 0)
    
    def _check_quality(self, embedding):
        # Simple quality check based on embedding norm
        norm = np.linalg.norm(embedding)
        return norm > self.min_quality_score
    
    def _verify_face(self, embedding, candidate_idx):
        # Get the stored embeddings for the candidate ID
        if candidate_idx >= len(self.embeddings_history):
            return False
        
        stored_embeddings = np.array(self.embeddings_history[candidate_idx])
        if len(stored_embeddings) == 0:
            return False
        
        # Compare with all stored embeddings for this ID
        similarities = np.dot(stored_embeddings, embedding.T).squeeze()
        avg_similarity = np.mean(similarities)
        return avg_similarity > self.similarity_threshold
    
    def add_face(self, embedding):
        if not self._check_quality(embedding):
            return -1  # Invalid face
            
        embedding = self._normalize_embedding(embedding)
        
        if self.index.ntotal > 0:
            S, I = self.index.search(embedding, 3)  # Get top 3 matches
            for similarity, idx in zip(S[0], I[0]):
                if similarity > self.similarity_threshold and self._verify_face(embedding, idx):
                    # Update embedding history
                    self.embeddings_history[idx].append(embedding.flatten())
                    if len(self.embeddings_history[idx]) > 5:  # Keep last 5 embeddings
                        self.embeddings_history[idx].pop(0)
                    return self.face_ids[idx]
        
        # Add new face
        self.index.add(embedding)
        self.face_ids.append(self.next_id)
        self.embeddings_history.append([embedding.flatten()])
        self.next_id += 1
        return self.face_ids[-1]
    
    def save(self, filepath):
        data = {
            'index': faiss.serialize_index(self.index),
            'face_ids': self.face_ids,
            'embeddings_history': self.embeddings_history,
            'next_id': self.next_id
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.index = faiss.deserialize_index(data['index'])
            self.face_ids = data['face_ids']
            self.embeddings_history = data.get('embeddings_history', [[] for _ in self.face_ids])
            self.next_id = data['next_id']
