import numpy as np
import os
from pathlib import Path
import cv2

try:
    import faiss
except ImportError:
    print("Error: FAISS is not installed.")
    exit(1)

class FaceMemory:
    def __init__(self, app=None, embedding_dim=512, similarity_threshold=0.5):  # Match detect_faces.py threshold
        self.app = app
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.face_ids = []
        self.embeddings_history = []  # Store embeddings for each ID
        self.next_id = 0
        self.min_quality_score = 0.2  # Lower quality threshold
        self.known_face_threshold = 0.6  # Separate threshold for known faces
        self.face_times = {}  # Track time spent in ROI for each face
        self.last_seen = {}   # Track last seen timestamp for each face
        self.known_faces = {}  # Store known face embeddings
        self.known_faces_dir = Path("known_faces")
        if not self.known_faces_dir.exists():
            self.known_faces_dir.mkdir(exist_ok=True)
            print(f"Created known_faces directory at {self.known_faces_dir}")
        if self.app:  # Only load known faces if app is provided
            self.load_known_faces()
    
    def load_known_faces(self):
        if not self.app:
            return
            
        if self.known_faces_dir.exists():
            print("\nLoading known faces:")
            for face_path in self.known_faces_dir.glob("*.jpg"):
                name = face_path.stem
                img = cv2.imread(str(face_path))
                if img is None:
                    continue
                faces = self.app.get(img)
                if faces:
                    embedding = faces[0].embedding  # Don't normalize reference embeddings
                    self.known_faces[name] = embedding
                    print(f"  ✓ Loaded known face: {name}")
            print(f"\nTotal known faces loaded: {len(self.known_faces)}")
        else:
            print("Warning: known_faces directory not found")
    
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
            return "Low_Quality"
            
        embedding = self._normalize_embedding(embedding)
        
        # Use same comparison logic as detect_faces.py
        for name, known_embedding in self.known_faces.items():
            cos_similarity = np.dot(embedding.flatten(), known_embedding) / \
                (np.linalg.norm(embedding.flatten()) * np.linalg.norm(known_embedding))
            if cos_similarity > self.similarity_threshold:
                return name
        
        # Only create new unknown if no match found
        new_id = f"Unknown_{self.next_id}"
        self.next_id += 1
        return new_id

    def update_time(self, face_id, timestamp, in_roi=False):
        if face_id not in self.face_times:
            self.face_times[face_id] = 0
            
        if in_roi:
            if face_id in self.last_seen:
                time_diff = timestamp - self.last_seen[face_id]
                self.face_times[face_id] += time_diff
            self.last_seen[face_id] = timestamp
        else:
            self.last_seen.pop(face_id, None)
    
    def get_dwell_time(self, face_id):
        return self.face_times.get(face_id, 0)
        new_id = f"Unknown_{self.next_id}"
        self.face_ids.append(new_id)
        self.embeddings_history.append([embedding.flatten()])
        self.next_id += 1
        return new_id
    
    def update_time(self, face_id, timestamp, in_roi=False):
        if face_id not in self.face_times:
            self.face_times[face_id] = 0
            
        if in_roi:
            if face_id in self.last_seen:
                time_diff = timestamp - self.last_seen[face_id]
                self.face_times[face_id] += time_diff
            self.last_seen[face_id] = timestamp
        else:
            self.last_seen.pop(face_id, None)
    
    def get_dwell_time(self, face_id):
        return self.face_times.get(face_id, 0)
