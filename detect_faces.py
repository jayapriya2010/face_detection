import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path

# Initialize the model for CPU
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))  # Reduced detection size for better CPU performance

def get_face_embedding(face):
    return face.embedding

def compare_faces(known_embedding, face_embedding, threshold=0.5):
    cos_similarity = np.dot(known_embedding, face_embedding) / \
        (np.linalg.norm(known_embedding) * np.linalg.norm(face_embedding))
    return cos_similarity > threshold

# Load known faces from a directory
known_faces = {}
known_faces_dir = Path("known_faces")
if known_faces_dir.exists():
    for face_path in known_faces_dir.glob("*.jpg"):
        name = face_path.stem
        img = cv2.imread(str(face_path))
        faces = app.get(img)
        if faces:
            known_faces[name] = get_face_embedding(faces[0])

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit(1)

# Test GUI support
try:
    cv2.namedWindow("Test")
    cv2.destroyWindow("Test")
except cv2.error:
    print("Error: OpenCV GUI support not available.")
    print("Please reinstall opencv-python with GUI support:")
    print("conda install -c conda-forge opencv-python")
    cap.release()
    exit(1)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Detect faces
    faces = app.get(frame)
    
    # Draw bounding boxes and label recognized faces
    for face in faces:
        box = face.bbox.astype(int)
        current_embedding = get_face_embedding(face)
        recognized_name = "Unknown"
        
        for name, known_embedding in known_faces.items():
            if compare_faces(known_embedding, current_embedding):
                recognized_name = name
                break
        
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, recognized_name, (box[0], box[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
