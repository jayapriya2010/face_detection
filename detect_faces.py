import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from pathlib import Path

# Initialize the model for CPU
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

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

# Load the image
image_path = "your_image1.jpg"
if not Path(image_path).exists():
    print(f"Error: Image file '{image_path}' not found")
    print("Please update the image_path variable with your target image")
    exit(1)

img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image '{image_path}'")
    exit(1)

# Detect faces
faces = app.get(img)
print(f"Detected {len(faces)} face(s)")

# Draw bounding boxes and label recognized faces
for face in faces:
    box = face.bbox.astype(int)
    # Try to recognize the face
    current_embedding = get_face_embedding(face)
    recognized_name = "Unknown"
    
    for name, known_embedding in known_faces.items():
        if compare_faces(known_embedding, current_embedding):
            recognized_name = name
            break
    
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.putText(img, recognized_name, (box[0], box[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save instead of show
cv2.imwrite("output.jpg", img)
print("✅ Saved output image as output.jpg")
