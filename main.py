import sys

try:
    import onnxruntime
except ImportError:
    print("Error: onnxruntime is not installed. Please install it using:")
    print("pip install onnxruntime")
    sys.exit(1)

try:
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    from face_memory import FaceMemory
    import os
except ImportError as e:
    print(f"Error: Failed to import required modules: {e}")
    print("Please install all requirements using:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Initialize face analysis
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize face memory
memory = FaceMemory()
memory_file = 'face_memory.pkl'
if os.path.exists(memory_file):
    memory.load(memory_file)

# Initialize webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and analyze faces
        faces = app.get(frame)
        
        for face in faces:
            # Get face embedding
            embedding = face.embedding
            
            # Get or assign face ID
            face_id = memory.add_face(embedding)
            
            # Draw bounding box and ID
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {face_id}", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Real-time Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Save face memory
    memory.save(memory_file)
    cap.release()
    cv2.destroyAllWindows()
