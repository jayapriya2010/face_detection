import sys
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from face_memory import FaceMemory
import os

# Initialize face analysis for CPU only
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

# Initialize face memory with app
memory = FaceMemory(app=app)

# ROI selection
def select_roi(frame):
    print("Select ROI and press SPACE or ENTER when done")
    roi = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyWindow("Select ROI")
    return roi

# Initialize webcam
cap = cv2.VideoCapture(0)
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read from webcam")
    sys.exit(1)

# Let user select ROI
roi = select_roi(first_frame)
roi_x, roi_y, roi_w, roi_h = roi

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw ROI
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

        # Detect and analyze faces
        faces = app.get(frame)
        current_time = time.time()
        
        for face in faces:
            bbox = face.bbox.astype(int)
            face_center_x = (bbox[0] + bbox[2]) // 2
            face_center_y = (bbox[1] + bbox[3]) // 2
            
            # Check if face is in ROI
            in_roi = (roi_x < face_center_x < roi_x + roi_w and 
                     roi_y < face_center_y < roi_y + roi_h)
            
            # Get or assign face ID (using simpler matching like detect_faces.py)
            face_id = memory.add_face(face.embedding)
            
            # Update time tracking only for recognized faces
            if not face_id.startswith("Unknown_"):
                memory.update_time(face_id, current_time, in_roi)
            
            # Draw bounding box and information
            color = (0, 255, 0) if in_roi else (0, 165, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Display name and time
            dwell_time = memory.get_dwell_time(face_id)
            time_str = f" Time: {dwell_time:.1f}s" if not face_id.startswith("Unknown_") else ""
            cv2.putText(frame, f"{face_id}{time_str}", 
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Face Recognition with ROI', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
