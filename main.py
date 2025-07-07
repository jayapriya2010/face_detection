import sys
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from face_memory import FaceMemory
import os
from ultralytics import YOLO

# Initialize YOLO model
try:
    model = YOLO('yolov8n.pt')  # using YOLOv8 nano model
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Installing required packages...")
    os.system('pip install ultralytics')
    model = YOLO('yolov8n.pt')

class ROI:
    def __init__(self, x, y, w, h, name):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.name = name
        self.authorized_people = []

def get_video_source():
    video_path = input("Enter video file path (or press Enter for webcam): ").strip()
    if video_path and os.path.exists(video_path):
        return cv2.VideoCapture(video_path)
    print("Using webcam (no valid video file provided)")
    return cv2.VideoCapture(0)

# Initialize face analysis for CPU only
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

# Initialize face memory with app
memory = FaceMemory(app=app)

# ROI selection
def select_multiple_rois(frame):
    rois = []
    while True:
        # Select ROI
        print("\nSelect ROI and press SPACE or ENTER when done")
        print("Press 'q' when you're done adding ROIs")
        roi = cv2.selectROI("Select ROI", frame, False)
        if roi == (0, 0, 0, 0):
            break
            
        # Get ROI name
        name = input("Enter a name for this ROI: ").strip()
        if not name:
            name = f"ROI_{len(rois) + 1}"
        
        # Create ROI object
        new_roi = ROI(roi[0], roi[1], roi[2], roi[3], name)
        
        # Get authorized people for this ROI
        print(f"\nEnter authorized people for {name} (one name per line)")
        print("Press Enter twice when done:")
        while True:
            person = input().strip()
            if not person:
                break
            new_roi.authorized_people.append(person)
        
        rois.append(new_roi)
        
        # Draw existing ROIs
        temp_frame = frame.copy()
        for r in rois:
            cv2.rectangle(temp_frame, (r.x, r.y), (r.x + r.w, r.y + r.h), (255, 0, 0), 2)
            cv2.putText(temp_frame, r.name, (r.x, r.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Select ROI", temp_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyWindow("Select ROI")
    return rois

# Initialize video source
cap = get_video_source()
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read from video source")
    sys.exit(1)

# Let user select multiple ROIs
rois = select_multiple_rois(first_frame)
if not rois:
    print("No ROIs selected. Exiting.")
    sys.exit(1)

print("\nStarting facial recognition with the following ROIs:")
for roi in rois:
    print(f"- {roi.name} (Authorized: {', '.join(roi.authorized_people)})")

try:
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000/fps) if fps > 0 else 30  # milliseconds between frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        # Add instructions text
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset times", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Run YOLO detection
        results = model(frame)[0]
        people_detections = [det for det in results.boxes.data if int(det[5]) == 0]  # class 0 is person

        # Count people in each ROI
        roi_counts = {roi.name: 0 for roi in rois}
        for detection in people_detections:
            x1, y1, x2, y2 = detection[:4].int().cpu().numpy()
            person_center_x = (x1 + x2) // 2
            person_center_y = (y1 + y2) // 2
            
            for roi in rois:
                if (roi.x < person_center_x < roi.x + roi.w and 
                    roi.y < person_center_y < roi.y + roi.h):
                    roi_counts[roi.name] += 1

        # Draw all ROIs with overcrowding indication
        for roi in rois:
            # Red if overcrowded (more than 1 person), blue otherwise
            color = (0, 0, 255) if roi_counts[roi.name] > 1 else (255, 0, 0)
            cv2.rectangle(frame, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), color, 2)
            
            # Draw ROI name and count
            cv2.putText(frame, f"{roi.name} (Count: {roi_counts[roi.name]})", 
                       (roi.x, roi.y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add OVERCROWDED warning if more than 1 person
            if roi_counts[roi.name] > 1:
                cv2.putText(frame, "OVERCROWDED", 
                           (roi.x, roi.y - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)

        # Detect and analyze faces
        faces = app.get(frame)
        current_time = time.time()
        
        for face in faces:
            bbox = face.bbox.astype(int)
            face_center_x = (bbox[0] + bbox[2]) // 2
            face_center_y = (bbox[1] + bbox[3]) // 2
            
            # Get or assign face ID first
            face_id = memory.add_face(face.embedding)
            
            # Check if face is in any ROI and is authorized
            in_roi = False
            roi_name = ""
            is_authorized = False
            
            for roi in rois:
                if (roi.x < face_center_x < roi.x + roi.w and 
                    roi.y < face_center_y < roi.y + roi.h):
                    in_roi = True
                    roi_name = roi.name
                    is_authorized = face_id in roi.authorized_people
                    break
            
            # Update time tracking only for authorized faces in their assigned ROI
            if not face_id.startswith("Unknown_") and in_roi and is_authorized:
                memory.update_time(face_id, current_time, True)
            
            # Draw bounding box and information
            if in_roi:
                color = (0, 255, 0) if is_authorized else (0, 0, 255)  # Red for unauthorized
            else:
                color = (0, 165, 255)  # Orange for outside ROI
                
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Display name, ROI, authorization status, and time
            dwell_time = memory.get_dwell_time(face_id)
            auth_str = "" if not in_roi else " (Authorized)" if is_authorized else " (Unauthorized)"
            time_str = f" Time: {dwell_time:.1f}s" if not face_id.startswith("Unknown_") and is_authorized else ""
            roi_str = f" ({roi_name})" if in_roi else ""
            
            cv2.putText(frame, f"{face_id}{roi_str}{auth_str}{time_str}", 
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Face Recognition with ROIs', frame)
        
        # Handle key presses
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reset_msg = memory.reset_all_times()
            print(reset_msg)
            
finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
