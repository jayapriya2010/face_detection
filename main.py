import sys
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from face_memory import FaceMemory
import os
from ultralytics import YOLO
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

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
        # Initialize hand ROI coordinates as None
        self.hand_roi_x = None
        self.hand_roi_y = None
        self.hand_roi_w = None
        self.hand_roi_h = None
        self.working_time_start = None
        self.total_working_time = 0
        self.current_working_time = 0  # Add this new property

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

def select_hand_sub_roi(frame, main_roi):
    print(f"\nSelect hand detection sub-ROI for {main_roi.name}")
    print("This should be within the main ROI")
    # Create a cropped frame for the main ROI
    roi_frame = frame[main_roi.y:main_roi.y + main_roi.h, 
                     main_roi.x:main_roi.x + main_roi.w]
    
    if roi_frame.size > 0:
        sub_roi = cv2.selectROI("Select Hand Detection Area", roi_frame, False)
        cv2.destroyWindow("Select Hand Detection Area")
        
        # Convert sub-ROI coordinates relative to main ROI to global coordinates
        main_roi.hand_roi_x = main_roi.x + sub_roi[0]
        main_roi.hand_roi_y = main_roi.y + sub_roi[1]
        main_roi.hand_roi_w = sub_roi[2]
        main_roi.hand_roi_h = sub_roi[3]

# ROI selection
def select_multiple_rois(frame):
    rois = []
    while True:
        # Select ROI
        print("\nSelect main ROI and press SPACE or ENTER when done")
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
        
        # Select hand detection sub-ROI
        select_hand_sub_roi(frame, new_roi)
        
        # Get authorized people for this ROI
        print(f"\nEnter authorized people for {name} (one name per line)")
        print("Press Enter twice when done:")
        while True:
            person = input().strip()
            if not person:
                break
            new_roi.authorized_people.append(person)
        
        rois.append(new_roi)
        
        # Draw existing ROIs and their sub-ROIs
        temp_frame = frame.copy()
        for r in rois:
            # Draw main ROI
            cv2.rectangle(temp_frame, (r.x, r.y), (r.x + r.w, r.y + r.h), (255, 0, 0), 2)
            cv2.putText(temp_frame, r.name, (r.x, r.y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw hand detection sub-ROI
            if r.hand_roi_x is not None:
                cv2.rectangle(temp_frame, 
                            (r.hand_roi_x, r.hand_roi_y),
                            (r.hand_roi_x + r.hand_roi_w, r.hand_roi_y + r.hand_roi_h),
                            (0, 255, 255), 2)
        
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

print("\nPress ESC to start detection...")
while cv2.waitKey(1) & 0xFF != 27:  # Wait for ESC key
    # Show the frame with ROIs
    temp_frame = first_frame.copy()
    for r in rois:
        cv2.rectangle(temp_frame, (r.x, r.y), (r.x + r.w, r.y + r.h), (255, 0, 0), 2)
        if r.hand_roi_x is not None:
            cv2.rectangle(temp_frame, 
                        (r.hand_roi_x, r.hand_roi_y),
                        (r.hand_roi_x + r.hand_roi_w, r.hand_roi_y + r.hand_roi_h),
                        (0, 255, 255), 2)
    cv2.imshow('Face Recognition with ROIs', temp_frame)

detection_active = True
try:
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000/fps) if fps > 0 else 30

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        # Add instructions text
        
        if detection_active:
            # Run YOLO detection for people
            results = model(frame)[0]
            # Filter only person class (class 0 in COCO dataset)
            people_detections = [det for det in results.boxes.data if int(det[5]) == 0 and float(det[4]) > 0.3]  # class 0 is person, confidence > 0.3

            # Count people in each ROI
            roi_counts = {roi.name: 0 for roi in rois}
            for detection in people_detections:
                x1, y1, x2, y2, conf, class_id = detection.int().cpu().numpy()  # Get detection details
                person_area = (x2 - x1) * (y2 - y1)
                
                # Draw person detection box for debugging
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Yellow box around detected people
                
                # Check overlap with each ROI
                for roi in rois:
                    # Calculate intersection coordinates
                    ix1 = max(x1, roi.x)
                    iy1 = max(y1, roi.y)
                    ix2 = min(x2, roi.x + roi.w)
                    iy2 = min(y2, roi.y + roi.h)
                    
                    # Calculate overlap area
                    if ix2 > ix1 and iy2 > iy1:
                        overlap_area = (ix2 - ix1) * (iy2 - iy1)
                        overlap_percentage = overlap_area / person_area
                        
                        # Only count if more than 50% of person is inside ROI
                        if overlap_percentage > 0.5:
                            roi_counts[roi.name] += 1
                            # Draw overlap indicator for debugging
                            cv2.putText(frame, f"{overlap_percentage:.2f}", 
                                      (x1, y1 - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (255, 255, 0), 1)

            # Draw all ROIs with overcrowding indication
            for roi in rois:
                # Update current working time if hands are detected
                if roi.working_time_start is not None:
                    roi.current_working_time = roi.total_working_time + (time.time() - roi.working_time_start)
                else:
                    roi.current_working_time = roi.total_working_time

                # Red if overcrowded (more than 1 person), blue otherwise
                color = (0, 0, 255) if roi_counts[roi.name] > 1 else (255, 0, 0)
                cv2.rectangle(frame, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), color, 2)
                
                # Draw working time in outer ROI with better visibility
                cv2.putText(frame, 
                          f"Working Time: {roi.current_working_time:.1f}s",
                          (roi.x + 5, roi.y + 25),  # Adjusted position
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,  # Slightly larger font
                          (0, 255, 0),  # Bright green color
                          2)  # Thicker font

                # Draw hand detection sub-ROI (yellow)
                if roi.hand_roi_x is not None:
                    cv2.rectangle(frame, 
                                (roi.hand_roi_x, roi.hand_roi_y),
                                (roi.hand_roi_x + roi.hand_roi_w, roi.hand_roi_y + roi.hand_roi_h),
                                (0, 255, 255), 1)

                    # Process hand detection in the sub-ROI
                    roi_frame = frame[roi.hand_roi_y:roi.hand_roi_y + roi.hand_roi_h, 
                                    roi.hand_roi_x:roi.hand_roi_x + roi.hand_roi_w]
                    
                    if roi_frame.size > 0:
                        rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(rgb_frame)

                        if results.multi_hand_landmarks:
                            # Hand detected - start or continue tracking working time
                            if roi.working_time_start is None:
                                roi.working_time_start = time.time()
                            
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_draw.draw_landmarks(
                                    roi_frame,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS
                                )
                        else:
                            # No hand detected - update total time if was tracking
                            if roi.working_time_start is not None:
                                roi.total_working_time += time.time() - roi.working_time_start
                                roi.working_time_start = None

                # Draw ROI name and count (moved after working time)
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
        else:
            # Just show the frame with ROIs when detection is paused
            for roi in rois:
                cv2.rectangle(frame, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (255, 0, 0), 2)
                if roi.hand_roi_x is not None:
                    cv2.rectangle(frame, 
                                (roi.hand_roi_x, roi.hand_roi_y),
                                (roi.hand_roi_x + roi.hand_roi_w, roi.hand_roi_y + roi.hand_roi_h),
                                (0, 255, 255), 2)

        # Show detection status only when paused
        if not detection_active:
            cv2.putText(frame, "Paused",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 0, 255), 2)

        cv2.imshow('Face Recognition with ROIs', frame)
        
        # Handle key presses
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            memory.reset_all_times()
            # Reset working times for all ROIs
            for roi in rois:
                roi.working_time_start = None
                roi.total_working_time = 0
            print("All times reset")
        elif key == 27:  # ESC key
            detection_active = not detection_active
            print(f"Detection {'started' if detection_active else 'paused'}")
            
finally:
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
