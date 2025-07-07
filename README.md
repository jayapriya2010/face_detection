# Face Recognition with ROI Tracking

## Quick Setup (Windows)
1. Run the setup script:
```bash
setup.bat
```

## Manual Installation
If the setup script doesn't work, install manually:
```bash
conda install -c conda-forge faiss-cpu --yes
conda install -c conda-forge opencv numpy --yes
pip install insightface onnxruntime
```

## Usage
1. When program starts, your webcam feed will open
2. Draw a rectangle to select your Region of Interest (ROI)
3. Press SPACE or ENTER to confirm
4. System will track faces and their time spent in ROI
5. Press 'q' to quit
4. Run the script:
```bash
python detect_faces.py
```

5. Run the program:
```bash
python main.py
```

## Usage Instructions
- When the program starts, your webcam will open
- First frame will appear with "Select ROI" window
- Draw a rectangle by clicking and dragging to select your Region of Interest
- Press SPACE or ENTER to confirm the ROI
- The system will then:
  - Track faces in real-time
  - Show green boxes for faces inside ROI
  - Show orange boxes for faces outside ROI
  - Display ID and dwell time for each face
- Press 'q' to quit the application

## Files
- `main.py` - Main program with ROI tracking
- `face_memory.py` - Face recognition and time tracking logic

## Git Setup

The following files are ignored in Git:
- `known_faces/*` - Directory containing reference face images
- `output.jpg` - Generated output images
- Various image files (*.jpg, *.jpeg, *.png)

To initialize the repository:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

Note: Make sure to create the `known_faces` directory locally and add your reference photos before running the script.
- Various image files (*.jpg, *.jpeg, *.png)

To initialize the repository:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

Note: Make sure to create the `known_faces` directory locally and add your reference photos before running the script.
