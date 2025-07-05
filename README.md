# Face Recognition Setup

1. Install required packages in your Conda environment:
```bash
# Remove any existing opencv installations
conda remove opencv-python opencv --yes
conda clean --all --yes

# Install packages
conda install -c conda-forge opencv --yes
conda install -c conda-forge insightface numpy --yes
```

2. Create directory structure:
```bash
mkdir known_faces
```

3. Add reference photos:
- Put clear face photos of known people in the `known_faces` folder
- Name each photo as the person's name (e.g., "john.jpg", "mary.jpg")
- Each photo should contain only one person's face

4. Run the script:
```bash
python detect_faces.py
```

## Usage
- The webcam will automatically open and start detecting faces
- Known faces will be labeled with their names
- Unknown faces will be labeled as "Unknown"
- Green boxes will appear around detected faces
- Press 'q' to quit the application

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
Note: Make sure to create the `known_faces` directory locally and add your reference photos before running the script.
