# Face Recognition Setup

1. Install required packages in your Conda environment:
```bash
conda install -c conda-forge insightface opencv numpy
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

Note: Before running, replace "your_image.jpg" in the script with the path to the image you want to analyze.
5. Run the script:
```bash
python detect_faces.py
```

Note: Before running, replace "your_image.jpg" in the script with the path to the image you want to analyze.

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
