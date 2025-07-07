@echo off
echo Installing dependencies...
call conda install -c conda-forge faiss-cpu --yes
call conda install -c conda-forge opencv --yes
call conda install numpy --yes
call pip install insightface
call pip install onnxruntime
echo Setup complete!
pause
