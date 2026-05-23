import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(BASE_DIR, "app.py")

subprocess.run([
    sys.executable,
    "-m",
    "streamlit",
    "run",
    app_path
])