# ============================================
# Live Assistant - Windows 11
$target_py_version = "3.12"
# ============================================


Write-Host "=== Checking Python version ==="
$pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pyVersion -ne $target_py_version) {
    Write-Host "ERROR: Python $pyVersion is not supported. Use Python $target_py_version"
    exit 1
}
Write-Host "Python version OK: $pyVersion"

Write-Host ""
Write-Host "=== Upgrading pip ==="
python -m pip install --upgrade pip

Write-Host ""
Write-Host "=== Installing required Python packages ==="
pip install lmstudio
pip install piper-tts
pip install faster-whisper
pip install soundfile
pip install sounddevice
pip install numpy

# CLI INPUT (line editing, persistent history, multi-line paste)
pip install prompt_toolkit

# TOOLS
pip install opencv-python
pip install yfinance
pip install matplotlib
pip install requests
pip install beautifulsoup4
pip install selenium
pip install webdriver-manager

# EMBEDDING
pip install -U "sentence-transformers[image]"
pip install nltk
pip install faiss-cpu

# DEV
pip install pytest


Write-Host ""
Write-Host "=== Installation complete ==="