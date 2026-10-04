# ============================================
# Live Assistant - Windows 11
$target_py_version = "3.12"
$environment_name = ".liveai-cpu"
# ============================================


Write-Host "=== Checking Python version ==="
$pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pyVersion -ne $target_py_version) {
    Write-Host "ERROR: Python $pyVersion is not supported. Use Python $target_py_version"
    exit 1
}
Write-Host "Python version OK: $pyVersion"

Write-Host ""
Write-Host "=== Activating $environment_name virtual environment ==="
if (-not (Test-Path ".\$environment_name\Scripts\Activate.ps1")) {
    Write-Host "ERROR: $environment_name virtual environment not found. Create it with: python -m venv $environment_name"
    exit 1
}
& ".\$environment_name\Scripts\Activate.ps1"
Write-Host "Active python: $(python -c 'import sys; print(sys.executable)')"

Write-Host ""
Write-Host "=== Upgrading pip ==="
python -m pip install --upgrade pip

Write-Host ""
Write-Host "=== Installing required Python packages ==="
pip install openai
pip install lmstudio
pip install ollama
pip install python-dotenv
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
pip install py7zr

# EMBEDDING
pip install -U "sentence-transformers[image]"
pip install nltk
pip install faiss-cpu

# TEXT-TO-IMAGE
# ============================================
# The default <CACHE_DIR> is ~/.cache/huggingface/hub. 
# It is customizable with the cache_dir argument on all methods, 
# or by specifying either HF_HOME or HF_HUB_CACHE environment variable.

# `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files.
# To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator.
# https://learn.microsoft.com/en-us/windows/advanced-settings/developer-mode

# From https://github.com/marcchicoine/runwayml-stable-diffusion-v1-5
# See https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py
# Patch 88 +,- [False for res in result]
# ============================================
pip install diffusers
pip install transformers
pip install accelerate
pip install imageio-ffmpeg
pip install torchvision

# DOCUMENT LOADERS (store_corpus: pdf incl. images, word, excel, html)
pip install pymupdf
pip install python-docx
pip install openpyxl

# DEV
pip install pytest


Write-Host ""
Write-Host "=== Installation complete ==="