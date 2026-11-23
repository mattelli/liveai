#!/usr/bin/env bash
# ============================================
# Live Assistant - Linux / macOS
# ============================================

set -euo pipefail

TARGET_PY_VERSION="3.12"
ENVIRONMENT_NAME=".liveai-cpu"

echo "=== Checking Python version ==="
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
if [[ "$PY_VERSION" != "$TARGET_PY_VERSION" ]]; then
    echo "ERROR: Python $PY_VERSION is not supported. Use Python $TARGET_PY_VERSION"
    exit 1
fi
echo "Python version OK: $PY_VERSION"

echo ""
echo "=== Activating $ENVIRONMENT_NAME virtual environment ==="
if [[ ! -f "./$ENVIRONMENT_NAME/bin/activate" ]]; then
    echo "ERROR: $ENVIRONMENT_NAME virtual environment not found."
    echo "Create it with: python3 -m venv $ENVIRONMENT_NAME"
    exit 1
fi
source "./$ENVIRONMENT_NAME/bin/activate"
echo "Active python: $(python -c 'import sys; print(sys.executable)')"

echo ""
echo "=== Upgrading pip ==="
python -m pip install --upgrade pip

echo ""
echo "=== Installing required Python packages ==="
pip install openai
pip install lmstudio
pip install ollama
pip install python-dotenv
pip install piper-tts
pip install faster-whisper
pip install "av==14.2.0"  # faster-whisper uses PyAV APIs incompatible with av>=15 wheels; 14.4+ ships source-only
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
pip install ddgs
pip install py7zr
pip install mcp
pip install uv

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

echo ""
echo "=== Installation complete ==="