# ============================================
# Stable Diffusion CPU Windows 11
# No way to install pakages to enable Windows 11 + AMD Radeon (680M)
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

# TOOLS
pip install opencv-python
pip install yfinance
pip install matplotlib
pip install beautifulsoup4



Write-Host ""
Write-Host "=== Installation complete ==="