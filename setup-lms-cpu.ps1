# ============================================
# Live Assistant - Windows 11
$target_py_version = "3.12"
$environment_name = ".lms"
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
pip install lmstudio
pip install openai
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

# EMBEDDING
pip install -U "sentence-transformers[image]"
pip install nltk
pip install faiss-cpu

# DOCUMENT LOADERS (store_corpus: pdf incl. images, word, excel, html)
pip install pymupdf
pip install python-docx
pip install openpyxl

# DEV
pip install pytest


Write-Host ""
Write-Host "=== Applying local lmstudio SDK patch (act: on_prediction_stream) ==="
# Exposes each prediction round's stream in .act() so responses can be
# cancelled cleanly with Ctrl+C. Re-run this block after any `pip install
# lmstudio` (a reinstall reverts the patch); it fails harmlessly if already
# applied.
$sdkPatch = Join-Path $PSScriptRoot "patches\lmstudio-1.5.0-act-on_prediction_stream.patch"
if (Test-Path $sdkPatch) {
    git -C $PSScriptRoot apply --directory="$environment_name\Lib\site-packages" --check $sdkPatch 2>$null
    if ($LASTEXITCODE -eq 0) {
        git -C $PSScriptRoot apply --directory="$environment_name\Lib\site-packages" $sdkPatch
        Write-Host "SDK patch applied."
    } else {
        Write-Host "SDK patch not applicable (already applied, or SDK version changed) - skipping."
    }
} else {
    Write-Host "No SDK patch file found - skipping."
}


Write-Host ""
Write-Host "=== Installation complete ==="