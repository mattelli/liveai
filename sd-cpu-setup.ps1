# ============================================
# Stable Diffusion CPU Windows 11
# No way to install pakages to enable Windows 11 + AMD Radeon (680M)
$target_py_version = "3.12"
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
pip install diffusers
pip install transformers
pip install accelerate
pip install imageio-ffmpeg
pip install torchvision

Write-Host ""
Write-Host "=== Installation complete ==="