# check.ps1 - one-shot validation gate for the `rust/` workspace (PLAN section 10.1).
#
# Usage (from any directory):
#   powershell -ExecutionPolicy Bypass -File rust\check.ps1
#
# Steps:
#   1. cargo fmt --all -- --check
#   2. cargo clippy --workspace --all-targets -- -D warnings
#   3. cargo check --workspace
#   4. cargo test --workspace
#
# All four steps must pass before a commit. Heavy native backends (whisper,
# sherpa, candle SD, ML encoders) are behind feature flags and are NOT built
# here, so this gate runs on any machine with a plain Rust toolchain.
# Optional: -Features also compiles the feature-enabled variants
# (much slower: whisper.cpp needs CMake + MSVC, candle needs a full build).

param(
    [switch]$Features
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ""
    Write-Host "== $Name ==" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "rust/ workspace validation gate" -ForegroundColor Yellow

Step "cargo fmt --all -- --check" { cargo fmt --all -- --check }
Step "cargo clippy -D warnings" { cargo clippy --workspace --all-targets -- -D warnings }
Step "cargo check --workspace" { cargo check --workspace }
Step "cargo test --workspace" { cargo test --workspace }

if ($Features) {
    Step "cargo check --features (ml/whisper/tts/sd)" {
        cargo check --workspace --features la/ml,la-listen/whisper,la-speak/tts,la-imggen/sd
    }
}

Write-Host ""
Write-Host "All gates passed." -ForegroundColor Green
