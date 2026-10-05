# setup_rust_res.ps1 - download every model/binary the Rust version needs into
# the repo's `.res/` folder (PLAN section 13.4). The required list is DERIVED
# from `config.toml` (each section's model keys) plus the Rust-specific needs
# table below, and artifacts that are already present are skipped.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File rust\setup_rust_res.ps1          # fetch what's missing
#   powershell -ExecutionPolicy Bypass -File rust\setup_rust_res.ps1 -WhatIf  # dry run: list actions
#   powershell -ExecutionPolicy Bypass -File rust\setup_rust_res.ps1 -Only speech_det
#   powershell -ExecutionPolicy Bypass -File rust\setup_rust_res.ps1 -Force    # re-download everything
#
# Sections (also accepted by -Only):
#   speech_det     whisper.cpp ggml model (Rust cannot read faster-whisper CT2 models, R1)
#   speech_gen     Piper voice .onnx + .onnx.json (sherpa-onnx VITS, same files as Python)
#   summarization  all-MiniLM-L6-v2 (safetensors + config + tokenizer)
#   document_query ms-marco-TinyBERT-L2 (+ one-time safetensors conversion, R3)
#   image_search   clip-ViT-B-32
#   embedding      all-MiniLM-L6-v2 + clip-ViT-B-32 (shared)
#   image_gen      Stable Diffusion models when [image_gen] holds HuggingFace ids
#   binaries       pdfium.dll (pdfium-render needs it next to the built binaries)
#
# Offline behaviour: artifacts already stored by the Python setup (faster-whisper
# aside, whose CT2 format is not loadable by whisper.cpp) are reused as-is.

param(
    [string]$Config = "",            # config.toml path; default: ../config.toml next to this script
    [string]$Only = "",              # restrict to one section (see header)
    [switch]$WhatIf,                 # dry run: list what would be fetched
    [switch]$Force,                  # re-download even when the artifact exists
    [string]$WhisperModel = "ggml-medium.bin"  # ggml file from ggerganov/whisper.cpp
)

$ErrorActionPreference = 'Continue'
# Windows PowerShell 5.1 defaults to pre-TLS1.2 protocols; HuggingFace needs TLS 1.2+.
try { [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor [System.Net.SecurityProtocolType]::Tls12 } catch {}
$repoRoot = Split-Path -Parent $PSScriptRoot          # live_assistant/
if ($Config -eq "" -or -not (Test-Path $Config)) {
    $Config = Join-Path $repoRoot "config.toml"
}
if (-not (Test-Path $Config)) {
    Write-Host "ERROR: config not found: $Config" -ForegroundColor Red
    exit 1
}
$resDir = Join-Path $repoRoot ".res"

# ---------------------------------------------------------------- helpers

function Parse-Toml {
    # Minimal TOML reader for the flat `key = value` sections of config.toml.
    param([string]$Path)
    $cfg = @{}
    $section = ""
    foreach ($line in (Get-Content -Path $Path)) {
        $t = $line.Trim()
        if ($t -eq "" -or $t.StartsWith("#")) { continue }
        if ($t -match '^\[(.+?)\]') {
            $section = $Matches[1].Trim()
            if (-not $cfg.ContainsKey($section)) { $cfg[$section] = @{} }
            continue
        }
        if ($t -match '^([A-Za-z0-9_-]+)\s*=\s*(.*)$') {
            $key = $Matches[1]
            $val = $Matches[2].Trim()
            if ($val -match '^"(.*)"') {
                $val = $Matches[1]                       # quoted string (last quote wins)
            } else {
                $val = ($val -replace '\s*#.*$', '').Trim()  # strip inline comment
            }
            if (-not $cfg.ContainsKey($section)) { $cfg[$section] = @{} }
            $cfg[$section][$key] = $val
        }
    }
    return $cfg
}

function Resolve-ConfigPath {
    # Mirror `load_config` (src/common.py): *_file / *_dir / *_path values and
    # list items are resolved relative to the config file's directory.
    param([string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) { return $Value }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Value))
}

function Invoke-Download {
    param([string]$Url, [string]$Out, [string]$Label)
    $dir = Split-Path -Parent $Out
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    if ($WhatIf) {
        Write-Host "  [what-if] $Label" -ForegroundColor Yellow
        Write-Host "            $Url" -ForegroundColor DarkGray
        Write-Host "        ->  $Out" -ForegroundColor DarkGray
        return $true
    }
    Write-Host "  downloading $Label" -ForegroundColor Cyan
    Write-Host "            $Url" -ForegroundColor DarkGray
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Out -UseBasicParsing
        Write-Host "        ->  $Out" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "        FAILED: $_" -ForegroundColor Red
        return $false
    }
}

function Test-UrlExists {
    param([string]$Url)
    try {
        $req = [System.Net.WebRequest]::Create($Url)
        $req.Method = "HEAD"
        $req.Timeout = 15000
        $resp = $req.GetResponse()
        $resp.Close()
        return $true
    } catch { return $false }
}

function Get-HfFileSet {
    # Download a set of files from a HuggingFace repo; skip 404s gracefully.
    # -Prefix maps a repo subfolder (e.g. `0_CLIPModel/`) onto the destination
    # root, mirroring how sentence-transformers nests the CLIPModel files.
    param([string]$Repo, [string]$Revision, [string]$Dest, [string[]]$Files, [string]$Label, [string]$Prefix = "")
    Write-Host "== $Label ($Repo) ==" -ForegroundColor Yellow
    $ok = $true
    foreach ($f in $Files) {
        $url = "https://huggingface.co/$Repo/resolve/$Revision/$Prefix$f"
        $out = Join-Path $Dest ($f -replace '/', '\')
        if ((Test-Path $out) -and -not $Force) {
            Write-Host "  present: $out" -ForegroundColor DarkGray
            continue
        }
        if (-not (Test-UrlExists $url)) {
            Write-Host "  not in repo (skipped): $f" -ForegroundColor DarkYellow
            continue
        }
        if (-not (Invoke-Download -Url $url -Out $out -Label $f)) { $ok = $false }
    }
    return $ok
}
$script:hadFailure = $false
function Register-Failure {
    param([string]$Message)
    $script:hadFailure = $true
    Write-Host "  MISSING: $Message" -ForegroundColor Red
}

# ---------------------------------------------------------------- parse config

$configDir = Split-Path -Parent (Resolve-Path $Config)
$cfg = Parse-Toml -Path $Config

function Get-CfgPath {
    param([string]$Section, [string]$Key)
    $v = Get-CfgValue -Section $Section -Key $Key
    if ($v) { return Resolve-ConfigPath -Value $v }
    return ""
}

function Get-CfgValue {
    # Safe access: sections/keys are optional ([rust] may be absent entirely).
    param([string]$Section, [string]$Key)
    if ($cfg.ContainsKey($Section) -and $cfg[$Section] -and $cfg[$Section].ContainsKey($Key)) {
        return $cfg[$Section][$Key]
    }
    return $null
}

function Invoke-Section {
    param([string]$Name, [scriptblock]$Body)
    if ($Only -ne "" -and $Only -ne $Name) { return }
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Body
}

$onlyValid = @("speech_det","speech_gen","summarization","document_query","image_search","embedding","image_gen","binaries")
if ($Only -ne "" -and $onlyValid -notcontains $Only) {
    Write-Host "ERROR: -Only must be one of: $($onlyValid -join ', ')" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------- speech_det

Invoke-Section "speech_det" {
    # R1: faster-whisper (CT2) models are NOT loadable by whisper.cpp. Rust
    # needs a ggml .bin; config points at one via the additive [rust] section,
    # otherwise we place ggml-medium.bin under .res/whisper/.
    $rustOverride = Get-CfgValue -Section "rust" -Key "speech_det_model_path"
    if ($rustOverride) {
        $target = Resolve-ConfigPath -Value $rustOverride
    } else {
        $target = Join-Path $resDir "whisper\$WhisperModel"
    }
    $haveModel = $false
    if (Test-Path $target -PathType Leaf) { $haveModel = $true }
    elseif (Test-Path $target -PathType Container) {
        $haveModel = ((Get-ChildItem $target -Filter *.bin -ErrorAction SilentlyContinue) | Measure-Object).Count -gt 0
    }
    if (-not $haveModel) {
        $url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$WhisperModel"
        if (-not (Invoke-Download -Url $url -Out $target -Label $WhisperModel)) {
            Register-Failure "whisper ggml model ($WhisperModel); later wire it via [rust] speech_det_model_path"
        }
    } else {
        Write-Host "  present: $target" -ForegroundColor DarkGray
    }
}

# ---------------------------------------------------------------- speech_gen

Invoke-Section "speech_gen" {
    # Piper voices are the SAME files as the Python version (rhasspy/piper-voices).
    $voice = Get-CfgPath -Section "speech_gen" -Key "model_path"
    if (-not $voice) { Register-Failure "[speech_gen] model_path missing from config"; return }
    if ((Test-Path "$voice") -and (Test-Path "$voice.json") -and -not $Force) {
        Write-Host "  present: $voice (+json)" -ForegroundColor DarkGray
        return
    }
    # en_US-hfc_female-medium.onnx -> en/en_US/hfc_female/medium/<file>
    $name = [System.IO.Path]::GetFileName($voice)
    if ($name -match '^([a-z]{2})_([A-Z]{2})-(.+)-([a-z]+)\.onnx$') {
        $lang = $Matches[1]; $region = $Matches[2]; $speaker = $Matches[3]; $quality = $Matches[4]
        $base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/$lang/${lang}_$region/$speaker/$quality/$name"
        if (-not (Invoke-Download -Url $base -Out $voice -Label $name)) { Register-Failure "piper voice $name" }
        if (-not (Invoke-Download -Url "$base.json" -Out "$voice.json" -Label "$name.json")) { Register-Failure "piper voice config $name.json" }
    } else {
        Write-Host "  cannot derive the piper-voices URL from '$name'; download .onnx + .onnx.json manually into $voice" -ForegroundColor Yellow
        Register-Failure "piper voice $name (manual)"
    }
}

function Invoke-TokenizerConversion {
    # Some repos only ship slow tokenizers (vocab.json + merges.txt); candle's
    # `tokenizers` crate needs a fast tokenizer.json. Convert once via the
    # Python env (transformers fast-tokenizer conversion).
    param([string]$Dir, [string]$Label)
    if (Test-Path (Join-Path $Dir "tokenizer.json")) {
        Write-Host "  present: tokenizer.json" -ForegroundColor DarkGray
        return
    }
    if (-not ((Test-Path (Join-Path $Dir "vocab.json")) -or (Test-Path (Join-Path $Dir "vocab.txt")))) {
        Write-Host "  cannot convert tokenizer (no vocab files) - skipping" -ForegroundColor DarkYellow
        return
    }
    if ($WhatIf) {
        Write-Host "  [what-if] convert slow tokenizer -> tokenizer.json ($Label)" -ForegroundColor Yellow
        return
    }
    $pyCandidates = @(
        (Join-Path $repoRoot ".liveai-cpu\Scripts\python.exe"),
        "python"
    )
    $py = $pyCandidates | Where-Object { Test-Path $_ -ErrorAction SilentlyContinue } | Select-Object -First 1
    if (-not $py) { Register-Failure "$Label conversion: no Python env (.liveai-cpu)"; return }
    Write-Host "  converting slow tokenizer -> tokenizer.json with $py ..." -ForegroundColor Cyan
    $code = "from transformers import AutoTokenizer; import sys; d=sys.argv[1]; AutoTokenizer.from_pretrained(d, use_fast=True).save_pretrained(d); print('ok')"
    & $py -c $code $Dir | Out-Null
    if ($LASTEXITCODE -eq 0 -and (Test-Path (Join-Path $Dir "tokenizer.json"))) {
        Write-Host "  converted -> tokenizer.json" -ForegroundColor Green
    } else {
        Register-Failure "$Label conversion failed"
    }
}

# ---------------------------------------------------------------- sentence-transformers models

$stEssential = @(
    "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
    "vocab.txt", "special_tokens_map.json", "1_Pooling/config.json"
)

Invoke-Section "summarization" {
    $mp = Get-CfgPath -Section "summarization" -Key "model_path"
    if (-not $mp) { Register-Failure "[summarization] model_path missing"; return }
    Get-HfFileSet -Repo "sentence-transformers/all-MiniLM-L6-v2" -Revision "main" -Dest $mp `
        -Files $stEssential -Label "all-MiniLM-L6-v2" | Out-Null
}

Invoke-Section "image_search" {
    $mp = Get-CfgPath -Section "image_search" -Key "model_path"
    if (-not $mp) { Register-Failure "[image_search] model_path missing"; return }
    # The repo nests the raw CLIPModel under 0_CLIPModel/ and has no
    # tokenizer.json; we flatten the files and convert the tokenizer once.
    Get-HfFileSet -Repo "sentence-transformers/clip-ViT-B-32" -Revision "main" -Dest $mp `
        -Files @("config.json", "model.safetensors", "tokenizer_config.json", "vocab.json", "special_tokens_map.json", "merges.txt", "preprocessor_config.json") `
        -Label "clip-ViT-B-32" -Prefix "0_CLIPModel/" | Out-Null
    Invoke-TokenizerConversion -Dir $mp -Label "clip-ViT-B-32 tokenizer.json"
}


Invoke-Section "embedding" {
    $mp = Get-CfgPath -Section "embedding" -Key "model_path"
    if ($mp) {
        Get-HfFileSet -Repo "sentence-transformers/all-MiniLM-L6-v2" -Revision "main" -Dest $mp `
            -Files $stEssential -Label "all-MiniLM-L6-v2" | Out-Null
    }
    $imp = Get-CfgPath -Section "embedding" -Key "image_model_path"
    $multimodal = (Get-CfgValue -Section "embedding" -Key "multimodal") -eq "true"
    if ($imp -and ($imp -ne $mp -or $multimodal)) {
        Get-HfFileSet -Repo "sentence-transformers/clip-ViT-B-32" -Revision "main" -Dest $imp `
            -Files @("config.json", "model.safetensors", "tokenizer_config.json", "vocab.json", "special_tokens_map.json", "merges.txt", "preprocessor_config.json") `
            -Label "clip-ViT-B-32" -Prefix "0_CLIPModel/" | Out-Null
        Invoke-TokenizerConversion -Dir $imp -Label "clip-ViT-B-32 tokenizer.json"
    }
}

# ---------------------------------------------------------------- document_query

Invoke-Section "document_query" {
    # R3: ms-marco-TinyBERT-L2 ships pytorch_model.bin only; candle reads
    # safetensors, so after the download we convert once via the Python env.
    $mp = Get-CfgPath -Section "document_query" -Key "model_path"
    if (-not $mp) { Register-Failure "[document_query] model_path missing"; return }
    Get-HfFileSet -Repo "cross-encoder/ms-marco-TinyBERT-L2" -Revision "main" -Dest $mp `
        -Files @("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json") `
        -Label "ms-marco-TinyBERT-L2" | Out-Null

    if (Test-Path (Join-Path $mp "model.safetensors")) {
        Write-Host "  present: model.safetensors (no conversion needed)" -ForegroundColor DarkGray
        return
    }
    $bin = Join-Path $mp "pytorch_model.bin"
    if (-not (Test-Path $bin)) {
        $url = "https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2/resolve/main/pytorch_model.bin"
        if (-not (Invoke-Download -Url $url -Out $bin -Label "pytorch_model.bin")) {
            Register-Failure "cross-encoder weights (pytorch_model.bin)"
            return
        }
    } else {
        Write-Host "  present: pytorch_model.bin" -ForegroundColor DarkGray
    }
    if ($WhatIf) {
        Write-Host "  [what-if] safetensors conversion via Python env" -ForegroundColor Yellow
        return
    }
    # Convert with the Python env (torch + transformers + safetensors).
    $pyCandidates = @(
        (Join-Path $repoRoot ".liveai-cpu\Scripts\python.exe"),
        "python"
    )
    $py = $pyCandidates | Where-Object { Test-Path $_ -ErrorAction SilentlyContinue } | Select-Object -First 1
    if (-not $py) {
        Register-Failure "safetensors conversion: no Python env found (.liveai-cpu); run:"
        Write-Host '    python -c "from transformers import AutoModelForSequenceClassification; import sys; d=r\'''"'$mp'\'''"'; m=AutoModelForSequenceClassification.from_pretrained(d); m.save_pretrained(d, safe_serialization=True)"' -ForegroundColor Yellow
        return
    }
    Write-Host "  converting weights to safetensors with $py ..." -ForegroundColor Cyan
    $code = "from transformers import AutoModelForSequenceClassification; import sys; d=sys.argv[1]; m=AutoModelForSequenceClassification.from_pretrained(d); m.save_pretrained(d, safe_serialization=True); print('ok')"
    & $py -c $code $mp
    if ($LASTEXITCODE -eq 0 -and (Test-Path (Join-Path $mp "model.safetensors"))) {
        Write-Host "  converted -> model.safetensors" -ForegroundColor Green
    } else {
        Register-Failure "safetensors conversion failed for $mp"
    }
}

# ---------------------------------------------------------------- image_gen

Invoke-Section "image_gen" {
    # Candle reads a curated file set from a diffusers repo. When the config
    # holds a HuggingFace id we fetch those files into .res/sd/<org>__<name>/;
    # point [image_gen] at that folder for fully offline use.
    $sdFiles = @(
        "model_index.json",
        "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json", "text_encoder/model.safetensors",
        "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json",
        "scheduler/scheduler_config.json"
    )
    foreach ($key in @("t2i_model", "i2i_model")) {
        $v = Get-CfgValue -Section "image_gen" -Key $key
        if (-not $v) { continue }
        $asPath = Resolve-ConfigPath -Value $v
        if (Test-Path $asPath) {
            Write-Host "  present (local path): $asPath" -ForegroundColor DarkGray
            continue
        }
        if ($v -match '^[^/\\]+/[^/\\]+$') {
            $dest = Join-Path $resDir "sd\$($v -replace '/', '__')"
            Get-HfFileSet -Repo $v -Revision "main" -Dest $dest -Files $sdFiles -Label "$key ($v)" | Out-Null
        } else {
            Write-Host "  '$v' is neither an existing path nor an HF id; skipped" -ForegroundColor DarkYellow
        }
    }
}

# ---------------------------------------------------------------- binaries

Invoke-Section "binaries" {
    # pdfium-render loads pdfium.dll at runtime; place it next to the built
    # binaries (target/release, target/debug) so `la` can find it. The release
    # asset is a .tgz containing bin/pdfium.dll (extract with bsdtar/tar).
    $tgzUrl = "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-win-x64.tgz"
    $tgz = Join-Path $env:TEMP "pdfium-win-x64.tgz"
    $targets = @(
        (Join-Path $PSScriptRoot "target\release"),
        (Join-Path $PSScriptRoot "target\debug")
    )
    $need = $targets | Where-Object { -not (Test-Path (Join-Path $_ "pdfium.dll")) }
    if ($need.Count -eq 0 -and -not $Force) {
        Write-Host "  present: pdfium.dll in all target dirs" -ForegroundColor DarkGray
        return
    }
    if ($WhatIf) {
        Write-Host "  [what-if] $tgzUrl -> pdfium.dll into $($need -join ', ')" -ForegroundColor Yellow
        return
    }
    if (-not (Invoke-Download -Url $tgzUrl -Out $tgz -Label "pdfium-win-x64.tgz")) {
        Register-Failure "pdfium.dll download"; return
    }
    $extractDir = Join-Path $env:TEMP "pdfium-extract"
    if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
    New-Item -ItemType Directory -Path $extractDir -Force | Out-Null
    # Windows ships bsdtar (System32); resolve it explicitly so an MSYS/Git-Bash
    # tar (first on PATH in dev shells) cannot mis-parse the Windows path.
    $tarExe = Join-Path $env:SystemRoot "System32\tar.exe"
    if (-not (Test-Path $tarExe)) { $tarExe = "tar" }
    & $tarExe -xzf $tgz -C $extractDir
    if ($LASTEXITCODE -ne 0) { Register-Failure "could not extract $tgz"; return }
    $dll = Get-ChildItem $extractDir -Recurse -Filter "pdfium.dll" | Select-Object -First 1
    if (-not $dll) { Register-Failure "pdfium.dll not inside the release archive"; return }
    foreach ($dir in $need) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Copy-Item $dll.FullName (Join-Path $dir "pdfium.dll") -Force
        Write-Host "  installed: $(Join-Path $dir 'pdfium.dll')" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------- summary

Write-Host ""
if ($script:hadFailure) {
    Write-Host "Setup finished WITH missing artifacts - see the MISSING lines above." -ForegroundColor Yellow
    exit 2
}
Write-Host "Setup finished: all required artifacts present." -ForegroundColor Green
