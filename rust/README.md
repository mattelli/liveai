# Live Assistant — Rust port

Re-implementation of `main.py` + `src/` in Rust (see `../PLAN.md` for the full
plan and parity checklist). The Python code is frozen; the two versions share
the **same `config.toml`** and the **same `.res/` model folders**.

## Modes

| Command | Behavior |
|---|---|
| `la` | voice loop: mic → silence-threshold capture → Whisper STT → streaming LLM (+tools) → optional Piper TTS. Ctrl+C while waiting quits; Ctrl+C during generation cancels. |
| `la --chat` | interactive chat: line editing, persistent history, inline suggestions, `/`-completion, `/paste`. Ctrl+C clears the line. |
| `la --prompt "..."` | single turn, no stdin reads. |
| `la --no-tools` | disable agent tool usage. |
| `la --config <path>` | use an alternate config.toml. |

Session commands: `/help`, `/quit`, `/clear`, `/voice on|off`, `/tools`, `/info`,
`exit` — plus the bootstrap prompt after the banner, tool launcher with
required-param prompting, and turn-end `context: X.XX % [Nk]` usage line.

## Build

```powershell
cd rust

# Quick build (no native ML backends; encoders/tools answer with hints)
cargo build --release

# Full build with every backend (whisper.cpp needs CMake + MSVC Build Tools;
# sherpa-onnx and pdfium use prebuilt binaries; candle compiles natively)
cargo build --release `
    -p la        --features ml `
    -p la-listen --features whisper `
    -p la-speak  --features tts `
    -p la-imggen --features sd
```

`cargo build` puts all four binaries into the same `target/(debug|release)/`
folder so the main binary can locate its workers.

| Feature | Crate | Backend |
|---|---|---|
| `ml` | `la` / `la-ml` | candle: MiniLM + CLIP + TinyBERT encoders |
| `whisper` | `la-listen` | whisper.cpp (ggml models) |
| `tts` | `la-speak` | sherpa-onnx VITS (Piper voices) |
| `sd` | `la-imggen` | candle Stable Diffusion (txt2img / img2img / inpaint) |

### Validation gate

```powershell
powershell -ExecutionPolicy Bypass -File check.ps1            # fmt + clippy -D warnings + check + test
powershell -ExecutionPolicy Bypass -File check.ps1 -Features  # additionally check feature-enabled variants
```

## Model preparation

Fetch everything the Rust version needs into `.res/` (skips what the Python
setup already stored, `-WhatIf` dry-runs, `-Only <section>` restricts):

```powershell
powershell -ExecutionPolicy Bypass -File setup_rust_res.ps1
```

What it derives from `config.toml`:

| Section | Artifact | Notes |
|---|---|---|
| `[speech_det]` | whisper.cpp **ggml** model → `.res/whisper/ggml-medium.bin` | faster-whisper CT2 models are **not** loadable by whisper.cpp (R1). Wire it via the additive `[rust]` section: `speech_det_model_path = ".res/whisper/ggml-medium.bin"` — Python ignores that section. |
| `[speech_gen]` | Piper voice `.onnx` + `.onnx.json` | same files as the Python install (rhasspy/piper-voices). |
| `[summarization]`, `[embedding]` | all-MiniLM-L6-v2 (safetensors + tokenizer) | candle needs safetensors. |
| `[image_search]`, `[embedding].image_model_path` | clip-ViT-B-32 | repo nests files under `0_CLIPModel/`; the script flattens them and converts the slow tokenizer to `tokenizer.json` via the Python env. |
| `[document_query]` | ms-marco-TinyBERT-L2 | ships `pytorch_model.bin` only (R3): the script downloads it and converts to `model.safetensors` once with the Python env. |
| `[image_gen]` | SD t2i/i2i repos (when the config holds HF ids) | candle file set under `.res/sd/<org>__<name>/`; point the config at that folder for offline use. |
| binaries | `pdfium.dll` (bblanchon/pdfium-binaries) | installed next to the built binaries for `pdfium-render`. |

`.env` support: `[brain].api_key_env` names the variable holding the API key;
values from the `.env` next to the config never override real environment
variables (Python `load_dotenv(override=False)` parity).

## Architecture

Four cooperating processes (PLAN §3), one JSON object per line over stdio
(`la-proto`), every request carrying a monotonic `id`:

```
la (main) ──▶ la-listen   mic + RMS calibration + capture + whisper STT
          ──▶ la-speak    Piper TTS + cpal playback (volume = digital gain)
          ──▶ la-imggen   candle SD, pipeline stays warm across calls
```

* workers spawn on demand (listen/speak at startup in voice mode, imggen at the
  first image request), are restarted when they die, and exit on stdin EOF
  (parent-death watchdog);
* shutdown sends `stop`, waits ≤ 2 s, then kills; on Windows every process runs
  inside a Job object with KILL_ON_JOB_CLOSE, so a crashed main cannot leave
  orphans;
* Ctrl+C during generation closes the HTTP stream, keeps the partial output and
  repairs the history (placeholder padding) — parity with `agent.cancel_generation()`.

The tool set (`crates/la-core/src/tools/`) mirrors Python's `_default_tools`
order; a checked-in snapshot test (`tests/tool_parity.rs`) guards name order
and required-parameter order against the reference.

## Workspace layout

```
rust/
├── Cargo.toml            workspace: crates/* + bin/*
├── check.ps1             fmt + clippy + check + test gate
├── setup_rust_res.ps1    model/binary fetcher (config.toml driven)
├── crates/
│   ├── la-proto/         IPC envelopes + framing (serde_json lines)
│   ├── la-core/          config, agent core, tools, document/ML tool backends
│   ├── la-audio/         cpal devices, RMS, WAV helpers
│   └── la-ml/            candle encoders (behind features)
└── bin/
    ├── la/               main binary (chat/voice/one-shot)
    ├── la-listen/        capture + whisper worker
    ├── la-speak/         Piper TTS worker
    └── la-imggen/        Stable Diffusion worker
```

## Tests

`cargo test --workspace` runs ~130 unit + integration tests, including a fake
OpenAI-compatible SSE server (`wiremock`-style hand-rolled in
`crates/la-core/tests/fake_openai.rs`) covering streaming, tool-call rounds and
cancellation. Heavy-model tests run only behind feature flags with real models
on disk.

## Known deviations from the Python version

* `src/local_model_handler.py` (LM Studio / Ollama model management) is not
  ported — accepted scope decision (PLAN §1, §12 R8).
* Embedding corpora are stored as Rust-native `corpus.bin` instead of Python
  `corpus.pkl` (PLAN §12 R4): same folder layout (`metadata.json`,
  `processed.md`, `images/`), but corpora must be re-`store_corpus`-ed once.
* CLIP scores can differ slightly from `sentence-transformers` (pooling
  details, PLAN §12 R5); the Rust pooling is validated against golden values.