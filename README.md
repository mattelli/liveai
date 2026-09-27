# Live Assistant

Simple example to create locally a live AI assistant.

Mic → Whisper (STT) → LM Studio (LLM) → Piper (TTS)


## Whisper
Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. 
See [Whisper](https://huggingface.co/openai/whisper-medium)

```
git lfs install
git clone https://huggingface.co/guillaumekln/faster-whisper-medium
```

## LM Studio SDK
Run AI models, locally and privately. See [LM Studio](https://lmstudio.ai/)
`lmstudio-python` provides a set APIs to interact with LLMs, embeddings models, and agentic flows.

### CLI 
To interact with the LM Studio Server use `lms -h` in the CLI.


## Piper
Piper is a local neural text-to-speech engine, see [Piper](https://github.com/OHF-Voice/piper1-gpl/tree/main)

```
python3 -m piper.download_voices en_US-hfc_female-medium
```


## Diffusers
Text to Image and Image to Image generation through Diffusion pipelines.

Considered models

- runwayml/stable-diffusion-v1-5
- UnfilteredAI/NSFW-gen-v2


## Text summarization

The `summarize_text` tool reads a **text file** and condenses it into its most
central sentences using sentence embeddings (`all-MiniLM-L6-v2`) and LexRank.
Call it with a file path (e.g. `summarize_text(file="notes.txt")`); the file
content is read as UTF-8. The model is loaded lazily on first tool use from a
local folder configured in `config.toml`:

```toml
[summarization]
model_path = ".res/sentence-transformers/all-MiniLM-L6-v2"  # resolved relative to config.toml
device = "cpu"            # "cpu" or "cuda"
ratio = 0.1               # fraction of sentences to keep (0 < ratio <= 1)
threshold = 0.5           # LexRank similarity threshold in [0, 1)
max_sentences = 0         # 0 -> derive from ratio; >0 -> hard cap
```

The tool is disabled/fails gracefully with a clear message until the model is
stored in `model_path`. Pick one of the following ways to store it:

```bash
# A) Hugging Face CLI (recommended)
pip install -U huggingface_hub
hf download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir .res/sentence-transformers/all-MiniLM-L6-v2

# B) git + git-lfs
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 \
    .res/sentence-transformers/all-MiniLM-L6-v2

# C) Python (downloads into the HF cache, then saves a local copy)
python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('.res/sentence-transformers/all-MiniLM-L6-v2')"
```

Notes:

- `model_path` is resolved relative to `config.toml`; it must be a folder
  containing `modules.json`/`config_sentence_transformers.json` and the model
  weights (`pytorch_model.bin` or `model.safetensors`).
- `.res/` is git-ignored, so downloaded weights are never committed.
- Sentence splitting uses NLTK's `punkt_tab` data when available and falls back
  to a regex splitter otherwise, so no runtime download is required.
- Tune the summary length with `ratio` or the per-call `max_sentences` argument.


## Setup (Python 3.12.10)

OS Windows 11 (powershell)

```
# Ai agent based on LM Studio API
.\setup-lms-cpu.ps1

# Stable-Diffusers pipeline
.\setup-sd-cpu.ps1  

```


## Interactive chat input

`python main.py --chat` uses [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/)
as its input layer, so the same behaviour works on Windows and POSIX:

- **Line editing** — arrows, Home/End, word operations, Ctrl-K/U/W, selection.
- **Persistent history** — Up/Down recalls previous turns, and history survives
  across sessions. It is stored in `.chat_history` (git-ignored). Delete the file
  to reset; it has no size cap.
- **Inline history suggestions** — faint grey text suggests a previous entry;
  press Right/End to accept.
- **Multi-line paste** — pasting a block inserts it as one buffer; Enter submits
  the whole block as a single turn (never split into one turn per line). A paste
  whose first line looks like `/quit` is sent to the model, not executed.
- **Completion** — type `/` and press Tab to complete session commands.
- **Explicit paste fallback** — on terminals that do not send bracketed-paste
  markers, start a line with `/paste`, type the block, then a line containing
  only `.` to submit it.

Configure via the `[input]` block in `config.toml` (all keys optional):

```toml
[input]
history_file = ".chat_history"   # relative to config.toml; empty/omitted -> in-memory
prompt_toolkit = true            # false -> builtin input() fallback
auto_suggest = true
completion = true
explicit_paste = true
multiline = false                # true -> Enter inserts newline, Esc+Enter accepts
```

`Ctrl-C` exits the loop cleanly (exit code 0); `Ctrl-D` (EOF) exits via the EOF
path. If prompt_toolkit is missing or disabled, input falls back to the builtin
`input()` with no line editing or history.


## System Notes

```
CPU - AMD Ryzen 7 7735HS with Radeon Graphics - 8 Cores
RAM - 24 GB

APU - AMD Radeon(TM) Graphics - Primary/Integrated
VRAM - 4096 MB - LPDDR5 800 MHz

Windows Edition - Windows 11 Professional (64 bit)
Windows Version - 25H2
```

The processing is done through CPU.
GPU processing is supported only through Direct ML (very limited - not usable)


## License

See [LICENSE](LICENSE) file.