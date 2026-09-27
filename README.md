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
    --local-dir .res/embedding/all-MiniLM-L6-v2

# B) git + git-lfs
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 \
    .res/embedding/all-MiniLM-L6-v2

# C) Python (downloads into the HF cache, then saves a local copy)
python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('.res/embedding/all-MiniLM-L6-v2')"
```

Notes:

- `model_path` is resolved relative to `config.toml`; it must be a folder
  containing `modules.json`/`config_sentence_transformers.json` and the model
  weights (`pytorch_model.bin` or `model.safetensors`).
- `.res/` is git-ignored, so downloaded weights are never committed.
- Sentence splitting uses NLTK's `punkt_tab` data when available and falls back
  to a regex splitter otherwise, so no runtime download is required.
- Tune the summary length with `ratio` or the per-call `max_sentences` argument.


## Semantic search (corpus + FAISS)

The `store_corpus` tool indexes the documents in a directory and saves the
embedded passages and images to a single corpus file. The `query_corpus` tool
then returns the most relevant passages (and images). Both use a FAISS exact
inner-product index (`IndexFlatIP`), which is cosine similarity on L2-normalized
vectors.

**Supported formats**

- Text-like: `.txt`, `.md`, `.markdown`, `.tex`, `.rst`, `.log`, `.csv`
- Markup: `.html`, `.htm`
- Office: `.pdf` (text plus embedded images), `.docx` (Word), `.xlsx` (Excel)
- Images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`

Legacy binary formats `.doc` and `.xls` are not supported (convert them to
`.docx`/`.xlsx` first).

**Embedding modes**

- `multimodal = true` (default): text and images are embedded into one shared
  space with the CLIP model at `image_model_path`
  (`.res/embedding/clip-ViT-B-32`), so a text query can return images too.
- `multimodal = false`: only text is embedded, with `model_path`
  (`.res/embedding/all-MiniLM-L6-v2`); image files are skipped.

**Per-file summaries**

After a successful build, a markdown summary is appended to `SUMMARY.md` next to
the corpus file, with one `###` section per input file. Summaries are **not**
stored in the corpus pickle. Set `summary_name` to change the filename.

The model, corpus file and knobs are configured in `config.toml` (paths are
resolved relative to `config.toml`; `summary_name` is relative to the corpus
file):

```toml
[embedding]
model_path = ".res/embedding/all-MiniLM-L6-v2"     # text-only-mode encoder
image_model_path = ".res/embedding/clip-ViT-B-32"  # shared text+image encoder (multimodal)
multimodal = true          # true -> use image_model_path for BOTH text and images
device = "cpu"             # "cpu" or "cuda"
corpus_path = ".res/corpus/corpus.pkl"  # resolved relative to config.toml
summary_name = "SUMMARY.md"  # per-file summaries, next to the corpus file (NOT in the pkl)
extensions = "txt, md, markdown, tex, rst, log, csv, html, htm, pdf, docx, xlsx, png, jpg, jpeg, webp, bmp, tiff"
image_extensions = "png, jpg, jpeg, webp, bmp, tiff"
chunk_size = 3            # sentences per indexed passage
top_k = 5                 # default number of results returned by query_corpus
```

The tools fail gracefully with a clear message until the model is stored in
`model_path` (text mode) / `image_model_path` (multimodal). Store a model with
one of the following methods (shown for `all-MiniLM-L6-v2`; substitute
`sentence-transformers/clip-ViT-B-32` and `.res/embedding/clip-ViT-B-32` for the
multimodal model):

```bash
# A) Hugging Face CLI (recommended)
pip install -U huggingface_hub
hf download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir .res/embedding/all-MiniLM-L6-v2
hf download sentence-transformers/clip-ViT-B-32 \
    --local-dir .res/embedding/clip-ViT-B-32

# B) git + git-lfs
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 \
    .res/embedding/all-MiniLM-L6-v2
git clone https://huggingface.co/sentence-transformers/clip-ViT-B-32 \
    .res/embedding/clip-ViT-B-32

# C) Python (downloads into the HF cache, then saves a local copy)
python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('.res/embedding/all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/clip-ViT-B-32').save('.res/embedding/clip-ViT-B-32')"
```

For multilingual text corpora, use the sample's aligned model instead by
changing `model_path` (and re-building the corpus):

```bash
hf download sentence-transformers/quora-distilbert-multilingual \
    --local-dir .res/embedding/quora-distilbert-multilingual
```

Usage:

- `store_corpus(directory="docs")` writes the configured `corpus_path` and its
  `SUMMARY.md`; pass `corpus_file=...` to write elsewhere and `overwrite=True`
  to replace an existing file.
- `query_corpus(query="what is quantum computing?")` searches the configured
  corpus; pass `corpus_file=...` for a different corpus and `top_k=...` to change
  the number of results. Tune `chunk_size` to control passage length.

Notes:

- Requires `faiss-cpu`, `pymupdf`, `python-docx` and `openpyxl` (installed into
  `.lms` by `setup-lms-cpu.ps1`).
- The corpus file is a pickle containing the records and their normalized
  embeddings; rebuild it whenever a model path, `extensions`, `multimodal` or
  `chunk_size` changes.
- `.res/` is git-ignored, so downloaded weights, corpora and `SUMMARY.md` are
  never committed.


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