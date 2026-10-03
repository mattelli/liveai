# Live Assistant

Simple example to create locally a live AI assistant.

Mic → Whisper (STT) → LLM (OpenAI-compatible API) → Piper (TTS)


## Whisper
Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. 
See [Whisper](https://huggingface.co/openai/whisper-medium)

```
git lfs install
git clone https://huggingface.co/guillaumekln/faster-whisper-medium
```

## LLM backend (OpenAI-compatible API)
Run AI models from API-key providers through any OpenAI-compatible chat API
(OpenAI, Mistral, Groq, OpenRouter, DeepSeek, Together, Gemini, xAI, ...)
with the `openai` client. See `config.toml` `[brain]` for model, `base_url`
and `api_key_env`; the API key itself lives in the environment or a `.env`
file (see `.env.example`), never in config.toml.

### Local models (LM Studio / Ollama servers)
The same Brain also runs fully local models: both LM Studio and Ollama expose
an OpenAI-compatible endpoint, so only `base_url` (and a dummy key) change.

| Server | Start | base_url |
|---|---|---|
| LM Studio | `lms server start` (or the app's Developer tab) | `http://localhost:1234/v1` |
| Ollama | `ollama serve` | `http://localhost:11434/v1` |

```toml
[brain]
model = "qwen/qwen3.5-8b"              # LM Studio model key (see `lms ls`)
# model = "qwen3:8b"                   # Ollama model tag (see `ollama ls`)
base_url = "http://localhost:1234/v1"  # or "http://localhost:11434/v1"
api_key_env = "LMSTUDIO_API_KEY"       # or OLLAMA_API_KEY
```

Local servers ignore the key, but the client must still send one: put a dummy
value in `.env` (e.g. `LMSTUDIO_API_KEY=localhost`).
Streaming, tool calling and vision work as with cloud providers, depending on
the model's capabilities.

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

The `summarize_text` tool reads a **document** and condenses its most central
sentences using sentence embeddings (`all-MiniLM-L6-v2`) and LexRank. Call it
with a file path (e.g. `summarize_text(file="notes.txt")`); any type supported
by `src/document_loader.py` works -- text-like (`.txt`, `.md`, `.tex`, `.csv`,
...), PDF, DOCX, XLSX, HTML and images (image files only report that they
contain no extractable text). The model is loaded lazily on first tool use from a
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


## Semantic search (named corpora + FAISS)

`store_corpus` builds a **named corpus** from a directory of documents and saves
it under `.res/corpora/<corpus_name>/`. `query_corpus` searches one named corpus,
and `list_corpora` lists the corpora that exist. Everything uses a FAISS exact
inner-product index (`IndexFlatIP`) - cosine similarity on L2-normalized vectors.

**Layout**

```
.res/corpora/
  autosar/
    corpus.pkl        # records + embeddings
    metadata.json     # name, mode, counts, source dir, built time
    processed.md      # absolute paths of the processed documents
    images/           # extracted figures (only referenced files)
```

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

**Tools**

- `list_corpora()` - lists the available corpora (from `metadata.json`, without
  loading any pickle). Call it first when the user refers to a topic.
- `store_corpus(directory, corpus_name, overwrite=False)` - builds the named
  corpus. `corpus_name` is **required**. Refuses to overwrite an existing corpus
  unless `overwrite=True`.
- `query_corpus(query, corpus_name, kind="", top_k=0)` - searches the named
  corpus. `corpus_name` is **required**; unknown names return a friendly error.
  `kind="text"` or `kind="image"` restricts results (empty searches both).

Example:

```
store_corpus(directory=".tmp/corpus", corpus_name="autosar")
query_corpus(query="cross cluster communication", corpus_name="autosar", top_k=3)
```

The tools fail gracefully with a clear message until the model is stored in
`model_path` (text mode) / `image_model_path` (multimodal). Store a model with
one of the following methods (shown for `all-MiniLM-L6-v2`; substitute
`sentence-transformers/clip-ViT-B-32` and `.res/embedding/clip-ViT-B-32` for the
multimodal model):

```bash
# A) Hugging Face CLI (recommended)
pip install -U huggingface_hub
hf download sentence-transformers/all-MiniLM-L6-v2     --local-dir .res/embedding/all-MiniLM-L6-v2
hf download sentence-transformers/clip-ViT-B-32     --local-dir .res/embedding/clip-ViT-B-32

# B) git + git-lfs
git lfs install
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2     .res/embedding/all-MiniLM-L6-v2
git clone https://huggingface.co/sentence-transformers/clip-ViT-B-32     .res/embedding/clip-ViT-B-32

# C) Python (downloads into the HF cache, then saves a local copy)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('.res/embedding/all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/clip-ViT-B-32').save('.res/embedding/clip-ViT-B-32')"
```

For multilingual text corpora, use the sample's aligned model instead by
changing `model_path` (and rebuilding the corpus):

```bash
hf download sentence-transformers/quora-distilbert-multilingual     --local-dir .res/embedding/quora-distilbert-multilingual
```

The corpora root is configured in `config.toml` (resolved relative to it):

```toml
[embedding]
model_path = ".res/embedding/all-MiniLM-L6-v2"     # text-only-mode encoder
image_model_path = ".res/embedding/clip-ViT-B-32"  # shared text+image encoder (multimodal)
multimodal = true          # true -> use image_model_path for BOTH text and images
device = "cpu"             # "cpu" or "cuda"
corpora_dir = ".res/corpora"  # resolved relative to config.toml
extensions = "txt, md, markdown, tex, rst, log, csv, html, htm, pdf, docx, xlsx, png, jpg, jpeg, webp, bmp, tiff"
image_extensions = "png, jpg, jpeg, webp, bmp, tiff"
chunk_size = 5            # sentences per indexed passage
top_k = 5                 # default number of results returned by query_corpus
```

**Image search and cleanup**

- Because text-to-text CLIP scores are much higher than text-to-image scores,
  images are hidden when searching both. Use
  `query_corpus(query="software cluster integration diagram", corpus_name="autosar", kind="image")`
  to rank only image records; matches are returned as `[image] <path>` and the
  files live under `.res/corpora/<corpus_name>/images/`.
- Embedded PDF images are filtered on extraction - soft masks, tiny icons
  (`< 64 px` on a side) and near-blank fills are skipped - and byte-identical
  images are de-duplicated, so `images/` holds only distinct, usable figures.

Each successful build also writes `.res/corpora/<corpus_name>/processed.md` with
one absolute path per processed document (failed files are annotated); it is not
stored in the pickle.

Notes:

- Inspect a named corpus with `python samples/inspect_corpus.py --list` and
  `python samples/inspect_corpus.py --corpus autosar` (filters: `--sources`,
  `--kind image`, `--source`, `--grep`, `--record N`, `--json`).
- Requires `faiss-cpu`, `pymupdf`, `python-docx` and `openpyxl` (installed into
  `.lms` by `setup-lms-cpu.ps1`).
- Rebuild a corpus whenever a model path, `extensions`, `multimodal` or
  `chunk_size` changes.
- `.res/` is git-ignored, so downloaded weights and corpora are never committed.


## Setup (Python 3.12.10)

OS Windows 11 (powershell)

```
# Ai agent based on an OpenAI-compatible LLM API
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

`Ctrl-C` stops the response being generated (the chat session keeps running);
at the input prompt it clears the line (re-prompt). `Ctrl-D` (EOF) exits via
the EOF path. If prompt_toolkit is missing or disabled, input falls back to the
builtin `input()` with no line editing or history.

While a response generates, a braille-dots spinner (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏) animates whenever nothing
visible is happening: during prompt processing (`processing prompt 42%`),
tool execution and while fenced code blocks are buffered for syntax
highlighting (`writing python code (812 chars)`). It is erased automatically
before each piece of real output and at the end of the turn.

Stopping a response closes the in-flight HTTP stream (best effort): the
partial answer stays in the chat history (a minimal placeholder is recorded
if the model was stopped before generating anything, keeping user/assistant
role alternation intact for the next request), and any further tool-call
rounds are skipped. Before each new prompt, the agent also repairs the
history from older cancelled/failed turns (drops unanswered prompts, pads
empty assistant turns).


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