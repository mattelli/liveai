# Live Assistant

Simple example to create locally a live AI assistant.

Mic → Whisper (STT) → LLM (OpenAI-compatible API) → Piper (TTS)

Chat mode is supported, several tools are available.


## Runtime data root (`.liveai`)

Every run keeps its own runtime state in a **`.liveai/` folder created in the
directory `main.py` is launched from** (not the project root), so several
sessions started from different folders never share files:

```
<cwd>/.liveai/
├── memory.log      # personal memory            ([brain] memory_path_cwd)
├── sessions/       # /session store|list|load   ([brain] sessions_dir_cwd)
├── agents.jsonl    # activity monitor log       ([agent] monitor_log_file_cwd)
├── input.log       # prompt_toolkit history     ([input] history_file_cwd)
├── schedule.json   # task persistence           ([delegation.scheduler] persist_file_cwd)
├── tmp/            # scratch output (downloads, charts, sandbox)  ([cwd] tmp_dir_cwd)
└── behaviour/      # markdown instruction files ([cwd] behaviour_dir_cwd)
```

**Assets vs. state.** Shared assets stay in the project root `.res/` (model
weights, corpora, voices, embeddings) and the config files stay in the project
root too; only *per-run state* moves into `.liveai/`.

**Resolution rule.** Any config key ending in `_cwd` is resolved relative to
the launch directory, while every other `*_file` / `*_dir` / `*_path` key
resolves relative to the config file. `[cwd] data_dir_cwd` is the data root
itself; every **other** `_cwd` key is resolved *under* it, so the values stay
short and the root is defined once:

```toml
[cwd]
data_dir_cwd = ".liveai"       # the data root (relative to the launch directory)
tmp_dir_cwd = "tmp"            # -> <cwd>/.liveai/tmp
behaviour_dir_cwd = "behaviour"  # -> <cwd>/.liveai/behaviour

[brain]
memory_path_cwd = "memory.log" # -> <cwd>/.liveai/memory.log
sessions_dir_cwd = "sessions"
```

Change `data_dir_cwd` in one place and every state file follows.

**Behaviour folder.** Put free-form markdown instruction files in
`<cwd>/.liveai/behaviour/`; the `[agent] bootstrap_prompt` references the
folder with a `{behaviour_dir}` placeholder (resolved at startup), so the
agent reads them on the first turn. No files → nothing to follow.

The data root and its `tmp` / `behaviour` subfolders are created on startup;
existing root-level `memory.log` / `.sessions/` / `input.log` are **not**
migrated (move them manually if needed).


## Whisper
Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. 
See [Whisper](https://huggingface.co/openai/whisper-medium)

```
# A) git + git-lfs
git lfs install
git clone https://huggingface.co/guillaumekln/faster-whisper-medium

# B) Python 
from faster_whisper import WhisperModel
model = WhisperModel("medium", device="auto", compute_type="int8")

```

## LLM backend (OpenAI-compatible API)
Run AI models from API-key providers through any OpenAI-compatible chat API
(OpenAI, Mistral, Groq, OpenRouter, DeepSeek, Together, Gemini, xAI, ...)
with the `openai` client. See `config.toml` `[brain]` for the models JSON file
(`models_file`) holding each model's `base_url` and `api_key_env`; the API
keys themselves live in the environment or a `.env` file (see
`.env.example`), never in config.toml. The first entry of the models file
boots the session.

Several models can be prepared in a JSON file and switched at runtime with the
`/model` command - see [Model switching (`/model`)](#model-switching-model)
below.

`[brain] request_timeout_s` (default `600`) is the maximum number of seconds
the client waits **between stream chunks** before aborting the request. It is a
backstop against a server that hangs mid-generation: without it a stalled
stream blocks the turn forever, with nothing for `Ctrl+C` to interrupt. A
slow-but-alive generation is not cut off (the timer resets on every chunk);
set it to `0` to keep the SDK default (no timeout).

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


## Image generation (Stable Diffusion)

`generate_image` creates an image from a text prompt with a local
Stable-Diffusion pipeline (`diffusers`, see `src/image_generator.py`), mirroring
`playground/simple_img_gen.py`. The first call loads the model, which can take a
while and needs several GB of free RAM; later calls in the same session reuse
the loaded pipeline. The image is saved to disk and the tool answers with the
file's absolute path plus the seed used - repeat the seed to reproduce the same
image, change it to get a variation:

```
generate_image(prompt="a white dog sitting on a pillow")
-> C:/Users/Max/Documents/workspace/live_assistant/.out_images/generated_20261004-124630.png (seed=42)
```

- Parameters left empty fall back to the `[image_gen]` defaults in `config.toml`
  (size 512x512, 50 steps, configured negative prompt).
- `width`/`height` must be multiples of 8, `steps` trades quality for speed and
  a `seed` < 0 picks a random seed for every image.
- `unload_image_generator()` frees the RAM/VRAM again when no more images are
  needed (the model stays resident until then).

```toml
[image_gen] # local image generation with Stable Diffusion (src/image_generator.py, diffusers)
t2i_model = "UnfilteredAI/NSFW-GEN-ANIME"     # text-to-image model (HuggingFace id or local folder)
i2i_model = "runwayml/stable-diffusion-v1-5"  # image-to-image / inpaint model
device = "auto"         # "auto" (cuda when available), "cpu" or "cuda"
negative_prompt = "blurry, low quality, deformed, extra limbs, watermark, text"
width = 512             # default image width in pixels (multiple of 8)
height = 512            # default image height in pixels (multiple of 8)
steps = 50              # default diffusion steps (higher = slower, more refined)
seed = 42               # default seed; negative -> random seed for every image
guidance = 5.0          # default img2img/inpaint prompt guidance (3-12)
denoise = 0.5           # default img2img/inpaint denoise strength (0-1)
out_dir = ".out_images" # folder for generated images (resolved relative to this file)
```

The tool needs the diffusers stack in the Python environment that runs the
assistant. `setup-liveai-cpu.ps1` installs it; for existing environments:

```
pip install diffusers transformers accelerate safetensors
```

The models are downloaded from the Hugging Face Hub automatically on first use
(~2 GB for the default text-to-image model) into the local HF cache, so nothing
has to be stored manually. To run offline instead, pre-download the model once
and point the config at the folder:

```
hf download UnfilteredAI/NSFW-GEN-ANIME --local-dir .res/sd/nsfw-gen-anime
```

```toml
[image_gen]
t2i_model = "C:/Users/Max/Documents/workspace/live_assistant/.res/sd/nsfw-gen-anime"
```

Notes:

- Missing packages fail gracefully: the tool answers with the `pip install`
  command to run instead of a stack trace.
- `t2i_model`/`i2i_model` accept a Hugging Face id **or** a local model folder.
  Unlike the `*_path` keys they are not resolved relative to `config.toml` -
  local folders need an absolute path (or one relative to the launch directory).
- Image-to-image and inpainting (editing a masked region of an input image with
  `runwayml/stable-diffusion-v1-5`) are available through the
  `ImageGenerator.image2image` API (`src/image_generator.py`,
  `playground/simple_img_gen.py`); the agent currently exposes text-to-image as
  a tool only.
- Generated images land in `.out_images/` (git-ignored via the `.*/` pattern)
  with a timestamped name, so nothing is overwritten; pass `out_file` to choose
  name and folder explicitly.
- On the CPU-only reference machine (see System Notes) a 512x512 image with 50
  steps takes minutes: lower `steps` (20-30) and/or the size for quicker
  iteration, or set `seed = -1` in the config for a fresh image every time.


## Text summarization

The `summarize_doc` tool reads a **document** and condenses its most central
sentences using sentence embeddings (`all-MiniLM-L6-v2`) and LexRank. Call it
with a file path (e.g. `summarize_doc(file="notes.txt")`); any type supported
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

# C) Python
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('.res/embedding/all-MiniLM-L6-v2')
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

# C) Python 
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('.res/embedding/all-MiniLM-L6-v2')
SentenceTransformer('sentence-transformers/clip-ViT-B-32').save('.res/embedding/clip-ViT-B-32')
```

For multilingual text corpora, use the sample's aligned model instead by
changing `model_path` (and rebuilding the corpus):

```bash
# A) Hugging Face CLI (recommended)
hf download sentence-transformers/quora-distilbert-multilingual  --local-dir .res/embedding/quora-distilbert-multilingual

# B) git + git-lfs
git lfs install
git clone https://huggingface.co/sentence-transformers/quora-distilbert-multilingual .res/embedding/quora-distilbert-multilingual
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

- Inspect a named corpus with `python playground/inspect_corpus.py --list` and
  `python playground/inspect_corpus.py --corpus autosar` (filters: `--sources`,
  `--kind image`, `--source`, `--grep`, `--record N`, `--json`).
- Requires `faiss-cpu`, `pymupdf`, `python-docx` and `openpyxl` (installed into Python environment).
- Rebuild a corpus whenever a model path, `extensions`, `multimodal` or
  `chunk_size` changes.
- `.res/` is git-ignored, so downloaded weights and corpora are never committed.


## In-document search (CrossEncoder)

`query_document` answers "where in this file is ...?" for a **single document**
without building a corpus. Mirroring
`playground/sentence_transformers/in_document_search_crossencoder.py`, the document
is read with `src/document_loader.py` (any supported type: text-like, PDF,
DOCX, XLSX, HTML), split into passages of up to `chunk_size` consecutive
sentences (PDF passages keep their page number), and every `[query, passage]`
pair is scored with a CrossEncoder. The `top_k` highest-scoring passages are
returned in descending score order.

```toml
[document_query]
model_path = ".res/crossencoder/ms-marco-TinyBERT-L2"  # resolved relative to config.toml
device = "cpu"            # "cpu" or "cuda"
chunk_size = 3            # consecutive sentences per passage
top_k = 5                 # default number of results returned by query_document
```

Store the model locally before first use (`.res/` is git-ignored):

```bash
# A) Hugging Face CLI (recommended)
hf download cross-encoder/ms-marco-TinyBERT-L2 --local-dir .res/crossencoder/ms-marco-TinyBERT-L2

# B) git + git-lfs
git lfs install
git clone https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2 .res/crossencoder/ms-marco-TinyBERT-L2
```

Example:

```
query_document(file="C:/docs/paper.pdf", query="How large is Europe?", top_k=3)
```

Notes:

- The CrossEncoder scores all passages of the document per query, so very large
  documents make each query slower (passages are cached per file mtime).
- Requires `sentence-transformers` (already used by summarization/semantic
  search) and NLTK sentence splitting with the same regex fallback as the other
  tools.


## Image search (CLIP)

`image_search` finds the pictures inside a **folder** that best match a text
query. Mirroring `playground/sentence_transformers/image_search.py`, every image
under the folder (recursively; `.png .jpg .jpeg .webp .bmp .tif .tiff`) is
embedded with the CLIP model, the query text is embedded into the same shared
space, and cosine similarity ranks the images. The tool returns the **top_k
matches as score/path pairs**, one per line, best match first:

```
1) [0.912] C:/Users/Max/Pictures/two_dogs_in_snow.jpg
2) [0.034] C:/Users/Max/Pictures/vacation.png
```

Unreadable or corrupt files are skipped.

```toml
[image_search]
model_path = ".res/embedding/clip-ViT-B-32"  # resolved relative to config.toml
device = "cpu"            # "cpu" or "cuda"
top_k = 3                 # default number of results returned by image_search
```

The model is the same CLIP encoder as the multimodal semantic search mode, so
if it is already stored there is nothing new to download. Otherwise:

```bash
# A) Hugging Face CLI (recommended)
hf download sentence-transformers/clip-ViT-B-32 --local-dir .res/embedding/clip-ViT-B-32

# B) git + git-lfs
git lfs install
git clone https://huggingface.co/sentence-transformers/clip-ViT-B-32 .res/embedding/clip-ViT-B-32
```

Example:

```
image_search(folder="C:/Users/Max/Pictures", query="two dogs in the snow", top_k=3)
```

Notes:

- Image embeddings are cached per (path, mtime), so repeated queries over an
  unchanged folder do not re-encode the images.
- Querying a large folder encodes every image at least once; start with a
  narrower folder when possible.




## Web search and page extraction (DDGS)

The web tools are thin wrappers around the [DDGS](https://github.com/deedy5/ddgs)
metasearch library (`pip install ddgs`), which aggregates results from multiple
engines (bing, duckduckgo, google, yahoo, ...). No browser or web driver is
needed — the old Selenium/ChromeDriver stack was removed.

| Tool | DDGS call | Returns |
|---|---|---|
| `web_search(query, n_results=5, region, safesearch, timelimit)` | `text()` | page results: `title`, `href`, `body` |
| `web_images(query, n_results=5, ..., size, color, type_image, layout, license_image)` | `images()` | image URLs: `title`, `image`, `thumbnail`, `width`, `height`, `source` |
| `web_videos(query, n_results=5, ..., resolution, duration)` | `videos()` | video results: `title`, `content` (URL), `duration`, `publisher`, `uploader` |
| `web_news(query, n_results=5, timelimit)` | `news()` | news results: `date`, `title`, `body`, `url`, `source` |
| `web_books(query, n_results=5)` | `books()` | book results: `title`, `author`, `publisher`, `info`, `url` |
| `extract_webpage(url, fmt="text_markdown")` | `extract()` | the page content as markdown/plain/rich text (truncated at 4000 chars) |

`timelimit` accepts `d` (day), `w` (week), `m` (month), `y` (year); `safesearch`
is `on`, `moderate` or `off`. Every tool returns a numbered results block as a
string, or an `Error: ...` string when the search fails — failures never raise.

Examples:

```python
web_search("python programming", n_results=3)
web_images("two dogs in the snow", n_results=3, size="Large")
web_videos("lo-fi jazz", n_results=2)
web_news("space exploration", n_results=3, timelimit="w")
web_books("the sea wolf jack london")
extract_webpage("https://example.com")
```

Notes:

- Image and video URLs returned by `web_images`/`web_videos` can be saved to
  disk with `download_file(url, out_dir=...)`.
- `extract_webpage` supports `fmt` of `text_markdown` (default, keeps links /
  headers / lists), `text_plain`, `text_rich` and `text` (raw HTML).
- The old `webpage_headlines` / `webpage_content` tools were replaced by
  `extract_webpage`.




## Interactive chat input

`python main.py --chat` uses [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/)
as its input layer, so the same behaviour works on Windows and POSIX:

- **Line editing** — arrows, Home/End, word operations, Ctrl-K/U/W, selection.
- **Persistent history** — Up/Down recalls previous turns, and history survives
  across sessions. It is stored in `<cwd>/.liveai/input.log` (git-ignored).
  Delete the file to reset; it has no size cap.
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
history_file_cwd = "input.log"   # under [cwd] data_dir_cwd; empty/omitted -> in-memory
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

### Interrupting a response (`Ctrl+C`)

`Ctrl+C` cancels the current generation and the session stays usable:

- **Immediate, even during prefill** — the completion request is issued from a
  helper thread, so the in-flight HTTP stream is reachable (and closable) from
  the very first moment. A cancel lands within ~a few hundred ms, whether the
  server is still prefilling or already streaming tokens; the spinner stops and
  the prompt returns.
- **Reported as a cancel, never an error** — you get `⏹ generation cancelled`
  (yellow). Aborting the stream can surface as an httpx/OpenAI exception; the
  pending cancel flag tells the two apart, so no `❌` or traceback is printed.
- **The partial answer is discarded** — a cancelled turn leaves no trace in the
  chat history (no partial text, no placeholder), and any trailing unfinished
  tool-call group is dropped. Before each new prompt the agent also repairs the
  history from older cancelled/failed turns (drops a trailing unfinished
  tool-call group and unanswered prompts, pads empty assistant turns).
- **A running tool is killed** — a long `run_command` / `python_eval` (or the
  foreground path of `spawn_process`) is terminated instead of running until
  its own timeout; its result is reported as cancelled. On Windows the whole
  process tree is killed (`taskkill /F /T`), since the command runs through a
  shell.
- **Speech first, generation second** — while speech audio is playing, the
  first `Ctrl+C` stops the audio only (buffered audio is dropped for immediate
  silence, the response continues, and later tool rounds may speak again); a
  **second** `Ctrl+C` then cancels the generation. The speech guard hands
  `Ctrl+C` back to the generation handler after silencing the audio.
- **A hung server cannot block forever** — `[brain] request_timeout_s`
  (default `600`) bounds the wait between stream chunks, so a stalled stream is
  aborted instead of blocking the turn with nothing to interrupt.

Voice mode / barge-in is out of scope: this applies to chat mode (`--chat`).

## Session files (`/session`)

The full chat history can be saved to and restored from JSON files with the
`/session` command (a plain session command, not an agent tool):

- `/session store [name]` — store the full conversation history (system
  prompt, user/assistant turns and tool calls) as `<name>.json` in the
  configured folder; without a name a timestamped
  `session-YYYYMMDD-HHMMSS.json` file is created. An existing file is never
  overwritten.
- `/session list` — list the stored session files (message count, storage
  time, size).
- `/session load <name|index>` — replace the current conversation with the
  stored one; `<name>` is the file name with or without the `.json` suffix,
  or the 1-based index shown by `/session list`.

The storage folder comes from `[brain] sessions_dir_cwd` in `config.toml`
(resolved under `[cwd] data_dir_cwd`, created on first store):

```toml
[brain]
sessions_dir_cwd = "sessions"   # -> <cwd>/.liveai/sessions
```

## Model switching (`/model`)

The active model can be changed at runtime with the `/model` command (a plain
session command, not an agent tool) from a JSON file listing the selectable
models. Each entry carries the connection details of one model:

- `model` (required) — the model id sent to the server.
- `api_key_env` (optional) — name of the environment variable holding the API
  key (resolved from the environment and `.env`, never stored in the JSON).
- `base_url` (optional) — the OpenAI-compatible endpoint.
- `provider` (optional) — OpenRouter-style routing hint (e.g. `"wafer/us"`).
  When set, every completion request is pinned to that provider with
  fallbacks disabled (`extra_body={"provider": {"only": ["wafer/us"],
  "allow_fallbacks": false}}`); entries without it send requests unchanged.

Missing `api_key_env`/`base_url` fall back to the values configured under
`[brain]` in `config.toml`, so entries of one provider can carry only `model`.
The file location comes from `[brain] models_file` (resolved relative to
`config.toml`):

```toml
[brain]
models_file = "models.json"
```

The **first entry of the file boots the session**: at startup the assistant
connects to that entry's model/endpoint/key, so `[brain]` needs no duplicate
`model`/`base_url`/`api_key_env` (those keys are only used by the legacy
models-file-less setup). To change the startup model, reorder the entries in
the JSON file.

```json
{
  "glm-flash": {
    "api_key_env": "OPENROUTER_API_KEY",
    "base_url": "https://openrouter.ai/api/v1",
    "model": "z-ai/glm-5.3-flash"
  },
  "llama32-local": {
    "api_key_env": "OLLAMA_API_KEY",
    "base_url": "http://localhost:11434/v1",
    "model": "llama3.2:3b"
  }
}
```

Commands:

- `/model` — list the selectable models (name, model id, endpoint) with the
  currently active one marked `→`, plus the usage hint.
- `/model <name>` — switch to that entry (name matching is case-insensitive).

Switching rebuilds the OpenAI client for the entry's endpoint and key, keeps
the conversation history, and re-resolves the model-dependent settings from
the new server: context window and vision support (server-reported when
available, `[brain]` overrides otherwise) and the per-generation token cap
(`max_response_tokens` ratio re-applied to the new window). A failed switch
(unknown name, missing file, missing API key, ...) never touches the current
model: the error is reported and the session continues with the previous one.

### Chain-of-thought visualization

Reasoning models expose their thinking while they generate (`reasoning_content`
or `reasoning` stream fields, or inline istine tags). With `[ui] show_thinking = true`
the thought chain is streamed live in dim text behind a `💭` marker, capped by
`[ui] max_thinking_tokens` (0 = unlimited). Thinking is display-only: it never
enters the chat history, the spoken reply or the returned answers.

When the cap is reached a `… (thinking truncated)` marker is shown once and
further reasoning is dropped from the display (the model keeps thinking). The
marker ends the line and the activity roller (`⠋ thinking`) resumes below it, so
a long truncated reasoning phase is visibly still running instead of looking
stalled.

Notes:

- The JSON file also accepts a list of entries (each carrying a `name`) and a
  top-level `"models"` wrapper around either shape; a bare string stands for a
  model id selectable under its own name.
- The API key of the target model must be resolvable at switch time (set it in
  `.env` or the environment); local servers still accept a dummy value.
- The switch is validated before any state changes: an entry without any
  `base_url`/`api_key_env` (entry and `[brain]` alike) is rejected.
- The same validation applies at startup for the first entry: a missing file,
  invalid JSON or a non-resolvable API key aborts the launch with a clear
  message.


## Task delegation (`[delegation.agents]`)

The main agent can delegate tasks to named sub-agents with the
**`delegate_task(agent_name, task)`** tool. Sub-agents are declared in
`config.toml`; with no `[delegation.agents]` entries the tool is not registered
and the
feature costs zero tokens.

```toml
[delegation.agents]
summarize_results = false            # true -> long child results get an LLM summary before being shown
# summarize_max_chars = 800          # summarize only results longer than this (0 = always)

[delegation.agents.researcher]
tool_groups = "web, files"           # tool topics for this child (see [agent] tool_groups)
model = "fast"                       # models.json entry name; unset = the main model
system_prompt = "You are a research assistant. Be concise."
max_act_rounds = 50                  # child round budget
description = "web research"         # shown to the main agent in its prompt
```

How it works:

- **Asynchronous (fire-and-forget)** — `delegate_task` returns an
  acknowledgment with a task id immediately; the parent's tool loop is never
  blocked. The child runs on a daemon thread as an isolated, quiet agent
  (fresh per-agent tool instances, like the scheduler executor).
- **Result delivery** — when the child finishes, its final text is queued and
  injected into the main chat history before the next turn (like scheduled-task
  results); the parent agent can then use or mention it. Child failures queue
  an error text instead of crashing the turn.
- **Flat topology** — children get no `delegate_task` tool and cannot spawn
  further agents; the delegation tool is registered for the main session agent
  only.
- **Child model/tools** — each child uses its configured `tool_groups` (smaller
  tool set = cheaper per-request payloads), optional `model` (a `models.json`
  entry) and optional `system_prompt`.
- **Result summarization (optional)** — with `summarize_results = true`,
  results longer than `summarize_max_chars` are condensed by an LLM summary
  pass (on the child's own model endpoint) before being queued; short results
  pass through unchanged and a summary failure queues the original text
  (fail-open, results are never dropped).
- **Known limitations (v1)** — no mid-run cancellation of a running child, and
  no hierarchical delegation (flat only). Sub-agent tool calls are visible in
  the activity monitor and the `agent_activity` tool.

## Reducing token consumption

Every request re-sends the system prompt, the tool schemas and the whole chat
history — the agents defined in `src/live_ai.py` register 43 tools (~21k chars
of JSON schemas ≈ 5.3k tokens) and a single big tool output is re-sent on every
round of a turn. Four config keys (all default to the historical behavior when
unset) keep that cost bounded:

```toml
[brain]
# max_tool_result_chars = 4000        # cap tool results stored in the history (head+tail kept with a truncation marker; 0 = uncapped)
# context_trim_ratio = 0.7            # prune the oldest history when context usage exceeds this ratio (0 = never prune)
# context_keep_messages = 40          # most recent messages never touched by pruning
# auto_compaction = false             # run compact_chat (summary) instead of plain pruning when context_trim_ratio is exceeded
# compact_keep_messages = 4           # recent turns kept by auto-compaction (default: context_keep_messages)

[agent]
# tool_doc_chars = 300                # cap tool descriptions in the JSON schemas (0 = full docstrings)
# tool_groups = "model, memory, scheduling, files, web, documents, images, system, mcp"  # keep only these tool topics (default "all")
```

- `max_tool_result_chars` — oversized tool results keep the first 75% and the
  last 25% (the tail preserves recent entries of append-ordered files like
  `memory.log`) plus an explicit `[truncated, N chars total]` marker, so the
  model can re-query with a targeted tool (`search_in_file`, `query_document`,
  `summarize_doc`) instead of re-reading everything. Truncation is preferred
  over eager summarization: it is free, deterministic and never distorts what
  it keeps.
- `tool_doc_chars` + no `default` values in schemas — shrinks the fixed
  per-request payload (measured: 21.9k → 19.1k chars for the full 43-tool set);
  `tool_groups` excludes whole topics (finance, archives, ...) for zero tokens.
- `context_trim_ratio` / `context_keep_messages` — at the start of each turn,
  when usage passes the ratio, the oldest *complete* tool rounds are dropped
  (never split: no orphan tool results) until usage is back under the ratio;
  the system prompt and the last `context_keep_messages` messages are always
  kept, and a `[history pruned: N messages dropped]` marker is appended to the
  system prompt. This turns an unbounded, quadratic per-turn token cost into a
  bounded, linear one.
- `auto_compaction` (default `false`) / `compact_keep_messages` — with
  `auto_compaction = true`, the same usage checkpoint runs a full compaction
  instead of the plain drop: the oldest messages are summarized into a single
  `[Compacted context ...]` message (via the chat model, with a built-in
  extractive fallback) while the system prompt, the recalled personal data and
  the last `compact_keep_messages` turns stay verbatim. Legacy pruning still
  runs afterwards if usage remains above the ratio. You can also compact
  manually with the `/compact [keep_last]` command or let the model call the
  `compact_chat` tool. Compaction is safe by construction:
  - the split is *group-aligned*: a tool round is never cut in half, so the
    resulting history stays valid for OpenAI-compatible servers (no orphan
    `tool` result, no dangling `assistant(tool_calls)`);
  - the re-seeded head (system prompt + recalled personal data) is never
    summarized, so the personal data is not duplicated into the summary; the
    summary itself is capped with `max_tool_result_chars` (0 = uncapped) and
    the re-seeded recall text with the same limit;
  - the summary is a `[Compacted context ...]` **user** message (mid-history
    `system` messages are rejected by some servers) but is excluded from
    personal-memory capture (`store_memory`) and from the `clear_chat` replay;
  - it is idempotent and never grows the history: a second `/compact` with no
    new messages, a range that is only a previous summary, or a *model*
    summary that is not smaller than the range it replaces are all no-ops
    ("Context is already compact; nothing to do."); the result string reports
    " (extractive fallback)" when no bound summarizer produced the summary;
  - isolated (background) agents get their own `MemoryTools` instance *and*
    their own summarizer binding, and auto-compaction uses that instance;
  - a round containing `clear_chat`/`compact_chat` always runs sequentially
    (never in the `max_parallel_tools` pool), and the memory instance is
    lock-guarded, so a compaction cannot race the session or its own round.
- The scheduler execution agent inherits the same settings (it copies the
  config). Recommended
  starting values: `max_tool_result_chars = 4000`, `tool_doc_chars = 300`,
  `context_trim_ratio = 0.7`, `context_keep_messages = 40`.
- Runtime toggling: `/tools off <tool|topic>` drops a tool's schema from every
  subsequent request for the rest of the session (e.g. `/tools off finance`
  removes both finance tools; `/tools on finance` restores them), and
  `/tools` lists the tools with `[off]` markers. Toggles are session-only —
  `[agent] tool_groups` remains the boot-time default — and survive
  `/mcp refresh`.

## Sub-agent activity monitoring

Background agents (the scheduler's execution agent) are quiet by default, so
their work is invisible. The `[agent] monitor` setting makes their activity
visible through independent, combinable flags
(comma-separated; unset/empty/`"none"` = fully quiet, as always):

```toml
[agent]
monitor = "calls,log"                 # "calls" and/or "log"; empty = quiet
# monitor_log_file_cwd = "agents.jsonl"  # JSONL event log path (monitor = "log"; under [cwd] data_dir_cwd)
# monitor_buffer = 100                # in-memory events kept for the agent_activity tool
```

- `calls` — while a scheduled task runs, its tool executions print the usual
  `[CALL]` blocks live on the console, each labeled with the agent name
  (e.g. `[CALL] web_search (scheduler)(...`). The main agent's own
  `[CALL]` blocks are unchanged.
- `log` — every agent event is appended as one JSON object per line to
  `monitor_log_file_cwd` (default `agents.jsonl`, under `[cwd] data_dir_cwd`):
  run `start`/`answer`
  (with prompt/answer and duration) and per-tool-call `tool_call`/`tool_result`
  events. Lines are written atomically under a lock, so worker threads never
  interleave half-lines; the file is created lazily and a write failure
  disables the file sink without breaking the agents.
- Both channels feed a bounded **in-memory ring buffer** (`monitor_buffer`,
  default 100 events) that is always populated when monitoring is enabled.

The main agent can consult this buffer through the **`agent_activity(last=20)`**
tool: it returns the recent events as human-readable one-liners (or
`No sub-agent activity recorded.` when the buffer is empty), so you can simply
ask *"what did the scheduled tasks do?"*.

In the console you can also type **`/monitor [N|clear]`** to print the last N
events (default 20), or empty the buffer (the JSONL file is not touched).

## MCP servers (optional)

The assistant can use tools exposed by [MCP (Model Context Protocol)](https://modelcontextprotocol.io)
servers. Every server tool becomes a **native agent tool** (`mcp_<server>_<tool>`,
built from the server's `inputSchema`), callable in the normal tool loop; two
extra bridge tools (`mcp_list_tools`, `mcp_call_tool`) cover runtime discovery
and servers that are offline at startup. Requires the official SDK in the same
environment:

```
pip install mcp
```

Servers are configured in `config.toml` (the block ships commented out, so the
feature stays off by default):

```toml
[mcp]
connect_timeout_s = 10.0   # per-server connect + tools/list budget
call_timeout_s    = 120.0  # per-tool-call budget

[mcp.servers.filesystem]      # stdio server: local process
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/Max/data"]

[mcp.servers.deepwiki]        # HTTP server: streamable HTTP endpoint
url = "https://mcp.deepwiki.com/mcp"
```

A server entry needs `command` (+ optional `args` list) **or** `url`; anything
else is skipped with a warning. Windows note for stdio servers: if spawning
`npx` directly fails, use `command = "cmd"`, `args = ["/c", "npx", "-y", ...]`.

**Authentication (static tokens).** Per-server keys:

- `env = { NAME = "value" }` — environment variables passed to a stdio server
  process (merged over the SDK's safe whitelist: PATH, SYSTEMROOT, ...).
- `headers = { Name = "value" }` — HTTP headers sent with every request to a
  `url` server (e.g. `Authorization` or vendor-specific API-key headers).
- `token_env = "VAR"` — convenience: reads the token from environment
  variable `VAR` and applies it as `Authorization: Bearer <token>` for a
  `url` server, or as the env entry `<VAR>=<token>` for a `command` server.
- Any `env`/`headers` value may reference other variables with `${VAR}`
  (expanded at load time; unknown variables expand to the empty string).

Secrets are resolved from the environment or the project `.env` file at
startup and never stored in config.toml. Example Telegram bot server:

```toml
[mcp.servers.telegram]
command = "python"
args = ["-m", "telegram_bot_mcp"]
env = { TELEGRAM_BOT_TOKEN = "${TELEGRAM_BOT_TOKEN}" }
```

Example remote server with a bearer token:

```toml
[mcp.servers.github]
url = "https://api.githubcopilot.com/mcp"
token_env = "GITHUB_MCP_TOKEN"
```

Not supported: interactive OAuth flows (the SDK ships the machinery, but the
browser-based authorization step does not fit the synchronous tool model yet).

- `/mcp` lists the configured servers with transport, connection status and
  tool count; `/mcp refresh` reconnects all servers and rebuilds the injected
  tools.
- A failing/unreachable server never prevents startup: it degrades to an
  error string (the agent keeps answering) and can be retried with
  `/mcp refresh`.
- Without an `[mcp]` section (or without the `mcp` package) behavior is
  unchanged: no new tools beyond the two bridge tools, no startup cost.

**Security note:** MCP servers execute code with your user privileges (a stdio
server is a local process; a URL server is a remote service). Only configure
servers you trust — the same class of trust as the built-in `run_command`
tool.


## Process spawning (`spawn_process`)

The `system` topic includes **`spawn_process(cmd, cwd="")`**: it starts a
command without waiting for it (stdout/stderr/stdin detached) and returns
either the spawned **Process ID (PID)** — e.g. `Process started with PID 12345`
— or an error message (`Error spawning process: ...`) if the process could not
be started. Useful for long-running programs (a local server, a watcher, a
player) that must keep running while the conversation continues; inspect or
stop the process later through `run_command` (e.g. `taskkill /PID 12345` on
Windows, `kill 12345` on POSIX). Unlike `run_command` (blocking, captures
output), `spawn_process` returns immediately and discards the child output.

With `background=False` the command runs in the foreground and the call waits
for it (like `run_command`); a `Ctrl+C` cancel then terminates it (see
[Interrupting a response](#interrupting-a-response-ctrlc)). Background spawns
are detached and are **not** affected by a cancel.


## Task scheduler (optional)

The assistant can run prompts at a future time: one-shot reminders or
recurring jobs. Three tools are exposed to the model:

- `schedule_task(prompt, at, repeat="")` — run `prompt` at `at` (ISO 8601 or
  `YYYY-MM-DD HH:MM`, local time); `repeat` makes it recurring: `daily`,
  `hourly`, `every 30m`, or a 5-field cron expression like `0 9 * * MON`.
- `list_scheduled_tasks()` — pending tasks with ids and next run times.
- `cancel_scheduled_task(task_id)` — cancel a task by id.

All three belong to the `scheduling` tool topic, together with
`agent_activity(last=20)` — the tool that reports what the scheduled runs did
(see [Sub-agent activity monitoring](#sub-agent-activity-monitoring)).

Scheduled prompts run on a **dedicated isolated agent** (quiet, own tool
state), never on the main session agent, so the live conversation is never
disturbed. Tasks persist to a JSON file and survive restarts.

The feature ships off by default; enable it in `config.toml`:

```
[delegation.scheduler]
enabled = true                       # false/missing -> tools are no-ops
persist_file_cwd = "schedule.json"   # task persistence (survives restarts; under [cwd] data_dir_cwd)
on_missed = "skip"                   # missed one-shots after a restart:
                                     # "run" executes them, "skip" drops them
```

## Completion notifications (`[delegation.notify]`)

Scheduled tasks and delegated sub-agents finish in the background: their
result is injected into the chat history before the next turn. With
`[delegation.notify]` enabled the user also gets an immediate one-line
terminal hint the
moment a task completes — no result text is shown, only the task kind and id
(so long results are not echoed twice and stay out of the terminal):

```
🔔 Scheduled task t-1 completed.
🔔 Sub-agent task researcher (agent-1) completed.
```

```toml
[delegation.notify]
terminal = true  # missing section (or false) -> no hint, results on next turn only
```

The hint prints from the background thread; in chat mode prompt_toolkit's
`patch_stdout` renders it above the prompt line, so the input is never
garbled. Voice mode prints it between prompts.

## Setup (Python 3.12.10)

### Environment Variable `LIVEAI_DIR`

The `LIVEAI_DIR` environment variable must be set to the project root directory.
All launcher scripts (`bin/liveai`, `bin/liveai-code`, `bin/local-servers`) use it
to locate `main.py`, config files, and the virtual environment.
The `LIVEAI_DIR/bin` folder should be added in system path environment variable to launch available  `liveai ` commands from any directory.


### Launch setup scripts

The setup script needs Python 3.12 environment `.liveai-cpu/` to be created at first.

```bash
# OS Windows 11 (powershell)
.liveai-cpu\Scripts\activate.bat     # Active the Python environment
.\setup-liveai-cpu.ps1

# OS Linux / macOS (bash)
.liveai-cpu/bin/activate             # Active the Python environment
./setup-liveai-cpu.sh
```


## Unit tests

The test suite is standard **pytest** (no LLM, no network: every test uses
stubbed clients and fake agents, so it runs offline in ~30s).

Run everything from the project root:

```
python -m pytest tests/
```

Useful variants:

```
python -m pytest tests/ -q                 # quiet output, one dot per test
python -m pytest tests/ -x                 # stop at the first failure
python -m pytest tests/ -v --tb=short      # verbose + short tracebacks
python -m pytest tests/ --durations=10     # show the slowest tests
python -m pytest tests/ -q -rs             # show the skipped ones with skip reasons
```

Run a subset:

```
python -m pytest tests/test_live_ai.py                 # one file
python -m pytest tests/test_session.py -k tools        # tests matching a name
python -m pytest tests/test_session.py::test_tools_lists_available_tools
```

Notes:

- The setup script creates the project virtual environment
  (`.liveai-cpu/`); if your shell is not inside it, activate it first
  (`.liveai-cpu\Scripts\activate` on Windows) or invoke its interpreter
  directly: `.liveai-cpu\Scripts\python.exe -m pytest tests/`.
- A few scheduler tests use real threads with short waits (~150 ms), which is
  where most of the ~30s runtime comes from.

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
