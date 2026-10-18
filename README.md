# Live Assistant

Simple example to create locally a live AI assistant.

Mic → Whisper (STT) → LLM (OpenAI-compatible API) → Piper (TTS)

Chat mode is supported, several tools are available.


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
with the `openai` client. See `config.toml` `[brain]` for the models JSON file
(`models_file`) holding each model's `base_url` and `api_key_env`; the API
keys themselves live in the environment or a `.env` file (see
`.env.example`), never in config.toml. The first entry of the models file
boots the session.

Several models can be prepared in a JSON file and switched at runtime with the
`/model` command - see [Model switching (`/model`)](#model-switching-model)
below.

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

- Inspect a named corpus with `python samples/inspect_corpus.py --list` and
  `python samples/inspect_corpus.py --corpus autosar` (filters: `--sources`,
  `--kind image`, `--source`, `--grep`, `--record N`, `--json`).
- Requires `faiss-cpu`, `pymupdf`, `python-docx` and `openpyxl` (installed into Python environment).
- Rebuild a corpus whenever a model path, `extensions`, `multimodal` or
  `chunk_size` changes.
- `.res/` is git-ignored, so downloaded weights and corpora are never committed.


## In-document search (CrossEncoder)

`query_document` answers "where in this file is ...?" for a **single document**
without building a corpus. Mirroring
`samples/sentence_transformers/in_document_search_crossencoder.py`, the document
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
query. Mirroring `samples/sentence_transformers/image_search.py`, every image
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
at the input prompt it clears the line (re-prompt). While speech audio is
playing, `Ctrl-C` stops the audio only: buffered audio is dropped for
immediate silence, the response continues, and later tool rounds may speak
again. `Ctrl-D` (EOF) exits via the EOF path. If prompt_toolkit is missing or disabled, input falls back to the
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

The storage folder comes from `[brain] sessions_dir` in `config.toml`
(resolved relative to the config file and created on first store):

```toml
[brain]
sessions_dir = ".sessions"
```

## Model switching (`/model`)

The active model can be changed at runtime with the `/model` command (a plain
session command, not an agent tool) from a JSON file listing the selectable
models. Each entry carries the connection details of one model:

- `model` (required) — the model id sent to the server.
- `api_key_env` (optional) — name of the environment variable holding the API
  key (resolved from the environment and `.env`, never stored in the JSON).
- `base_url` (optional) — the OpenAI-compatible endpoint.

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

## Multi-agent pool (delegation + parallel sub-agents)

The assistant can run extra **worker agents** beside the main one (agent-1,
the one the user talks to). The main agent delegates tasks to them through two
tools, and each worker runs its own tool loop with its own model, history and
tool instances (the agent-as-tool pattern):

| Tool | Behavior |
|---|---|
| `ask_agent(agent, prompt)` | Delegates one task to a named sub-agent; the worker runs its own `act()` round loop (using its own tools, e.g. `web_search`) and its **final answer** is returned to the main agent as a normal tool result. |
| `run_parallel_agents(tasks)` | Fans out several independent tasks concurrently (thread pool; LLM calls are I/O-bound). Each task is `{"agent": name, "prompt": text}`; returns a JSON list of `{"agent", "prompt", "answer"}` in task order for aggregation. |

The pool is declared in `config.toml` under `[agent]`:

```toml
[agent]
bootstrap_prompt = "..."            # unchanged
parallel_agents = 1                 # extra agents beside the running one (0 = off)
agent_models = ["ollama-llama32"]   # models.json entry names, one per worker
# agent_system_prompts = []         # optional per-agent system prompts
# agent_quiet = true                # workers don't print/speak (default true)
```

- `agent_models` entries are **models.json entry names** (see
  [Model switching (`/model`)](#model-switching-model)); each worker boots from
  its entry, so a worker can use a different provider or a local server than
  the main agent. Missing keys default to `parallel_agents = 0` → the app
  behaves exactly as the single-agent setup.
- With `parallel_agents = 0` nothing is built at startup.
- Workers are **quiet**: they never print to the console and never speak, so
  only agent-1's output and voice reach the user.
- Each worker owns **its own tool instances** (memory/vision state is
  per-agent, never shared module state), so parallel workers cannot read or
  clear each other's history.
- **Nested delegation is refused**: a worker calling `ask_agent` /
  `run_parallel_agents` gets an error instead of deadlocking the pool.
- Unknown model entry names or a missing `models_file` fail fast at startup
  with a clear message; `ask_agent` with an unknown agent name returns an
  error string to the model (never raises).

Example flow: the user asks about the weather; agent-1 calls
`ask_agent(agent="ollama-llama32", prompt="Get current weather in Torino")`,
the worker runs `web_search` on its own and returns its summary, and agent-1
merges it into its reply. Several research questions can be answered at once
with `run_parallel_agents`.

Notes:

- Only final answers cross the agent boundary (no partial streaming), which
  keeps each agent's context small.
- Ctrl+C cancels the main agent's turn; running workers finish their current
  round.
- On local servers (LM Studio/Ollama) workers share the server's capacity —
  point workers at cloud endpoints in models.json for true parallelism, or
  keep `parallel_agents = 1` on constrained machines.

## Setup (Python 3.12.10)

OS Windows 11 (powershell)

```
# Ai agent based on an OpenAI-compatible LLM API
.\setup-liveai-cpu.ps1

```

OS Linux / macOS (bash)

```
# Install Python 3.12, then run the setup script
./setup-liveai-cpu.sh
```



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