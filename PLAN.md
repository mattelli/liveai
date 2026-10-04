# PLAN — Rust port of `main.py` (Live Assistant)

Status: draft for implementation
Target location: new Cargo workspace in **`live_assistant/rust/`** (Python code, incl. `main.py` and `src/`, is **frozen and must not change**).
Reference entry point: `main.py` + `src/` (modules listed in §2).
Crate research: all packages verified on **https://docs.rs/** (see §5).

---

## 1. Goal and scope

Re-implement, in Rust, **at least the same functionality** that `main.py` delivers today:

| Mode | CLI | Behavior to reproduce |
|---|---|---|
| Voice | `la` (no args) | mic → silence-threshold speech capture → Whisper STT → Agent (OpenAI-compatible LLM, streaming, tools) → optional Piper TTS. Ctrl+C while waiting quits; Ctrl+C during generation cancels the response. |
| Chat | `la --chat` | rustyline/prompt_toolkit-like interactive input with history, suggestions, completion, `/paste`; Ctrl+C clears the line, Ctrl+D/EOF quits. |
| One-shot | `la --prompt "..."` | single turn, no stdin reads. |
| Switches | `--no-tools`, `--config <path>` | same semantics. |

Plus the **session loop** (`src/session.py`): banner, bootstrap prompt, commands
(`/help`, `/quit`, `/clear`, `/voice on|off`, `/tools`, `/info`, `exit`), per-turn error
recovery, generation cancel, shutdown ("👋 Bye!", speech off).

Non-goals: modifying any Python file, porting `samples/`, `playground/`, `bin/*.bat`,
or bit-exact file-format interop with Python-only artifacts (see §10, corpora).
Also **not ported**: `src/local_model_handler.py` (LM Studio / Ollama model
management — it is not part of `main.py`'s dependency graph; stays Python-only).

---

## 2. Reference inventory (what must exist in Rust)

| Python source | Functionality | Rust counterpart (crate) |
|---|---|---|
| `main.py` | arg parsing, config load, tool config injection, reader wiring (chat/voice/one-shot) | `clap`, `la-core` |
| `src/common.py` | `cprint` (ANSI colors), `prompt_input`, `load_config` (TOML→ns, path resolution for `*_file/_dir/_path` and list items), JSON io, `dump_func_call` logger, **`StreamPrinter`** (streaming printer: fence detection, syntax-highlighted code blocks, activity spinner) | `toml`, `serde`, `crossterm`, `syntect` |
| `src/session.py` | `ConversationSession`: prompt loop, commands, tool launcher (prompt for required params), Ctrl+C/EOF semantics, shutdown | `la-core::session` |
| `src/input_reader.py` | `InputReader`: line editing, persistent file history, inline history suggestions (HistoryHinter), completer for `/commands`, multiline paste fallback | `rustyline` (`with-file-history`) |
| `src/live_ai.py` | `OpenAICore`/`Brain`/`Agent`: OpenAI-compatible client (base_url + api_key_env + `.env`), model metadata (`/models`, raw fallback), vision detection, context window resolution, `max_response_tokens` ratio cap, streaming, **tool-calling `act()` rounds**, history repair, cancel, `model_info`, JSON-schema tool definitions from type hints | `async-openai`, `reqwest`, `serde_json` |
| `src/audio_wrapper.py` | input device discovery, RMS, WAV buffers/files, silence-threshold calibration, `capture_speech` (chunked, silence timeout, level callback, clear timeout), `SpeechDetector` (Whisper transcribe, beam_size, language) | `cpal`, `hound`, `whisper-rs` |
| `src/audio_generator.py` | `SpeechGenerator`: Piper voice load (lazy), samplerate check, `speak` (stream chunks to device), `to_wav`, `clean_for_speech` (markdown/emoji stripping) | `sherpa-rs` (VITS/Piper), `hound` |
| `src/image_generator.py` | `ImageGenerator`: lazy pipeline load/unload, txt2img/img2img(+inpaint), defaults from `[image_gen]`, `check_size` (multiple of 8), seed handling, save to disk | `candle-*` (in separate `imggen` worker process) |
| `src/document_loader.py` | extract text(+images) from txt/md/tex/csv/log/rst/markdown, html, pdf (pages + embedded images), docx, xlsx, images | `pdfium-render`, `quick-xml`, `calamine`, `html2text`, `image` |
| `src/embedding_search.py` | corpus build/query: passages, L2 norm, FAISS IndexFlatIP ranking, multimodal CLIP space, metadata.json, processed.md, image dedup | `candle` (BERT/CLIP), `ndarray`, `md-5` |
| `src/summarizer.py` + `src/lex_rank.py` | sentence split (regex fallback), embeddings, cosine similarity, LexRank degree centrality (Markov chain + power method) | `candle` (BERT), `ndarray` |
| `src/document_query.py` | CrossEncoder `[query, passage]` scoring, top-k, per-mtime passage cache | `candle` (TinyBERT seq-cls) |
| `src/image_search.py` | CLIP folder image search with per-(path,mtime) cache | `candle` (CLIP), `image` |
| `src/local_model_handler.py` | LM Studio / Ollama model management | **not ported** — out of scope, Python-only (see §1 non-goals; §12 R8) |
| `src/tools/*` | 30+ agent tools (see §7) | see §5/§7 |

---

## 3. Architecture — multi-process design

### 3.1 Process model (per user requirement)

Four cooperating processes, all stoppable with **Ctrl+C**:

```
                    JSON-lines over stdio (requests ↓ / events ↑)
┌─────────────┐  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  la (main)  │──▶ la-listen      │   la-speak        │ la-imggen    │
│ session+LLM │  │ mic+VAD+STT  │   │ Piper TTS+audio │ SD pipeline  │
└─────────────┘  └──────────────┘   └──────────────┘   └──────────────┘
```

- **`la` (main)** — owns config, session loop, Brain/Agent (HTTP to the LLM server),
  tool registry. Spawns workers and brokers requests to them.
- **`la-listen`** — `cpal` capture at 16 kHz mono int16, RMS calibration
  (`calibration_chunks`, `guard_factor`), speech capture loop (silence timeout,
  clear timeout, RMS level events), then `whisper-rs` transcription
  (language, beam_size). Heavy model stays out of the main process.
- **`la-speak`** — `sherpa-rs` VITS (Piper `.onnx` voices) synthesis + `cpal` playback;
  stream acks when playback finished; `enable/disable` toggles the generator;
  volume applied as digital gain in the playback path.
- **`la-imggen`** — `candle` Stable-Diffusion txt2img/img2img(+inpaint); the pipeline is
  loaded **once per process** (mirrors Python module-level cache) and survives across
  calls; `unload` drops it. Spawned lazily at the first `generate_image` call.

Spawn policy: `la-listen` at startup in voice mode; `la-speak` at startup in voice mode
(lazily in chat mode on `/voice on`); `la-imggen` lazily on first image request.

### 3.2 IPC protocol (`la-proto`)

One JSON object per line (UTF-8, `\n` delimited) on each direction of stdio.
Typed with serde enums, versioned by a `v` field:

```jsonc
// main -> listen
{"v":1,"type":"listen","silence_timeout_s":2.0,"show_levels":true}
// listen -> main
{"v":1,"type":"level","rms":0.021,"threshold":0.032}
{"v":1,"type":"transcript","text":"what time is it"}   // or {"text":""} when silent
{"v":1,"type":"error","message":"No audio input device available"}

// main -> speak: {"v":1,"type":"speak","text":"...","strip":true} | {"type":"wav","path":...}
// speak -> main: {"v":1,"type":"spoken","bytes":48120} | {"type":"error",...}

// main -> imggen: {"v":1,"type":"t2i","prompt":...,"negative":...,"width":512,"height":512,"seed":-1,"steps":50,"out":"..."}
//                 {"v":1,"type":"i2i",...,"image":...,"mask":...,"denoise":0.5,"guidance":5.0}
//                 {"v":1,"type":"unload"}
// imggen -> main: {"v":1,"type":"done","path":"...","seed":42,"seconds":123.4} | {"type":"error",...}
```

Each request carries a `id` (monotonic u64) so responses can be matched; progress
messages (`level`, spinner notes) are unsolicited events.

### 3.3 Ctrl+C semantics (parity with Python)

| Situation | Python behavior | Rust behavior |
|---|---|---|
| Ctrl+C during generation (chat/voice) | SIGINT handler → `agent.cancel_generation()`: close the HTTP stream, keep partial output, skip tool rounds, print " ⏹ generation cancelled" | `tokio::signal::ctrl_c` monitored in `select!` alongside the SSE stream: set cancel flag + drop the stream (closes the connection), keep partial text in history |
| Ctrl+C while waiting for input (voice) | quits the session | same: break loop → shutdown |
| Ctrl+C while waiting for input (chat) | clears the line and re-prompts | rustyline returns `Interrupted` → re-prompt |
| Ctrl+D / EOF | exits | `ReadlineError::Eof` / `prompt_input` returns `None` |
| Exit / quit anywhere | Ctrl+C terminates the process group | shutdown sequence: send `{"type":"stop"}` to workers, wait ≤ 2 s, then kill; workers also exit on **stdin EOF** (parent-death watchdog) |

Hardening (optional, Windows): put every child in a Win32 **Job object** with
KILL_ON_JOB_CLOSE so orphans cannot survive a main-process crash (`win32job` crate).

---

## 4. Rust project structure

```
live_assistant/
└── rust/
    ├── Cargo.toml                 # [workspace] members = ["crates/*", "bin/*"]
    ├── README.md                  # build/run docs, model notes (ggml whisper, safetensors)
    ├── rust-toolchain.toml        # stable, MSVC
    ├── crates/
    │   ├── la-proto/              # IPC message enums + framing (serde_json lines)
    │   ├── la-core/               # agent core library
    │   │   └── src/
    │   │       ├── config.rs      #   load_config (serde structs, path resolution)
    │   │       ├── common.rs      #   cprint, prompt_input, dump_func_call, json io
    │   │       ├── printer.rs     #   StreamPrinter: fences, syntect, spinner
    │   │       ├── input.rs       #   InputReader over rustyline (+ /paste, completer)
    │   │       ├── brain.rs       #   OpenAICore: client, metadata, vision, context
    │   │       ├── agent.rs       #   Brain + Agent: streaming, act() rounds, repair, model_info
    │   │       ├── session.rs     #   ConversationSession + commands + tool launcher
    │   │       ├── tools/         #   registry + one module per Python tools/*.py
    │   │       ├── document_loader.rs
    │   │       ├── embedding_search.rs
    │   │       ├── summarizer.rs
    │   │       ├── lex_rank.rs
    │   │       ├── document_query.rs
    │   │       ├── image_search.rs
    │   │       └── lib.rs
    │   ├── la-ml/                 # candle-based encoders (lazy): MiniLM, CLIP, cross-encoder
    │   ├── la-audio/              # shared: cpal devices, rms, wav (hound), capture_speech
    ├── bin/
    │   ├── la/                    # main binary: main() parity (args, wiring, session)
    │   ├── la-listen/             # worker: capture + whisper-rs
    │   ├── la-speak/              # worker: sherpa-rs VITS + cpal output
    │   └── la-imggen/             # worker: candle stable-diffusion
    └── tests/                     # workspace integration tests (fake servers, worker e2e)
```

Rationale: separate crates keep heavy native deps (whisper.cpp, sherpa-onnx, candle,
pdfium) out of the main binary and out of each other's compile; workers are thin
binaries over `la-proto` + the relevant lib. `cargo build` produces `la`,
`la-listen`, `la-speak`, `la-imggen` (same folder for workers to locate each other).

---

## 5. Selected crates (verified on docs.rs)

### 5.1 Core / async / CLI / config
| Crate (docs.rs) | Purpose |
|---|---|
| [`tokio`](https://docs.rs/tokio) 1.x (`rt-multi-thread`,`macros`,`signal`,`process`,`io-util`,`sync`,`time`,`fs`) | async runtime; `signal::ctrl_c`; spawn+pipe workers; timeouts |
| [`futures`](https://docs.rs/futures) 0.3 | SSE stream combinators |
| [`clap`](https://docs.rs/clap) 4 (derive) | `--chat`, `--no-tools`, `--config`, `--prompt` (parity with `parse_args`) |
| [`serde`](https://docs.rs/serde) + [`serde_json`](https://docs.rs/serde_json) 1 | config structs, IPC, tool args, tool schemas |
| [`toml`](https://docs.rs/toml) 0.9 | read the **same** `config.toml` |
| [`dotenvy`](https://docs.rs/dotenvy) 0.15 | `.env` loading, existing env vars win (like `override=False`) |
| [`anyhow`](https://docs.rs/anyhow) / [`thiserror`](https://docs.rs/thiserror) | app vs. library errors |
| [`log`](https://docs.rs/log) + [`env_logger`](https://docs.rs/env_logger) | logging at `config.logging.level` |
| [`chrono`](https://docs.rs/chrono) 0.4 | `current_date_time`, timestamps, model `created` date |

### 5.2 LLM client
| Crate | Purpose |
|---|---|
| [`async-openai`](https://docs.rs/async-openai) 0.41 | OpenAI-compatible chat completions: **SSE streaming**, **tool calls** (streamed deltas with `tool_calls`), **vision** (`image_url` content parts), custom `base_url` + api key — covers LM Studio/Ollama/Together/Ollama endpoints |
| [`reqwest`](https://docs.rs/reqwest) 0.12 (`json`,`stream`,`gzip`) | raw `GET /models` fallback (bare-array servers, like Python's httpx fallback), web tools, downloads |

### 5.3 Terminal / input
| Crate | Purpose |
|---|---|
| [`crossterm`](https://docs.rs/crossterm) 0.29 | ANSI colors for `cprint`, CR+K-erase for spinner line |
| [`syntect`](https://docs.rs/syntect) 5 | syntax highlighting of fenced code blocks (monokai theme) |
| [`rustyline`](https://docs.rs/rustyline) 18 (`with-file-history`) | chat input: file history, `HistoryHinter` (auto-suggest), `Completer` for `/commands`, multiline |

### 5.4 Web / files / archives
| Crate | Purpose |
|---|---|
| [`aurora_websearch`](https://docs.rs/aurora_websearch) 0.1.0 | **replaces Selenium + scraper for all web tools**: DuckDuckGo search via `AuroraSearch::search_with_limit(query, n)` (HTML client, browser fingerprints, rate-limit awareness, `SafeSearch` levels) and page extraction via `ContentExtractor` (`extract(url)` → `ExtractedContent { title, content, description, links }`, `extract_text`, `extract_markdown`); markdown output preserves headings, so `webpage_headlines` derives h1/h2 parity from `#`/`##` lines; `extract_from_html(url, html)` enables offline fixture tests |
| [`walkdir`](https://docs.rs/walkdir) 2 | recursive sorted file traversal |
| [`zip`](https://docs.rs/zip) | extract (zip-slip guarded) + `zip_files` |
| [`tar`](https://docs.rs/tar) + [`flate2`](https://docs.rs/flate2) + [`bzip2`](https://docs.rs/bzip2) + [`xz2`](https://docs.rs/xz2) | tar/tar.gz/tbz2/txz |
| [`sevenz-rust2`](https://docs.rs/sevenz-rust2) 0.23 (maintained fork of `sevenz-rust` 0.6.1) | .7z extraction |
| [`regex`](https://docs.rs/regex) 1 | tex strip, dot leaders, safe filename, emoji strip |

### 5.5 Documents / images
| Crate | Purpose |
|---|---|
| [`html2text`](https://docs.rs/html2text) 0.17 | HTML→plain text for **local** `.html` files in `document_loader::extract_html` (html5ever-based, no browser) |
| [`pdfium-render`](https://docs.rs/pdfium-render) 0.9 | PDF per-page text + embedded image extraction (needs bundled `pdfium.dll`) |
| [`calamine`](https://docs.rs/calamine) | xlsx sheets→rows |
| [`quick-xml`](https://docs.rs/quick-xml) | docx paragraphs/tables from `word/document.xml` |
| [`image`](https://docs.rs/image) 0.25 | decode/encode png/jpg/webp/bmp/tiff; save generated/captured images |

### 5.6 Audio / STT / TTS
| Crate | Purpose |
|---|---|
| [`cpal`](https://docs.rs/cpal) 0.15+ | input/output streams (WASAPI on Windows), default input device discovery |
| [`hound`](https://docs.rs/hound) 3.5 | WAV read/write (in-memory + files) |
| [`whisper-rs`](https://docs.rs/whisper-rs) 0.16 | whisper.cpp STT: `WhisperContext::new_with_params` + `full` with language + beam; **ggml models** (see §10 risk R1) |
| [`sherpa-rs`](https://docs.rs/sherpa-rs) 0.6.8 (`tts`,`download-binaries`) | sherpa-onnx bindings: `VitsTts` runs **Piper voices** (`.onnx` + `.onnx.json` — same files as `[speech_gen] model_path`); Windows prebuilt binaries |

### 5.7 ML (worker + encoders)
| Crate | Purpose |
|---|---|
| [`candle-core`](https://docs.rs/candle-core), [`candle-nn`](https://docs.rs/candle-nn), [`candle-transformers`](https://docs.rs/candle-transformers) 0.11 | BERT (`models::bert` — has an `_all_mini_lm_l6_v2` config), CLIP (`models::clip`), **Stable Diffusion** (`models::stable_diffusion`: txt2img, img2img+strength, inpaint+mask, guidance, seed — used by `la-imggen`) |
| [`tokenizers`](https://docs.rs/tokenizers) 0.21 | HF tokenizers for BERT/CLIP/SD |
| [`hf-hub`](https://docs.rs/hf-hub) 0.4 | optional Hub download of SD weights when not present locally |
| [`ndarray`](https://docs.rs/ndarray) 0.16 | embeddings, cosine/inner-product ranking (replaces FAISS IndexFlatIP — exact IP = matrix product), LexRank power method, connected components |
| [`md-5`](https://docs.rs/md-5) | image dedup (Python used `hashlib.md5`) |

### 5.8 Finance
| Crate | Purpose |
|---|---|
| [`yahoo_finance_api`](https://docs.rs/yahoo_finance_api) 5.0 (`blocking`) | `get_quote_range`/`get_quote_history_interval` (period+interval parity) |
| [`plotters`](https://docs.rs/plotters) 0.3 | line chart → jpg for `save_financial_chart` |

### 5.9 Process management / testing
| Crate | Purpose |
|---|---|
| [`win32job`](https://docs.rs/win32job) | optional Windows Job-Object hardening (children die with parent) |
| [`wiremock`](https://docs.rs/wiremock) | fake OpenAI-compatible server (SSE + tool calls) in tests |
| [`tempfile`](https://docs.rs/tempfile) 3 | test fixtures |

RMS levels, silence threshold, `clean_for_speech` stripping, passage building,
`has_content` filters, history repair, etc. are small pure functions — implemented
with std + `ndarray`/`regex`, no extra crates.

---

## 6. Config strategy

- `la --config <path>` defaults to `../config.toml` relative to the binary's
  parent dir (i.e. the same file as Python's `CONFIG_FILE`).
- Strongly-typed serde structs mirror every section: `[logging]`, `[speech_det]`,
  `[brain]`, `[agent]`, `[speech_gen]`, `[input]`, `[summarization]`,
  `[document_query]`, `[image_search]`, `[image_gen]`, `[embedding]`.
- Path resolution mirrors `load_config`: keys ending `_file`/`_dir`/`_path` and
  string list items are resolved relative to the config file's dir.
- Additive-only extension: optional `[rust]` section (e.g. `whisper_model =
  "path/to/ggml-medium.bin"`) for knobs that cannot map 1:1. Python ignores
  unknown sections, so the shared file stays compatible and unchanged.
- `.env` via `dotenvy` (never overrides real env vars) → `api_key_env` resolved
  like Python; missing key = startup error with the same guidance message.

---

## 7. Tool parity matrix (`src/tools` → Rust registry)

Tools are registered in a `ToolRegistry`: each entry = `{ name, description,
json_schema, required_params, handler }` (schema hand-written to keep the
docstring-grade descriptions; args deserialized via serde into typed structs).
Order matches `_default_tools` in `src/live_ai.py`.

| Python tool | Rust implementation | Phase |
|---|---|---|
| `model_info` | Brain method: `/models` metadata + context usage % (chars/4 estimate) | 1 |
| `recall_personal`, `store_personal`, `clear_chat` | `[brain].history_path` file + live-history provider binding | 1 |
| `image_response` (vision) | `async-openai` multimodal content (base64 data URL or URL) | 1 |
| `allow_speech` | route to `la-speak` worker (enable/disable) | 2 |
| `summarize_doc` | `la-core::summarizer` + document_loader | 4 |
| `query_document` | `la-core::document_query` (cross-encoder) | 4 |
| `image_search` | `la-core::image_search` (CLIP) | 4 |
| `list_corpora`, `store_corpus`, `query_corpus` | `la-core::embedding_search` (ndarray IP ranking, own `corpus.bin` format — see §10 R4) | 4 |
| `capture_image` | `nokhwa` (`input-native` MSMF) frame → jpg | 3 |
| `system_info`, `current_date_time`, `get_working_directory`, `exit_gracefully` | `std`, `chrono`; exit via a tool `Action::Exit` handled by the session loop (parity with `SystemExit`) | 1 |
| `run_command` | `tokio::process::Command` shell, timeout, cwd | 1 |
| `search_in_file`, `search_pattern_in_files`, `read_text_file`, `replace_string_in_file`, `write_text_file` | std fs + regex | 1 |
| `web_search` | `aurora_websearch::AuroraSearch::search_with_limit(query, n_results)` → `format_search_results` parity (numbered block, ` › ` → `/`) | 3 |
| `webpage_headlines`, `webpage_content` | `aurora_websearch::ContentExtractor`: headlines = `extract_markdown(url)` filtered to `#`/`##` lines (+ page `title`), min length applied; content = `extract_text(url)`/`extract(url)` filtered by `relevant_text_length`; both prefixed `\nFrom [<url>]\n` | 3 |
| `download_file` | reqwest stream → file; Content-Disposition / RFC 5987 filename, content-type extension completion, overwrite flag | 3 |
| `extract_archive`, `zip_files` | zip/tar/flate2/bzip2/xz2/sevenz-rust2; zip-slip guards; exclude-glob | 3 |
| `fetch_financial_data`, `save_financial_chart` | `yahoo_finance_api` + `plotters` | 3 |
| `generate_image`, `unload_image_generator` | route to `la-imggen` worker; default out path `<out_dir>/generated_image_<stamp>.png`; reply `"<abs path> (seed=NN)"` | 2 |

`/tools` launcher: list tools (name + first description line), select by number or
name, prompt for required params (schema properties without a default), coerce
int/float/bool like `_convert_argument`.

---

## 8. Streaming UX parity (`StreamPrinter`)

Ported state machine, unit-tested with golden outputs:

- plain text streamed in `cyan`; per-line classification; ambiguous prefixes
  (indentation + 1-2 backticks) held back until classified;
- fenced blocks: buffer until closing fence, then render via `syntect`
  (language tag from fence, default `text`, monokai theme);
- activity spinner: braille frames ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ every 120 ms, notes
  (`thinking`, `working`, `writing rust code (N chars)`), erased before real
  output and around raw `cprint` banners (output hook), `mark_line_clean`
  between act() rounds;
- turn-end separator: blank line + `context: X.XX % [Nk]` grey line.

---

## 9. Implementation phases

Each phase ends with green validation (§10) and a parity checklist entry (§11).

- **Phase 0 — scaffold**: workspace, toolchain, `cargo fmt/clippy` gates, config
  structs + path resolution, `cprint`/`prompt_input`, logging level, IPC framing
  lib, CI-style script (`rust/check.ps1`).
- **Phase 1 — chat parity (core)**: `Brain`/`Agent` (async-openai: streaming, tool
  rounds, history repair, cancel, model_info, metadata resolution + raw fallback,
  token-ratio cap), `ToolRegistry` + Phase-1 tools, `StreamPrinter`, `InputReader`
  (rustyline), `ConversationSession` (commands, launcher, Ctrl+C/EOF), `la --chat`,
  `--prompt`, `--no-tools`, `--config`; fake-server integration tests.
- **Phase 2 — process workers + voice**: `la-audio` (devices, RMS, wav),
  `la-listen` (calibration, capture loop, whisper-rs), `la-speak` (sherpa-rs VITS,
  gain, acks), `la-imggen` (candle SD t2i; i2i/inpaint as stretch), supervisor
  (spawn, request/response, events, stop, respawn-on-death), voice mode wiring
  (`SpeechDetector` parity: show_levels, beam_size, language, silence timeout),
  Ctrl+C shutdown path, Windows Job object hardening.
- **Phase 3 — utility tools**: web/download/archives/camera/finance + `/tools`
  launcher completion + graceful-exit tool.
- **Phase 4 — ML tool parity**: document_loader (pdf/docx/xlsx/html), summarizer
  (MiniLM via candle + LexRank ndarray port), embedding corpora (build/query,
  metadata.json, processed.md, dedup, orphan prune), document_query
  (cross-encoder), image_search (CLIP + mtime cache).
- **Phase 5 — hardening & docs**: perf checks on the CPU reference machine,
  `rust/README.md` (model prep: ggml whisper, safetensors conversions, pdfium
  dll, sherpa binaries), final parity run (§11), version 1.0 tag.

---

## 10. Codebase validation

### 10.1 Static gates (every commit)
```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo check --workspace
cargo test --workspace
```

### 10.2 Unit tests (mirror the Python suite, seam-for-seam)
Rust tests use the same dependency-injection style as the Python seams
(`stream_factory`, `model_factory`, ... become trait objects / generic params):

| Python test file | Rust test target | Key cases |
|---|---|---|
| `test_common.py` | `la-core::printer` (+config) | fence state machine, spinner suppression, config path resolution |
| `test_input_reader.py` | `la-core::input` | `/paste` accumulation, argument conversion, history file creation |
| `test_session.py` | `la-core::session` | commands, EOF, tool launcher, shutdown, banner |
| `test_live_ai.py` | `la-core::agent` | act() rounds, streamed tool-call accumulation, history repair, cancel mid-stream, token cap math, metadata/context-window resolution |
| `test_main.py` | `bin/la` | arg parsing defaults/flags, no side effects at import (pure `main()`), single-prompt turn |
| `test_audio_wrapper.py` | `la-audio` (+`listen` worker) | `capture_speech` with fake chunk streams (threshold, timeouts, trimming, clear timeout), RMS, wav buffers |
| `test_audio_generator.py` | `la-speak` | clean_for_speech, samplerate mismatch error, empty-text no-op |
| `test_image_generator.py` | `la-imggen` | `check_size`, defaults, out-path stamping (pipeline itself behind an `#[ignore]` heavy test) |
| `test_lex_rank.py` | `la-core::lex_rank` | synthetic matrices → centrality |
| `test_embedding_search.py` | `la-core::embedding_search` | build_passages, has_content, normalize_rows, sanitize names, corpus save/load/query with fake embedder/index |
| `test_document_loader.py` | `la-core::document_loader` | per-extractor golden tests, tex strip, dot leaders, image usefulness, iter order |
| `test_summarizer.py` / `test_document_query.py` / `test_image_search.py` | same-named modules | ratio/threshold/max-sentences logic, k resolution, caching by mtime, error strings (fake encoders) |
| `test_tools.py` | per-tool module tests | string outputs parity (error messages included) |

Parity guard: a test compares the registered tool list (names in order) against a
checked-in snapshot generated from `_default_tools`.

### 10.3 Integration tests
- **Fake OpenAI server** (`wiremock`): SSE chunks incl. partial `tool_calls`
  deltas; asserts streamed fragments, round loop, tool result messages, cancel.
- **Worker e2e**: spawn built worker binaries from tests, drive the JSON-line
  protocol (listen with a fake capture backend; speak with a null output device;
  imggen `unload` round-trip).
- **Web tools offline**: aurora_websearch `ContentExtractor::extract_from_html` on checked-in HTML fixtures (headline/content formatting parity); live DuckDuckGo search behind `#[ignore]`d tests (network).
- **CLI e2e**: `la --prompt "hi"` against the fake server asserts the final
  history message and exit code 0.

### 10.4 Manual parity runbook (needs real hardware/models)
1. `la` (voice): silence calibration prints, RMS line, transcript → answer → `/voice on` speaks.
2. Ctrl+C during generation cancels; partial text kept; history stays valid for the next turn.
3. `la --chat`: history recall, `/`-completion, paste, `/clear`, `/info`, `/quit`.
4. `generate_image` → file under `.out_images/` + seed reply; `unload_image_generator` frees it.
5. `store_corpus`/`query_corpus`/`summarize_doc`/`query_document`/`image_search` on the sample folders.
6. Long-context server (LM Studio) → context-window auto-detection; Together-style bare-array `/models` → raw fallback.

### 10.5 Acceptance criteria (v1.0)
- `cargo test --workspace` green (heavy-model tests behind `#[ignore]`/features).
- All §11 checklist rows implemented or explicitly deferred with rationale.
- `main.py`, `src/`, `config.toml`, `samples/`, `playground/` untouched (`git status` clean for those paths).

---

## 11. Behavior parity checklist

- [ ] `--chat`, `--prompt`, `--no-tools`, `--config` (+ defaults)
- [ ] voice loop: calibrate → listen → transcribe → act → speak-if-enabled
- [ ] `/help /quit /clear /voice /tools /info`, `exit` alias, unknown-command error
- [ ] `/tools` interactive launcher with required-param prompting + type coercion
- [ ] bootstrap prompt after banner
- [ ] streaming output with fence highlighting + spinner + context line
- [ ] cancel generation (Ctrl+C) keeps partial output + placeholder on empty cancel
- [ ] history repair before each prompt; `clear_chat` reseeds with recalled personal data
- [ ] `.env` + `api_key_env` resolution; missing key → startup error
- [ ] server capability detection: vision + context window (+ raw fallback)
- [ ] `max_response_tokens` ratio cap
- [ ] model_info format (endpoint, provider, created, context, capabilities, usage %)
- [ ] all §7 tools present with matching names/descriptions/behavior
- [ ] graceful exit tool; speech toggle lifecycle; image model load/unload lifecycle
- [ ] workers started/stopped with Ctrl+C; respawn on death; stdin-EOF watchdog
- [ ] Windows-first (MSMF camera, WASAPI audio, Job object), POSIX-clean code paths

---

## 12. Risks & mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | `faster-whisper` (CTranslate2) models are **not** loadable by whisper.cpp — it needs ggml `.bin` | `la-listen` accepts a directory and auto-selects `*.bin`; README documents `ggml-medium` download (Hugging Face `ggerganov/whisper.cpp`); optional `[rust] speech_det.model_path` override. compute_type maps to quantization choice (`int8` → quantized ggml if present); `device=auto` → CPU (CUDA/Vulkan possible via whisper-rs features later) |
| R2 | Piper synthesis knobs differ in sherpa-onnx (no `volume`/`normalize_audio` in VITS config) | volume = digital gain in the playback path; `normalize_audio` = best-effort RMS normalize in the worker; `speech_speed` → `length_scale = 1/speed`; samplerate enforced from the voice's `.onnx.json` (same mismatch error as Python) |
| R3 | Cross-encoder `ms-marco-TinyBERT-L2` ships only `pytorch_model.bin` (candle reads safetensors) | one-time prep documented in `rust/README.md`: export weights to safetensors via the Python env (no code changes) into `.res/crossencoder/<name>-safetensors/`; Phase-4 tests use fake encoders |
| R4 | Python corpora are `pickle` (Python-only format) | Rust writes its own `corpus.bin` (bincode-style records+embeddings) next to `metadata.json`/`processed.md`; same folder layout and semantics; re-`store_corpus` required once. Python corpora remain usable from Python |
| R5 | CLIP scores may differ slightly from sentence-transformers (pooling/projection details) | implement sentence-transformers pooling exactly (mean-pool + projection) and validate against Python on a fixed folder (golden scores within tolerance) |
| R6 | Native build complexity on Windows (whisper.cpp needs CMake/MSVC; pdfium needs the DLL; sherpa prebuilt via `download-binaries`) | phased crates so core compiles without ML backends; feature flags; prebuilt binary downloads documented; CI-lite `check.ps1` |
| R7 | CPU-only reference machine (8 cores, 24 GB): SD on CPU takes minutes | worker process keeps the pipeline warm across calls (parity with Python cache); docs recommend lower steps; generation time printed like Python |
| R8 | `src/local_model_handler.py` is deliberately not ported (scope decision, §1) | no action; if ever needed later, implement over REST (`/models` for LM Studio, `/api/tags`/`/api/ps`/`/api/generate`/`/api/show` for Ollama) as a future extension — recorded here as accepted non-parity |
| R9 | DuckDuckGo HTML endpoint changes / rate limits (`aurora_websearch` 0.1.0 is a young crate) | crate ships browser fingerprints + rate-limit handling; wrap calls in retry/backoff; contingency: hand-roll the DuckDuckGo HTML client with `reqwest` if the crate breaks |

---

## 13. Deliverables

1. `rust/` Cargo workspace (§4) with all binaries and crates.
2. `rust/README.md`: build, model preparation (R1/R3), run instructions, feature flags.
3. `rust/check.ps1`: fmt+clippy+test one-shot gate (Windows PowerShell).
4. `rust/setup_rust_res.ps1`: downloads every model/binary required by the Rust
   version into the repo's `.res/` folder; the required list is **derived** by
   parsing `config.toml` (each section's model keys) plus the Rust-specific needs
   table, and skips artifacts that are already present:
   - `[speech_det]` → whisper.cpp ggml model (e.g. `ggerganov/whisper.cpp` `ggml-medium.bin`) stored under `.res/whisper/` (CT2 models are not loadable by whisper.cpp — R1; wired via the `[rust]` override, §6)
   - `[speech_gen]` → Piper voice `.onnx` + `.onnx.json` (rhasspy/piper-voices) into the configured `model_path`
   - `[summarization]`/`[embedding]` → all-MiniLM-L6-v2 and clip-ViT-B-32 (safetensors + config + tokenizer) into the configured `model_path`s
   - `[document_query]` → ms-marco-TinyBERT-L2 weights + one-time safetensors conversion via the existing Python env (R3)
   - `[image_gen]` → t2i/i2i models when given as HF ids (`hf download` or direct HTTPS resolve)
   - non-config binary need → prebuilt `pdfium.dll` (bblanchon/pdfium-binaries release) next to the built binaries
   - flags: `-SkipExisting` (default), `-WhatIf` dry-run listing, `-Only <section>`; works offline for artifacts already stored by the Python setup
5. Test suites per §10 and the parity checklist §11 updated as phases land.
6. `PLAN.md` (this file) updated with phase-completion statuses as work progresses.
