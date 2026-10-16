import inspect
import json
import os
import pathlib
import time
import types
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints
from logging import getLogger
from datetime import datetime, timezone
from urllib.parse import urlsplit
from src.common import SimpleNamespace, dump_func_call, cprint, crule, StreamPrinter
from src.tools import (
    exit_gracefully, system_info, current_date_time, get_working_directory,
    read_text_file, write_text_file, run_command,
    web_search, web_images, web_videos, web_news, web_books, extract_webpage,
    capture_image,
    generate_image, unload_image_generator,
    summarize_doc, query_document, image_search,
    list_corpora, store_corpus, query_corpus,
    recall_personal, store_personal, clear_chat, allow_speech, image_response,
    download_file, extract_archive, zip_files,
    search_pattern_in_files, replace_string_in_file, search_in_file,
    fetch_financial_data, save_financial_chart,
)
from src.tools import speech as speech_tools


# Padded into the history when a generation is cancelled before any text was
# produced: an empty assistant turn would be dropped by the model's prompt
# template and break user/assistant role alternation on the next request.
CANCELLED_PLACEHOLDER = "(generation cancelled by user)"


def _validated_style(value: str | None, default: str = "cyan") -> str:
    """`[ui]` assistant color: any rich style name, falling back to `default`."""
    value = value or default
    try:
        from rich.style import Style
        Style.parse(value)
    except Exception:
        return default
    return value


def _validated_code_theme(value: str | None, default: str = "monokai") -> str:
    """`[ui]` code theme: any Pygments style name, falling back to `default`."""
    value = value or default
    try:
        from pygments.styles import get_style_by_name
        get_style_by_name(value)
    except Exception:
        return default
    return value


def __getattr__(name: str):
    """Backward-compatible access to ``SpeechGenerator``.

    The speech engine is owned by ``src.tools.speech`` now; this shim keeps
    older callers (and tests) that import it from this module working without
    pulling in the heavy ``piper``/``sounddevice`` stack at import time.
    """
    if name == "SpeechGenerator":
        from src.audio_generator import SpeechGenerator
        return SpeechGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _response_token_cap(max_response_tokens: float, context_window: int) -> int:
    """Resolve ``[brain] max_response_tokens`` into a per-generation cap.

    The config value is a ratio of the model's context window, so the cap
    self-adapts to whichever model is configured: ``0.1`` on a 4k-window model
    caps a response at ~409 tokens. Valid range is 0 < ratio < 1; unset/0
    keeps the historical default of the full context window (no practical
    cap). The cap bounds any single generation - including one the provider
    cannot stop instantly after Ctrl+C (e.g. cancelled during prompt
    processing) - so a stale prediction can never monopolise the model.
    """
    if not max_response_tokens:
        return context_window
    try:
        ratio = float(max_response_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"brain.max_response_tokens must be a number (got {max_response_tokens!r})"
        ) from exc
    if not 0 < ratio < 1:
        raise ValueError(
            "brain.max_response_tokens must be a ratio with 0 < ratio < 1, "
            f"got {max_response_tokens!r}"
        )
    return max(1, int(context_window * ratio))


def _default_tools(agent: "Brain") -> list[Callable]:
    """Tool set of the agent.

    Only ``model_info`` stays a session-bound method (model/endpoint/usage
    state); every other tool lives in a ``src.tools`` submodule and gets its
    config injected via ``src.tools.configure`` (called by ``main.py``).
    """
    return [
        agent.model_info,
        recall_personal, store_personal, clear_chat,
        image_response, allow_speech,
        summarize_doc, query_document, image_search,
        list_corpora, store_corpus, query_corpus,
        capture_image, system_info, current_date_time, get_working_directory,
        run_command, search_in_file, search_pattern_in_files,
        read_text_file, replace_string_in_file, write_text_file,
        web_search, web_images, web_videos, web_news, web_books,
        extract_webpage, download_file, extract_archive, zip_files,
        fetch_financial_data, save_financial_chart,
        generate_image, unload_image_generator,
        exit_gracefully
    ]


def _load_env_file(dotenv_path: pathlib.Path | None = None) -> None:
    """Load ``.env`` from the project root (next to ``main.py``).

    Optional convenience: only active when ``python-dotenv`` is installed, so
    secrets can live in a ``.env`` file instead of the shell. Existing
    environment variables always win over file entries (``override=False``),
    and the file is never required.
    """
    if dotenv_path is None:
        dotenv_path = pathlib.Path(__file__).resolve().parents[1] / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=dotenv_path, override=False)


class OpenAICore:
    """
    Core client for an OpenAI-compatible chat API (API-key providers).

    Talks to an OpenAI-compatible endpoint with the ``openai`` client, so
    cloud models (OpenAI, Mistral, Groq, OpenRouter, DeepSeek, Together,
    Gemini, xAI, Anthropic, ...) and local servers (Ollama, LM Studio) share
    one implementation: connection handling, model metadata resolution
    (context window, vision) from the OpenAI models endpoint - plus the
    native APIs of local servers, whose OpenAI /models data is thin - and the
    function-calling helpers (JSON Schema tool definitions, stream chunk
    parsing).
    The connection comes from the models JSON file (``[brain] models_file``):
    its first entry boots the session and ``/model`` switches between the
    entries at runtime; without a models file the legacy ``[brain]``
    model/base_url/api_key_env keys are used. The API key is read from the
    environment variable named by the entry's ``api_key_env`` (also resolved
    from a project ``.env`` file) and must never be stored in config.toml or
    the models file.
    """

    DEFAULT_CONTEXT_WINDOW = 128_000

    def __init__(self, config: SimpleNamespace):
        self.logger = getLogger(self.__class__.__name__)
        self.config = config
        try:
            from openai import OpenAI  # lazy import: keeps startup light
        except ImportError as exc:
            raise ImportError("pip install openai  # required for the OpenAI-based Brain") from exc

        model, base_url, api_key = self._boot_connection()

        self.logger.info("🚀 Connecting to remote model '%s' @ %s ...", model, base_url)
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model_name = model
        self._base_url = base_url
        self._context_window = self._resolve_context_window(config)
        self._vision = self._resolve_vision_flag(config)
        self._temperature = getattr(config.brain, "temperature", 0.7)
        self._max_tokens = _response_token_cap(
            getattr(config.brain, "max_response_tokens", 0),
            self._context_window,
        )
        self._messages: list[dict] = [{"role": "system", "content": config.brain.system_prompt}]
        ui = getattr(config, "ui", None)
        self._stream = StreamPrinter(
            text_color=_validated_style(getattr(ui, "assistant_color", None)),
            code_theme=_validated_code_theme(getattr(ui, "code_theme", None)),
        )
        self._cancel_requested = False
        self._active_stream = None
        # Session-bound tool modules: their config comes from main.py via
        # src.tools.configure; the live state is bound here (getter first:
        # the messages list is re-read on every access, so clearing or
        # reassigning the session history stays visible to the tools).
        from src.tools import memory as memory_tools, vision as vision_tools
        memory_tools.bind_history(lambda: self._messages)
        vision_tools.bind_session(self)
        self.logger.info("✅ Remote model '%s' connected.", self._model_name)

    ## models file: boot connection + /model switching ##

    def _models_file(self) -> pathlib.Path | None:
        """Models JSON file from ``[brain] models_file`` (None when unset)."""
        value = getattr(self.config.brain, "models_file", None)
        return pathlib.Path(value) if value else None

    def _load_model_entries(self) -> dict[str, dict]:
        """Read the models JSON file into ``{name: entry}`` (file order kept).

        Accepted shapes: a ``name -> entry`` mapping, a list of entries each
        carrying a ``name``, and both wrapped in a top-level ``"models"`` key.
        A bare string stands for a model id selectable under its own name.
        Every usable entry needs a ``model`` id; anything else is skipped.
        """
        path = self._models_file()
        if path is None:
            raise ValueError(
                "No models file configured: set [brain] models_file in config.toml")
        if not path.is_file():
            raise FileNotFoundError(f"Models file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in models file {path}: {exc}") from exc
        if isinstance(data, dict) and isinstance(data.get("models"), (list, dict)):
            data = data["models"]
        if isinstance(data, dict):
            pairs = list(data.items())
        elif isinstance(data, list):
            pairs = []
            for entry in data:
                if isinstance(entry, dict):
                    pairs.append((entry.get("name"), entry))
                else:  # bare model id: selectable under its own name
                    text = str(entry).strip()
                    pairs.append((text, {"model": text}))
        else:
            raise ValueError(
                f"Models file {path} must hold a name->entry mapping or a list of entries")
        entries: dict[str, dict] = {}
        for name, entry in pairs:
            if isinstance(entry, str):  # shorthand mapping value: {"alias": "model-id"}
                entry = {"model": entry.strip()}
            name = str(name or "").strip()
            if name and isinstance(entry, dict) and str(entry.get("model") or "").strip():
                entries[name] = entry
        if not entries:
            raise ValueError(
                f"No usable model entries in models file {path} "
                "(every entry needs a name and a 'model' id)")
        return entries

    def list_models(self) -> dict[str, dict]:
        """Selectable models from the models JSON file (``{name: entry}``)."""
        return self._load_model_entries()

    def _entry_connection(self, display_name: str, entry: dict) -> tuple[str, str, str]:
        """Connection triple ``(model, base_url, api_key)`` for a models entry.

        Entry values win; missing ``base_url``/``api_key_env`` fall back to the
        ``[brain]`` config (legacy setups keep them there). The API key is
        resolved from the environment (``.env`` included) exactly as at
        startup and is never stored in the JSON file.
        """
        brain = self.config.brain
        model = str(entry.get("model") or "").strip()
        base_url = str(entry.get("base_url")
                       or getattr(brain, "base_url", "") or "").strip()
        api_key_env = str(entry.get("api_key_env")
                          or getattr(brain, "api_key_env", "") or "").strip()
        if not model:
            raise ValueError(f"Model entry {display_name!r} has no 'model' id")
        if not base_url:
            raise ValueError(
                f"Model entry {display_name!r} has no 'base_url' and none is configured")
        if not api_key_env:
            raise ValueError(
                f"Model entry {display_name!r} has no 'api_key_env' and none is configured")
        _load_env_file()  # fill unset variables from .env (real env vars win)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found for model {display_name!r}: set the '{api_key_env}' "
                "environment variable, e.g. put "
                f"'{api_key_env}=sk-...' in the project .env file "
                f"or run: setx {api_key_env} 'sk-...'")
        return model, base_url, api_key

    def _boot_connection(self) -> tuple[str, str, str]:
        """Resolve the startup connection: ``(model, base_url, api_key)``.

        With ``[brain] models_file`` configured, the **first entry** of the
        models JSON file boots the session: the startup model is always one of
        the ``/model``-switchable entries and ``[brain]`` carries no duplicate
        ``model``/``base_url``/``api_key_env``. Without a models file the
        legacy ``[brain]`` keys are used directly.
        """
        brain = self.config.brain
        if getattr(brain, "models_file", None):
            entries = self._load_model_entries()
            display_name, entry = next(iter(entries.items()))
            self._model_entry = display_name
            self.logger.info("Boot model: first entry '%s' of the models file.",
                             display_name)
            return self._entry_connection(display_name, entry)
        model = str(getattr(brain, "model", "") or "").strip()
        base_url = str(getattr(brain, "base_url", "") or "").strip()
        api_key_env = str(getattr(brain, "api_key_env", "") or "").strip()
        if not base_url:
            raise ValueError("Missing [brain] base_url")
        if not model:
            raise ValueError("Missing [brain] model")
        _load_env_file()  # fill unset variables from .env (real env vars win)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found: set the '{api_key_env}' environment variable, "
                f"e.g. put '{api_key_env}=sk-...' in the project .env file "
                f"or run: setx {api_key_env} 'sk-...'")
        return model, base_url, api_key

    def select_model(self, name: str) -> str:
        """Switch the live session to a model entry from the models JSON file.

        Every entry carries ``model`` (the model id) plus optional
        ``api_key_env`` and ``base_url``; missing connection fields keep the
        values currently configured under ``[brain]``. The API key is resolved
        from the environment (``.env`` included) exactly as at startup and is
        never stored in the JSON file. On success the client is rebuilt for
        the new endpoint, the cached model metadata is dropped and the context
        window / vision support are re-resolved from the new server (the
        per-generation token cap follows the new window). The conversation
        history is kept. Any failure raises before the current model is
        touched; success returns a human-readable confirmation.
        """
        entries = self._load_model_entries()
        lookup = {key.lower(): (key, entry) for key, entry in entries.items()}
        match = lookup.get(str(name or "").strip().lower())
        if match is None:
            raise KeyError(
                f"Unknown model {name!r} in the models file. Available: "
                + ", ".join(entries))
        display_name, entry = match
        model, base_url, api_key = self._entry_connection(display_name, entry)

        self.logger.info("🔄 Switching to model '%s' @ %s ...", model, base_url)
        from openai import OpenAI  # installed: __init__ already imported it
        client = OpenAI(base_url=base_url, api_key=api_key)

        # Validation passed: swap the connection, drop the cached metadata and
        # re-resolve the model-dependent state from the new server.
        self._client = client
        self._model_name = model
        self._base_url = base_url
        self._model_entry = display_name
        self.__dict__.pop("_model_meta", None)
        self._context_window = self._resolve_context_window(self.config)
        self._vision = self._resolve_vision_flag(self.config)
        self._max_tokens = _response_token_cap(
            getattr(self.config.brain, "max_response_tokens", 0), self._context_window)
        self.logger.info(
            "✅ Model '%s' is now active (context %d, vision %s).",
            model, self._context_window, "yes" if self._vision else "no",
        )
        return (f"Model switched to {display_name}: {model} @ {base_url} "
                f"(context window {self._context_window}, "
                f"vision {'yes' if self._vision else 'no'})")

    def _resolve_vision_flag(self, config) -> bool:
        """Derive vision support from the server-reported model capabilities.

        The OpenAI models endpoint may include a ``capabilities`` list (e.g.
        LM Studio: ["completion", "tools", "vision"]); when the server
        reports it, that value is authoritative. Local servers are covered by
        the native-API fallback in ``_fetch_model_metadata``; servers still
        reporting nothing (most cloud providers) fall back to the optional
        ``[brain] vision`` config flag (default False), so the key stays
        usable as an override but is no longer required.
        """
        configured = bool(getattr(config.brain, "vision", False))
        capabilities = self._fetch_model_metadata().get("capabilities")
        if not isinstance(capabilities, (list, tuple)) or not capabilities:
            self._vision_source = "config"
            self.logger.info(
                "Model '%s' vision: %s (config; server did not report capabilities)",
                self._model_name, configured,
            )
            return configured
        self._vision_source = "server"
        has_vision = "vision" in [str(c).lower() for c in capabilities]
        self.logger.info(
            "Model '%s' capabilities (server): %s -> vision=%s",
            self._model_name, ", ".join(map(str, capabilities)), has_vision,
        )
        return has_vision

    def _resolve_context_window(self, config) -> int:
        """Resolve the context window: explicit config > server report > default.

        ``[brain] context_window`` stays authoritative when set. When unset
        (0 or missing), the window is derived from the server-reported model
        metadata via the OpenAI /models endpoint - e.g. LM Studio's
        ``max_context_length`` or Together's ``context_length`` - preferring
        explicit maximums over the currently loaded allocation
        (``loaded_context_length``). Servers reporting nothing (e.g. Ollama)
        fall back to ``DEFAULT_CONTEXT_WINDOW``.
        """
        try:
            configured = int(getattr(config.brain, "context_window", 0) or 0)
        except (TypeError, ValueError):
            configured = 0
        if configured:
            self._context_window_source = "config"
            return configured
        meta = self._fetch_model_metadata()
        context_fields = {
            k: v for k, v in meta.items()
            if "context" in k.lower() and isinstance(v, int)
        }
        for key in self._CONTEXT_MAX_KEYS:
            if key in context_fields:
                return self._adopt_server_context_window(key, context_fields[key])
        for key in sorted(context_fields):
            if not key.lower().startswith("loaded"):
                return self._adopt_server_context_window(key, context_fields[key])
        self._context_window_source = "default"
        self.logger.info(
            "Model '%s' context window: default %d (server did not report one)",
            self._model_name, self.DEFAULT_CONTEXT_WINDOW,
        )
        return self.DEFAULT_CONTEXT_WINDOW

    def _adopt_server_context_window(self, key: str, value: int) -> int:
        """Adopt a server-reported context length and log where it came from."""
        self._context_window_source = "server"
        self.logger.info(
            "Model '%s' context window from server (%s): %d",
            self._model_name, key, value,
        )
        return value

    _MODEL_RECORD_SKIP = {"id", "object", "root", "parent", "permission"}

    # Server metadata keys describing the model's maximum context, in the
    # order they are preferred when several are reported.
    _CONTEXT_MAX_KEYS = (
        "max_context_length", "context_length", "context_window", "max_model_len",
    )

    @staticmethod
    def _model_record_fields(record) -> dict:
        """Flatten an OpenAI SDK model record into a plain dict.

        Accepts pydantic records (``model_dump()``, used by the real SDK
        client), plain dicts (raw fallback) and SimpleNamespace-based fakes
        (``__dict__``, tests), so server-specific extra fields survive the
        round-trip.
        """
        if isinstance(record, dict):
            return dict(record)
        dump = getattr(record, "model_dump", None)
        data = dump() if callable(dump) else None
        if not isinstance(data, dict) or not data:
            data = dict(getattr(record, "__dict__", {}) or {})
        return data

    def _fetch_model_metadata(self, use_raw_fallback: bool = True) -> dict:
        """Query the server for model metadata via the OpenAI models endpoint.

        Only OpenAI-style requests are used: ``GET /models/{id}`` first, then
        ``GET /models`` + lookup by id as fallback (some servers, e.g. Ollama,
        only implement the list variant). The standard fields are id/object/
        created/owned_by; any extra fields a server returns beyond those (e.g.
        LM Studio: max_context_length, loaded_context_length, capabilities,
        quantization, arch) are passed through untouched.

        When the OpenAI endpoint reports neither capabilities nor context
        lengths (true for every local server), the server's native API is
        probed as well (see ``_fetch_native_model_metadata``) and fills the
        missing keys (existing keys are never overwritten).

        ``use_raw_fallback=False`` skips the tolerant re-request for servers
        returning a bare JSON array. Failures degrade silently. Successful
        results are cached on the instance (model metadata is static for the
        session), while a total failure stores no cache so a later call can
        retry. Returns a dict of non-empty scalar/list-of-scalar values,
        possibly {}.
        """
        cached = getattr(self, "_model_meta", None)
        if cached is not None:
            return cached
        record = None
        try:
            record = self._client.models.retrieve(self._model_name)
        except Exception as exc:
            self.logger.debug("GET /models/%s failed: %s", self._model_name, exc)
        if record is None:
            try:
                models = self._client.models.list().data
                record = next((m for m in models if m.id == self._model_name), None)
            except Exception as exc:
                self.logger.debug("GET /models failed: %s", exc)
                if use_raw_fallback:
                    record = self._find_in_raw_models_list()
        meta: dict = {}
        if record is not None:
            for key, value in self._model_record_fields(record).items():
                if key in self._MODEL_RECORD_SKIP or value in (None, ""):
                    continue
                if isinstance(value, (list, tuple)):
                    # Keep lists of scalars (capabilities); drop object lists
                    # (OpenAI's permission entries) - model info stays flat.
                    if not value or not all(
                        isinstance(v, (str, int, float, bool)) for v in value
                    ):
                        continue
                elif not isinstance(value, (str, int, float, bool)):
                    continue  # skip nested structures
                meta[key] = value
        if not meta.get("capabilities") or not any(
                "context" in k.lower() for k in meta):
            # Local servers report neither through the OpenAI endpoint; fill
            # the gaps from their native APIs (no-op for cloud hosts).
            for key, value in self._fetch_native_model_metadata().items():
                meta.setdefault(key, value)
        if meta:
            self._model_meta = meta
        return meta

    def _find_in_raw_models_list(self):
        """Lookup fallback for servers the OpenAI SDK cannot parse.

        Some OpenAI-compatible servers (e.g. Together.ai) answer ``GET
        /models`` with a bare JSON array instead of the standard
        ``{"object": "list", "data": [...]}`` envelope, which makes the SDK
        fail while parsing. This retries the same endpoint with the SDK's own
        httpx client and accepts both shapes. Returns the matching record
        (plain dict) or None.
        """
        try:
            import httpx
            resp = httpx.get(
                self._base_url.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {self._client.api_key}"},
                timeout=5.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self.logger.debug("Raw GET /models failed: %s", exc)
            return None
        if isinstance(data, dict):
            data = data.get("data") or []
        return next(
            (m for m in data if isinstance(m, dict) and m.get("id") == self._model_name),
            None,
        )

    ## native-API metadata (local servers) ##

    _LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

    def _server_root(self) -> str:
        """The server root URL: the OpenAI base_url without a trailing '/v1'."""
        base = self._base_url.rstrip("/")
        return base[:-3] if base.endswith("/v1") else base

    def _fetch_native_model_metadata(self) -> dict:
        """Model metadata from a local server's native API.

        The OpenAI ``/models`` endpoints of local servers are thin (no
        capabilities, no context lengths), while their native APIs carry the
        full picture: Ollama (``POST /api/show`` + ``GET /api/ps``) and LM
        Studio (``GET /api/v0/models`` - ``type == "vlm"`` adds a "vision"
        capability, the REST capability list only covers tool use). Tries
        Ollama first, then LM Studio; probes only loopback hosts and returns
        {} when neither server answers, so cloud providers are unaffected and
        callers degrade to the OpenAI-endpoint data.
        """
        if urlsplit(self._base_url).hostname not in self._LOOPBACK_HOSTS:
            return {}  # the native APIs are a local-server thing
        for fetcher in (self._ollama_native_metadata, self._lmstudio_native_metadata):
            meta = fetcher()
            if meta:
                return meta
        return {}

    def _ollama_native_metadata(self) -> dict:
        """Ollama-native metadata for the model ({} when not an Ollama server)."""
        try:
            import httpx
            resp = httpx.post(
                self._server_root() + "/api/show",
                json={"model": self._model_name},
                # local servers need no auth; tolerate clients without api_key
                headers={"Authorization": f"Bearer {getattr(self._client, 'api_key', '')}"},
                timeout=5.0,
            )
            resp.raise_for_status()
            shown = resp.json()
            if not isinstance(shown, dict):
                return {}
        except Exception as exc:
            self.logger.debug("GET /api/show failed: %s", exc)
            return {}
        meta: dict = {}
        capabilities = shown.get("capabilities")
        if isinstance(capabilities, (list, tuple)) and capabilities:
            meta["capabilities"] = [str(c) for c in capabilities]
        modelinfo = shown.get("model_info")
        if isinstance(modelinfo, dict):
            max_ctx = self._ollama_context_length(modelinfo)
            if max_ctx:
                meta["max_context_length"] = max_ctx
        loaded_ctx = self._ollama_loaded_context()
        if loaded_ctx:
            meta["loaded_context_length"] = loaded_ctx
        return meta

    @staticmethod
    def _ollama_context_length(modelinfo: dict) -> int | None:
        """Trained maximum context from Ollama's modelinfo blob.

        Stored as ``<architecture>.context_length``, with the architecture
        also under ``general.architecture`` (mirrors src.local_model_handler).
        """
        arch = modelinfo.get("general.architecture")
        if arch and isinstance(modelinfo.get(f"{arch}.context_length"), int):
            return int(modelinfo[f"{arch}.context_length"])
        for key, value in modelinfo.items():
            if key.endswith(".context_length") and isinstance(value, int):
                return int(value)
        return None

    def _ollama_loaded_context(self) -> int | None:
        """The model's currently allocated context (``GET /api/ps``), if loaded."""
        try:
            import httpx
            resp = httpx.get(self._server_root() + "/api/ps", timeout=5.0)
            resp.raise_for_status()
            models = resp.json().get("models") or []
        except Exception as exc:
            self.logger.debug("GET /api/ps failed: %s", exc)
            return None
        for entry in models:
            if isinstance(entry, dict) and self._model_name in (
                    entry.get("model"), entry.get("name")):
                if isinstance(entry.get("context_length"), int):
                    return entry["context_length"]
        return None

    def _lmstudio_native_metadata(self) -> dict:
        """LM Studio-native metadata for the model ({} when not LM Studio)."""
        try:
            import httpx
            resp = httpx.get(self._server_root() + "/api/v0/models", timeout=5.0)
            resp.raise_for_status()
            records = resp.json().get("data") or []
        except Exception as exc:
            self.logger.debug("GET /api/v0/models failed: %s", exc)
            return {}
        record = next(
            (r for r in records
             if isinstance(r, dict) and r.get("id") == self._model_name),
            None,
        )
        if record is None:
            return {}
        meta: dict = {}
        for key in ("type", "publisher", "arch", "quantization", "state",
                    "max_context_length", "loaded_context_length"):
            value = record.get(key)
            if value not in (None, ""):
                meta[key] = value
        capabilities = [str(c) for c in (record.get("capabilities") or [])]
        if record.get("type") == "vlm" and "vision" not in capabilities:
            capabilities.append("vision")  # LM Studio marks vision models by type
        if capabilities:
            meta["capabilities"] = capabilities
        return meta

    ## OpenAI function-calling helpers (tool definitions, stream chunks) ##

    _JSON_TYPES: dict[type, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}

    @staticmethod
    def _json_schema(annotation: Any) -> dict:
        """Best-effort JSON Schema for a Python type annotation.

        ``Annotated[T, "description"]`` maps the string metadata to the schema's
        ``description``, so tools can document parameters for the model.
        """
        metadata = getattr(annotation, "__metadata__", None)  # typing.Annotated
        if metadata:
            schema = OpenAICore._json_schema(get_args(annotation)[0])
            for meta in metadata:
                if isinstance(meta, str):
                    schema["description"] = meta
                    break
            return schema
        origin = get_origin(annotation)
        if origin is Literal:
            return {"type": "string", "enum": list(get_args(annotation))}
        if origin in (list, set, tuple):
            args = get_args(annotation)
            return {"type": "array", "items": OpenAICore._json_schema(args[0]) if args else {}}
        if origin is dict:
            return {"type": "object"}
        if origin is Union or origin is types.UnionType:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            return OpenAICore._json_schema(args[0]) if len(args) == 1 else {"type": "string"}
        json_type = OpenAICore._JSON_TYPES.get(annotation)
        return {"type": json_type} if json_type else {"type": "string"}

    @staticmethod
    def _openai_tool_definition(tool: Callable) -> dict:
        """Describe ``tool`` in OpenAI function-calling (JSON Schema) format.

        Parameter names keep their snake_case spelling: OpenAI-compatible models
        handle it natively, and the arguments come back as a JSON blob that can be
        passed to the implementation directly.
        """
        signature = inspect.signature(tool)
        try:
            hints = get_type_hints(tool, include_extras=True)
        except Exception:
            hints = getattr(tool, "__annotations__", {}) or {}

        properties: dict = {}
        required: list[str] = []
        for name, param in signature.parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            schema = OpenAICore._json_schema(hints.get(name, str))
            if param.default is inspect.Parameter.empty:
                required.append(name)
            else:
                schema.setdefault("default", param.default)
            properties[name] = schema
        return {
            "type": "function",
            "function": {
                "name": getattr(tool, "__name__", "tool"),
                "description": (getattr(tool, "__doc__", "") or "").strip(),
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }

    @staticmethod
    def _stream_delta(chunk: Any):
        """First choice's delta of an OpenAI-compatible stream chunk (or None)."""
        choices = getattr(chunk, "choices", None)
        return choices[0].delta if choices else None


class Brain(OpenAICore):
    """
    Tool-bearing assistant built on top of OpenAICore.

    Owns the conversation history (plumbing and repair across cancelled or
    failed turns), streaming generation with cancellation, and the model
    info tool. All other LLM tools live in the ``src.tools`` submodules:
    their config is injected once by ``main.py`` via ``src.tools.configure``,
    and their session-bound state (chat history, LLM client) is bound to
    this Brain instance in ``__init__`` via the ``bind_*`` functions.
    """

    ## history plumbing (plain OpenAI-style dicts) ##

    def _add_user_prompt(self, prompt: str):
        self._repair_history()
        self._messages.append({"role": "user", "content": prompt})
        self.logger.debug("Prompt:\n%s", prompt)

    def _repair_history(self) -> None:
        """Keep the chat history template-valid across cancelled/failed turns.

        Prompt templates require user/assistant roles to alternate (tool-call
        /result sequences excepted). Two failure modes break that:

        * a Ctrl+C-cancelled generation can end in an *empty* assistant
          message; templates drop empty turns, which leaves a tool result
          directly followed by a user message and fails rendering with
          "conversation roles must alternate ...";
        * a turn that errored or was cancelled before any response leaves the
          user prompt unanswered, so the next prompt would render as two
          consecutive user messages.

        Called before each new prompt: pad the empty assistant turn with a
        minimal placeholder and drop a trailing unanswered user prompt.
        """
        messages = self._messages
        while messages:
            role = messages[-1].get("role")
            if role == "assistant":
                if not str(messages[-1].get("content") or "").strip():
                    messages.pop()
                    messages.append({"role": "assistant", "content": CANCELLED_PLACEHOLDER})
                break
            if role == "user":
                messages.pop()  # unanswered prompt from a failed/aborted turn
                continue
            break

    ## session view for the session-bound tool modules (src.tools.vision) ##

    @property
    def client(self):
        """OpenAI-compatible chat client (read-only session view)."""
        return self._client

    @property
    def model_name(self) -> str:
        """Configured model id (read-only session view)."""
        return self._model_name

    @property
    def temperature(self) -> float:
        """Configured sampling temperature (read-only session view)."""
        return self._temperature

    @property
    def vision(self) -> bool:
        """Whether the model supports image input (read-only session view)."""
        return self._vision

    @property
    def vision_source(self) -> str:
        """Where the vision flag came from: 'server' or 'config'."""
        return self._vision_source

    @property
    def messages(self) -> list[dict]:
        """The live conversation history (tools may append/pop entries)."""
        return self._messages

    ## generation ##

    @property
    def cancel_requested(self) -> bool:
        """True while a user cancel (Ctrl+C) is pending for the current turn."""
        return self._cancel_requested

    def _stream_completion(self, tools: list[dict] | None = None):
        kwargs: dict = {
            "model": self._model_name,
            "messages": self._messages,
            "stream": True,
            "temperature": self._temperature,
            # Legacy name, still accepted by all OpenAI-compatible endpoints.
            "max_tokens": self._max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        return self._client.chat.completions.create(**kwargs)

    def model_fragments(self, prompt: str):
        """
        Get response fragments as stream from the model for a given prompt.
        Tools are not supported in this method.
        """
        self.logger.info("Sending to Model [🤖: %s] ...", self._model_name)
        self._add_user_prompt(prompt)
        self._cancel_requested = False  # fresh turn
        self._stream.start_spinner("thinking")
        stream = self._stream_completion()
        self._active_stream = stream
        parts: list[str] = []
        try:
            for chunk in stream:
                if self._cancel_requested:
                    break
                delta = self._stream_delta(chunk)
                if delta is not None and delta.content:
                    parts.append(delta.content)
                    yield SimpleNamespace(content=delta.content)
        finally:
            self._active_stream = None
            # Keep partial (or cancelled) output in the history.
            text = "".join(parts)
            if text:
                self._messages.append({"role": "assistant", "content": text})
            elif self._cancel_requested:
                self._messages.append({"role": "assistant", "content": CANCELLED_PLACEHOLDER})

    def print_fragment(self, fragment):
        """
        Print a response fragment to the console.
        Fenced code blocks are syntax-highlighted; plain text is cyan.
        This can be used as a callback for streaming responses.
        """
        self._stream.feed(fragment.content)

    def flush_stream(self) -> None:
        """Flush buffered text and stop the activity spinner (turn boundary)."""
        self._stream.flush()
        self._stream.stop_spinner()

    def cancel_generation(self) -> str:
        """
        Stop the in-flight model response as soon as possible (best effort).
        Not exposed to the LLM; invoked by the session's Ctrl+C handling.
        Closes the active HTTP stream for both streamed responses and ``act()``
        rounds; the pending flag covers a cancel raised while a tool is
        running (between rounds). Partial output is kept in the chat history.
        """
        self._cancel_requested = True
        stream, self._active_stream = self._active_stream, None
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:
                self.logger.debug("Closing prediction stream failed: %s", exc)
        return "Cancel requested"

    def _context_usage(self) -> tuple[float, int]:
        """Approximate context usage (no server-side tokenizer available)."""
        chars = sum(len(str(m.get("content") or "")) for m in self._messages)
        tokens = chars // 4  # rough ~4 chars per token estimate
        return float(tokens / self._context_window) * 100, self._context_window


    ## TOOLS ##

    @dump_func_call
    def model_info(self) -> str:
        """
        Returns detailed info about the loaded model: endpoint, provider and
        everything the server reports through the OpenAI models endpoint (or
        through its native API for local servers: ownership, context lengths,
        capabilities, quantization, ...), plus the current session settings.
        Useful when user ask for information about the loaded model.
        """
        meta = self._fetch_model_metadata()
        usage_pct, context_window = self._context_usage()
        lines = [
            f"model={self._model_name}",
            f"base_url={self._base_url} (OpenAI-compatible API)",
        ]
        if meta.get("owned_by"):
            lines.append(f"provider={meta['owned_by']}")
        created = meta.get("created")
        if created:
            if isinstance(created, (int, float)):
                created = datetime.fromtimestamp(
                    created, tz=timezone.utc
                ).strftime("%Y-%m-%d")
            lines.append(f"created={created}")
        window_labels = {
            "config": "configured",
            "server": "server-reported",
            "default": "default (server did not report)",
        }
        window_source = getattr(self, "_context_window_source", "config")
        lines.append(
            f"context_window={context_window} tokens"
            f" ({window_labels.get(window_source, 'configured')})"
        )
        context_fields = {
            k: v for k, v in meta.items()
            if "context" in k.lower() and isinstance(v, int)
        }
        if context_fields:
            lines.append("server_context=" + ", ".join(
                f"{k}={v}" for k, v in sorted(context_fields.items())))
            max_fields = [v for k, v in context_fields.items()
                          if not k.lower().startswith("loaded")]
            if any(v < context_window for v in max_fields):
                lines.append(
                    "warning: configured context_window exceeds the "
                    "server-reported maximum"
                )
        if meta.get("capabilities"):
            lines.append(f"capabilities={', '.join(map(str, meta['capabilities']))}")
        rendered = {"owned_by", "created", "capabilities"} | set(context_fields)
        for key in sorted(k for k in meta if k not in rendered):
            value = meta[key]
            if isinstance(value, (list, tuple)):
                value = ", ".join(map(str, value))
            lines.append(f"{key}={value}")
        lines += [
            f"temperature={self._temperature}",
            f"max_response_tokens={self._max_tokens} (per-generation cap)",
            f"vision={'yes' if self._vision else 'no'}",
            f"vision_source={self._vision_source}",
            f"session_messages={len(self._messages)} (context ~{usage_pct:.1f}% used)",
        ]
        return "\n".join(lines)


class Agent(Brain):
    """
    Agent that extends the Brain with OpenAI function-calling tool usage.

    Implements the round-based ``act()`` loop: stream a round, run the
    requested tools, feed the results back, and repeat until the model
    answers without tool calls.
    """

    MAX_ACT_ROUNDS = 20

    def _build_tools(self) -> list[Callable]:
        return _default_tools(self)

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self._fragments_buffer: list[str] = []
        self.logger.info(
            "✅ Agent with Model '%s' initialized. Setting up tools ...",
            self._model_name
        )
        self.tools: list[Callable] = self._build_tools()
        self.tool_definitions: list[dict] = [self._openai_tool_definition(tool) for tool in self.tools]
        self._tool_impls = {tool.__name__: tool for tool in self.tools}
        self.logger.info(
            "✅ Agent with Model '%s' is ready with [%d] tools.",
            self._model_name, len(self.tools)
        )

    def bootstrap(self) -> None:
        """Run the configured bootstrap prompt (called by the session, not on init)."""
        if self.config.agent.bootstrap_prompt:
            self.act(self.config.agent.bootstrap_prompt)

    def _handle_fragment(self, content: str) -> None:
        self._fragments_buffer.append(content)
        self.print_fragment(SimpleNamespace(content=content))

    def _handle_round_end(self, round_index: int = 0) -> None:
        context_usage_percentage, context_window = self._context_usage()
        self._stream.flush()  # keep the spinner alive across rounds
        cprint("")
        stats = f"context {context_usage_percentage:.1f}% [{int(context_window / 1000)}k]"
        started = getattr(self, "_turn_started", None)
        if started is not None:
            stats += f" · {time.monotonic() - started:.1f}s"
        crule(stats)
        self._stream.mark_line_clean()  # spinner may resume for the next round
        self._stream.note("working")
        response = " ".join(self._fragments_buffer).strip()
        if not self._cancel_requested:
            speech_tools.speak_if_enabled(response)
        self._fragments_buffer.clear()

    def _stream_round(self, tools: list[dict]) -> tuple[str, list[dict]]:
        """Stream one completion round; return (text, accumulated tool calls)."""
        stream = self._stream_completion(tools)
        self._active_stream = stream
        parts: list[str] = []
        calls: dict[int, dict] = {}
        try:
            for chunk in stream:
                if self._cancel_requested:
                    break
                delta = self._stream_delta(chunk)
                if delta is None:
                    continue
                if delta.content:
                    parts.append(delta.content)
                    self._handle_fragment(delta.content)
                for tc in getattr(delta, "tool_calls", None) or []:
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        slot["name"] += tc.function.name or ""
                        slot["arguments"] += tc.function.arguments or ""
        finally:
            self._active_stream = None
        return "".join(parts), [calls[i] for i in sorted(calls)]

    def _execute_tool(self, call: dict) -> str:
        tool = self._tool_impls.get(call["name"])
        if tool is None:
            return f"Error: unknown tool '{call['name']}'"
        try:
            args = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError as exc:
            return f"Error: malformed arguments for '{call['name']}': {exc}"
        try:
            result = tool(**args)
        except Exception as exc:
            result = f"Error: {exc}"
        return result if isinstance(result, str) else str(result)

    def act(self, prompt: str):
        """
        Perform an action based on the input prompt.
        The response may involve using tools or just generating a text.
        """
        cprint(f"\n🤖 [{self._model_name}]", "grey")
        self.logger.info("Sending to Agent [🤖: %s] ...", self._model_name)
        self._add_user_prompt(prompt)
        self._cancel_requested = False  # fresh turn
        self._turn_started = time.monotonic()
        self._stream.start_spinner("thinking")
        try:
            for round_index in range(self.MAX_ACT_ROUNDS):
                text, tool_calls = self._stream_round(self.tool_definitions)
                if tool_calls and not self._cancel_requested:
                    assistant_call = {
                        "role": "assistant",
                        "content": text or None,
                        "tool_calls": [
                            {
                                "id": call["id"] or f"call_{index}",
                                "type": "function",
                                "function": {
                                    "name": call["name"],
                                    "arguments": call["arguments"] or "{}",
                                },
                            }
                            for index, call in enumerate(tool_calls)
                        ],
                    }
                    self._messages.append(assistant_call)
                    for index, call in enumerate(tool_calls):
                        result = self._execute_tool(call)
                        self.logger.debug("Tool %s -> %s", call["name"], str(result)[:200])
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"] or f"call_{index}",
                            "content": result,
                        })
                    self._handle_round_end(round_index)
                    continue
                if not text and self._cancel_requested:
                    text = CANCELLED_PLACEHOLDER
                self._messages.append({"role": "assistant", "content": text})
                self._handle_round_end(round_index)
                break
        finally:
            self._stream.stop_spinner()
            self._active_stream = None
