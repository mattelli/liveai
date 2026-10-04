import base64
import inspect
import json
import mimetypes
import os
import pathlib
import sys
import types
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints
from logging import getLogger
from datetime import datetime, timezone
from src.common import SimpleNamespace, dump_func_call, cprint, StreamPrinter
from src.tools import (
    exit_gracefully, system_info, current_date_time, get_working_directory,
    read_text_file, write_text_file, run_command,
    web_search, webpage_headlines, webpage_content, capture_image,
    fetch_financial_data, save_financial_chart, download_file, extract_archive,
    search_pattern_in_files, replace_string_in_file, search_in_file
)
from src.summarizer import TextSummarizer
from src.document_query import DEFAULT_CHUNK_SIZE as QUERY_CHUNK_SIZE, DEFAULT_TOP_K as QUERY_TOP_K, DocumentQuery
from src.image_search import DEFAULT_TOP_K as IMAGE_SEARCH_TOP_K, ImageSearch
from src.embedding_search import (
    DEFAULT_CHUNK_SIZE, DEFAULT_CORPORA_DIR, DEFAULT_EXTENSIONS, DEFAULT_TOP_K,
    EmbeddingCorpus, list_corpora, resolve_corpus_dir
)
from src.document_loader import IMAGE_EXTENSIONS

Role = Literal['system', 'user', 'assistant']

# Padded into the history when a generation is cancelled before any text was
# produced: an empty assistant turn would be dropped by the model's prompt
# template and break user/assistant role alternation on the next request.
CANCELLED_PLACEHOLDER = "(generation cancelled by user)"


def __getattr__(name: str):
    """Lazily expose ``SpeechGenerator`` so importing this module does not pull
    in the heavy ``piper``/``sounddevice`` stack. The name stays patchable on
    the module for callers (and tests) that override it.
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
    """Tool set of the agent."""
    return [
        agent.model_info, agent.recall_personal, agent.store_personal,
        agent.image_response, agent.allow_speech, agent.clear_chat,
        agent.summarize_text, agent.query_document, agent.image_search,
        agent.list_corpora, agent.store_corpus, agent.query_corpus,
        capture_image, system_info, current_date_time, get_working_directory,
        run_command, search_in_file, search_pattern_in_files,
        read_text_file, replace_string_in_file, write_text_file,
        web_search, webpage_content, webpage_headlines, download_file, extract_archive,
        fetch_financial_data, save_financial_chart,
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

    Talks to ``[brain] base_url`` with the ``openai`` client, so cloud models
    (OpenAI, Mistral, Groq, OpenRouter, DeepSeek, Together, Gemini, xAI,
    Anthropic, ...) and local servers (Ollama, LM Studio) share one
    implementation: connection handling, model metadata resolution (context
    window, vision) from the OpenAI models endpoint, and the function-calling
    helpers (JSON Schema tool definitions, stream chunk parsing).
    The API key is read from the environment variable named by
    ``[brain] api_key_env`` (default ``<PROVIDER>_API_KEY``, also resolved
    from a project ``.env`` file) and must never be stored in config.toml.
    """

    DEFAULT_CONTEXT_WINDOW = 128_000

    def __init__(self, config: SimpleNamespace):
        self.logger = getLogger(self.__class__.__name__)
        self.config = config
        try:
            from openai import OpenAI  # lazy import: keeps startup light
        except ImportError as exc:
            raise ImportError("pip install openai  # required for the OpenAI-based Brain") from exc

        base_url = getattr(config.brain, "base_url", "")
        if not base_url:
            raise ValueError(f"Missing [brain] base_url")
        api_key_env = getattr(config.brain, "api_key_env", "")
        _load_env_file()  # fill unset variables from .env (real env vars win)
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found: set the '{api_key_env}' environment variable, "
                f"e.g. put '{api_key_env}=sk-...' in the project .env file "
                f"or run: setx {api_key_env} 'sk-...'")

        self.logger.info("🚀 Connecting to remote model '%s' @ %s ...", config.brain.model, base_url)
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model_name = config.brain.model
        self._base_url = base_url
        self._context_window = self._resolve_context_window(config)
        self._vision = self._resolve_vision_flag(config)
        self._temperature = getattr(config.brain, "temperature", 0.7)
        self._max_tokens = _response_token_cap(
            getattr(config.brain, "max_response_tokens", 0),
            self._context_window,
        )
        self._messages: list[dict] = [{"role": "system", "content": config.brain.system_prompt}]
        self._stream = StreamPrinter(text_color="cyan")
        self._cancel_requested = False
        self._active_stream = None
        self.logger.info("✅ Remote model '%s' connected.", self._model_name)

    def _resolve_vision_flag(self, config) -> bool:
        """Derive vision support from the server-reported model capabilities.

        The OpenAI models endpoint may include a ``capabilities`` list (e.g.
        LM Studio: ["completion", "tools", "vision"]); when the server
        reports it, that value is authoritative. Servers without capability
        info (Ollama, most cloud providers) fall back to the optional
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
    failed turns), streaming generation with cancellation, and the built-in
    tool set (model info, image responses, personal memory, summarization,
    corpora search, speech, ...) that an Agent exposes to the model via
    OpenAI function calling.
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

    def _get_messages(self) -> list[dict]:
        # Shape expected by store_personal/_get_messages_from: text at content[0]["text"].
        return [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in self._messages
            if isinstance(m.get("content"), str)
        ]

    @staticmethod
    def _get_messages_from(role: Role, msg:list) -> list[str]:
        try:
            return [m["content"][0]["text"] for m in msg if m["role"] == role]
        except:
            return []

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


    ## TOOLS CALLS ##

    @dump_func_call
    def clear_chat(self) -> bool:
        """
        Useful when user ask for cleaning conversation history.
        """
        try:
            system = next(
                (m["content"] for m in self._messages if m["role"] == "system"),
                self.config.brain.system_prompt,
            )
            user_msgs = [m["content"] for m in self._messages if m["role"] == "user"]
            self._messages = [{"role": "system", "content": system}]
            self._messages.append({"role": "user", "content": self.recall_personal()})
            if user_msgs:
                self._messages.append({"role": "user", "content": user_msgs[-1]})
            return "Chat cleared successfully"
        except Exception as e:
            return f"Error: {e}"

    @dump_func_call
    def model_info(self) -> str:
        """
        Returns detailed info about the loaded model: endpoint, provider and
        everything the server reports through the OpenAI models endpoint
        (ownership, context lengths, capabilities, quantization, ...), plus
        the current session settings.
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

    @dump_func_call
    def image_response(self, image: str, prompt: str = "describe the image") -> str:
        """
        Provide text response form an input image.
        Useful when user asks for getting a description of the input image content.
        """
        if not self._vision:
            if self._vision_source == "server":
                return (
                    "Cannot handle images: the server reports the model has no "
                    "'vision' capability"
                )
            return (
                "Cannot handle images: no 'vision' capability reported by the "
                "server and no [brain] vision = true override set"
            )
        if str(image).startswith(("http://", "https://")):
            image_url = {"url": str(image)}
        else:
            data = pathlib.Path(image).read_bytes()
            mime = mimetypes.guess_type(str(image))[0] or "image/png"
            image_url = {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}
        self._messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url},
            ],
        })
        try:
            resp = self._client.chat.completions.create(
                model=self._model_name,
                messages=self._messages,
                temperature=self._temperature,
            )
        except Exception as e:
            self._messages.pop()  # drop the image turn: provider rejected it
            return f"Error: {e}"
        answer = resp.choices[0].message.content or ""
        self._messages.append({"role": "assistant", "content": answer})
        return answer

    @dump_func_call
    def summarize_text(self, file: str, max_sentences: int = 0) -> str:
        """
        Summarize the content of a document into its most important sentences.
        Useful when the user asks to summarize, condense, shorten or outline a
        document. ``file`` is the path of the document to read; text, PDF, DOCX,
        XLSX, HTML and image files are supported via ``src.document_loader``.
        ``max_sentences`` = 0 uses the ratio configured in the
        ``[summarization]`` section.
        """
        try:
            if getattr(self, "_summarizer", None) is None:
                cfg = self.config.summarization
                self._summarizer = TextSummarizer(
                    model_path=cfg.model_path,
                    device=getattr(cfg, "device", "cpu"),
                    ratio=getattr(cfg, "ratio", 0.1),
                    threshold=getattr(cfg, "threshold", 0.5),
                    max_sentences=getattr(cfg, "max_sentences", 0),
                )
            return self._summarizer.summarize_file(file, max_sentences or None)
        except Exception as e:
            return f"Error: {e}"

    @dump_func_call
    def query_document(self, file: str, query: str, top_k: int = 0) -> str:
        """
        Search one document for the passages most relevant to a query.
        Useful when the user asks to find, locate or look up something inside a
        specific file without building a corpus. Any type supported by
        ``src.document_loader`` works: text, PDF (results keep page numbers),
        DOCX, XLSX, HTML, ... A CrossEncoder scores query/passage pairs.
        ``top_k`` = 0 uses the value configured in the ``[document_query]``
        section.
        """
        try:
            if not query or not query.strip():
                return "Error: a query is required."
            return self._document_query().query(file, query, top_k or None)
        except Exception as e:
            return f"Error: {e}"

    @dump_func_call
    def image_search(self, folder: str, query: str, top_k: int = 0) -> str:
        """
        Search the images inside a folder for the ones best matching a text query,
        using the CLIP model. Useful when the user asks to find, locate or show a
        picture/photo/screenshot in a folder by describing its content. Returns
        the top_k matches as '<rank>) [<cosine score>] <absolute image path>'
        lines, best match first; ``top_k`` = 0 uses the value configured in the
        ``[image_search]`` section (default 3).
        """
        try:
            if not query or not query.strip():
                return "Error: a query is required."
            hits = self._image_search().search(folder, query, top_k or None)
            if not hits:
                return f"No readable images found in '{folder}'."
            return "\n".join(
                f"{rank}) [{score:.3f}] {path}"
                for rank, (score, path) in enumerate(hits, start=1)
            )
        except Exception as e:
            return f"Error: {e}"

    def _image_search(self):
        if getattr(self, "_img_search", None) is None:
            cfg = self.config.image_search
            self._img_search = ImageSearch(
                model_path=cfg.model_path,
                device=getattr(cfg, "device", "cpu"),
                top_k=getattr(cfg, "top_k", IMAGE_SEARCH_TOP_K),
            )
        return self._img_search

    def _document_query(self):
        if getattr(self, "_doc_query", None) is None:
            cfg = self.config.document_query
            self._doc_query = DocumentQuery(
                model_path=cfg.model_path,
                device=getattr(cfg, "device", "cpu"),
                chunk_size=getattr(cfg, "chunk_size", QUERY_CHUNK_SIZE),
                top_k=getattr(cfg, "top_k", QUERY_TOP_K),
            )
        return self._doc_query

    def _embedding_corpus(self):
        if getattr(self, "_emb_corpus", None) is None:
            cfg = self.config.embedding
            self._emb_corpus = EmbeddingCorpus(
                model_path=cfg.model_path,
                device=getattr(cfg, "device", "cpu"),
                extensions=getattr(cfg, "extensions", DEFAULT_EXTENSIONS),
                chunk_size=getattr(cfg, "chunk_size", DEFAULT_CHUNK_SIZE),
                top_k=getattr(cfg, "top_k", DEFAULT_TOP_K),
                image_model_path=getattr(cfg, "image_model_path", None),
                multimodal=getattr(cfg, "multimodal", False),
                image_extensions=getattr(cfg, "image_extensions", IMAGE_EXTENSIONS),
            )
        return self._emb_corpus

    @dump_func_call
    def list_corpora(self) -> str:
        """
        List the available local corpora (name, mode, record counts, sources, built time).
        Useful before query_corpus when the user refers to a topic instead of a corpus name.
        """
        entries = list_corpora(self.config.embedding.corpora_dir)
        if not entries:
            return "No corpora available."
        lines = [f"Available corpora ({len(entries)}):"]
        for e in entries:
            lines.append(
                f"- {e['name']}: mode={e['mode']}, records={e['records']} "
                f"(text {e['text_records']} / image {e['image_records']}), "
                f"sources={e['sources']}, built={e['built_at']}"
            )
        return "\n".join(lines)

    @dump_func_call
    def store_corpus(self, directory: str, corpus_name: str, overwrite: bool = False) -> str:
        """
        Build a named corpus from all supported files in a directory.
        Useful when the user asks to index, embed, or store documents for later semantic search.
        """
        try:
            if not corpus_name or not corpus_name.strip():
                return "Error: a corpus name is required (call list_corpora to see available corpora)."
            corpus_dir = resolve_corpus_dir(corpus_name, self.config.embedding.corpora_dir)
            return self._embedding_corpus().store(directory, corpus_dir, overwrite)
        except Exception as e:
            return f"Error: {e}"

    @dump_func_call
    def query_corpus(self, query: str, corpus_name: str, kind: str = "", top_k: int = 0) -> str:
        """
        Search a named corpus for the content most relevant to a query text.
        Useful when the user asks a question that should be answered from indexed documents
        or when looking for a matching image.
        ``kind`` restricts results to 'text' or 'image'; leave it empty to search both.
        """
        try:
            if not corpus_name or not corpus_name.strip():
                return "Error: a corpus name is required."
            corpus_dir = resolve_corpus_dir(corpus_name, self.config.embedding.corpora_dir)
            return self._embedding_corpus().query(corpus_dir, query, top_k or None, kind or None)
        except Exception as e:
            return f"Error: {e}"

    @dump_func_call
    def allow_speech(self, enable:bool) -> str:
        """
        Enable or disable speech for Agent responses.
        When enabled, the Agent will use the Speech Generator.
        It can be toggled on the fly during the conversation.
        """
        if enable and not hasattr(self, "speech"):
            self.logger.info("Initializing Speech Generator ...")
            gen_cfg = self.config.speech_gen
            SpeechGenerator = getattr(sys.modules[__name__], "SpeechGenerator")
            self.speech = SpeechGenerator(
                model_path=gen_cfg.model_path,
                samplerate=gen_cfg.samplerate,
                volume=gen_cfg.volume,
                speech_speed=gen_cfg.speech_speed,
                speaker_id=getattr(gen_cfg, "speaker_id", None),
                noise_scale=getattr(gen_cfg, "noise_scale", 1.0),
                noise_w_scale=getattr(gen_cfg, "noise_w_scale", 1.0),
                normalize_audio=getattr(gen_cfg, "normalize_audio", False),
            )
            return "Speech Generator initialized"
        elif not enable and hasattr(self, "speech"):
            self.logger.info("Releasing Speech Generator ...")
            self.speech.close()
            del self.speech
            return "Speech Generator released"

    @dump_func_call
    def store_personal(self) ->  str:
        """
        Stores the user personal content to a text file.
        Useful when the users explicitly mention the prompted text is "personal".
        """
        history = pathlib.Path(self.config.brain.history_path).resolve()
        now = datetime.now(tz=timezone.utc)
        timestamp = now.strftime("%d/%m/%Y, %H:%M")
        messages = self._get_messages()
        user_msgs = self._get_messages_from(role="user", msg=messages)
        if user_msgs and user_msgs[0] == self.config.agent.bootstrap_prompt:
            user_msgs = user_msgs[1:]
        else:
            user_msgs = user_msgs
        selected_msgs = [m for m in user_msgs if "personal" in m.lower()]
        user_prompt = f"At {timestamp} user wrote: {selected_msgs[-1]} \n"
        if not history.exists():
            history.write_text(user_prompt, encoding="utf-8")
        else:
            with open(file=history, mode="a", encoding="utf-8") as f:
                f.write(user_prompt)
        return f"Conversation written to {history}"

    @dump_func_call
    def recall_personal(self) -> str:
        """
        Read users personal data from file.
        Useful when the users ask to recall their personal data.
        """
        history = pathlib.Path(self.config.brain.history_path).resolve()
        if not history.exists():
            return "No personal data found!"
        else:
            return history.read_text(encoding="utf-8")


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
        cprint("-"*50 + f"\ncontext: {context_usage_percentage:.2f} % [{int(context_window / 1000)}k]\n", "grey")
        self._stream.mark_line_clean()  # spinner may resume for the next round
        self._stream.note("working")
        response = " ".join(self._fragments_buffer).strip()
        if not self._cancel_requested and hasattr(self, "speech"):
            self.speech.speak(response)
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
