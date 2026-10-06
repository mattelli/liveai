"""
Handler classes for local model servers (LM Studio, Ollama).

Each backend handler exposes the same interface over its Python SDK:

    handler = LocalModelHandler("ollama")          # or "lmstudio", or "auto"
    models = handler.list_models()           # downloaded models (available)
    loaded = handler.list_loaded()           # models currently in memory
    info   = handler.load("llama3.2:3b")     # load into memory (blocks)
    info   = handler.model_info("llama3.2:3b")  # details of one model
    handler.unload("llama3.2:3b")            # free the memory

Backends:

* LM Studio uses the ``lmstudio`` SDK. By default the SDK targets the LM
  Studio app's daemon port (via its discovery file); when that is unresolved
  (LM Studio running headless via ``lms server start``) the handler retries
  against the standard API server port (localhost:1234). "Available" =
  downloaded model files; ``load()`` fetches-or-loads the model and returns
  the loaded instance info (identifier, context length, ...).
* Ollama uses the ``ollama`` SDK (REST API on localhost:11434).
  "Available" = models in the local registry; Ollama has no explicit load
  endpoint, so ``load()`` sends the documented empty-prompt generate request
  with a ``keep_alive`` TTL (and ``keep_alive=0`` unloads again).
* Every model entry/info carries ``capabilities`` ("vision", "tool_use",
  "tools", ...) and ``max_context_length`` (trained maximum); entries for
  loaded models additionally carry the active ``context_length``. Ollama
  capabilities come from its per-model ``show()`` response, LM Studio's are
  derived from the model info's ``vision`` / ``trained_for_tool_use`` flags.

Both SDKs are imported lazily so the module is importable without either
installed; instantiating the matching backend without it raises an
``ImportError`` with the pip install hint.
"""

import shlex
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from logging import getLogger
from typing import Any, Callable

__all__ = [
    "LocalModelHandlerError",
    "ModelNotFoundError",
    "BaseLocalModelHandler",
    "LMStudioLocalModelHandler",
    "OllamaLocalModelHandler",
    "LocalModelHandler",
    "human_size",
]


class LocalModelHandlerError(RuntimeError):
    """A model server request failed (server unreachable, unknown model, ...)."""


class ModelNotFoundError(LocalModelHandlerError):
    """The requested model is neither loaded nor downloaded/available."""


def _to_dict(obj: Any) -> Any:
    """Recursively convert SDK records to plain dicts / JSON-friendly scalars.

    Field order matters: msgspec structs (lmstudio SDK) are read attribute-wise
    via ``__struct_fields__`` (snake_case), because their ``to_dict()`` returns
    camelCase wire-format keys; pydantic models (ollama SDK) use ``model_dump()``.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_dict(v) for v in obj]
    fields = getattr(obj, "__struct_fields__", None)  # msgspec struct (lmstudio)
    if fields:
        return {str(f): _to_dict(getattr(obj, f)) for f in fields}
    for method_name in ("model_dump", "to_dict"):  # pydantic (ollama), fallbacks
        method = getattr(obj, method_name, None)
        if callable(method):
            data = method()
            if isinstance(data, dict):
                return _to_dict(data)
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict) and data:
        return _to_dict(data)
    return str(obj)  # last resort: keep the value readable


# SDK exception class names meaning "this model does not exist on the server".
_NOT_FOUND_CLASS_NAMES = {"LMStudioModelNotFoundError"}


def _is_not_found(exc: Exception) -> bool:
    """Heuristic: did the server report the model as missing?

    Recognises SDK exception types (lmstudio.LMStudioModelNotFoundError),
    HTTP 404 statuses (the ollama SDK's ResponseError carries ``status_code``)
    and "not found" error messages across servers.
    """
    if type(exc).__name__ in _NOT_FOUND_CLASS_NAMES:
        return True
    if getattr(exc, "status_code", None) == 404:
        return True
    return "not found" in str(exc).lower()


def _lmstudio_capabilities(info: dict) -> list[str]:
    """Normalized capability tags for an LM Studio model-info dict.

    LM Studio reports no capability list; it flags vision/tool training on
    the model info instead (``vision`` / ``trained_for_tool_use`` booleans).
    """
    capabilities = []
    if info.get("vision"):
        capabilities.append("vision")
    if info.get("trained_for_tool_use"):
        capabilities.append("tool_use")
    return capabilities


def _ollama_max_context(modelinfo: dict, details: Any) -> int | None:
    """The trained maximum context length from an Ollama ``show()`` response.

    It lives in the ``modelinfo`` blob as ``<architecture>.context_length``;
    ``details.family`` usually names the same architecture.
    """
    arch = modelinfo.get("general.architecture") or getattr(details, "family", None)
    if arch and isinstance(modelinfo.get(f"{arch}.context_length"), int):
        return int(modelinfo[f"{arch}.context_length"])
    for key, value in modelinfo.items():  # fallback: any "<arch>.context_length"
        if key.endswith(".context_length") and isinstance(value, int):
            return int(value)
    return None


def human_size(num_bytes: float) -> str:
    """Render a byte count as a human-readable string ("3.4 GiB")."""
    size = float(num_bytes or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


class BaseLocalModelHandler(ABC):
    """Common interface implemented by every local-server backend handler."""

    backend: str = "base"

    def __init__(self) -> None:
        self.logger = getLogger(self.__class__.__name__)

    def _server_call(self, description: str, func: Callable[[], Any]) -> Any:
        """Run an SDK call; convert transport/SDK failures to handler errors.

        "Model not found" failures become :class:`ModelNotFoundError`; every
        other server failure (unreachable server, API not started, ...) becomes
        :class:`LocalModelHandlerError` with the backend and action in the message.
        Already-classified handler errors pass through unchanged.
        """
        try:
            return func()
        except LocalModelHandlerError:
            raise  # already classified (includes ModelNotFoundError)
        except Exception as exc:
            if _is_not_found(exc):
                raise ModelNotFoundError(f"{self.backend}: {exc}") from exc
            raise LocalModelHandlerError(f"{self.backend} {description} failed: {exc}") from exc

    def list_models(self) -> list[dict]:
        """List available (downloaded / pulled) models, newest first."""
        models = self._list_available()
        models.sort(key=lambda m: str(m.get("modified_at") or ""), reverse=True)
        loaded_keys = {
            m.get("model_key") or m.get("model") for m in self.list_loaded()
        }
        for model in models:
            key = model.get("model_key") or model.get("model")
            model["loaded"] = key in loaded_keys
        return models

    def _list_available(self) -> list[dict]:
        """Backend-specific available-model listing (unsorted, no load flags)."""
        raise NotImplementedError

    @abstractmethod
    def list_loaded(self) -> list[dict]:
        """List models currently loaded in memory."""

    @abstractmethod
    def load(self, model: str, **options) -> dict:
        """Load ``model`` into memory and return its instance info."""

    @abstractmethod
    def unload(self, model: str) -> bool:
        """Unload ``model`` (by key or instance identifier). True if found."""

    @abstractmethod
    def model_info(self, model: str) -> dict:
        """Fetch info for ``model`` (loaded preferred, else available)."""

    @abstractmethod
    def is_available(self) -> bool:
        """True when the server answers a cheap API call."""


class LMStudioLocalModelHandler(BaseLocalModelHandler):
    """
    Model manager for a running LM Studio instance (``lmstudio`` SDK).

    ``host`` pins the SDK connection (e.g. "localhost:1234"); by default the
    SDK resolves the LM Studio app's daemon port through its discovery file,
    and when that is unresolved (LM Studio running headless via ``lms server
    start``) the handler retries once against the standard API server port.
    ``ttl`` is the unload timeout LM Studio applies to instances created by
    :meth:`load` (JIT model management); ``load_config`` optionally carries
    load knobs accepted by the SDK (e.g. ``{"context_length": 8192}``).
    """

    backend = "lmstudio"

    # Standard HTTP API server port (`lms server start`); used when the SDK's
    # app-daemon port discovery is unresolved (no GUI app running).
    FALLBACK_HOST = "localhost:1234"
    _UNRESOLVED_PORT_MARKER = "port is not yet resolved"

    def __init__(self, client=None, host: str | None = None, ttl: int = 3600,
                 load_config: dict | None = None):
        super().__init__()
        self._client = client
        self._host = host
        self._injected_client = client is not None  # injected clients are never replaced
        if client is None:
            try:
                import lmstudio as lms  # lazy import: keeps startup light
            except ImportError as exc:
                raise ImportError("pip install lmstudio  # required for LM Studio models") from exc
        self._ttl = ttl
        self._load_config = load_config

    def _get_client(self):
        """The bound client, created lazily (``host=None`` -> SDK discovery)."""
        if self._client is None:
            import lmstudio as lms
            self._client = lms.Client(self._host) if self._host is not None else lms.Client()
        return self._client

    def _llm_session(self):
        """The LLM session of the bound client (connects on first use)."""
        return self._get_client().llm

    def _server_call(self, description: str, func: Callable[[], Any]) -> Any:
        """Base wrapping plus a one-shot retry on the standard API server port.

        Without an explicit ``host`` and with a self-created client the SDK
        targets the LM Studio app's daemon port (via its discovery file). With
        a headless ``lms server start`` that resolution fails ("Local API host
        port is not yet resolved."); rebind to ``FALLBACK_HOST`` once and
        retry. An explicitly injected ``client`` is never replaced.
        """
        try:
            return super()._server_call(description, func)
        except LocalModelHandlerError as exc:
            if (self._host is None and not self._injected_client
                    and self._UNRESOLVED_PORT_MARKER in str(exc).lower()):
                self.logger.info(
                    "LM Studio app port unresolved; retrying against %s ...", self.FALLBACK_HOST)
                self._host = self.FALLBACK_HOST
                self._client = None  # rebind lazily to the fallback host
                return super()._server_call(description, func)
            raise

    def _list_available(self) -> list[dict]:
        models: list[dict] = []

        def list_downloaded():
            # Lazily resolve the client inside the call: a fallback-host retry
            # (see _server_call) must rebind _client before the call runs.
            return self._get_client().list_downloaded_models()

        for downloaded in self._server_call("listing downloaded models", list_downloaded):
            info = _to_dict(downloaded.info)
            info["path"] = str(downloaded.path)
            info["capabilities"] = _lmstudio_capabilities(info)
            models.append(info)
        return models

    def _fetch_loaded_infos(self) -> list[dict]:
        """Infos of loaded instances (server calls wrapped, identifier merged)."""
        def fetch() -> list[dict]:
            infos = []
            for handle in self._llm_session().list_loaded():
                info = _to_dict(handle.get_info())
                info["identifier"] = getattr(handle, "identifier", info.get("identifier"))
                info["capabilities"] = _lmstudio_capabilities(info)
                infos.append(info)
            return infos
        return self._server_call("listing loaded models", fetch)

    def list_loaded(self) -> list[dict]:
        return self._fetch_loaded_infos()

    def load(self, model: str, ttl: int | None = None, config: dict | None = None,
             on_load_progress: Callable[[float], Any] | None = None) -> dict:
        """
        Load ``model`` (by model key) or return the already loaded instance.

        Returns the loaded instance info; ``on_load_progress`` (if given)
        receives load fractions in [0, 1] while the model loads.
        """
        self.logger.info("Loading model '%s' (ttl=%s) ...", model, ttl or self._ttl)

        def load_model():
            return self._llm_session().model(
                model,
                ttl=ttl if ttl is not None else self._ttl,
                config=config if config is not None else self._load_config,
                on_load_progress=on_load_progress,
            )

        handle = self._server_call(f"loading model '{model}'", load_model)
        info = _to_dict(self._server_call("fetching instance info", handle.get_info))
        info["identifier"] = handle.identifier
        info["capabilities"] = _lmstudio_capabilities(info)
        info["loaded"] = True
        self.logger.info("Model '%s' loaded as instance '%s'.", model, handle.identifier)
        return info

    def unload(self, model: str) -> bool:
        """Unload by instance identifier or model key (embedding models too)."""
        for namespace in ("llm", "embedding"):
            session = self._server_call(
                f"accessing {namespace} session",
                lambda ns=namespace: getattr(self._get_client(), ns),
            )
            for handle in self._server_call("listing loaded models", session.list_loaded):
                if model in (getattr(handle, "identifier", None), handle.get_info().model_key):
                    self._server_call(f"unloading instance '{handle.identifier}'", handle.unload)
                    self.logger.info("Unloaded instance '%s'.", handle.identifier)
                    return True
        return False

    def model_info(self, model: str) -> dict:
        """Info for a loaded instance (identifier or key) or downloaded model."""
        for info in self._fetch_loaded_infos():
            if model in (info.get("identifier"), info.get("model_key")):
                info["loaded"] = True
                return info
        def list_downloaded():
            return self._get_client().list_downloaded_models()

        for downloaded in self._server_call("listing downloaded models", list_downloaded):
            if model in (downloaded.model_key, getattr(downloaded.info, "display_name", None)):
                info = _to_dict(downloaded.info)
                info["path"] = str(downloaded.path)
                info["capabilities"] = _lmstudio_capabilities(info)
                info["loaded"] = False
                return info
        raise ModelNotFoundError(f"LM Studio model not found: {model!r}")

    def is_available(self) -> bool:
        try:
            self._server_call("probing the server", lambda: self._llm_session().list_loaded())
            return True
        except LocalModelHandlerError as exc:
            self.logger.debug("LM Studio not available: %s", exc)
            return False


class OllamaLocalModelHandler(BaseLocalModelHandler):
    """
    Model manager for an Ollama server (``ollama`` SDK).

    ``host`` falls back to the SDK default (``OLLAMA_HOST`` env var, else
    http://127.0.0.1:11434). ``keep_alive`` is the memory TTL applied by
    :meth:`load` ("10m", -1 = forever, 0 = unload immediately).
    """

    backend = "ollama"

    def __init__(self, host: str | None = None, keep_alive: str | int = "10m", client=None):
        super().__init__()
        if client is None:
            try:
                import ollama  # lazy import: keeps startup light
            except ImportError as exc:
                raise ImportError("pip install ollama  # required for Ollama models") from exc
            client = ollama.Client(host=host)
        self._client = client
        self._keep_alive = keep_alive

    def _show_enrichment(self, model: str) -> dict:
        """Capabilities + trained max context from a per-model ``show()`` call."""
        shown = self._server_call(
            f"fetching capabilities of '{model}'", lambda: self._client.show(model))
        modelinfo = _to_dict(shown.modelinfo) or {}
        return {
            "capabilities": list(getattr(shown, "capabilities", None) or []),
            "max_context_length": _ollama_max_context(modelinfo, shown.details),
        }

    def _list_available(self) -> list[dict]:
        records = self._server_call("listing available models", lambda: self._client.list().models)
        models = []
        for record in records:
            details = record.details
            models.append({
                "model": record.model,
                "modified_at": _to_dict(record.modified_at),
                "digest": record.digest,
                "size": int(record.size or 0),
                "family": getattr(details, "family", None),
                "families": list(getattr(details, "families", None) or []),
                "parameter_size": getattr(details, "parameter_size", None),
                "quantization_level": getattr(details, "quantization_level", None),
            })
        # Capabilities/context live in the per-model show() response; enrich
        # best-effort so a single failing model does not break the listing.
        for entry in models:
            try:
                enrichment = self._show_enrichment(entry["model"])
            except LocalModelHandlerError as exc:
                self.logger.debug("Skipping info enrichment for %r: %s", entry["model"], exc)
                continue
            entry.update({k: v for k, v in enrichment.items() if v is not None})
        return models

    def list_loaded(self) -> list[dict]:
        records = self._server_call("listing loaded models", lambda: self._client.ps().models)
        models = []
        for record in records:
            details = record.details
            models.append({
                "model": record.model,
                "digest": record.digest,
                "size": int(record.size or 0),
                "size_vram": int(record.size_vram or 0),
                "context_length": record.context_length,
                "expires_at": _to_dict(record.expires_at),
                "family": getattr(details, "family", None),
                "parameter_size": getattr(details, "parameter_size", None),
                "quantization_level": getattr(details, "quantization_level", None),
            })
        return models

    def load(self, model: str, keep_alive: str | int | None = None) -> dict:
        """
        Load ``model`` into memory: an empty generate request (no prompt) with
        a ``keep_alive`` TTL is the documented Ollama preloading mechanism.
        """
        ttl = self._keep_alive if keep_alive is None else keep_alive
        self.logger.info("Loading model '%s' (keep_alive=%s) ...", model, ttl)
        response = self._server_call(
            f"loading model '{model}'",
            lambda: self._client.generate(model=model, keep_alive=ttl),
        )
        return {
            "model": model,
            "loaded": True,
            "keep_alive": ttl,
            "done": bool(response.done),
            "total_duration": response.total_duration,
            "load_duration": response.load_duration,
        }

    def unload(self, model: str) -> bool:
        """Free ``model`` from memory (keep_alive=0, documented unload call)."""
        self._server_call(
            f"unloading model '{model}'",
            lambda: self._client.generate(model=model, keep_alive=0),
        )
        self.logger.info("Unloaded model '%s'.", model)
        return True

    def model_info(self, model: str) -> dict:
        """Details of ``model``: capabilities, parsed parameters, modelinfo."""
        response = self._server_call(
            "fetching model info", lambda: self._client.show(model)
        )
        if response.modelinfo is None:
            raise ModelNotFoundError(f"Ollama model not found: {model!r}")
        details = response.details
        modelinfo = _to_dict(response.modelinfo)
        return {
            "model": model,
            "modified_at": _to_dict(response.modified_at),
            "capabilities": list(response.capabilities or []),
            "max_context_length": _ollama_max_context(modelinfo, details),
            "family": getattr(details, "family", None),
            "families": list(getattr(details, "families", None) or []),
            "parameter_size": getattr(details, "parameter_size", None),
            "quantization_level": getattr(details, "quantization_level", None),
            "parameters": self._parse_parameters(response.parameters),
            "modelinfo": modelinfo,
        }

    @staticmethod
    def _parse_parameters(parameters: str | None) -> dict:
        """Parse the ``parameters`` block ("temperature 0.6", 'stop "..."', ...) into a dict.

        Repeated keys (several ``stop`` sequences) accumulate into a list.
        """
        parsed: dict = {}
        for line in (parameters or "").splitlines():
            tokens = shlex.split(line.strip())
            if len(tokens) < 2:
                continue
            key, values = tokens[0], tokens[1:]
            def _convert(value: str):
                for cast in (int, float):
                    try:
                        return cast(value)
                    except ValueError:
                        continue
                return value

            converted = [_convert(v) for v in values]
            value = converted[0] if len(converted) == 1 else converted
            if key in parsed:  # repeated key: keep every occurrence
                previous = parsed[key]
                parsed[key] = [*previous, value] if isinstance(previous, list) else [previous, value]
            else:
                parsed[key] = value
        return parsed

    def is_available(self) -> bool:
        try:
            self._server_call("probing the server", lambda: self._client.list())
            return True
        except LocalModelHandlerError as exc:
            self.logger.debug("Ollama not available: %s", exc)
            return False


_BACKEND_CLASSES = {"lmstudio": LMStudioLocalModelHandler, "ollama": OllamaLocalModelHandler}


class LocalModelHandler:
    """
    Facade over the backend handlers with a single entry point.

    ``LocalModelHandler("lmstudio")`` / ``LocalModelHandler("ollama")`` bind one server;
    ``LocalModelHandler("auto")`` probes both servers (LM Studio first) and binds to
    the first that answers; ``LocalModelHandler("all")`` binds to *every* reachable
    server and aggregates results across them:

    * list results carry a ``"backend"`` tag on each entry;
    * model-targeting calls accept qualified names ("ollama:llama3.2:3b",
      "lmstudio:qwen/qwen3.5-4b"); unqualified names are located in the
      backends' available lists and must be unambiguous.
    """

    def __init__(self, backend: str = "auto", **kwargs):
        self.logger = getLogger(self.__class__.__name__)
        backend = str(backend).lower()
        if backend in _BACKEND_CLASSES:
            self._impls: list[BaseLocalModelHandler] = [_BACKEND_CLASSES[backend](**kwargs)]
        elif backend == "auto":
            self._impls = [self._probe(**kwargs)[0]]  # first reachable, LM Studio first
        elif backend == "all":
            self._impls = self._probe(**kwargs)
        else:
            raise ValueError(
                f"Unknown backend {backend!r} (use 'lmstudio', 'ollama', 'auto' or 'all')")
        self.backend = backend if backend == "all" else self._impls[0].backend
        for impl in self._impls:
            impl.logger.info("LocalModelHandler bound to %s backend.", impl.backend)

    @staticmethod
    def _probe(**kwargs) -> list[BaseLocalModelHandler]:
        """Probe both servers; return handlers for every reachable one (LM Studio first)."""
        impls: list[BaseLocalModelHandler] = []
        errors: list[str] = []
        for handler_class in (LMStudioLocalModelHandler, OllamaLocalModelHandler):
            try:
                handler = handler_class(**kwargs)
            except ImportError as exc:  # SDK not installed: remember and try next
                errors.append(f"{handler_class.backend}: {exc}")
                continue
            if handler.is_available():
                impls.append(handler)
            else:
                errors.append(f"{handler_class.backend}: server not reachable")
        if not impls:
            raise LocalModelHandlerError(
                "No local model server reachable (LM Studio :1234, Ollama :11434): "
                + "; ".join(errors)
            )
        return impls

    @property
    def backends(self) -> list[str]:
        """Names of the bound backend(s)."""
        return [impl.backend for impl in self._impls]

    def list_models(self) -> list[dict]:
        """Available models of every bound backend, entries tagged with ``"backend"``."""
        models = []
        for impl in self._impls:
            for entry in impl.list_models():
                entry["backend"] = impl.backend
                models.append(entry)
        return models

    def list_loaded(self) -> list[dict]:
        """Loaded models of every bound backend, entries tagged with ``"backend""."""
        models = []
        for impl in self._impls:
            for entry in impl.list_loaded():
                entry["backend"] = impl.backend
                models.append(entry)
        return models

    def load(self, model: str, **options) -> dict:
        """Load ``model`` on its backend (qualified names allowed)."""
        impl, name = self._resolve_impl(model)
        info = impl.load(name, **options)
        if len(self._impls) > 1:
            info["backend"] = impl.backend
        return info

    def unload(self, model: str) -> bool:
        """Unload ``model`` from its backend (qualified names allowed)."""
        impl, name = self._resolve_impl(model)
        return impl.unload(name)

    def model_info(self, model: str) -> dict:
        """Info for ``model`` from its backend (qualified names allowed)."""
        impl, name = self._resolve_impl(model)
        info = impl.model_info(name)
        if len(self._impls) > 1:
            info["backend"] = impl.backend
        return info

    def is_available(self) -> bool:
        """True when at least one bound server answers."""
        return any(impl.is_available() for impl in self._impls)

    def _by_backend(self, name: str) -> BaseLocalModelHandler:
        """The bound handler for backend ``name`` (must be bound)."""
        for impl in self._impls:
            if impl.backend == name:
                return impl
        raise LocalModelHandlerError(
            f"backend {name!r} is not bound (bound: {', '.join(self.backends)})")

    def _resolve_impl(self, model: str) -> tuple[BaseLocalModelHandler, str]:
        """Backend + bare model name for a (possibly backend-qualified) model.

        ``"ollama:llama3.2:3b"`` routes explicitly. An unqualified name routes
        directly when a single backend is bound; with several, it is located
        in the backends' available lists and must be unambiguous.
        """
        prefix, sep, rest = model.partition(":")
        if sep and prefix in _BACKEND_CLASSES:
            return self._by_backend(prefix), rest
        if len(self._impls) == 1:
            return self._impls[0], model
        matches = []
        for impl in self._impls:
            try:
                keys = {m.get("model_key") or m.get("model") for m in impl.list_models()}
            except LocalModelHandlerError as exc:  # backend down: skip in the search
                self.logger.debug("Skipping %s while resolving %r: %s", impl.backend, model, exc)
                continue
            if model in keys:
                matches.append(impl)
        if not matches:
            raise ModelNotFoundError(f"model {model!r} not found on: {', '.join(self.backends)}")
        if len(matches) > 1:
            raise LocalModelHandlerError(
                f"model {model!r} is available on several backends "
                f"({', '.join(i.backend for i in matches)}); "
                f"qualify it, e.g. '{matches[0].backend}:{model}'")
        return matches[0], model
