"""Tests for the local model-server handler (fake lmstudio/ollama SDKs)."""

import sys
import types
from datetime import datetime, timezone

import pytest

from src.local_model_handler import (
    LMStudioLocalModelHandler,
    LocalModelHandler,
    LocalModelHandlerError,
    ModelNotFoundError,
    OllamaLocalModelHandler,
    human_size,
)


# --------------------------------------------------------------- fake lmstudio

class FakeStruct:
    """Stand-in for the SDK's msgspec structs (to_dict-based serialisation)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to_dict(self):
        return dict(self.__dict__)


class FakeLMStudioHandle:
    """Symbolic handle to a 'loaded' model instance (unload via its session)."""

    def __init__(self, identifier, info, session=None):
        self.identifier = identifier
        self._info = info
        self._session = session
        self.unloaded = False

    def get_info(self):
        return self._info

    def unload(self):
        self.unloaded = True
        if self._session is not None:
            self._session.unload(self.identifier)


class FakeLMStudioSession:
    """LLM session double: list_loaded / model (get-or-load) / unload."""

    def __init__(self):
        self.loaded: dict[str, FakeLMStudioHandle] = {}  # identifier -> handle
        self.model_calls: list[tuple] = []

    def list_loaded(self):
        return list(self.loaded.values())

    def model(self, model_key, /, *, ttl=3600, config=None, on_load_progress=None):
        self.model_calls.append((model_key, ttl, config))
        if on_load_progress:
            on_load_progress(1.0)
        identifier = f"{model_key}::instance"
        if identifier not in self.loaded:
            info = FakeStruct(
                identifier=identifier, model_key=model_key,
                context_length=4096, max_context_length=131072,
                vision=True, trained_for_tool_use=True,
            )
            self.loaded[identifier] = FakeLMStudioHandle(identifier, info, self)
        return self.loaded[identifier]

    def unload(self, identifier):
        self.loaded.pop(identifier, None)


class FakeLMStudioClient:
    """Client double: downloaded models + llm/embedding session properties."""

    def __init__(self, downloaded=()):
        self._downloaded = list(downloaded)
        self._llm = FakeLMStudioSession()
        self._embedding = FakeLMStudioSession()

    @property
    def llm(self):
        return self._llm

    @property
    def embedding(self):
        return self._embedding

    def list_downloaded_models(self, namespace=None):
        return self._downloaded


class DeadLMStudioClient(FakeLMStudioClient):
    """Client whose server connection is down."""

    @property
    def llm(self):
        raise ConnectionError("LM Studio is not running")


class FakeLMStudioModule(types.ModuleType):
    """``lmstudio`` module double exposing Client()."""

    Client = FakeLMStudioClient


class FakeDownloaded:
    """DownloadedModel double."""

    def __init__(self, model_key, **info_fields):
        fields = dict(
            model_key=model_key, display_name=model_key.title(),
            architecture="qwen2", params_string="4B", max_context_length=131072,
            vision=False, trained_for_tool_use=True, size_bytes=2_300_000_000,
            type="llm", format="safetensors",
        )
        fields.update(info_fields)
        self.info = FakeStruct(**fields)
        self.model_key = model_key
        self.type = fields["type"]
        self.path = f"C:/models/{model_key}.gguf"


# ----------------------------------------------------------------- fake ollama

class FakeOllamaRecord:
    """ListResponse.Model / ProcessResponse.Model double."""

    def __init__(self, model, *, size=2_000_000_000, details=None, **extra):
        self.model = model
        self.size = size
        self.digest = "sha256:fake"
        self.modified_at = datetime(2026, 1, 15, tzinfo=timezone.utc)
        self.details = details
        for key, value in extra.items():
            setattr(self, key, value)


class FakeOllamaDetails:
    def __init__(self, family="llama", families=("llama",),
                 parameter_size="3.2B", quantization_level="Q4_K_M"):
        self.family = family
        self.families = families
        self.parameter_size = parameter_size
        self.quantization_level = quantization_level


class FakeShowResponse:
    """ShowResponse double with a realistic parameters block."""

    def __init__(self):
        self.modified_at = datetime(2026, 1, 15, tzinfo=timezone.utc)
        self.capabilities = ["completion", "tools", "vision"]
        self.details = FakeOllamaDetails()
        self.modelinfo = {"general.architecture": "llama", "llama.context_length": 131072}
        self.parameters = 'num_ctx 4096\ntemperature 0.6\nstop "<|im_end|>"\nstop "<|endoftext|>"'
        self.license = "MIT"
        self.template = "fake template"
        self.modelfile = "FROM llama3.2"


class FakeOllamaClient:
    """Client double recording generate() calls (load/unload probes)."""

    def __init__(self, host=None, models=("llama3.2:3b", "qwen2.5:7b")):
        self.host = host
        self.models = list(models)
        self.generate_calls: list[dict] = []

    def list(self):
        records = [FakeOllamaRecord(m, details=FakeOllamaDetails()) for m in self.models]
        return types.SimpleNamespace(models=records)

    def ps(self):
        records = [FakeOllamaRecord(
            self.models[0], details=FakeOllamaDetails(),
            size_vram=1_500_000_000, context_length=4096,
            expires_at=datetime(2026, 1, 15, 12, 30, tzinfo=timezone.utc),
        )]
        return types.SimpleNamespace(models=records)

    def show(self, model):
        if model not in self.models:
            raise FakeResponseError(f"model '{model}' not found", status_code=404)
        return FakeShowResponse()

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return types.SimpleNamespace(done=True, total_duration=1_000, load_duration=500)


class FakeOllamaModule(types.ModuleType):
    """``ollama`` module double exposing Client() and ResponseError()."""

    Client = FakeOllamaClient
    ResponseError = None  # set below: the real module exposes ResponseError


class FakeResponseError(Exception):
    """Mirror of ollama.ResponseError(error, status_code)."""

    def __init__(self, error: str, status_code: int = -1):
        super().__init__(error)
        self.status_code = status_code


FakeOllamaModule.ResponseError = FakeResponseError


# -------------------------------------------------------------------- fixtures

@pytest.fixture
def fake_lmstudio(monkeypatch):
    client = FakeLMStudioClient(downloaded=[
        FakeDownloaded("qwen/qwen3.5-4b"),
        FakeDownloaded("nomic-embed", type="embedding", vision=None,
                       trained_for_tool_use=None, architecture="nomic-bert"),
    ])
    module = FakeLMStudioModule("lmstudio")
    module.Client = lambda *args, **kwargs: client  # every Client() gets the shared instance
    monkeypatch.setitem(sys.modules, "lmstudio", module)
    return client


def kill_lmstudio(monkeypatch, client_class=DeadLMStudioClient):
    """Make the installed fake lmstudio module hand out an unreachable client."""
    monkeypatch.setattr(sys.modules["lmstudio"], "Client",
                        lambda *args, **kwargs: client_class())


@pytest.fixture
def fake_ollama(monkeypatch):
    client = FakeOllamaClient()
    module = FakeOllamaModule("ollama")
    module.Client = lambda *args, **kwargs: client  # every Client() gets the shared instance
    monkeypatch.setitem(sys.modules, "ollama", module)
    return client


# ------------------------------------------------------------------ LM Studio

def test_lmstudio_lists_downloaded_models_with_load_flags(fake_lmstudio):
    handler = LMStudioLocalModelHandler(client=fake_lmstudio)
    models = handler.list_models()
    assert [m["model_key"] for m in models] == ["qwen/qwen3.5-4b", "nomic-embed"]
    assert models[0]["loaded"] is False
    assert models[0]["params_string"] == "4B"
    assert models[1]["type"] == "embedding"


def test_lmstudio_list_loaded_reports_instances(fake_lmstudio):
    handler = LMStudioLocalModelHandler(client=fake_lmstudio)
    fake_lmstudio.llm.loaded["qwen/qwen3.5-4b::instance"] = FakeLMStudioHandle(
        "qwen/qwen3.5-4b::instance",
        FakeStruct(identifier="qwen/qwen3.5-4b::instance", model_key="qwen/qwen3.5-4b",
                   context_length=4096, max_context_length=131072),
    )
    loaded = handler.list_loaded()
    assert loaded[0]["identifier"] == "qwen/qwen3.5-4b::instance"
    assert loaded[0]["context_length"] == 4096


def test_lmstudio_load_returns_instance_info_and_flags(fake_lmstudio):
    handler = LMStudioLocalModelHandler(client=fake_lmstudio, ttl=600, load_config={"context_length": 8192})
    info = handler.load("qwen/qwen3.5-4b")
    assert info["identifier"] == "qwen/qwen3.5-4b::instance"
    assert info["loaded"] is True
    assert info["context_length"] == 4096
    # load() call recorded with the handler defaults
    assert fake_lmstudio.llm.model_calls == [("qwen/qwen3.5-4b", 600, {"context_length": 8192})]


def test_lmstudio_load_per_call_overrides(fake_lmstudio):
    handler = LMStudioLocalModelHandler(client=fake_lmstudio)
    handler.load("qwen/qwen3.5-4b", ttl=60, config={"gpu": "off"})
    assert fake_lmstudio.llm.model_calls == [("qwen/qwen3.5-4b", 60, {"gpu": "off"})]


def test_lmstudio_unload_by_model_key_and_identifier(fake_lmstudio):
    handler = LMStudioLocalModelHandler(client=fake_lmstudio)
    handler.load("qwen/qwen3.5-4b")
    assert handler.unload("qwen/qwen3.5-4b") is True  # via model key
    assert fake_lmstudio.llm.loaded == {}
    handler.load("qwen/qwen3.5-4b")
    assert handler.unload("qwen/qwen3.5-4b::instance") is True  # via identifier
    assert fake_lmstudio.llm.loaded == {}
    assert handler.unload("nope") is False


def test_lmstudio_model_info_loaded_then_downloaded(fake_lmstudio):
    handler = LMStudioLocalModelHandler(client=fake_lmstudio)
    handler.load("qwen/qwen3.5-4b")
    info = handler.model_info("qwen/qwen3.5-4b")  # loaded instance wins
    assert info["loaded"] is True and info["vision"] is True
    info = handler.model_info("qwen/qwen3.5-4b::instance")  # by identifier too
    assert info["loaded"] is True


def test_lmstudio_model_info_downloaded_only(fake_lmstudio):
    handler = LMStudioLocalModelHandler(client=fake_lmstudio)
    info = handler.model_info("qwen/qwen3.5-4b")
    assert info["loaded"] is False
    assert info["path"].endswith("qwen/qwen3.5-4b.gguf")


def test_lmstudio_model_info_unknown_raises(fake_lmstudio):
    handler = LMStudioLocalModelHandler(client=fake_lmstudio)
    with pytest.raises(ModelNotFoundError):
        handler.model_info("ghost")


def test_lmstudio_is_available_reflects_server(fake_lmstudio):
    handler = LMStudioLocalModelHandler(client=fake_lmstudio)
    assert handler.is_available() is True
    assert LMStudioLocalModelHandler(client=DeadLMStudioClient()).is_available() is False


# -------------------------------------------------------------------- Ollama

def test_ollama_lists_registry_models_with_load_flags(fake_ollama):
    handler = OllamaLocalModelHandler(client=fake_ollama)
    models = handler.list_models()
    assert [m["model"] for m in models] == ["llama3.2:3b", "qwen2.5:7b"]
    assert models[0]["loaded"] is True  # running per ps()
    assert models[1]["loaded"] is False
    assert models[0]["parameter_size"] == "3.2B"
    assert models[0]["modified_at"] == "2026-01-15T00:00:00+00:00"


def test_ollama_list_loaded_reports_vram_and_context(fake_ollama):
    handler = OllamaLocalModelHandler(client=fake_ollama)
    loaded = handler.list_loaded()
    assert loaded[0]["model"] == "llama3.2:3b"
    assert loaded[0]["size_vram"] == 1_500_000_000
    assert loaded[0]["context_length"] == 4096


def test_ollama_load_sends_empty_generate_with_keep_alive(fake_ollama):
    handler = OllamaLocalModelHandler(client=fake_ollama, keep_alive="30m")
    info = handler.load("qwen2.5:7b")
    assert info == {"model": "qwen2.5:7b", "loaded": True, "keep_alive": "30m",
                    "done": True, "total_duration": 1_000, "load_duration": 500}
    call = fake_ollama.generate_calls[-1]
    assert call["model"] == "qwen2.5:7b"
    assert call["keep_alive"] == "30m"
    assert "prompt" not in call  # empty generate = documented preload call


def test_ollama_load_default_keep_alive(fake_ollama):
    OllamaLocalModelHandler(client=fake_ollama).load("qwen2.5:7b")
    assert fake_ollama.generate_calls[-1]["keep_alive"] == "10m"


def test_ollama_unload_uses_zero_keep_alive(fake_ollama):
    handler = OllamaLocalModelHandler(client=fake_ollama)
    assert handler.unload("llama3.2:3b") is True
    assert fake_ollama.generate_calls[-1] == {"model": "llama3.2:3b", "keep_alive": 0}


def test_ollama_model_info_parses_show_response(fake_ollama):
    handler = OllamaLocalModelHandler(client=fake_ollama)
    info = handler.model_info("llama3.2:3b")
    assert info["capabilities"] == ["completion", "tools", "vision"]
    assert info["parameter_size"] == "3.2B"
    assert info["quantization_level"] == "Q4_K_M"
    # parameters block parsed: numerics converted, repeated keys collected
    assert info["parameters"]["temperature"] == 0.6
    assert info["parameters"]["num_ctx"] == 4096
    assert info["parameters"]["stop"] == ["<|im_end|>", "<|endoftext|>"]
    assert info["modelinfo"]["general.architecture"] == "llama"


def test_ollama_model_info_unknown_raises(fake_ollama):
    handler = OllamaLocalModelHandler(client=fake_ollama)
    with pytest.raises(ModelNotFoundError):
        handler.model_info("ghost")


def test_ollama_is_available_reflects_server(fake_ollama):
    class DeadClient(FakeOllamaClient):
        def list(self):
            raise ConnectionError("cannot connect")

    assert OllamaLocalModelHandler(client=fake_ollama).is_available() is True
    assert OllamaLocalModelHandler(client=DeadClient()).is_available() is False


# -------------------------------------------- LM Studio host fallback

class UnresolvedPortClient(FakeLMStudioClient):
    """SDK default-host client: LM Studio app daemon port is unresolved."""

    def list_downloaded_models(self, namespace=None):
        raise ConnectionError("Local API host port is not yet resolved.")

    @property
    def llm(self):
        raise ConnectionError("Local API host port is not yet resolved.")


class FallbackLMStudioModule(types.ModuleType):
    """``lmstudio`` double: Client() -> unresolved daemon, Client(host) -> live server."""

    @staticmethod
    def Client(host=None):
        if host is None:
            return UnresolvedPortClient()  # app-daemon discovery: unresolved
        if host == "localhost:1234":
            return FakeLMStudioClient(downloaded=[FakeDownloaded("qwen/qwen3.5-4b")])
        return UnresolvedPortClient()  # any other explicit host: down


def test_lmstudio_falls_back_to_api_server_port(monkeypatch):
    # Headless `lms server start`: app-daemon discovery fails, the handler
    # retries once against the standard API server port and remembers it.
    monkeypatch.setitem(sys.modules, "lmstudio", FallbackLMStudioModule("lmstudio"))
    handler = LMStudioLocalModelHandler()  # no host -> SDK default discovery is unresolved
    assert handler.is_available() is True
    models = handler.list_models()
    assert [m["model_key"] for m in models] == ["qwen/qwen3.5-4b"]
    assert handler._host == LMStudioLocalModelHandler.FALLBACK_HOST


def test_lmstudio_explicit_host_binds_directly(monkeypatch):
    monkeypatch.setitem(sys.modules, "lmstudio", FallbackLMStudioModule("lmstudio"))
    handler = LMStudioLocalModelHandler(host="localhost:1234")
    assert handler.list_models()[0]["model_key"] == "qwen/qwen3.5-4b"
    assert handler._client is not None  # bound directly, no fallback needed
    assert handler._host == "localhost:1234"


def test_lmstudio_no_fallback_for_explicit_host(monkeypatch):
    monkeypatch.setitem(sys.modules, "lmstudio", FallbackLMStudioModule("lmstudio"))
    handler = LMStudioLocalModelHandler(host="localhost:9999")
    assert handler.is_available() is False  # explicit host: no fallback retry
    assert handler._host == "localhost:9999"


# ------------------------------------------------- server-failure wrapping

def test_lmstudio_unreachable_raises_local_model_handler_error(fake_lmstudio):
    # LM Studio app reachable but loaded-model listing fails (e.g. API server
    # not started: "Local API host port is not yet resolved.").
    handler = LMStudioLocalModelHandler(client=DeadLMStudioClient())
    with pytest.raises(LocalModelHandlerError, match="lmstudio listing loaded models failed"):
        handler.list_models()


def test_lmstudio_download_listing_failure_wrapped(fake_lmstudio):
    class DeadDownloadedClient(FakeLMStudioClient):
        def list_downloaded_models(self, namespace=None):
            raise ConnectionError("Local API host port is not yet resolved.")

    with pytest.raises(LocalModelHandlerError, match="lmstudio listing downloaded models failed"):
        LMStudioLocalModelHandler(client=DeadDownloadedClient()).list_models()


def test_lmstudio_load_unknown_model_raises_not_found(fake_lmstudio):
    class LMStudioModelNotFoundError(Exception):  # name mimics the SDK class
        pass

    class NotFoundSession(FakeLMStudioSession):
        def model(self, model_key, /, *, ttl=3600, config=None, on_load_progress=None):
            raise LMStudioModelNotFoundError(f"model '{model_key}' not found")

    class ClientWithNotFoundSession(FakeLMStudioClient):
        @property
        def llm(self):
            return NotFoundSession()

    with pytest.raises(ModelNotFoundError):
        LMStudioLocalModelHandler(client=ClientWithNotFoundSession()).load("ghost")


def test_ollama_unreachable_raises_local_model_handler_error(fake_ollama):
    class DeadClient(FakeOllamaClient):
        def list(self):
            raise ConnectionError("cannot connect")

    with pytest.raises(LocalModelHandlerError, match="ollama listing available models failed"):
        OllamaLocalModelHandler(client=DeadClient()).list_models()


def test_ollama_load_unknown_model_raises_not_found(fake_ollama):
    class NotFoundClient(FakeOllamaClient):
        def generate(self, **kwargs):
            raise FakeResponseError("model 'ghost' not found", status_code=404)

    with pytest.raises(ModelNotFoundError):
        OllamaLocalModelHandler(client=NotFoundClient()).load("ghost")


# -------------------------------------------------------------------- facade

def test_facade_binds_explicit_backend(fake_ollama):
    handler = LocalModelHandler("ollama")
    assert handler.backend == "ollama"
    assert handler.list_models()[0]["model"] == "llama3.2:3b"


def test_facade_auto_detects_alive_server(fake_lmstudio, fake_ollama):
    handler = LocalModelHandler("auto")
    assert handler.backend == "lmstudio"  # probed first, alive -> wins


def test_facade_auto_falls_back_to_ollama(fake_lmstudio, fake_ollama, monkeypatch):
    kill_lmstudio(monkeypatch)
    handler = LocalModelHandler("auto")
    assert handler.backend == "ollama"


def test_facade_raises_when_no_server_answers(fake_lmstudio, fake_ollama, monkeypatch):
    class DeadOllamaClient(FakeOllamaClient):
        def list(self):
            raise ConnectionError("cannot connect")

    kill_lmstudio(monkeypatch)
    monkeypatch.setattr(sys.modules["ollama"], "Client",
                        lambda *args, **kwargs: DeadOllamaClient())
    with pytest.raises(LocalModelHandlerError, match="No local model server reachable"):
        LocalModelHandler("all")


def test_facade_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        LocalModelHandler("vllm")


# ------------------------------------------------ multi-backend ("all")

def test_facade_all_binds_both_servers(fake_lmstudio, fake_ollama):
    handler = LocalModelHandler("all")
    assert handler.backends == ["lmstudio", "ollama"]
    assert handler.backend == "all"


def test_facade_all_merges_and_tags_lists(fake_lmstudio, fake_ollama):
    handler = LocalModelHandler("all")
    available = handler.list_models()
    assert {m["backend"] for m in available} == {"lmstudio", "ollama"}
    assert any(m.get("model_key") == "qwen/qwen3.5-4b" for m in available)
    assert any(m.get("model") == "llama3.2:3b" for m in available)
    loaded = handler.list_loaded()  # ollama fake runs llama3.2:3b
    assert [m["model"] for m in loaded] == ["llama3.2:3b"]
    assert loaded[0]["backend"] == "ollama"


def test_facade_all_routes_unqualified_to_single_owner(fake_lmstudio, fake_ollama):
    handler = LocalModelHandler("all")
    info = handler.load("qwen2.5:7b")  # only in the ollama fake's registry
    assert info["backend"] == "ollama"
    assert fake_ollama.generate_calls[-1]["model"] == "qwen2.5:7b"


def test_facade_all_qualified_name_forces_backend(fake_lmstudio, fake_ollama):
    fake_lmstudio._downloaded.append(FakeDownloaded("shared"))  # ambiguous name
    fake_ollama.models.append("shared")
    handler = LocalModelHandler("all")
    with pytest.raises(LocalModelHandlerError, match="several backends"):
        handler.load("shared")
    info = handler.load("ollama:shared")  # qualified -> routed to ollama
    assert info["backend"] == "ollama"
    assert fake_ollama.generate_calls[-1]["model"] == "shared"
    lm_info = handler.model_info("lmstudio:shared")  # qualified -> routed to lmstudio
    assert lm_info["backend"] == "lmstudio"


def test_facade_all_model_info_unknown_across_backends(fake_lmstudio, fake_ollama):
    handler = LocalModelHandler("all")
    with pytest.raises(ModelNotFoundError, match="lmstudio, ollama"):
        handler.model_info("ghost")


def test_facade_all_skips_dead_backend(fake_lmstudio, fake_ollama, monkeypatch):
    kill_lmstudio(monkeypatch)
    handler = LocalModelHandler("all")
    assert handler.backends == ["ollama"]
    assert {m["backend"] for m in handler.list_models()} == {"ollama"}


# -------------------------------------------------------------------- utils

def test_human_size():
    assert human_size(0) == "0.0 B"
    assert human_size(512) == "512.0 B"
    assert human_size(2300000000) == "2.1 GiB"
    assert human_size(None) == "0.0 B"
