"""Tests for the transcribe_audio tool (smolagents SpeechToTextTool port)."""
import src.tools.stt as stt_mod


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, path, language=None):
        return iter(self._segments), None


def test_transcribe_audio_returns_text(tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"RIFF....")
    stt_mod._STT_FACTORY = lambda: _FakeModel(
        [_FakeSegment("hello"), _FakeSegment("world")])
    try:
        result = stt_mod.transcribe_audio(str(audio))
        assert result == "hello world"
    finally:
        stt_mod._STT_FACTORY = None


def test_transcribe_audio_missing_file():
    stt_mod._STT_FACTORY = lambda: _FakeModel([])
    try:
        result = stt_mod.transcribe_audio("Z:/no/such/file.wav")
        assert result.startswith("Error: audio file not found")
    finally:
        stt_mod._STT_FACTORY = None


def test_transcribe_audio_empty_path():
    assert stt_mod.transcribe_audio("").startswith("Error: no audio file")


def test_transcribe_audio_too_large(tmp_path):
    big = tmp_path / "big.wav"
    big.write_bytes(b"x" * 10)  # tiny, but we patch the limit
    original = stt_mod._MAX_FILE_BYTES
    stt_mod._MAX_FILE_BYTES = 5
    try:
        result = stt_mod.transcribe_audio(str(big))
        assert result.startswith("Error: audio file too large")
    finally:
        stt_mod._MAX_FILE_BYTES = original


def test_transcribe_audio_missing_dependency(tmp_path, monkeypatch):
    """A missing faster-whisper surfaces as an Error line, not a crash."""
    import builtins
    audio = tmp_path / "s.wav"
    audio.write_bytes(b"ok")
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("No module named 'faster_whisper'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    stt_mod._STT_FACTORY = None
    stt_mod._model_cache = None
    try:
        result = stt_mod.transcribe_audio(str(audio))
        assert result.startswith("Error: faster-whisper is not installed")
    finally:
        stt_mod._model_cache = None


def test_transcribe_audio_no_speech(tmp_path):
    audio = tmp_path / "silence.wav"
    audio.write_bytes(b"ok")
    stt_mod._STT_FACTORY = lambda: _FakeModel([])
    try:
        result = stt_mod.transcribe_audio(str(audio))
        assert result == "(no speech recognized)"
    finally:
        stt_mod._STT_FACTORY = None


def test_transcribe_audio_in_build_tools():
    import src.live_ai as live_ai_mod
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "transcribe_audio" in names


def test_stt_configure_reads_section():
    from types import SimpleNamespace

    class _Rec:
        calls = []

        @staticmethod
        def record(section):
            _Rec.calls.append(section)

    config = SimpleNamespace(stt=SimpleNamespace(model_size="tiny",
                                                 device="cpu"))
    stt_mod.configure(config)
    assert stt_mod._config_cache is config.stt
    assert stt_mod._model_cache is None  # cache reset on reconfigure
