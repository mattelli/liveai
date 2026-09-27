import wave
from types import SimpleNamespace

import pytest

pytest.importorskip("piper", reason="audio_generator requires piper to import")
pytest.importorskip("sounddevice", reason="audio_generator requires sounddevice to import")

import src.audio_generator as ag  # noqa: E402
from tests.conftest import (  # noqa: E402
    FakeOutputStream,
    FakePiperVoice,
)


def make_model(tmp_path):
    """Create a placeholder .onnx path so model_path.exists() is True."""
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"not-a-real-model")
    return model


def make_generator(tmp_path, **kw):
    kw.setdefault("voice", FakePiperVoice())
    kw.setdefault("stream", FakeOutputStream())
    return ag.SpeechGenerator(model_path=make_model(tmp_path), **kw)


# ---------------------------------------------------------------- G0 fakes


def test_fakes_are_usable():
    voice = FakePiperVoice()
    chunks = list(voice.synthesize("hi"))
    assert chunks[0].audio_int16_bytes == b"\x01\x02"
    stream = FakeOutputStream(a=1)
    stream.start()
    stream.stop()
    stream.close()
    assert (stream.started, stream.stopped, stream.closed) == (1, 1, 1)


def test_make_generator_constructs_without_hardware(tmp_path):
    gen = make_generator(tmp_path)
    assert gen.voice is not None and isinstance(gen.stream, FakeOutputStream)


# ---------------------------------------------------------------- G1 clean


def test_clean_removes_markdown_and_parens():
    cleaned = ag.SpeechGenerator.clean_for_speech("**b** _i_ (p) [x] #t `c` > q ~s~")
    assert cleaned == "b i p x t c q s"


def test_clean_removes_emoji():
    assert ag.SpeechGenerator.clean_for_speech("hi 👋 there") == "hi there"


def test_clean_collapses_whitespace():
    assert ag.SpeechGenerator.clean_for_speech("a\n\n  b\tc") == "a b c"


def test_clean_empty_returns_empty():
    assert ag.SpeechGenerator.clean_for_speech("") == ""


# ---------------------------------------------------------------- G2 lifecycle


def test_close_releases_stream_and_is_idempotent(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)
    gen.close()
    gen.close()
    assert stream.closed == 1 and stream.stopped == 1
    assert gen.stream is None


def test_context_manager_closes_stream(tmp_path):
    stream = FakeOutputStream()
    with make_generator(tmp_path, stream=stream) as gen:
        assert gen is not None
    assert stream.closed == 1


# ---------------------------------------------------------------- G3 lazy voice


def test_voice_is_lazy_and_loaded_once(monkeypatch, tmp_path):
    calls = []

    def fake_load(path):
        calls.append(path)
        return FakePiperVoice(sample_rate=22050)

    monkeypatch.setattr(ag, "PiperVoice", SimpleNamespace(load=fake_load))
    gen = ag.SpeechGenerator(model_path=make_model(tmp_path), stream=FakeOutputStream())
    assert calls == []  # not loaded at construction
    gen._ensure_voice()
    gen._ensure_voice()
    assert len(calls) == 1


def test_injected_voice_is_not_reloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ag, "PiperVoice", SimpleNamespace(load=lambda p: pytest.fail("load"))
    )
    voice = FakePiperVoice()
    gen = make_generator(tmp_path, voice=voice)
    assert gen._ensure_voice() is voice


# ---------------------------------------------------------------- G4 speak


def test_speak_empty_text_is_a_noop(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)
    assert gen.speak("   ") == 0
    assert stream.started == 0 and stream.stopped == 0


def test_speak_writes_chunks_and_returns_bytes(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)
    written = gen.speak("hello", strip_formatting=False)
    assert written == 4  # two 2-byte chunks
    assert stream.written == [b"\x01\x02", b"\x01\x02"]
    assert stream.started == 1 and stream.stopped == 1


def test_speak_stops_stream_when_synthesis_fails(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, voice=FakePiperVoice(fail_after=1), stream=stream)
    with pytest.raises(RuntimeError):
        gen.speak("hello", strip_formatting=False)
    assert stream.started == 1 and stream.stopped == 1


def test_speak_cleans_text_by_default(tmp_path):
    voice = FakePiperVoice()
    gen = make_generator(tmp_path, voice=voice)
    gen.speak("**hi**", strip_formatting=True)
    assert voice.last_text == "hi"


# ---------------------------------------------------------------- G5 config


@pytest.mark.parametrize("speed", [0, -1.0])
def test_invalid_speech_speed_raises(tmp_path, speed):
    with pytest.raises(ValueError):
        ag.SpeechGenerator(
            model_path=make_model(tmp_path), speech_speed=speed,
            stream=FakeOutputStream(),
        )


@pytest.mark.parametrize("volume", [-0.1, 1.5])
def test_invalid_volume_raises(tmp_path, volume):
    with pytest.raises(ValueError):
        ag.SpeechGenerator(
            model_path=make_model(tmp_path), volume=volume,
            stream=FakeOutputStream(),
        )


def test_synthesis_config_receives_params(monkeypatch, tmp_path):
    seen = {}

    class RecorderConfig:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(ag, "SynthesisConfig", RecorderConfig)
    ag.SpeechGenerator(
        model_path=make_model(tmp_path),
        volume=0.25, speech_speed=2.0,
        speaker_id=3, noise_scale=0.5, noise_w_scale=0.6, normalize_audio=True,
        stream=FakeOutputStream(),
    )
    assert seen == {
        "speaker_id": 3, "volume": 0.25, "length_scale": 0.5,
        "noise_scale": 0.5, "noise_w_scale": 0.6, "normalize_audio": True,
    }


# ---------------------------------------------------------------- G6 samplerate


def test_samplerate_mismatch_raises(tmp_path):
    gen = make_generator(
        tmp_path, voice=FakePiperVoice(sample_rate=16000), samplerate=22050
    )
    with pytest.raises(ValueError):
        gen._ensure_voice()


def test_samplerate_match_is_accepted(tmp_path):
    gen = make_generator(
        tmp_path, voice=FakePiperVoice(sample_rate=16000), samplerate=16000
    )
    assert gen._ensure_voice() is gen.voice


def test_missing_voice_rate_is_tolerated(tmp_path):
    voice = FakePiperVoice()
    voice.config = SimpleNamespace()  # no sample_rate attribute
    gen = make_generator(tmp_path, voice=voice)
    assert gen._ensure_voice() is voice


# ---------------------------------------------------------------- G7 to_wav


def test_to_wav_creates_parent_and_returns_path(tmp_path):
    gen = make_generator(tmp_path)
    out = gen.to_wav("hello", tmp_path / "nested" / "out.wav", strip_formatting=False)
    assert out == tmp_path / "nested" / "out.wav" and out.exists()
    with wave.open(str(out), "rb") as wav_file:
        assert wav_file.getnframes() == 2  # FakePiperVoice writes two frames


def test_to_wav_empty_text_raises(tmp_path):
    gen = make_generator(tmp_path)
    with pytest.raises(ValueError):
        gen.to_wav("   ", tmp_path / "e.wav")


def test_to_wav_cleans_text_by_default(tmp_path):
    voice = FakePiperVoice()
    gen = make_generator(tmp_path, voice=voice)
    gen.to_wav("**hi**", tmp_path / "c.wav", strip_formatting=True)
    assert voice.last_text == "hi"
    assert voice.wav_calls == 1


# ---------------------------------------------------------------- G9 acceptance


def test_end_to_end_speak_and_export(tmp_path):
    stream = FakeOutputStream()
    voice = FakePiperVoice(chunks=(b"\x01\x02", b"\x03\x04"))
    gen = make_generator(tmp_path, voice=voice, stream=stream)

    written = gen.speak("hello", strip_formatting=False)
    out = gen.to_wav("bye", tmp_path / "out.wav", strip_formatting=False)

    assert written == 4 and stream.written == [b"\x01\x02", b"\x03\x04"]
    assert stream.started == 1 and stream.stopped == 1
    assert out.exists() and voice.synth_calls == 1 and voice.wav_calls == 1


def test_module_exports_speech_generator():
    assert ag.__all__ == ["SpeechGenerator"]
