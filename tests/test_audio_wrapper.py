import io
from logging import getLogger
from types import SimpleNamespace

import numpy as np
import pytest

from tests.conftest import (  # noqa: E402
    FakeDevice,
    FakeStream,
    FakeWhisperModel,
    require_import,
)

# audio_wrapper imports sounddevice/faster_whisper at module top; these may be
# absent, or present but unusable (no audio server on a headless runner).
require_import("sounddevice", "faster_whisper")

import soundfile as sf  # noqa: E402
import src.audio_wrapper as aw  # noqa: E402
from src.audio_wrapper import (  # noqa: E402
    audio_to_wav,
    audio_to_wav_buffer,
    capture_speech,
    compute_silence_threshold,
    format_level,
    get_audio_tone,
    rms,
)


class _Clock:
    def __init__(self, step=0.0):
        self.t = 0.0
        self.step = step

    def __call__(self):
        self.t += self.step
        return self.t


def _silent(n):
    return np.zeros(n, dtype=np.int16)


def _loud(n):
    return np.full(n, 5000, dtype=np.int16)


def test_format_level_hidden():
    assert format_level(0.5, 0.1, show=False) == ""


def test_format_level_shown():
    s = format_level(0.5, 0.1, show=True)
    assert "0.500" in s and "0.100" in s


# ---------------------------------------------------------------- A0 fakes


def test_fake_stream_is_context_managed_and_reads():
    chunks = [np.ones(4, dtype=np.int16)]
    with FakeStream(chunks) as s:
        chunk, _ = s.read(4)
    assert chunk.tolist() == [1, 1, 1, 1]
    assert s.read_count == 1


def test_fake_whisper_model_returns_text():
    segments, _ = FakeWhisperModel("hi").transcribe("ignored")
    assert segments[0].text == "hi"


# ------------------------------------------------------------- A1 tone


def test_tone_is_not_silent():
    tone = get_audio_tone(duration=0.1, samplerate=16000, frequency=440.0)
    assert tone.dtype == np.int16
    assert int(np.abs(tone).max()) > 1000


def test_tone_length_matches_samplerate():
    tone = get_audio_tone(duration=0.25, samplerate=8000)
    assert len(tone) == 2000


def test_tone_zero_amplitude_is_silent():
    assert int(np.abs(get_audio_tone(0.05, amplitude=0.0)).max()) == 0


# ------------------------------------------------------------- A2 wav


def test_audio_to_wav_roundtrip(tmp_path):
    audio = get_audio_tone(duration=0.05, samplerate=8000)
    out = audio_to_wav(audio, tmp_path / "t.wav", samplerate=8000)
    assert out == tmp_path / "t.wav" and out.exists()
    data, sr = sf.read(str(out))
    assert sr == 8000 and len(data) == len(audio)


def test_audio_to_wav_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        audio_to_wav(np.array([], dtype=np.int16), tmp_path / "e.wav")


# ------------------------------------------------------------- A3 rms/threshold


def test_rms_empty_is_zero():
    assert rms(np.array([], dtype=np.int16)) == 0.0


def test_rms_constant_signal():
    assert rms(np.full(10, 100, dtype=np.int16)) == pytest.approx(100.0, rel=1e-6)


def test_threshold_uses_exactly_requested_chunks():
    stream = FakeStream([np.zeros(8, dtype=np.int16)] * 5)
    threshold, mean_rms, std_rms = compute_silence_threshold(
        FakeDevice(), samplerate=8000, chunk_time_s=0.001,
        chunks=5, stream_factory=lambda **kw: stream,
    )
    assert stream.read_count == 5
    assert (threshold, mean_rms, std_rms) == (0.0, 0.0, 0.0)


def test_threshold_scales_with_guard_factor():
    stream = FakeStream([np.full(8, 1000, dtype=np.int16)] * 4)
    t1, _, _ = compute_silence_threshold(
        FakeDevice(), samplerate=8000, chunk_time_s=0.001, chunks=4,
        guard_factor=1.0, stream_factory=lambda **kw: stream,
    )
    stream2 = FakeStream([np.full(8, 1000, dtype=np.int16)] * 4)
    t2, _, _ = compute_silence_threshold(
        FakeDevice(), samplerate=8000, chunk_time_s=0.001, chunks=4,
        guard_factor=2.0, stream_factory=lambda **kw: stream2,
    )
    assert t2 == pytest.approx(2 * t1)


# ------------------------------------------------------------- A4 device errors


def test_audio_device_error_when_query_fails(monkeypatch):
    def boom(**kw):
        raise RuntimeError("no backend")

    monkeypatch.setattr(aw.sd, "query_devices", boom)
    with pytest.raises(aw.AudioDeviceError):
        aw.InputAudioDevice()


def test_audio_device_error_when_no_devices(monkeypatch):
    monkeypatch.setattr(aw.sd, "query_devices", lambda **kw: [])
    with pytest.raises(aw.AudioDeviceError):
        aw.InputAudioDevice()


def test_source_contains_no_sys_exit():
    import pathlib

    text = pathlib.Path(aw.__file__).read_text(encoding="utf-8")
    assert "sys.exit" not in text


# ------------------------------------------------------------- A5 capture loop


def test_capture_stops_after_consecutive_silence():
    stream = FakeStream([_loud(4), _silent(4), _silent(4)])
    audio = capture_speech(
        threshold=100.0, device_index=0, chunk_time_s=0.1,
        silence_timeout_s=0.2, stream_factory=lambda **kw: stream,
        time_func=_Clock(), max_record_chunks=10,
    )
    assert audio.size > 0
    assert int(np.abs(audio).max()) > 100


def test_capture_returns_empty_when_no_speech():
    stream = FakeStream([_silent(4)] * 5)
    audio = capture_speech(
        threshold=100.0, device_index=0, chunk_time_s=0.1,
        silence_timeout_s=0.2, stream_factory=lambda **kw: stream,
        time_func=_Clock(), max_record_chunks=3,
    )
    assert audio.size == 0 and audio.dtype == np.int16


def test_capture_honours_max_record_chunks():
    stream = FakeStream([_loud(4)] * 100)
    audio = capture_speech(
        threshold=100.0, device_index=0, chunk_time_s=0.1,
        silence_timeout_s=100.0, stream_factory=lambda **kw: stream,
        time_func=_Clock(), max_record_chunks=3,
    )
    assert stream.read_count == 3 and audio.size > 0


def test_capture_clears_buffer_after_clear_timeout():
    clock = _Clock(step=10.0)
    stream = FakeStream([_silent(4), _silent(4), _loud(4)])
    audio = capture_speech(
        threshold=100.0, device_index=0, chunk_time_s=0.1,
        silence_timeout_s=0.2, clear_timeout_s=5.0,
        stream_factory=lambda **kw: stream, time_func=clock,
        max_record_chunks=3,
    )
    assert audio.size > 0


def test_level_callback_called_per_chunk():
    seen = []
    stream = FakeStream([_silent(4), _loud(4), _silent(4), _silent(4)])
    capture_speech(
        threshold=100.0, device_index=0, chunk_time_s=0.1,
        silence_timeout_s=0.2, stream_factory=lambda **kw: stream,
        time_func=_Clock(), on_level=seen.append, max_record_chunks=10,
    )
    assert len(seen) == 4


# ------------------------------------------------------------- A6 in-memory wav


def test_wav_buffer_is_valid_and_honours_samplerate():
    tone = get_audio_tone(0.05, samplerate=8000)
    buf = audio_to_wav_buffer(tone, samplerate=8000)
    assert isinstance(buf, io.BytesIO) and buf.tell() == 0
    info = sf.info(buf)
    assert info.samplerate == 8000 and info.frames == len(tone)


def test_wav_buffer_rejects_empty():
    with pytest.raises(ValueError):
        audio_to_wav_buffer(np.array([], dtype=np.int16))


def test_transcribe_accepts_file_like_buffer():
    model = FakeWhisperModel("hello")
    fake_self = SimpleNamespace(
        model=model, language="en", beam_size=5,
        _ensure_model=lambda: model, logger=getLogger("test"),
    )
    buf = audio_to_wav_buffer(get_audio_tone(0.02, samplerate=8000), 8000)
    text = aw.SpeechDetector.transcribe(fake_self, buf)
    assert text == "hello" and model.calls == 1


def test_listen_and_transcribe_empty_audio_skips_model():
    calls = {"transcribe": 0}
    fake_self = SimpleNamespace(
        listen=lambda **kw: np.array([], dtype=np.int16),
        transcribe=lambda audio: calls.__setitem__("transcribe", calls["transcribe"] + 1),
    )
    assert aw.SpeechDetector.listen_and_transcribe(fake_self, 1.0) == ""
    assert calls["transcribe"] == 0


# ------------------------------------------------------------- A7 lazy model


def test_model_is_lazy_and_loaded_once(monkeypatch, tmp_path):
    model_dir = tmp_path / "faster-whisper"
    model_dir.mkdir()
    calls = []

    def fake_backend(path, device=None, compute_type=None):
        calls.append((path, device, compute_type))
        return FakeWhisperModel()

    monkeypatch.setattr(aw, "WhisperModel", fake_backend)
    stream = FakeStream([np.zeros(4, dtype=np.int16)] * 2)

    detector = aw.SpeechDetector(
        model_path=model_dir,
        device=FakeDevice(),
        stream_factory=lambda **kw: stream,
        time_func=_Clock(),
        calibration_chunks=2,
    )
    assert calls == []  # not loaded at construction
    detector._ensure_model()
    detector._ensure_model()
    assert len(calls) == 1


def test_model_backend_and_compute_type_passed_through(monkeypatch, tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    seen = {}

    def fake_backend(path, device=None, compute_type=None):
        seen["device"] = device
        seen["compute_type"] = compute_type
        return FakeWhisperModel()

    monkeypatch.setattr(aw, "WhisperModel", fake_backend)
    detector = aw.SpeechDetector(
        model_path=model_dir, device=FakeDevice(),
        stream_factory=lambda **kw: FakeStream([np.zeros(4, dtype=np.int16)] * 2),
        time_func=_Clock(), calibration_chunks=2,
        model_backend="cpu", compute_type="int8_float16",
    )
    detector._ensure_model()
    assert seen == {"device": "cpu", "compute_type": "int8_float16"}


# ------------------------------------------------------------- A9 end to end


def test_end_to_end_capture_then_transcribe(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()

    detector = aw.SpeechDetector(
        model_path=model_dir,
        device=FakeDevice(),
        stream_factory=lambda **kw: FakeStream([_loud(8), _silent(8), _silent(8)]),
        time_func=_Clock(),
        model=FakeWhisperModel("end to end"),
        calibration_chunks=2,
    )
    detector.silence_threshold = 100.0  # bypass ambient calibration
    text = detector.listen_and_transcribe(silence_timeout_s=0.2, samplerate=8000)
    assert text == "end to end"


def test_no_cprint_import_inside_loop():
    import pathlib

    src = pathlib.Path(aw.__file__).read_text(encoding="utf-8")
    loop_start = src.index("def capture_speech")
    loop_end = src.index("class SpeechDetector", loop_start)
    assert "import cprint" not in src[loop_start:loop_end]
