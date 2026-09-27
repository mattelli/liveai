# PLAN-AUDIO-WRAPPER.md — Improve `src/audio_wrapper.py`

Addendum to [`PLAN.md`](PLAN.md). Everything in `PLAN.md` §0 (rules for the executing
agent), §2 (test strategy and conventions) and §3 (design constraints) still applies.
Read those sections before starting any task here.

**Scope.** Correctness bugs, robustness and testability of `src/audio_wrapper.py` plus the
`[speech_det]` wiring in `main.py`/`config.toml`. No change to `Brain`/`Agent`,
`ConversationSession`, input handling or tools.

**Target files.** `src/audio_wrapper.py`, `main.py`, `config.toml`,
`tests/test_audio_wrapper.py`, `tests/conftest.py`.

**Hardware rule (PLAN.md §0.3).** No test may open a real microphone, load a real Whisper
model, or touch `/dev/*`. Every stream, device and model used in tests is a fake injected
through the seams added in A0/A3/A5/A7.

**Baseline (current behaviour, for reference).** Line numbers are from the current file and
may shift as tasks land.

| # | Location | Issue |
|---|----------|-------|
| B1 | `get_audio_tone` (line 32) | `0.5 * sin()` cast to `int16` truncates to **all zeros** |
| B2 | `audio_to_wav` (line 52) | file handle never closed; crashes on empty array |
| B3 | `compute_silence_threshold` (line 65) | annotated `-> float` but returns a 3-tuple; collects `chunks + 1` samples (off-by-one) |
| B4 | `InputAudioDevice.__init__` (line 17) | calls `sys.exit(1)` — kills the host process, untestable |
| B5 | `listen` (line 108) | silence-timeout branch is nested under `audio_rms > threshold` where `audio_rms` is the cumulative mean; it rarely fires, so capture can run far past the intended stop |
| B6 | `listen_and_transcribe` (line 181) | hardcodes 16000 Hz; always overwrites `input_audio.wav` on disk; ignores configured samplerate |
| B7 | `SpeechDetector.__init__` (line 104) | loads `WhisperModel` eagerly; device/compute type hardcoded, not configurable |
| B8 | `listen` (line 108) | no upper bound on recording length |

---

## 0. Conventions for the executing agent

1. One task per commit, in the order of §1. Each task is self-contained.
2. Write the test **first**, watch it fail, then implement the change. A task is done only
   when its validation command passes.
3. Keep public signatures exactly as specified. Downstream callers (`main.py`,
   `playground/simple_speech_det.py`, `playground/simple_speech_runner.py`) must keep
   working; if a signature changes, update all callers in the same task.
4. Never call `sys.exit` inside `src/`. Raise an exception instead.
5. Do not import `faster_whisper`, `sounddevice`, or any model at test-collection time
   beyond the existing `pytest.importorskip` guard in `tests/test_audio_wrapper.py`.
6. Prefer small pure functions; test them directly.
7. `python -m pytest tests/ -q` must stay green at the end of every task (the full suite,
   not only the task's file).

---

## 1. Task index

| ID | Title | Depends on | Phase |
|----|-------|-----------|-------|
| A0 | Test harness for audio: fakes + import guard | — | Foundations |
| A1 | Fix `get_audio_tone` amplitude scaling | A0 | Correctness |
| A2 | Fix `audio_to_wav` (close handle, reject empty) | A0 | Correctness |
| A3 | Add `rms()` helper and fix `compute_silence_threshold` | A0 | Correctness |
| A4 | Replace `sys.exit` with `AudioDeviceError` | A0 | Correctness |
| A5 | Extract and fix the capture loop (`capture_speech`) | A3 | Correctness |
| A6 | In-memory WAV + correct samplerate in `listen_and_transcribe` | A2 | Correctness |
| A7 | Lazy, configurable Whisper model loading | A0, A5 | Robustness |
| A8 | Wire `[speech_det]` config into `SpeechDetector` | A7 | Integration |
| A9 | Typing/cleanup + end-to-end acceptance | A1–A8 | Verification |

A1–A6 fix real bugs. A7–A8 make the module testable and configurable. A9 closes the plan.

---

## A0 — Test harness for audio: fakes and import guard

**Goal.** Give the audio tests deterministic fakes so no task below needs a microphone,
sound card, or Whisper model.

**Current behaviour.** `tests/test_audio_wrapper.py` only tests `format_level` and guards
the whole module with `pytest.importorskip("sounddevice")`. There is no fake stream,
device or model.

**Expected behaviour.** In `tests/conftest.py` add (import `numpy` lazily, mirroring the
existing test file):

```python
class FakeDevice:
    """Stand-in for InputAudioDevice; no hardware touched."""
    name = "Fake Mic"
    index = 0
    default_samplerate = 16000
    max_channel = 1


class FakeStream:
    """Context-managed InputStream double.

    chunks: iterable of fixed-size numpy arrays returned by read().
    Raises StopIteration when exhausted (the caller must bound the loop).
    """
    def __init__(self, chunks):
        self._it = iter(chunks)
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, chunk_size):
        self.read_count += 1
        return next(self._it), False


class FakeWhisperModel:
    """Returns one fixed segment so transcribe() can be tested without weights."""
    def __init__(self, text="fake transcript"):
        self.text = text
        self.calls = 0

    def transcribe(self, audio, language=None, beam_size=None):
        self.calls += 1
        return [SimpleNamespace(text=self.text)], None
```

`tests/test_audio_wrapper.py` keeps its `importorskip` guard and imports the fakes from
`tests.conftest`. Its import block should look like:

```python
import io
from logging import getLogger
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("sounddevice", reason="audio_wrapper requires sounddevice to import")
pytest.importorskip("faster_whisper", reason="audio_wrapper requires faster_whisper to import")

import src.audio_wrapper as aw
from src.audio_wrapper import format_level  # noqa: E402
from tests.conftest import FakeDevice, FakeStream, FakeWhisperModel  # noqa: E402
```

Later tasks may import the specific functions they test from `src.audio_wrapper` alongside
these; keep the `importorskip` calls before the `src.audio_wrapper` import.

**Files.** `tests/conftest.py`, `tests/test_audio_wrapper.py`.

**Tests.** Add to `tests/test_audio_wrapper.py`:

```python
import numpy as np
from tests.conftest import FakeStream, FakeDevice, FakeWhisperModel


def test_fake_stream_is_context_managed_and_reads():
    chunks = [np.ones(4, dtype=np.int16)]
    with FakeStream(chunks) as s:
        chunk, _ = s.read(4)
    assert chunk.tolist() == [1, 1, 1, 1]
    assert s.read_count == 1


def test_fake_whisper_model_returns_text():
    segments, _ = FakeWhisperModel("hi").transcribe("ignored")
    assert segments[0].text == "hi"
```

**Validation.** `python -m pytest tests/test_audio_wrapper.py -q`

**Done when.** The two new tests pass without importing `sounddevice` beyond the guard.

---

## A1 — Fix `get_audio_tone` amplitude scaling

**Goal.** A generated tone must actually contain audio, not silence.

**Current behaviour (`get_audio_tone`, line 32).**

```python
audio = 0.5 * np.sin(2 * np.pi * frequency * t)   # values in [-0.5, 0.5]
return audio.astype('int16')                       # every sample truncates to 0
```

Verified: the current return value has `max() == 0`.

**Expected behaviour.** Return a peak-scaled `int16` array, with an optional amplitude:

```python
def get_audio_tone(
    duration: float = 2.0,
    samplerate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate an int16 sine tone. Peak sample is amplitude * 32767."""
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return (wave * np.iinfo(np.int16).max).astype(np.int16)
```

**Files.** `src/audio_wrapper.py`; `tests/test_audio_wrapper.py`.

**Tests.**

```python
from src.audio_wrapper import get_audio_tone


def test_tone_is_not_silent():
    tone = get_audio_tone(duration=0.1, samplerate=16000, frequency=440.0)
    assert tone.dtype == np.int16
    assert int(np.abs(tone).max()) > 1000


def test_tone_length_matches_samplerate():
    tone = get_audio_tone(duration=0.25, samplerate=8000)
    assert len(tone) == 2000


def test_tone_zero_amplitude_is_silent():
    assert int(np.abs(get_audio_tone(0.05, amplitude=0.0)).max()) == 0
```

**Validation.** `python -m pytest tests/test_audio_wrapper.py -q`

**Done when.** `test_tone_is_not_silent` fails on the old code and passes on the new code.

---

## A2 — Fix `audio_to_wav` (close handle, reject empty)

**Goal.** Writing a WAV must not leak a file handle or crash on empty input.

**Current behaviour (`audio_to_wav`, line 52).** `open(...)` is never closed and
`sf.write` raises when `audio` is empty. The default `out_file` is a `str` but annotated
`Path`.

**Expected behaviour.**

```python
def audio_to_wav(
    audio: np.ndarray,
    out_file: str | Path = "input_audio.wav",
    samplerate: int = 16000,
) -> Path:
    """Write audio to a WAV file and return its resolved Path.

    Raises ValueError when there are no samples to write.
    """
    if np.asarray(audio).size == 0:
        raise ValueError("Cannot write WAV: no audio samples")
    path = Path(out_file)
    sf.write(str(path), audio, samplerate)
    return path
```

**Files.** `src/audio_wrapper.py`; `tests/test_audio_wrapper.py`.

**Tests.**

```python
import soundfile as sf
import pytest
from src.audio_wrapper import audio_to_wav


def test_audio_to_wav_roundtrip(tmp_path):
    audio = get_audio_tone(duration=0.05, samplerate=8000)
    out = audio_to_wav(audio, tmp_path / "t.wav", samplerate=8000)
    assert out == tmp_path / "t.wav" and out.exists()
    data, sr = sf.read(str(out))
    assert sr == 8000 and len(data) == len(audio)


def test_audio_to_wav_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        audio_to_wav(np.array([], dtype=np.int16), tmp_path / "e.wav")
```

**Validation.** `python -m pytest tests/test_audio_wrapper.py -q`

**Done when.** Both tests pass and the returned type is `Path` for `str` and `Path` inputs.

---

## A3 — Add `rms()` helper and fix `compute_silence_threshold`

**Goal.** One tested RMS helper, and a threshold function with a correct signature and no
off-by-one.

**Current behaviour (`compute_silence_threshold`, line 65).**

- Annotated `-> float` but returns `(threshold, mean_rms, std_rms)`.
- `while len(chunk_rms_values) <= chunks:` collects `chunks + 1` samples.
- Hardcodes `channels=1, dtype='int16'` and `samplerate=16000`, ignoring
  `device.default_samplerate`.
- RMS is computed inline in three places.

**Expected behaviour.**

```python
def rms(samples: np.ndarray) -> float:
    """Root-mean-square level of an int16/float array. 0.0 for empty input."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def compute_silence_threshold(
    device,
    samplerate: int | None = None,
    chunk_time_s: float = 0.1,
    chunks: int = 20,
    guard_factor: float = 1.0,
    stream_factory=sd.InputStream,
) -> tuple[float, float, float]:
    """Return (threshold, mean_rms, std_rms) sampled from ambient noise."""
    if samplerate is None:
        samplerate = device.default_samplerate
    chunk_size = int(chunk_time_s * samplerate)
    values: list[float] = []
    with stream_factory(
        samplerate=samplerate, channels=1, dtype="int16", device=device.index
    ) as stream:
        for _ in range(chunks):
            chunk, _ = stream.read(chunk_size)
            values.append(rms(chunk))
    mean_rms = float(np.mean(values))
    std_rms = float(np.std(values))
    return guard_factor * (mean_rms + 2 * std_rms), mean_rms, std_rms
```

`SpeechDetector.listen` and `SpeechDetector.__init__` must use `rms(...)` instead of the
inline expression.

**Files.** `src/audio_wrapper.py`; `tests/test_audio_wrapper.py`.

**Tests.**

```python
from src.audio_wrapper import rms, compute_silence_threshold
from tests.conftest import FakeDevice, FakeStream


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
```

**Validation.** `python -m pytest tests/test_audio_wrapper.py -q`

**Done when.** The return annotation is `tuple[float, float, float]`, exactly `chunks`
reads happen, and all tests pass.

---

## A4 — Replace `sys.exit` with `AudioDeviceError`

**Goal.** A library must never terminate the host process.

**Current behaviour (`InputAudioDevice.__init__`, line 17).** On a failed
`sd.query_devices`, it logs and calls `sys.exit(1)`.

**Expected behaviour.**

```python
class AudioDeviceError(RuntimeError):
    """Raised when no usable audio input device is available."""


class InputAudioDevice:
    def __init__(self):
        self.logger = getLogger(self.__class__.__name__)
        try:
            devices = sd.query_devices(kind="input")
        except Exception as exc:
            self.logger.critical("%s", exc)
            raise AudioDeviceError(
                "No audio input device available. Check microphone settings."
            ) from exc
        ...
        if len(devices) == 0:
            raise AudioDeviceError("No audio input devices found")
        ...
```

`main.py` may catch `AudioDeviceError` and print a friendly message (that wiring is A8);
`src/` code must not call `sys.exit`.

**Files.** `src/audio_wrapper.py`; `tests/test_audio_wrapper.py`.

**Tests.** Do not open a device; monkeypatch `sounddevice.query_devices`.

```python
import src.audio_wrapper as aw


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
```

**Validation.** `python -m pytest tests/test_audio_wrapper.py -q`

**Done when.** The three tests pass and `sys.exit`/`sys` handling is removed from the
module if no longer needed.

---

## A5 — Extract and fix the capture loop (`capture_speech`)

**Goal.** A deterministic, bounded, correctly-terminating speech-capture loop that can be
unit-tested with `FakeStream` and a fake clock.

**Current behaviour (`listen`, line 108).** The silence-timeout check
`elif np.mean(chunk_rms_values[-timeout_chunks:]) < self.silence_threshold:` lives *inside*
`if audio_rms > self.silence_threshold`, where `audio_rms` is the mean over **all** chunks.
Once speech raises that average it stays high, so the elapsed-time stop is delayed or never
fires. There is also no upper bound on recording length. `listen()` mixes device I/O,
streaming, level printing and trimming, so it cannot be tested.

**Expected behaviour.** Add a module-level pure-ish function and make `listen` a thin
wrapper:

```python
def capture_speech(
    threshold: float,
    device_index: int,
    stream_factory=sd.InputStream,
    time_func=time.time,
    on_level=None,                 # Callable[[float], None] | None
    silence_timeout_s: float = 1.5,
    channels: int = 1,
    samplerate: int = 16000,
    chunk_time_s: float = 0.1,
    datatype: str = "int16",
    clear_timeout_s: float = 30.0,
    max_record_chunks: int | None = None,
) -> np.ndarray:
    """Record until silence_timeout_s of consecutive sub-threshold chunks.

    Returns the trimmed speech samples, or an empty int16 array if no speech
    was detected. max_record_chunks bounds recording for safety/testing.
    """
    chunk_size = int(chunk_time_s * samplerate)
    timeout_chunks = max(1, int(silence_timeout_s / chunk_time_s))
    chunks: list[np.ndarray] = []
    speech_started = False
    silent_chunks = 0
    start_chunk = 0
    start_time = time_func()
    read_count = 0

    with stream_factory(
        samplerate=samplerate, channels=channels, dtype=datatype,
        device=device_index,
    ) as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            read_count += 1
            chunks.append(chunk)
            level = rms(chunk)
            if on_level is not None:
                on_level(level)

            if level > threshold:
                if not speech_started:
                    speech_started = True
                    start_chunk = len(chunks) - 1
                silent_chunks = 0
            elif speech_started:
                silent_chunks += 1
                if silent_chunks >= timeout_chunks:
                    break

            if not speech_started and (time_func() - start_time) > clear_timeout_s:
                chunks.clear()
                silent_chunks = 0
                start_time = time_func()

            if max_record_chunks is not None and read_count >= max_record_chunks:
                break

    empty = np.empty(0, dtype=datatype)
    if not speech_started or not chunks:
        return empty
    tolerance = timeout_chunks // 2
    begin = start_chunk - tolerance if start_chunk > tolerance else 0
    end = len(chunks) - tolerance if len(chunks) > tolerance else len(chunks)
    audio = np.concatenate(chunks[begin:end], axis=0)
    if rms(audio) < threshold:
        return empty
    return audio
```

`SpeechDetector.listen` keeps its signature and delegates:

```python
def listen(self, silence_timeout_s=1.5, channels=1, samplerate=16000,
           chunk_time_s=0.1, datatype="int16", clear_timeout_s=30.0) -> np.ndarray:
    def on_level(level: float) -> None:
        line = format_level(level, self.silence_threshold, self.show_levels)
        if line:
            cprint(line, "reset", end="", flush=True)
    return capture_speech(
        threshold=self.silence_threshold,
        device_index=self.device.index,
        stream_factory=self._stream_factory,     # added in A0; see A7
        time_func=self._time_func,
        on_level=on_level,
        silence_timeout_s=silence_timeout_s,
        channels=channels,
        samplerate=samplerate,
        chunk_time_s=chunk_time_s,
        datatype=datatype,
        clear_timeout_s=clear_timeout_s,
    )
```

If `_stream_factory`/`_time_func` are introduced in A7 instead, use the module defaults
here and adjust in A7; either way `listen` must not access `sd.InputStream` directly.

**Files.** `src/audio_wrapper.py`; `tests/test_audio_wrapper.py`.

**Tests.** Use a fake clock so no real time passes.

```python
from src.audio_wrapper import capture_speech


class _Clock:
    def __init__(self, step=0.0):
        self.t = 0.0
        self.step = step
    def __call__(self):
        self.t += self.step
        return self.t


def _silent(n):  return np.zeros(n, dtype=np.int16)
def _loud(n):    return np.full(n, 5000, dtype=np.int16)


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
```

**Validation.** `python -m pytest tests/test_audio_wrapper.py -q`

**Done when.** A stream of `loud, silent, silent` stops after the second silent chunk
(`timeout_chunks == 2`) and the timeout branch fires **without** a pre-existing loud
average; all tests pass.

---

## A6 — In-memory WAV and correct samplerate in `listen_and_transcribe`

**Goal.** No temp file on disk, no silent samplerate mismatch.

**Current behaviour (`listen_and_transcribe`, line 181).**

```python
audio = self.listen(silence_timeout_s=silence_timeout_s)
text = self.transcribe(audio_to_wav(audio))     # always 16000 Hz, always "input_audio.wav"
```

`listen` can record at any samplerate, but `audio_to_wav` defaults to 16000, so a custom
samplerate yields a pitch/tempo shift. The file is left behind and every call overwrites
the same path.

**Expected behaviour.**

```python
import io

def audio_to_wav_buffer(audio: np.ndarray, samplerate: int = 16000) -> io.BytesIO:
    """Return a seekable in-memory WAV. Raises ValueError for empty audio."""
    if np.asarray(audio).size == 0:
        raise ValueError("Cannot encode WAV: no audio samples")
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate, format="WAV")
    buf.seek(0)
    return buf
```

- `transcribe(self, audio)` accepts `str | Path | io.BufferedIOBase` and passes it to
  `self.model.transcribe(audio=audio, ...)` unchanged (no `str()` coercion for file-like
  objects). It calls `self._ensure_model()` (A7).
- `listen_and_transcribe(self, silence_timeout_s, samplerate=16000)` calls
  `self.listen(silence_timeout_s=silence_timeout_s, samplerate=samplerate)`, returns `""`
  for empty audio, otherwise transcribes `audio_to_wav_buffer(audio, samplerate)`.

**Files.** `src/audio_wrapper.py`; `tests/test_audio_wrapper.py`.

**Tests.**

```python
import io
import soundfile as sf
from src.audio_wrapper import audio_to_wav_buffer, SpeechDetector
from tests.conftest import FakeWhisperModel


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
    text = SpeechDetector.transcribe(fake_self, buf)
    assert text == "hello" and model.calls == 1


def test_listen_and_transcribe_empty_audio_skips_model():
    calls = {"transcribe": 0}
    fake_self = SimpleNamespace(
        listen=lambda **kw: np.array([], dtype=np.int16),
        transcribe=lambda audio: calls.__setitem__("transcribe", calls["transcribe"] + 1),
    )
    assert SpeechDetector.listen_and_transcribe(fake_self, 1.0) == ""
    assert calls["transcribe"] == 0
```

**Validation.** `python -m pytest tests/test_audio_wrapper.py -q`

**Done when.** No task path writes a WAV file to disk and `audio_to_wav_buffer` reports the
samplerate that was passed.

---

## A7 — Lazy, configurable Whisper model loading

**Goal.** Do not load the model during construction; make device/compute type injectable
and configurable; make `SpeechDetector` constructible in tests without hardware.

**Current behaviour (`SpeechDetector.__init__`, line 104).**

```python
self.device = InputAudioDevice()
self.model = WhisperModel(str(model_path), device="auto", compute_type="int8")
```

The model loads eagerly (slow startup), device/compute type are hardcoded, and the
constructor cannot be exercised without a microphone and model weights.

**Expected behaviour.**

```python
class SpeechDetector:
    def __init__(
        self,
        model_path,
        language: str = "en",
        beam_size: int = 5,
        show_levels: bool = False,
        device=None,                       # injectable InputAudioDevice
        stream_factory=sd.InputStream,     # injectable for tests
        time_func=time.time,               # injectable for tests
        model=None,                        # injectable for tests
        model_backend: str = "auto",       # -> WhisperModel(device=...)
        compute_type: str = "int8",
        chunk_time_s: float = 0.1,
        calibration_chunks: int = 20,
        guard_factor: float = 1.0,
    ):
        self.show_levels = show_levels
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")
        self.logger = getLogger(self.__class__.__name__)
        self.device = device or InputAudioDevice()
        self._stream_factory = stream_factory
        self._time_func = time_func
        self._model = model
        self._model_backend = model_backend
        self._compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.silence_threshold, mean_rms, std_rms = compute_silence_threshold(
            self.device, chunk_time_s=chunk_time_s, chunks=calibration_chunks,
            guard_factor=guard_factor, stream_factory=stream_factory,
        )

    def _ensure_model(self):
        if self._model is None:
            self.logger.info("Loading Whisper model from %s ...", self._model_path)
            self._model = WhisperModel(
                str(self._model_path), device=self._model_backend,
                compute_type=self._compute_type,
            )
        return self._model
```

`transcribe` uses `model = self._ensure_model()`.

**Files.** `src/audio_wrapper.py`; `tests/test_audio_wrapper.py`.

**Tests.**

```python
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
    assert calls == []                       # not loaded at construction
    detector._ensure_model()
    detector._ensure_model()
    assert len(calls) == 1                   # loaded exactly once


def test_model_backend_and_compute_type_passed_through(monkeypatch, tmp_path):
    model_dir = tmp_path / "m"; model_dir.mkdir()
    seen = {}
    def fake_backend(path, device=None, compute_type=None):
        seen["device"] = device; seen["compute_type"] = compute_type
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
```

**Validation.** `python -m pytest tests/test_audio_wrapper.py -q`

**Done when.** Constructing `SpeechDetector` with injected seams performs zero
`WhisperModel` calls; `_ensure_model()` is idempotent; all tests pass.

---

## A8 — Wire `[speech_det]` config into `SpeechDetector`

**Goal.** expose the new knobs, and fail friendly when there is no microphone.

**Current behaviour.** `config.toml`'s `[speech_det]` has only `model_path`,
`silence_timeout_s`, `language`. `main.py` constructs `SpeechDetector(model_path=...,
language=..., show_levels=True)` and lets `AudioDeviceError` (if raised) crash with a
traceback.

**Expected behaviour.**

1. `config.toml` `[speech_det]` gains optional keys with these defaults:

   ```toml
   beam_size = 5
   show_levels = true
   device = "auto"
   compute_type = "int8"
   calibration_chunks = 20
   guard_factor = 1.0
   ```

   (Keys that are not `*_file`/`*_dir`/`*_path` remain plain values, so `load_config`
   leaves them untouched.)

2. `main.py` reads them with `getattr(..., default)` so a minimal config still works:

   ```python
   detector = SpeechDetector(
       model_path=config.speech_det.model_path,
       language=config.speech_det.language,
       beam_size=getattr(config.speech_det, "beam_size", 5),
       show_levels=getattr(config.speech_det, "show_levels", True),
       model_backend=getattr(config.speech_det, "device", "auto"),
       compute_type=getattr(config.speech_det, "compute_type", "int8"),
       calibration_chunks=getattr(config.speech_det, "calibration_chunks", 20),
       guard_factor=getattr(config.speech_det, "guard_factor", 1.0),
   )
   ```

3. `main.py` catches `AudioDeviceError` and prints one friendly line via `cprint`, then
   returns exit code `1` (`from src.audio_wrapper import AudioDeviceError` lazily inside the
   voice branch).

**Files.** `config.toml`, `main.py`, `tests/test_main.py`.

**Tests.** Extend `tests/test_main.py` using the existing stub pattern.

```python
def test_voice_passes_speech_det_config(monkeypatch):
    import types
    import src.audio_wrapper as aw

    seen = {}

    class RecorderDetector:
        def __init__(self, **kw):
            seen.update(kw)
        def listen_and_transcribe(self, silence_timeout_s):
            return "/quit"

    class StubSession:
        COMMANDS = ()
        def __init__(self, agent, read_prompt, use_tools=True):
            pass
        def run(self):
            return 0

    cfg = types.SimpleNamespace(
        brain=types.SimpleNamespace(model="m"),
        agent=types.SimpleNamespace(bootstrap_prompt=""),
        speech_det=types.SimpleNamespace(
            model_path="m", language="en", silence_timeout_s=2.0,
            beam_size=7, show_levels=False, device="cpu",
            compute_type="int8", calibration_chunks=3, guard_factor=2.0,
        ),
    )
    monkeypatch.setattr(aw, "SpeechDetector", RecorderDetector)
    monkeypatch.setattr(session_mod, "ConversationSession", StubSession)
    monkeypatch.setattr(live_ai_mod, "Agent", lambda config: object())
    monkeypatch.setattr(main_mod, "load_config", lambda p: cfg)

    assert main_mod.main([]) == 0
    assert seen["beam_size"] == 7 and seen["show_levels"] is False
    assert seen["model_backend"] == "cpu" and seen["calibration_chunks"] == 3
    assert seen["guard_factor"] == 2.0


def test_voice_missing_device_returns_error(monkeypatch, capsys):
    import types
    import src.audio_wrapper as aw

    class NoDeviceDetector:
        def __init__(self, **kw):
            raise aw.AudioDeviceError("no mic")

    class StubSession:
        COMMANDS = ()
        def __init__(self, *a, **k): pass
        def run(self): return 0

    cfg = types.SimpleNamespace(
        brain=types.SimpleNamespace(model="m"),
        agent=types.SimpleNamespace(bootstrap_prompt=""),
        speech_det=types.SimpleNamespace(
            model_path="m", language="en", silence_timeout_s=2.0,
        ),
    )
    monkeypatch.setattr(aw, "SpeechDetector", NoDeviceDetector)
    monkeypatch.setattr(session_mod, "ConversationSession", StubSession)
    monkeypatch.setattr(live_ai_mod, "Agent", lambda config: object())
    monkeypatch.setattr(main_mod, "load_config", lambda p: cfg)

    assert main_mod.main([]) == 1
    assert "no mic" in capsys.readouterr().out.lower() or "microphone" in capsys.readouterr().out.lower()
```

**Validation.** `python -m pytest tests/test_main.py -q`

**Done when.** Both tests pass and `config.toml` documents the new keys.

---

## A9 — Typing/cleanup and end-to-end acceptance

**Goal.** Remove the last rough edges and prove the whole capture→transcribe path works
end-to-end with fakes.

**Current behaviour.** `from src.common import cprint` is re-imported inside the capture
loop; `transcribe` is annotated `wav_path: Path`; docstrings are missing on several public
functions; `audio_to_wav`'s default arg type was inconsistent with its annotation.

**Expected behaviour.**

- Hoist `from src.common import cprint` to module scope (or import it once in
  `SpeechDetector.__init__`) — never inside the per-chunk loop.
- `transcribe(self, audio: str | Path | io.BufferedIOBase) -> str` with a docstring.
- Add concise docstrings to `rms`, `capture_speech`, `compute_silence_threshold`,
  `audio_to_wav`, `audio_to_wav_buffer`, `format_level`.
- Remove any now-unused `sys` import after A4.
- Verify with `ruff`/`python -m pyflakes` if available; otherwise a manual pass.

**Files.** `src/audio_wrapper.py`; `tests/test_audio_wrapper.py`.

**Tests.** End-to-end with fakes (no hardware, no weights):

```python
def test_end_to_end_capture_then_transcribe(monkeypatch, tmp_path):
    model_dir = tmp_path / "m"; model_dir.mkdir()

    detector = aw.SpeechDetector(
        model_path=model_dir,
        device=FakeDevice(),
        stream_factory=lambda **kw: FakeStream([_loud(8), _silent(8), _silent(8)]),
        time_func=_Clock(),
        model=FakeWhisperModel("end to end"),
        calibration_chunks=2,
    )
    detector.silence_threshold = 100.0     # bypass ambient calibration
    text = detector.listen_and_transcribe(silence_timeout_s=0.2, samplerate=8000)
    assert text == "end to end"


def test_no_cprint_import_inside_loop():
    import pathlib
    src = pathlib.Path(aw.__file__).read_text(encoding="utf-8")
    loop_start = src.index("def capture_speech")
    loop_end = src.index("class SpeechDetector", loop_start)
    assert "import cprint" not in src[loop_start:loop_end]
```

**Validation.** `python -m pytest tests/ -q` (full suite) and, if available,
`python -m ruff check src/audio_wrapper.py main.py`.

**Done when.** Full suite green and the end-to-end test exercises
`listen`→`audio_to_wav_buffer`→`transcribe` without touching disk or hardware.

---

## 2. Global acceptance criteria

Run from the repo root:

```
python -m pytest tests/ -q
```

Must pass. Additionally verify:

1. `python -c "import src.audio_wrapper"` makes no sound and opens no device.
2. `get_audio_tone(0.1).max() != 0`.
3. `audio_to_wav(empty)` raises `ValueError`; a round-tripped WAV keeps its samplerate.
4. `compute_silence_threshold` performs exactly `chunks` reads and returns a 3-tuple.
5. `InputAudioDevice()` raises `AudioDeviceError`, never `SystemExit`; `sys.exit` does not
   appear in `src/audio_wrapper.py`.
6. `capture_speech` with `[loud, silent, silent]` and `silence_timeout_s=2*chunk_time_s`
   returns non-empty audio and stops immediately after the second silent chunk.
7. `listen_and_transcribe` never writes `input_audio.wav` to the working directory.
8. Constructing `SpeechDetector` with an injected model performs zero `WhisperModel`
   loads; `_ensure_model()` loads exactly once.
9. `python main.py` still starts the voice loop; a missing microphone exits with code `1`
   and one friendly line, not a traceback.
10. Every task above has at least one test that fails before its change and passes after.

---

## 3. Out of scope (do not implement under this plan)

- Voice-activity detection beyond RMS thresholding (no WebRTC VAD, no Silero).
- Streaming/partial transcription, word timestamps, diarization.
- Changing the Whisper model family or downloading weights.
- Changes to `ConversationSession`, `Brain`/`Agent`, tools, or the `[input]` block.
- Audio playback (see `src/audio_generator.py`).
- Microphone hot-plug / re-selection at runtime.
