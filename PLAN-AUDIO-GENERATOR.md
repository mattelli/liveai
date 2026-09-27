# PLAN-AUDIO-GENERATOR.md — Improve `src/audio_generator.py`

Addendum to [`PLAN.md`](PLAN.md). Everything in `PLAN.md` §0 (rules for the executing
agent), §2 (test strategy and conventions) and §3 (design constraints) still applies.
Read those sections before starting any task here.

**Scope.** Correctness, robustness and testability of `src/audio_generator.py`
(`SpeechGenerator`) plus the `[speech_gen]` wiring in `src/live_ai.py` and `config.toml`.
No change to `Brain`/`Agent` beyond `allow_speech`, `ConversationSession`, audio capture, or
tools.

**Target files.** `src/audio_generator.py`, `src/live_ai.py` (`allow_speech` only),
`config.toml`, `tests/conftest.py`, new `tests/test_audio_generator.py`.

**Hardware rule (PLAN.md §0.3).** No test may load a real Piper voice, open a speaker, or
play audio. `PiperVoice`, the synthesis config and the output stream are all injectable;
tests use the fakes added in G0.

**Baseline (current behaviour, for reference).** Line numbers are from the current file and
may shift as tasks land.

| # | Location | Issue |
|---|----------|-------|
| C1 | `clean_for_speech` (line 51) | The regex removes `[ ]` but not `( )`, despite the comment on line 50 |
| C2 | `speak` (line 61) | Records nothing when the cleaned text is empty; if `synthesize` raises, `stream.stop()` is never called and the output stream is left running |
| C3 | class (line 8) | `__exit__` exists but there is no `__enter__`, so `with SpeechGenerator(...)` fails and the stream is never closed except by GC |
| C4 | `__init__` (line 23) | `PiperVoice.load` runs eagerly; construction cannot be tested without model weights |
| C5 | `__init__` (lines 24-30) | `1.0 / speech_speed` raises `ZeroDivisionError` for `speech_speed == 0`; `volume` is not validated; noise scales and `speaker_id` are hardcoded |
| C6 | `__init__` (line 31) | Output `samplerate` (22050) is never checked against the voice's real `config.sample_rate`, so a mismatch silently plays at the wrong pitch |
| C7 | `to_wav` (line 42) | Does not create the parent directory, does not clean formatting, returns `None`, and writes a zero-frame file for empty text |

---

## 0. Conventions for the executing agent

1. One task per commit, in the order of §1. Each task is self-contained.
2. Write the test **first**, watch it fail, then implement the change. A task is done only
   when its validation command passes.
3. Keep the public signatures exactly as specified in §2. Downstream callers
   (`src/live_ai.py:148`, `playground/simple_speech_gen.py`,
   `playground/simple_speech_runner.py`) must keep working; if a signature changes, update
   all callers in the same task.
4. Never call `sys.exit` inside `src/`. Raise an exception instead.
5. Import `piper`/`sounddevice` lazily or guard tests with `pytest.importorskip`; never let
   a missing audio stack break test collection.
6. `python -m pytest tests/ -q` must stay green at the end of every task.

---

## 1. Task index

| ID | Title | Depends on | Phase |
|----|-------|-----------|-------|
| G0 | Test harness for audio generation: fakes + import guard | — | Foundations |
| G1 | Fix `clean_for_speech` parenthesis removal | G0 | Correctness |
| G2 | Lifecycle: `close()`, `__enter__`, `__exit__` | G0 | Correctness |
| G3 | Lazy, injectable voice loading | G0 | Robustness |
| G4 | Robust `speak()`: empty guard, always stop, bytes return | G3 | Correctness |
| G5 | Validate and parameterize `SynthesisConfig` | G0 | Robustness |
| G6 | Validate output samplerate against the voice | G3 | Correctness |
| G7 | Harden `to_wav` (parent dir, return Path, clean, empty guard) | G3 | Correctness |
| G8 | Wire `[speech_gen]` config and `live_ai.allow_speech` cleanup | G3, G2 | Integration |
| G9 | Typing/docstrings + end-to-end acceptance | G1–G8 | Verification |

G1–G2 and G5–G7 fix real bugs; G3–G4 make the module testable and safe.

---

## 2. `SpeechGenerator` contract to build (used by all tasks)

```python
class SpeechGenerator:
    def __init__(
        self,
        model_path: str | Path,
        volume: float = 0.5,
        speech_speed: float = 1.0,
        samplerate: int = 22050,
        channels: int = 1,
        datatype: str = "int16",
        speaker_id: int | None = None,
        noise_scale: float = 1.0,
        noise_w_scale: float = 1.0,
        normalize_audio: bool = False,
        voice=None,                              # test seam / preloaded voice
        stream=None,                             # test seam / prebuilt output stream
        stream_factory=sd.RawOutputStream,       # used when stream is None
    ):
        """Validate params and model path, build syn_config and the output stream.

        The Piper voice is loaded lazily on first use (G3). Raises ValueError for
        an invalid speech_speed/volume and FileNotFoundError for a missing model.
        """

    def _ensure_voice(self): ...        # loads PiperVoice once (G3); validates rate (G6)
    def clean_for_speech(text: str) -> str          # static (G1)
    def speak(self, text: str, strip_formatting: bool = True) -> int: ...  # bytes (G4)
    def to_wav(self, text: str, file_path: str | Path,
               strip_formatting: bool = True) -> Path: ...   # (G7)
    def close(self) -> None: ...        # idempotent (G2)
    def __enter__(self): ...            # returns self (G2)
    def __exit__(self, *exc) -> bool: ...  # calls close(); returns False (G2)
    voice                                              # may be None until first use
    stream                                             # output stream
    syn_config                                         # SynthesisConfig
```

---

## G0 — Test harness for audio generation: fakes and import guard

**Goal.** Deterministic fakes so no task below needs Piper weights or a speaker.

**Current behaviour.** There is no test file for `audio_generator.py`.

**Expected behaviour.** Add to `tests/conftest.py`:

```python
class FakeAudioChunk:
    """Stands in for piper.voice.AudioChunk."""
    def __init__(self, data: bytes = b"\x00\x00"):
        self.audio_int16_bytes = data


class FakePiperVoice:
    """Stand-in for PiperVoice; records calls, never loads a model."""
    def __init__(self, chunks=(b"\x01\x02", b"\x01\x02"),
                 sample_rate=22050, fail_after=None, wav_bytes=b"\x01\x02"):
        self._chunks = list(chunks)
        self.sample_rate = sample_rate
        self.config = SimpleNamespace(sample_rate=sample_rate)
        self._fail_after = fail_after
        self._wav_bytes = wav_bytes
        self.synth_calls = 0
        self.wav_calls = 0
        self.last_text = None

    def synthesize(self, text, syn_config=None):
        self.synth_calls += 1
        self.last_text = text
        for i, data in enumerate(self._chunks):
            if self._fail_after == i:
                raise RuntimeError("synth failed")
            yield FakeAudioChunk(data)

    def synthesize_wav(self, text, wav_file, syn_config=None):
        self.wav_calls += 1
        self.last_text = text
        wav_file.setparams((1, 2, self.sample_rate, 0, "NONE", "not compressed"))
        wav_file.writeframes(self._wav_bytes * 2)


class FakeOutputStream:
    """Stand-in for sounddevice.RawOutputStream."""
    def __init__(self, **kw):
        self.kw = kw
        self.started = 0
        self.stopped = 0
        self.closed = 0
        self.written = []

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    def write(self, data):
        self.written.append(data)

    def close(self):
        self.closed += 1
```

New `tests/test_audio_generator.py` starts with import guards and a shared helper:

```python
import wave
from logging import getLogger
from types import SimpleNamespace

import pytest

pytest.importorskip("piper", reason="audio_generator requires piper to import")
pytest.importorskip("sounddevice", reason="audio_generator requires sounddevice to import")

import src.audio_generator as ag  # noqa: E402
from tests.conftest import FakeAudioChunk, FakeOutputStream, FakePiperVoice  # noqa: E402


def make_model(tmp_path):
    """Create a placeholder .onnx path so model_path.exists() is True."""
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"not-a-real-model")
    return model


def make_generator(tmp_path, **kw):
    kw.setdefault("voice", FakePiperVoice())
    kw.setdefault("stream", FakeOutputStream())
    return ag.SpeechGenerator(model_path=make_model(tmp_path), **kw)
```

**Files.** `tests/conftest.py`, new `tests/test_audio_generator.py`.

**Tests.**

```python
def test_fakes_are_usable(tmp_path):
    voice = FakePiperVoice()
    chunks = list(voice.synthesize("hi"))
    assert chunks[0].audio_int16_bytes == b"\x01\x02"
    stream = FakeOutputStream(a=1)
    stream.start(); stream.stop(); stream.close()
    assert (stream.started, stream.stopped, stream.closed) == (1, 1, 1)


def test_make_generator_constructs_without_hardware(tmp_path):
    gen = make_generator(tmp_path)
    assert gen.voice is not None and isinstance(gen.stream, FakeOutputStream)
```

**Validation.** `python -m pytest tests/test_audio_generator.py -q`

**Done when.** Both tests pass and no test touches the real audio stack or Piper model.

---

## G1 — Fix `clean_for_speech` parenthesis removal

**Goal.** `clean_for_speech` removes exactly the characters its comment promises.

**Current behaviour (`clean_for_speech`, line 47).**

```python
# Remove markdown formatting characters: **, __, *, _, #, `, >, ~, [ ] ( )
text = re.sub(r"[*_#`>~\[\]]", "", text)   # '(' and ')' are missing
```

**Expected behaviour.** Add `()` to the character class and keep the rest unchanged:

```python
text = re.sub(r"[*_#`>~\[\]()]", "", text)
```

The emoji/whitespace passes stay as they are. Note that removing `( )` also removes
parentheses in ordinary prose — this is intentional per the comment (they are read aloud as
noise by the TTS).

**Files.** `src/audio_generator.py`; `tests/test_audio_generator.py`.

**Tests.**

```python
def test_clean_removes_markdown_and_parens():
    cleaned = ag.SpeechGenerator.clean_for_speech("**b** _i_ (p) [x] #t `c` > q ~s~")
    assert cleaned == "b i p x t c q s"


def test_clean_removes_emoji():
    assert ag.SpeechGenerator.clean_for_speech("hi 👋 there") == "hi there"


def test_clean_collapses_whitespace():
    assert ag.SpeechGenerator.clean_for_speech("a\n\n  b\tc") == "a b c"


def test_clean_empty_returns_empty():
    assert ag.SpeechGenerator.clean_for_speech("") == ""
```

**Validation.** `python -m pytest tests/test_audio_generator.py -q`

**Done when.** `test_clean_removes_markdown_and_parens` fails on the old regex and passes on
the new one.

---

## G2 — Lifecycle: `close()`, `__enter__`, `__exit__`

**Goal.** The output stream can be released deterministically and the object works as a
context manager.

**Current behaviour (lines 37-40).** Only `__exit__` exists; without `__enter__`,
`with SpeechGenerator(...) as sg:` raises `TypeError`. Calling `__exit__` twice calls
`stream.close()` twice. `live_ai.allow_speech(False)` releases the generator by `del`
(line 155), relying on garbage collection to close the stream.

**Expected behaviour.**

```python
def close(self) -> None:
    """Stop and close the output stream. Safe to call more than once."""
    if self._closed:
        return
    self._closed = True
    if self.stream is not None:
        self.logger.info("🧹 Closing audio output stream...")
        try:
            self.stream.stop()
        except Exception:                      # already stopped / no device
            pass
        self.stream.close()
        self.stream = None


def __enter__(self):
    return self


def __exit__(self, exc_type, exc_value, traceback):
    self.close()
    return False
```

`__init__` sets `self._closed = False` before the stream is assigned.

**Files.** `src/audio_generator.py`; `tests/test_audio_generator.py`.

**Tests.**

```python
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
```

**Validation.** `python -m pytest tests/test_audio_generator.py -q`

**Done when.** Both tests pass and a second `close()` performs no further stream calls.

---

## G3 — Lazy, injectable voice loading

**Goal.** Construction validates inputs and prepares the stream, but does not load the
Piper model until speech is actually requested.

**Current behaviour (`__init__`, line 23).** `self.voice = PiperVoice.load(model_path)` runs
during construction, so an agent that enables speech but never speaks still pays the model
load, and the class cannot be constructed in a test without weights.

**Expected behaviour.**

- `__init__` accepts `voice=None`. When a voice is supplied, it is used as-is
  (`self.voice = voice`); otherwise `self.voice = None`.
- Store `self.model_path = Path(model_path)`.
- Add:

```python
def _ensure_voice(self):
    """Load the Piper voice once and return it."""
    if self.voice is None:
        self.logger.info("🚀 Loading Audio Generator model from %s ...", self.model_path)
        self.voice = PiperVoice.load(self.model_path)
    return self.voice
```

- `speak` and `to_wav` call `self._ensure_voice()` before synthesizing (implemented in G4
  and G7; for this task add the call to `to_wav` only if it does not conflict).

**Files.** `src/audio_generator.py`; `tests/test_audio_generator.py`.

**Tests.**

```python
def test_voice_is_lazy_and_loaded_once(monkeypatch, tmp_path):
    calls = []

    def fake_load(path):
        calls.append(path)
        return FakePiperVoice(sample_rate=22050)

    monkeypatch.setattr(ag, "PiperVoice", SimpleNamespace(load=fake_load))
    gen = ag.SpeechGenerator(model_path=make_model(tmp_path), stream=FakeOutputStream())
    assert calls == []                    # not loaded at construction
    gen._ensure_voice()
    gen._ensure_voice()
    assert len(calls) == 1


def test_injected_voice_is_not_reloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(ag, "PiperVoice", SimpleNamespace(load=lambda p: pytest.fail("load")))
    voice = FakePiperVoice()
    gen = make_generator(tmp_path, voice=voice)
    assert gen._ensure_voice() is voice
```

**Validation.** `python -m pytest tests/test_audio_generator.py -q`

**Done when.** Construction performs zero `PiperVoice.load` calls and `_ensure_voice()` is
idempotent.

---

## G4 — Robust `speak()`: empty guard, always stop, bytes return

**Goal.** `speak()` does not touch the stream for empty text, always stops the stream even
on failure, and reports how many bytes it wrote.

**Current behaviour (`speak`, line 61).**

```python
self.stream.start()
for chunk in self.voice.synthesize(text, syn_config=self.syn_config):
    self.stream.write(chunk.audio_int16_bytes)
self.stream.stop()
```

If `text` is empty after cleaning, the stream is started/stopped for nothing. If
`synthesize` raises (bad text, ONNX error), `stream.stop()` is skipped and the stream stays
active. There is no return value.

**Expected behaviour.**

```python
def speak(self, text: str, strip_formatting: bool = True) -> int:
    """Synthesize text and stream it to the output device.

    Returns the number of audio bytes written. Empty (or fully stripped) text
    is a no-op: the stream is neither started nor stopped.
    """
    if strip_formatting:
        self.logger.debug("Removing formatted syntax...")
        text = self.clean_for_speech(text)
    if not text:
        self.logger.debug("Nothing to speak after cleaning; skipping.")
        return 0

    voice = self._ensure_voice()
    self.logger.info("🔊 Streaming output audio ...")
    written = 0
    self.stream.start()
    try:
        for chunk in voice.synthesize(text, syn_config=self.syn_config):
            data = chunk.audio_int16_bytes
            self.stream.write(data)
            written += len(data)
    finally:
        self.stream.stop()
    self.logger.info("🔊 Finished streaming audio.")
    return written
```

**Files.** `src/audio_generator.py`; `tests/test_audio_generator.py`.

**Tests.**

```python
def test_speak_empty_text_is_a_noop(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)
    assert gen.speak("   ") == 0
    assert stream.started == 0 and stream.stopped == 0


def test_speak_writes_chunks_and_returns_bytes(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)
    written = gen.speak("hello", strip_formatting=False)
    assert written == 4                              # two 2-byte chunks
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
```

**Validation.** `python -m pytest tests/test_audio_generator.py -q`

**Done when.** Empty text performs zero stream calls and a synthesis failure still calls
`stream.stop()` exactly once.

---

## G5 — Validate and parameterize `SynthesisConfig`

**Goal.** Invalid rate/volume fail loudly at construction; noise scales and speaker are
configurable.

**Current behaviour (`__init__`, lines 24-30).**

```python
self.syn_config = SynthesisConfig(
    volume=volume,
    length_scale=1.0 / speech_speed,   # ZeroDivisionError when speech_speed == 0
    noise_scale=1.0,
    noise_w_scale=1.0,
    normalize_audio=False,
)
```

`volume` is unvalidated (Piper expects `0..1`), and the other fields are hardcoded.

**Expected behaviour.**

- New keyword arguments with defaults:
  `speaker_id: int | None = None`, `noise_scale: float = 1.0`,
  `noise_w_scale: float = 1.0`, `normalize_audio: bool = False`.
- Validate before building the config:

```python
if speech_speed <= 0:
    raise ValueError("speech_speed must be greater than 0")
if not 0.0 <= volume <= 1.0:
    raise ValueError("volume must be between 0.0 and 1.0")
```

- Build the config with all fields:

```python
self.syn_config = SynthesisConfig(
    speaker_id=speaker_id,
    volume=volume,
    length_scale=1.0 / speech_speed,
    noise_scale=noise_scale,
    noise_w_scale=noise_w_scale,
    normalize_audio=normalize_audio,
)
```

**Files.** `src/audio_generator.py`; `tests/test_audio_generator.py`.

**Tests.**

```python
@pytest.mark.parametrize("speed", [0, -1.0])
def test_invalid_speech_speed_raises(tmp_path, speed):
    with pytest.raises(ValueError):
        ag.SpeechGenerator(model_path=make_model(tmp_path), speech_speed=speed,
                           stream=FakeOutputStream())


@pytest.mark.parametrize("volume", [-0.1, 1.5])
def test_invalid_volume_raises(tmp_path, volume):
    with pytest.raises(ValueError):
        ag.SpeechGenerator(model_path=make_model(tmp_path), volume=volume,
                           stream=FakeOutputStream())


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
```

**Validation.** `python -m pytest tests/test_audio_generator.py -q`

**Done when.** Invalid values raise `ValueError` and every synthesis field is forwarded.

---

## G6 — Validate output samplerate against the voice

**Goal.** A configured samplerate that disagrees with the voice must not play at the wrong
pitch silently.

**Current behaviour (`__init__`, line 31).** The output stream is opened with
`samplerate=22050`, but the Piper voice has its own `config.sample_rate`. Nothing checks
them.

**Expected behaviour.** Extend `_ensure_voice` (G3) to validate after loading the voice:

```python
def _ensure_voice(self):
    if self.voice is None:
        self.logger.info("🚀 Loading Audio Generator model from %s ...", self.model_path)
        self.voice = PiperVoice.load(self.model_path)
    voice_rate = getattr(getattr(self.voice, "config", None), "sample_rate", None)
    if voice_rate is not None and int(voice_rate) != int(self.samplerate):
        raise ValueError(
            f"Output samplerate {self.samplerate} does not match voice "
            f"samplerate {voice_rate}; set [speech_gen] samplerate = {voice_rate}"
        )
    return self.voice
```

`self.samplerate = samplerate` must be stored in `__init__`.

**Files.** `src/audio_generator.py`; `tests/test_audio_generator.py`.

**Tests.**

```python
def test_samplerate_mismatch_raises(tmp_path):
    gen = make_generator(tmp_path, voice=FakePiperVoice(sample_rate=16000),
                         samplerate=22050)
    with pytest.raises(ValueError):
        gen._ensure_voice()


def test_samplerate_match_is_accepted(tmp_path):
    gen = make_generator(tmp_path, voice=FakePiperVoice(sample_rate=16000),
                         samplerate=16000)
    assert gen._ensure_voice() is gen.voice


def test_missing_voice_rate_is_tolerated(tmp_path):
    voice = FakePiperVoice()
    voice.config = SimpleNamespace()          # no sample_rate attribute
    gen = make_generator(tmp_path, voice=voice)
    assert gen._ensure_voice() is voice
```

**Validation.** `python -m pytest tests/test_audio_generator.py -q`

**Done when.** A mismatch raises `ValueError` naming both rates, and a missing
`config.sample_rate` does not crash.

---

## G7 — Harden `to_wav`

**Goal.** `to_wav` writes a valid file to any path, cleans the text, and reports where it
wrote.

**Current behaviour (`to_wav`, line 42).**

```python
with wave.open(str(file_path), "wb") as wav_file:
    self.voice.synthesize_wav(text, wav_file, syn_config=self.syn_config)
self.logger.info(f"💾 Audio saved to {file_path.name}")
```

It fails when the parent directory does not exist, writes the raw text (markdown/emoji
included), returns `None`, and creates a zero-frame WAV for empty text.

**Expected behaviour.**

```python
def to_wav(self, text: str, file_path: str | Path,
           strip_formatting: bool = True) -> Path:
    """Synthesize text to a WAV file and return its Path.

    Applies clean_for_speech when strip_formatting is True and creates missing
    parent directories. Raises ValueError for empty (or fully stripped) text.
    """
    if strip_formatting:
        text = self.clean_for_speech(text)
    if not text:
        raise ValueError("Cannot synthesize WAV: empty text")
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    voice = self._ensure_voice()
    with wave.open(str(path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=self.syn_config)
    self.logger.info(f"💾 Audio saved to {path.name}")
    return path
```

**Files.** `src/audio_generator.py`; `tests/test_audio_generator.py`.

**Tests.**

```python
def test_to_wav_creates_parent_and_returns_path(tmp_path):
    gen = make_generator(tmp_path)
    out = gen.to_wav("hello", tmp_path / "nested" / "out.wav", strip_formatting=False)
    assert out == tmp_path / "nested" / "out.wav" and out.exists()
    with wave.open(str(out), "rb") as wav_file:
        assert wav_file.getnframes() == 2        # FakePiperVoice writes two frames


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
```

**Validation.** `python -m pytest tests/test_audio_generator.py -q`

**Done when.** Nested paths are created, the returned value is a `Path`, and empty text
raises before any file is opened.

---

## G8 — Wire `[speech_gen]` config and `live_ai.allow_speech` cleanup

**Goal.** Expose the new synthesis knobs and release the stream deterministically.

**Current behaviour.**
- `config.toml` `[speech_gen]` has only `model_path`, `samplerate`, `volume`,
  `speech_speed`.
- `src/live_ai.py:148-155` constructs `SpeechGenerator` with four arguments and releases it
  with `del self.speech` (no `close()`), so the output stream is not stopped/closed until
  GC.

**Expected behaviour.**

1. `config.toml` `[speech_gen]` gains optional documented keys:

   ```toml
   speaker_id = 0          # optional; omit or comment out for single-speaker voices
   noise_scale = 1.0
   noise_w_scale = 1.0
   normalize_audio = false
   ```

2. `Brain.allow_speech` (in `src/live_ai.py`) forwards them via `getattr` and calls
   `close()` before dropping the generator:

   ```python
   if enable and not hasattr(self, "speech"):
       gen_cfg = self.config.speech_gen
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
       self.speech.close()
       del self.speech
       return "Speech Generator released"
   ```

   Keep the `@dump_func_call` decorator and the return strings unchanged.

**Files.** `config.toml`, `src/live_ai.py`, new `tests/test_live_ai.py`.

**Tests.** Do not construct a real `Brain`; call the method on a bare instance and stub
`SpeechGenerator` in `src.live_ai`:

```python
from logging import getLogger
from types import SimpleNamespace

import src.live_ai as live_ai_mod


def test_allow_speech_forwards_config_and_closes(monkeypatch):
    seen = {}

    class Recorder:
        def __init__(self, **kw):
            seen.update(kw)
            self.closed = 0

        def close(self):
            self.closed += 1

    monkeypatch.setattr(live_ai_mod, "SpeechGenerator", Recorder)

    brain = live_ai_mod.Brain.__new__(live_ai_mod.Brain)
    brain.logger = getLogger("test")
    brain.config = SimpleNamespace(speech_gen=SimpleNamespace(
        model_path="m", samplerate=22050, volume=0.3, speech_speed=1.5,
        speaker_id=2, noise_scale=0.4, noise_w_scale=0.7, normalize_audio=True,
    ))

    assert brain.allow_speech(True) == "Speech Generator initialized"
    speech = brain.speech
    assert seen["speaker_id"] == 2 and seen["noise_scale"] == 0.4
    assert seen["noise_w_scale"] == 0.7 and seen["normalize_audio"] is True
    assert seen["speech_speed"] == 1.5

    assert brain.allow_speech(False) == "Speech Generator released"
    assert speech.closed == 1
    assert not hasattr(brain, "speech")
```

Also add a config smoke test asserting a `[speech_gen]` block without the new keys still
constructs a namespace (all keys optional via `getattr`).

**Validation.** `python -m pytest tests/test_live_ai.py tests/test_audio_generator.py -q`

**Done when.** `allow_speech(False)` calls `close()` exactly once and the new keys are
forwarded.

---

## G9 — Typing/docstrings + end-to-end acceptance

**Goal.** Clean up the module surface and prove `speak`/`to_wav` work end to end with fakes.

**Current behaviour.** `to_wav` is annotated `-> None`, methods lack docstrings, and the
module has no `__all__`.

**Expected behaviour.**

- `to_wav` returns `Path` (done in G7); add return annotations and concise docstrings to
  `_ensure_voice`, `speak`, `to_wav`, `close`, `clean_for_speech`.
- Add `__all__ = ["SpeechGenerator"]` at module scope.
- Keep imports as they are (all are used) but import `sounddevice as sd` lazily if a test
  environment lacks it; the existing `importorskip` guard is sufficient for now.

**Files.** `src/audio_generator.py`; `tests/test_audio_generator.py`.

**Tests.**

```python
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
```

**Validation.** `python -m pytest tests/ -q` (full suite).

**Done when.** Full suite green and the end-to-end test exercises construction → speak →
to_wav with no hardware.

---

## 3. Global acceptance criteria

Run from the repo root:

```
python -m pytest tests/ -q
```

Must pass. Additionally verify:

1. `python -c "import src.audio_generator"` loads no Piper model and opens no audio device.
2. `SpeechGenerator` can be constructed with a `FakePiperVoice` and `FakeOutputStream`
   only; no weights required.
3. `with SpeechGenerator(...) as sg: pass` closes the stream exactly once.
4. `speak("")` writes nothing and does not start the stream; `speak("hi")` writes chunks and
   returns their total byte length.
5. If synthesis raises mid-stream, `stream.stop()` is still called.
6. `speech_speed=0` and `volume=1.5` raise `ValueError`.
7. `to_wav` creates nested parent directories, returns a `Path`, and raises `ValueError` for
   empty text.
8. A voice/samplerate mismatch raises `ValueError` naming both rates.
9. `Brain.allow_speech(False)` calls `SpeechGenerator.close()` once.
10. `clean_for_speech` removes `( )` as well as the other markdown characters.
11. Every task above has at least one test that fails before its change and passes after.

---

## 4. Out of scope (do not implement under this plan)

- Changing the TTS engine or voice format (Piper stays).
- Streaming to a file, resampling, or mixing multiple voices.
- SSML, phoneme control, or word-level alignments.
- Changes to `Brain` beyond `allow_speech`, audio capture, tools, or the chat input layer.
- Playing audio through anything other than `sounddevice.RawOutputStream`.
