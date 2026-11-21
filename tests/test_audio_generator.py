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


# ------------------------------------------------------------- G4b stop request


class StopRequestingVoice:
    """Fake piper voice: requests a stop while producing the second chunk."""

    def __init__(self, gen):
        self.gen = gen
        self.config = SimpleNamespace(sample_rate=22050)

    def synthesize(self, text, syn_config=None):
        yield SimpleNamespace(audio_int16_bytes=b"\x01\x02")
        self.gen.request_stop()  # stop arrives while chunk 2 is synthesized
        yield SimpleNamespace(audio_int16_bytes=b"\x03\x04")
        yield SimpleNamespace(audio_int16_bytes=b"\x05\x06")


def test_speak_stops_on_request_mid_stream(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)
    gen.voice = StopRequestingVoice(gen)  # injected fake voice is not reloaded
    written = gen.speak("hello", strip_formatting=False)
    assert written == 2  # only the first chunk was played
    assert stream.written == [b"\x01\x02"]
    assert stream.aborted == 1 and stream.stopped == 0  # buffered audio dropped
    assert gen._stop_event.is_set() is False  # cleared: the next speak() plays


def test_speak_clears_stale_stop_request(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)
    gen.request_stop()  # e.g. pressed during generation, before speech starts
    written = gen.speak("hello", strip_formatting=False)
    assert written == 4  # a stale request does not skip the next playback
    assert stream.aborted == 0 and stream.stopped == 1


def test_natural_finish_does_not_abort(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)
    gen.speak("hello", strip_formatting=False)
    assert stream.stopped == 1 and stream.aborted == 0


def test_large_chunk_is_written_in_slices(tmp_path):
    """A sentence-sized chunk goes out as ~50 ms slices, not one blocking write."""
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)  # 22050 Hz -> 2204 B per slice
    big = b"\x01\x02" * 5000  # 10000 bytes -> 5 slices

    class BigVoice:
        config = SimpleNamespace(sample_rate=22050)

        def synthesize(self, text, syn_config=None):
            yield SimpleNamespace(audio_int16_bytes=big)

    gen.voice = BigVoice()
    written = gen.speak("hello", strip_formatting=False)
    assert written == 10000
    assert len(stream.written) == 5  # sliced writes
    assert b"".join(stream.written) == big  # audio is preserved


def test_stop_is_honored_between_slices(tmp_path):
    """Ctrl+C during a long chunk stops within one slice, not after the chunk."""
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)  # 22050 Hz -> 2204 B per slice
    big = b"\x01\x02" * 5000  # 10000 bytes -> 5 slices

    class BigVoice:
        config = SimpleNamespace(sample_rate=22050)

        def synthesize(self, text, syn_config=None):
            yield SimpleNamespace(audio_int16_bytes=big)

    gen.voice = BigVoice()

    def write_and_stop(data):  # Ctrl+C arriving during the second slice
        stream.written.append(data)
        if len(stream.written) == 2:
            gen.request_stop()

    stream.write = write_and_stop
    written = gen.speak("hello", strip_formatting=False)
    assert written == 2 * 2204  # stopped after the second ~50 ms slice
    assert stream.aborted == 1  # the queued rest of the chunk is dropped


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


# ---------------------------------------------------------------- G10 pauses


def test_pause_defaults_and_storage(tmp_path):
    gen = make_generator(tmp_path)
    assert gen._pause_ms == {"comma": 150, "sentence": 350, "paragraph": 600}
    assert gen.non_prose_content is False
    assert gen.omitted_placeholder == "content omitted"


def test_pause_values_are_stored(tmp_path):
    gen = make_generator(
        tmp_path, pause_comma_ms=10, pause_sentence_ms=20,
        pause_paragraph_ms=30, non_prose_content=True,
        omitted_placeholder="see the code",
    )
    assert gen._pause_ms == {"comma": 10, "sentence": 20, "paragraph": 30}
    assert gen.non_prose_content is True
    assert gen.omitted_placeholder == "see the code"


def test_negative_pause_duration_raises(tmp_path):
    with pytest.raises(ValueError):
        make_generator(tmp_path, pause_comma_ms=-1)


def test_zero_pause_duration_is_allowed(tmp_path):
    gen = make_generator(tmp_path, pause_comma_ms=0)
    assert gen._pause_ms["comma"] == 0


# ------------------------------------------------------- G10b segmentation


FENCE = "```"


def seg(text, **kw):
    return ag.SpeechGenerator.segment_for_speech(text, **kw)


def test_segment_splits_on_sentence_and_paragraph():
    assert seg("A. B\n\nC") == [
        ("A.", "sentence"), ("B", "paragraph"), ("C", "none"),
    ]


def test_segment_keeps_terminator_attached():
    assert seg("hello, world.") == [("hello,", "comma"), ("world.", "none")]


def test_segment_bullets_and_headings_are_separate_sentences():
    assert seg("- a\n- b") == [("a", "sentence"), ("b", "none")]
    assert seg("# H\nx") == [("H", "sentence"), ("x", "none")]


def test_segment_strips_list_markers():
    assert seg("1. one\n2. two") == [("one", "sentence"), ("two", "none")]


def test_segment_drops_fenced_code_block():
    text = f"text\n\n{FENCE}\ncode line\nmore\n{FENCE}\nafter"
    assert seg(text) == [
        ("text", "paragraph"), ("content omitted", "sentence"), ("after", "none"),
    ]


def test_segment_drops_table():
    text = "| a | b |\n|---|---|\n| 1 | 2 |\nend"
    assert seg(text) == [("content omitted", "sentence"), ("end", "none")]


def test_segment_drops_indented_code_block():
    text = "    indented code\nmore code\nafter"
    assert seg(text) == [("content omitted", "sentence"), ("more code after", "none")]


def test_segment_empty_placeholder_is_silent():
    text = f"code\n\n{FENCE}\nx\n{FENCE}"
    assert seg(text, omitted_placeholder="") == [("code", "none")]


def test_segment_custom_placeholder_is_used():
    text = f"code\n\n{FENCE}\nx\n{FENCE}"
    assert seg(text, omitted_placeholder="see the code") == [
        ("code", "paragraph"), ("see the code", "none"),
    ]


def test_segment_non_prose_content_true_speaks_code():
    text = f"code\n\n{FENCE}\nprint(1)\n{FENCE}"
    assert seg(text, non_prose_content=True) == [
        ("code", "paragraph"), ("print1", "none"),
    ]


def test_segment_non_prose_content_true_speaks_table():
    text = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert seg(text, non_prose_content=True) == [("a b 1 2", "none")]


def test_segment_keeps_inline_code_text():
    assert seg("see `x` now") == [("see x now", "none")]


def test_segment_replaces_url_with_word():
    assert seg("go to http://example.com/page.") == [("go to link.", "none")]


def test_segment_collapses_whitespace_inside_segment():
    assert seg("a    b\tc") == [("a b c", "none")]


def test_segment_spaced_dash_becomes_clause_boundary():
    assert seg("one - two") == [("one,", "comma"), ("two", "none")]


def test_segment_empty_text_returns_nothing():
    assert seg("") == [] and seg("   ") == []


# ------------------------------------------------------- G10c speak pauses


def test_silence_bytes_length_and_zero():
    gen = ag.SpeechGenerator.__new__(ag.SpeechGenerator)
    gen.samplerate, gen.channels, gen.datatype = 22050, 1, "int16"
    # 350 ms at 22050 Hz -> 7717 frames -> 15434 bytes
    assert len(gen._silence_bytes(350)) == 7717 * 2
    assert gen._silence_bytes(0) == b""
    assert gen._silence_bytes(-5) == b""


def test_pause_ms_for_scales_with_speech_speed():
    gen = ag.SpeechGenerator.__new__(ag.SpeechGenerator)
    gen._pause_ms = {"comma": 150, "sentence": 350, "paragraph": 600}
    gen.speech_speed = 2.0
    assert gen._pause_ms_for("sentence") == 175
    assert gen._pause_ms_for("paragraph") == 300
    assert gen._pause_ms_for("none") == 0


def silence_total(stream):
    """Total silence bytes written (all-zero buffers)."""
    return sum(len(buf) for buf in stream.written if buf and not any(buf))


def test_speak_inserts_silence_between_segments(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)
    # "a. b" -> two segments ("a.", "b"), sentence pause after the first.
    written = gen.speak("a. b", strip_formatting=False)
    assert silence_total(stream) == int(22050 * 0.350) * 2
    assert written == 4 + int(22050 * 0.350) * 2 + 4
    # the audio of the second segment comes after the silence
    audio = [i for i, buf in enumerate(stream.written) if buf == b"\x01\x02"]
    silence = [i for i, buf in enumerate(stream.written) if buf and not any(buf)]
    assert max(silence) < max(audio) and min(audio) < min(silence)


def test_speak_comma_pause_is_shorter(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)
    gen.speak("a, b", strip_formatting=False)
    assert silence_total(stream) == int(22050 * 0.150) * 2


def test_paragraph_pause_is_longer_than_sentence(tmp_path):
    sentence = FakeOutputStream()
    make_generator(tmp_path, stream=sentence).speak("a. b", strip_formatting=False)
    paragraph = FakeOutputStream()
    make_generator(tmp_path, stream=paragraph).speak("a.\n\nb", strip_formatting=False)
    assert silence_total(paragraph) == int(22050 * 0.600) * 2
    assert silence_total(paragraph) > silence_total(sentence)


def test_pause_durations_scale_with_speech_speed(tmp_path):
    fast = FakeOutputStream()
    make_generator(tmp_path, stream=fast, speech_speed=2.0).speak(
        "a. b", strip_formatting=False)
    slow = FakeOutputStream()
    make_generator(tmp_path, stream=slow, speech_speed=1.0).speak(
        "a. b", strip_formatting=False)
    assert abs(silence_total(fast) - silence_total(slow) // 2) <= 2


def test_no_trailing_pause_after_last_segment(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)
    gen.speak("a. b", strip_formatting=False)
    # last written buffer is audio, not silence
    assert stream.written[-1] == b"\x01\x02"


def test_single_segment_has_no_silence(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream)
    gen.speak("hello", strip_formatting=False)
    assert stream.written == [b"\x01\x02", b"\x01\x02"]


def test_zero_pause_writes_no_silence(tmp_path):
    stream = FakeOutputStream()
    gen = make_generator(tmp_path, stream=stream, pause_sentence_ms=0)
    gen.speak("a. b", strip_formatting=False)
    assert silence_total(stream) == 0
    assert stream.written == [b"\x01\x02"] * 4  # two segments, no silence


def test_stop_request_during_pause_aborts_playback(tmp_path):
    """A stop arriving while the first pause is written stops the playback."""
    gen = make_generator(tmp_path, stream=FakeOutputStream())

    class StopOnSilenceStream(FakeOutputStream):
        def write(self, data):
            super().write(data)
            if data and not any(data):
                gen.request_stop()  # Ctrl+C during the pause

    gen.stream = StopOnSilenceStream()
    gen.speak("a. b", strip_formatting=False)
    played = gen.stream.written
    assert played[:2] == [b"\x01\x02", b"\x01\x02"]  # first segment played
    assert b"\x01\x02" not in played[2:]  # second segment never played
    assert gen.stream.aborted == 1 and gen.stream.stopped == 0
