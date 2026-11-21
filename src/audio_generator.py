import re
import threading
import wave
import sounddevice as sd
from pathlib import Path
from logging import getLogger
from piper import PiperVoice, SynthesisConfig

__all__ = ["SpeechGenerator"]


# --- Text segmentation patterns (see SpeechGenerator.segment_for_speech) -----

# Boundary kinds used as the ``pause_after`` tag of a segment.
_PAUSE_NONE = "none"
_PAUSE_COMMA = "comma"
_PAUSE_SENTENCE = "sentence"
_PAUSE_PARAGRAPH = "paragraph"

# Code fence (``` or ~~~), possibly indented.
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
# Markdown table row / separator row (``|---|:--:|``).
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
# Heading (``# Title``) and list marker (``- item``, ``1. item``).
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# Indented code block: 4 spaces or a tab.
_INDENT_RE = re.compile(r"^(?: {4}|\t)")

# A sentence ends on a run of terminators followed by whitespace or the end.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?…]+(?=\s|$)")
# A clause ends on a comma/semicolon/colon before whitespace, or on a spaced
# dash (the dash and its trailing spaces are consumed so they are not spoken).
_CLAUSE_BOUNDARY_RE = re.compile(r"[,;:](?=\s)|(?<=\s)[—–-]\s+")
# URLs are not spelled out; they are read as a short word.
_URL_RE = re.compile(r"https?://[^\s]*[^\s.,;:!?)\]]")
_URL_WORD = "link"


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
        pause_comma_ms: float = 150,
        pause_sentence_ms: float = 350,
        pause_paragraph_ms: float = 600,
        non_prose_content: bool = False,
        omitted_placeholder: str = "content omitted",
        voice=None,
        stream=None,
        stream_factory=sd.RawOutputStream,
    ):
        """
        Configure the speech generator.

        The Piper voice is loaded lazily on first use (``_ensure_voice``).
        Raises FileNotFoundError for a missing model and ValueError for an
        invalid ``speech_speed``, ``volume`` or negative pause duration.

        ``pause_*_ms`` are the base silences inserted after a clause, a
        sentence and a paragraph (scaled by ``speech_speed`` at playback).
        ``non_prose_content`` False (default) replaces code blocks and tables
        with ``omitted_placeholder`` instead of reading them aloud.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Voice model file not found: {model_path}")
        if speech_speed <= 0:
            raise ValueError("speech_speed must be greater than 0")
        if not 0.0 <= volume <= 1.0:
            raise ValueError("volume must be between 0.0 and 1.0")
        if min(pause_comma_ms, pause_sentence_ms, pause_paragraph_ms) < 0:
            raise ValueError("pause durations must not be negative")

        self.logger = getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.voice = voice
        self.samplerate = samplerate
        self.channels = channels
        self.datatype = datatype
        self._closed = False
        self.speech_speed = speech_speed
        # Base pause per boundary kind, in milliseconds; scaled by speech_speed
        # when the silences are written (see _pause_ms_for).
        self._pause_ms = {
            "comma": pause_comma_ms,
            "sentence": pause_sentence_ms,
            "paragraph": pause_paragraph_ms,
        }
        # Non-prose blocks (code, tables): False -> replaced by the placeholder
        # sentence, True -> read aloud as well.
        self.non_prose_content = non_prose_content
        self.omitted_placeholder = omitted_placeholder
        self.syn_config = SynthesisConfig(
            speaker_id=speaker_id,
            volume=volume,
            length_scale=1.0 / speech_speed,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
            normalize_audio=normalize_audio,
        )
        self.stream = stream if stream is not None else stream_factory(
            samplerate=samplerate,
            channels=channels,
            dtype=datatype,
        )
        # Stop request for the running playback (Ctrl+C during speech); an
        # Event is safely settable from a signal handler or any other thread.
        self._stop_event = threading.Event()

    # Audio written per blocking write call (~50 ms): small enough that a stop
    # request is honored quickly - the SIGINT handler can only run between
    # blocking write() calls, which block in real time for the buffered audio -
    # large enough to keep the per-call overhead negligible.
    _SLICE_SECONDS = 0.05
    _SAMPLE_SIZES = {"int8": 1, "uint8": 1, "int16": 2, "int32": 4, "float32": 4}

    def _slice_bytes(self) -> int:
        """Byte length of one write slice (about ``_SLICE_SECONDS`` of audio)."""
        frame = self.channels * self._SAMPLE_SIZES.get(self.datatype, 2)
        return max(frame, int(self.samplerate * self._SLICE_SECONDS) * frame)

    def _write_chunk(self, data: bytes) -> int:
        """Write one synthesis chunk in small slices; return bytes written.

        The stop event is checked between slices: piper yields one chunk per
        sentence (seconds of audio), and ``stream.write`` blocks until the
        buffer is consumed - a whole-chunk write would defer a Ctrl+C until
        the sentence finished playing.
        """
        written = 0
        for off in range(0, len(data), self._slice_bytes()):
            if self._stop_event.is_set():
                break
            piece = data[off:off + self._slice_bytes()]
            self.stream.write(piece)
            written += len(piece)
        return written

    def request_stop(self) -> None:
        """
        Ask the running speak() playback to stop as soon as possible.

        Stops the audio only: buffered audio is dropped for immediate
        silence, and a subsequent speak() call plays normally again. Safe to
        call from a signal handler or any other thread; a no-op while idle.
        """
        self._stop_event.set()

    def _ensure_voice(self):
        """Load the Piper voice once and return it."""
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

    def close(self) -> None:
        """Stop and close the output stream. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        if self.stream is not None:
            self.logger.info("🧹 Closing audio output stream...")
            try:
                self.stream.stop()
            except Exception:  # already stopped / no device
                pass
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def to_wav(
        self,
        text: str,
        file_path: str | Path,
        strip_formatting: bool = True,
    ) -> Path:
        """
        Synthesize text to a WAV file and return its Path.

        The text is split into the same clause/sentence/paragraph segments as
        :meth:`speak`, and the same speed-scaled silences are written between
        them, so an exported file matches live playback. Creates missing
        parent directories. Raises ValueError for empty (or fully stripped)
        text.
        """
        segments = self.segment_for_speech(
            text,
            non_prose_content=self.non_prose_content,
            omitted_placeholder=self.omitted_placeholder,
            strip_formatting=strip_formatting,
        )
        if not segments:
            raise ValueError("Cannot synthesize WAV: empty text")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        voice = self._ensure_voice()
        with wave.open(str(path), "wb") as wav_file:
            # The format is set here (not by the first chunk) so silence can be
            # written even for a segment that produces no audio.
            wav_file.setframerate(self.samplerate)
            wav_file.setsampwidth(2)
            wav_file.setnchannels(self.channels)
            for segment, pause_after in segments:
                voice.synthesize_wav(
                    segment, wav_file, syn_config=self.syn_config,
                    set_wav_format=False,
                )
                silence = self._silence_bytes(self._pause_ms_for(pause_after))
                if silence:
                    wav_file.writeframes(silence)
        self.logger.info(f"💾 Audio saved to {path.name}")
        return path

    @staticmethod
    def _strip_formatting(text: str) -> str:
        """Remove markdown formatting characters and emoji (no whitespace pass)."""
        # Remove markdown formatting characters: **, __, *, _, #, `, >, ~, [ ] ( )
        text = re.sub(r"[*_#`>~\[\]()]", "", text)
        # Remove emoji and other pictographic symbols
        return re.sub(
            r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\uFE0F\u200D\u2190-\u21FF\u2B00-\u2BFF]",
            "",
            text,
        )

    @staticmethod
    def clean_for_speech(text: str) -> str:
        """Remove formatting characters and emoji so they are not read aloud."""
        # Collapse leftover whitespace
        return re.sub(r"\s+", " ", SpeechGenerator._strip_formatting(text)).strip()

    @staticmethod
    def _blocks(text: str) -> list[tuple[list[str], bool, str]]:
        """Split ``text`` into ``(lines, is_paragraph, kind)`` triples.

        ``kind`` is ``"text"``, ``"fence"``, ``"table"`` or ``"indent"`` (the
        last three are non-prose). A blank line starts a new paragraph: the
        block *before* it gets ``is_paragraph`` True (so the paragraph pause
        lands at the end of the paragraph). A single newline, a heading or a
        list marker starts a new sentence (False). Fenced code blocks are kept
        together as one block, so their inner newlines never split them.
        """
        blocks: list[list] = []
        current: list[str] = []
        current_kind = "text"
        fence_char: str | None = None

        def flush():
            nonlocal current, current_kind
            if current:
                blocks.append([list(current), False, current_kind])
            current = []
            current_kind = "text"

        for line in text.splitlines():
            fence_match = _FENCE_RE.match(line)
            if fence_char is not None:
                # Inside a fenced block: keep the line verbatim until the
                # closing fence (which must use the same character).
                current.append(line.strip())
                if fence_match and fence_match.group(1)[0] == fence_char:
                    fence_char = None
                    flush()  # the code block is its own block
                continue
            if fence_match:
                flush()
                fence_char = fence_match.group(1)[0]
                current_kind = "fence"
                current.append(line.strip())
                continue
            if not line.strip():
                flush()
                if blocks:
                    blocks[-1][1] = True  # the block before the blank line
                continue
            if _HEADING_RE.match(line) or _LIST_RE.match(line):
                # A heading or a list item is its own sentence; the marker
                # itself is not spoken.
                flush()
                marker = _HEADING_RE.match(line) or _LIST_RE.match(line)
                current.append(line[marker.end():].strip())
                flush()
                continue
            kind = ("table" if _TABLE_ROW_RE.match(line)
                    else "indent" if _INDENT_RE.match(line) else "text")
            if kind != current_kind:
                flush()
                current_kind = kind
            current.append(line.strip())
        flush()
        return blocks

    @staticmethod
    def _split_keep(text: str, pattern: re.Pattern) -> list[str]:
        """Split on ``pattern`` keeping each delimiter attached to its part."""
        parts: list[str] = []
        last = 0
        for match in pattern.finditer(text):
            parts.append(text[last:match.end()])
            last = match.end()
        parts.append(text[last:])
        return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]

    @staticmethod
    def _sentences(block: str) -> list[str]:
        """Split a block into sentences, keeping each terminator attached."""
        return SpeechGenerator._split_keep(block, _SENTENCE_BOUNDARY_RE)

    @staticmethod
    def _clauses(sentence: str) -> list[str]:
        """Split a sentence into clauses, keeping each terminator attached."""
        # A spaced dash is a clause boundary too; turn it into a comma so it is
        # not read aloud as "dash".
        sentence = re.sub(r"\s+[—–-]\s+", ", ", sentence)
        return SpeechGenerator._split_keep(sentence, _CLAUSE_BOUNDARY_RE)

    @staticmethod
    def segment_for_speech(
        text: str,
        non_prose_content: bool = False,
        omitted_placeholder: str = "content omitted",
        strip_formatting: bool = True,
    ) -> list[tuple[str, str]]:
        """Split ``text`` into ``(segment_text, pause_after)`` pairs.

        ``pause_after`` is one of ``"none"``, ``"comma"``, ``"sentence"`` or
        ``"paragraph"``. Each segment keeps its punctuation terminator so the
        synthesizer still applies its natural intonation; the pause is added
        *after* the segment by the caller.

        Non-prose blocks (fenced/indented code, markdown tables) are replaced
        by ``omitted_placeholder`` when ``non_prose_content`` is False (the
        default), or read aloud when True. Inline code is read as plain text
        and URLs are read as the word ``"link"``. ``strip_formatting`` False
        keeps the raw text (markdown and URLs untouched). Pure function: no
        audio, no I/O.
        """
        segments: list[tuple[str, str]] = []
        for lines, is_paragraph, kind in SpeechGenerator._blocks(text):
            if kind == "text":
                block = " ".join(lines)
            else:
                if not non_prose_content:
                    if omitted_placeholder:
                        segments.append((omitted_placeholder, _PAUSE_SENTENCE))
                    continue
                block = SpeechGenerator._strip_code_markers(lines)
            if strip_formatting:
                block = _URL_RE.sub(_URL_WORD, SpeechGenerator._strip_formatting(block))
            for sentence in SpeechGenerator._sentences(block):
                clauses = SpeechGenerator._clauses(sentence)
                for i, clause in enumerate(clauses):
                    pause = _PAUSE_COMMA if i < len(clauses) - 1 else _PAUSE_SENTENCE
                    segments.append((clause, pause))
            if is_paragraph and segments:
                segments[-1] = (segments[-1][0], _PAUSE_PARAGRAPH)
        if segments:
            segments[-1] = (segments[-1][0], _PAUSE_NONE)
        return segments

    @staticmethod
    def _strip_code_markers(lines: list[str]) -> str:
        """Drop fence/table markers so a non-prose block can be read aloud."""
        kept = []
        for line in lines:
            if _FENCE_RE.match(line) or _TABLE_SEP_RE.match(line):
                continue
            kept.append(line.replace("|", " ").strip())
        return " ".join(kept).strip()

    def _pause_ms_for(self, pause_after: str) -> float:
        """Silence in milliseconds for a boundary kind, scaled by speech speed."""
        base = self._pause_ms.get(pause_after, 0)
        return base / self.speech_speed

    def _silence_bytes(self, ms: float) -> bytes:
        """``ms`` milliseconds of silence as raw PCM bytes (no frames if <= 0)."""
        frames = int(self.samplerate * (ms / 1000.0))
        if frames <= 0:
            return b""
        return b"\x00" * (frames * self.channels * self._SAMPLE_SIZES.get(self.datatype, 2))

    def speak(self, text: str, strip_formatting: bool = True) -> int:
        """
        Synthesize text and stream it to the output device.

        Returns the number of audio bytes written. Empty (or fully stripped)
        text is a no-op: the stream is neither started nor stopped.

        The text is split into clause/sentence/paragraph segments
        (``segment_for_speech``) and a silence - scaled by ``speech_speed`` -
        is written after each one, so punctuation and line breaks are audible
        as natural pauses.

        A stop request (Ctrl+C or ``request_stop``) breaks the loop between
        write slices (about 50 ms of audio each) and aborts the stream,
        dropping still-buffered audio so the silence is immediate; the written
        count reflects only the played audio. A later speak() call plays
        normally again.
        """
        segments = self.segment_for_speech(
            text,
            non_prose_content=self.non_prose_content,
            omitted_placeholder=self.omitted_placeholder,
            strip_formatting=strip_formatting,
        )
        if not segments:
            self.logger.debug("Nothing to speak after cleaning; skipping.")
            return 0

        voice = self._ensure_voice()
        self._stop_event.clear()  # a stale stop request must not skip playback
        self.logger.info("🔊 Streaming output audio ...")
        written = 0
        self.stream.start()
        try:
            for segment, pause_after in segments:
                if self._stop_event.is_set():
                    break
                for chunk in voice.synthesize(segment, syn_config=self.syn_config):
                    if self._stop_event.is_set():
                        break
                    written += self._write_chunk(chunk.audio_int16_bytes)
                if self._stop_event.is_set():
                    break
                silence = self._silence_bytes(self._pause_ms_for(pause_after))
                if silence:
                    written += self._write_chunk(silence)
        finally:
            if self._stop_event.is_set():
                self.logger.info("🔇 Speech playback stopped by request.")
                self._abort_stream()
                self._stop_event.clear()
            else:
                self.stream.stop()
        self.logger.info("🔊 Finished streaming audio.")
        return written

    def _abort_stream(self) -> None:
        """Stop the output stream immediately, dropping buffered audio."""
        abort = getattr(self.stream, "abort", None)
        try:
            if callable(abort):
                abort()
            else:
                self.stream.stop()  # stream without abort support
        except Exception:  # already stopped / no device
            self.logger.debug("Aborting audio stream failed.", exc_info=True)
