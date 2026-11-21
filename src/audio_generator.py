import re
import threading
import wave
import sounddevice as sd
from pathlib import Path
from logging import getLogger
from piper import PiperVoice, SynthesisConfig

__all__ = ["SpeechGenerator"]


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

        Applies ``clean_for_speech`` when ``strip_formatting`` is True and
        creates missing parent directories. Raises ValueError for empty (or
        fully stripped) text.
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

    @staticmethod
    def clean_for_speech(text: str) -> str:
        """Remove formatting characters and emoji so they are not read aloud."""
        # Remove markdown formatting characters: **, __, *, _, #, `, >, ~, [ ] ( )
        text = re.sub(r"[*_#`>~\[\]()]", "", text)
        # Remove emoji and other pictographic symbols
        text = re.sub(
            r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\uFE0F\u200D\u2190-\u21FF\u2B00-\u2BFF]",
            "",
            text,
        )
        # Collapse leftover whitespace
        return re.sub(r"\s+", " ", text).strip()

    def speak(self, text: str, strip_formatting: bool = True) -> int:
        """
        Synthesize text and stream it to the output device.

        Returns the number of audio bytes written. Empty (or fully stripped)
        text is a no-op: the stream is neither started nor stopped.

        A stop request (Ctrl+C or ``request_stop``) breaks the loop between
        write slices (about 50 ms of audio each) and aborts the stream,
        dropping still-buffered audio so the silence is immediate; the written
        count reflects only the played audio. A later speak() call plays
        normally again.
        """
        if strip_formatting:
            self.logger.debug("Removing formatted syntax...")
            text = self.clean_for_speech(text)
        if not text:
            self.logger.debug("Nothing to speak after cleaning; skipping.")
            return 0

        voice = self._ensure_voice()
        self._stop_event.clear()  # a stale stop request must not skip playback
        self.logger.info("🔊 Streaming output audio ...")
        written = 0
        self.stream.start()
        try:
            for chunk in voice.synthesize(text, syn_config=self.syn_config):
                if self._stop_event.is_set():
                    self.logger.info("🔇 Speech playback stopped by request.")
                    break
                written += self._write_chunk(chunk.audio_int16_bytes)
        finally:
            if self._stop_event.is_set():
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
