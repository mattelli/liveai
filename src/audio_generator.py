import re
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
        voice=None,
        stream=None,
        stream_factory=sd.RawOutputStream,
    ):
        """
        Configure the speech generator.

        The Piper voice is loaded lazily on first use (``_ensure_voice``).
        Raises FileNotFoundError for a missing model and ValueError for an
        invalid ``speech_speed`` or ``volume``.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Voice model file not found: {model_path}")
        if speech_speed <= 0:
            raise ValueError("speech_speed must be greater than 0")
        if not 0.0 <= volume <= 1.0:
            raise ValueError("volume must be between 0.0 and 1.0")

        self.logger = getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.voice = voice
        self.samplerate = samplerate
        self.channels = channels
        self.datatype = datatype
        self._closed = False
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
