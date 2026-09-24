import re
import wave
import sounddevice as sd
from pathlib import Path
from logging import getLogger
from piper import PiperVoice, SynthesisConfig

class SpeechGenerator:
    def __init__(
        self, 
        model_path: Path,
        volume: float = 0.5,
        speech_speed: float = 1.0,
        samplerate: int = 22050,
        channels: int = 1,
        datatype: str = "int16"
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"Voice model file not found: {model_path}")
        
        self.logger = getLogger(self.__class__.__name__)
        self.logger.info(f"🚀 Loading Audio Generator model from {model_path} ...")
        self.voice = PiperVoice.load(model_path)
        self.syn_config = SynthesisConfig(
            volume=volume,
            length_scale=1.0 / speech_speed,
            noise_scale=1.0,  # more audio variation
            noise_w_scale=1.0,  # more speaking variation
            normalize_audio=False, # use raw audio from voice
        )
        self.stream = sd.RawOutputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=datatype
        )
        
    def __exit__(self, exc_type, exc_value, traceback):
        if self.stream:
            self.logger.info("🧹 Closing audio output stream...")
            self.stream.close()

    def to_wav(self, text: str, file_path: Path):
        with wave.open(str(file_path), "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file, syn_config=self.syn_config)
        self.logger.info(f"💾 Audio saved to {file_path.name}")

    @staticmethod
    def clean_for_speech(text: str) -> str:
        """Remove formatting characters and emoji so they are not read aloud."""
        # Remove markdown formatting characters: **, __, *, _, #, `, >, ~, [ ] ( )
        text = re.sub(r"[*_#`>~\[\]]", "", text)
        # Remove emoji and other pictographic symbols
        text = re.sub(
            r"[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\uFE0F\u200D\u2190-\u21FF\u2B00-\u2BFF]",
            "",
            text,
        )
        # Collapse leftover whitespace
        return re.sub(r"\s+", " ", text).strip()

    def speak(self, text: str, strip_formatting: bool=True):
        if strip_formatting:
            self.logger.debug("Removing formatted syntax...")
            text = self.clean_for_speech(text)
        self.logger.info("🔊 Streaming output audio ...")
        self.stream.start()
        for chunk in self.voice.synthesize(text, syn_config=self.syn_config):
            self.stream.write(chunk.audio_int16_bytes)
        self.stream.stop()
        self.logger.info("🔊 Finished streaming audio.")
