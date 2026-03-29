import wave
import sounddevice as sd
from pathlib import Path
from logging import getLogger
from piper import PiperVoice, SynthesisConfig

class AudioGenerator:
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

    def save_audio(self, text: str, file_path: Path):
        with wave.open(str(file_path), "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file, syn_config=self.syn_config)
        self.logger.info(f"💾 Audio saved to {file_path.name}")

    def streaming_audio(self, text: str):
        self.logger.info("🔊 Streaming output audio ...")
        self.stream.start()
        for chunk in self.voice.synthesize(text, syn_config=self.syn_config):
            self.stream.write(chunk.audio_int16_bytes)
        self.stream.stop()
        self.logger.info("🔊 Finished streaming audio.")
