import time
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from logging import getLogger
from faster_whisper import WhisperModel


def get_audio_tone(
        duration: float = 2.0,    # seconds
        samplerate: int = 16000,  # samples per second
        frequency: float = 440.0  # Hz
    ) -> np.ndarray:
    """
    Generate a test audio tone (sine wave).
    """
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio.astype('int16')


def save_audio_wav(
        audio: np.ndarray, 
        out_file: Path="input_audio.wav", 
        samplerate: int = 16000
    ) -> Path:
    """
    Save audio to a WAV file.
    """
    audio_file = open(file=out_file, mode="wb")
    sf.write(audio_file, audio, samplerate)
    return out_file


@dataclass
class InputAudioDevice:
    name: str
    index: int
    default_samplerate: int
    max_channels: int


class AudioWrapper:
    def __init__(
            self, 
            model_path: str, 
            language: str = "en", 
            beam_size: int = 5,
            silence_threshold: float = 0.02,
            silence_timeout: float = 1.5,
        ):
        self.logger = getLogger(self.__class__.__name__)
        devices = sd.query_devices(kind='input')
        if devices and isinstance(devices, dict):
            devices = [devices]
        if len(devices) == 0:
            raise ValueError("No audio input devices found")
        else:
            self.logger.info(f"Found {len(devices)} audio input devices.")
            
        self.device = InputAudioDevice(
            name=devices[0].get("name", "Unknown Device"),
            index=int(devices[0].get("index", 0)),
            default_samplerate=int(devices[0].get("default_samplerate", 16000)),
            max_channels=int(devices[0].get("max_input_channels", 1))
        )
        self.logger.info(f"⚙️  Using audio input device: {self.device.name} (index {self.device.index})")
        self.logger.info(f"🚀 Loading Audio to Text model from {model_path} ...")
        self.model = WhisperModel(str(model_path), device="cpu", compute_type="int16")
        self.language = language
        self.beam_size = beam_size
        
        # Voice command detection settings
        self.silence_threshold = silence_threshold  # RMS level for silence detection
        self.silence_timeout = silence_timeout      # Seconds of silence before considering speech ended
    
    def _rms_level(self, audio: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) level of audio."""
        return np.sqrt(np.mean(audio.astype(float) ** 2))

    def record_audio_wait(
            self, 
            duration: float, 
            samplerate: int = 16000,
            channels: int = 1, 
            datatype: str = 'int16'
        ) -> np.ndarray:
        """
        Record audio from microphone for a given duration [seconds].
        """
        self.logger.info(f"🎤 Recording for {duration} seconds...")
        audio = sd.rec(
            frames=int(duration * samplerate),
            samplerate=samplerate,
            channels=channels,
            dtype=datatype,
            device=self.device.index
        )
        sd.wait()  # Wait until recording is finished
        self.logger.info(f"✅ Recorded {len(audio)} samples")
        return audio
    
    def record_audio(
        self,
        duration: float,
        samplerate: int = 16000,
        channels: int = 1,
        datatype: str = 'int16',
        silence_threshold: float = 0.01,
        silence_timeout: float = 1.5,
        chunk_size: float = 0.1
    ) -> np.ndarray:
        """
        Record audio until:
        - duration is reached, OR
        - RMS stays below silence_threshold for > silence_timeout seconds.
        Non-blocking, chunk-based recording.
        """
        self.logger.info(f"🎤 Recording up to {duration} seconds...")
        chunk_frames = int(chunk_size * samplerate)
        recorded_chunks = []
        silence_start = None
        start_time = time.time()

        with sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            dtype=datatype,
            device=self.device.index
        ) as stream:

            while True:
                chunk, _ = stream.read(chunk_frames)
                recorded_chunks.append(chunk)
                rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))
                print(f"\rIN-Audio RMS: {rms:.4f}", end="")
                if rms < silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= silence_timeout:
                        self.logger.info(f"⏹️  Stopping due to silence timeout: {silence_timeout} s")
                        break
                else:
                    silence_start = None
                if time.time() - start_time >= duration:
                    self.logger.info(f"⏹️  Max duration reached: {duration} s")
                    break
        audio = np.concatenate(recorded_chunks, axis=0)
        self.logger.info(f"✅ Recorded {len(audio)} samples")
        return audio
        
    def transcribe(self, wav_path: Path) -> str:
        self.logger.info("🧠 Transcribing with Whisper...")
        segments, _ = self.model.transcribe(
            audio=str(wav_path),
            language=self.language,
            beam_size=self.beam_size
        )
        text = "".join(seg.text for seg in segments).strip()
        self.logger.info(f"Text Length: {len(text)}")
        return text
