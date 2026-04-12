import sys
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
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


def audio_to_wav(
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


def compute_silence_threshold(
        device: InputAudioDevice,
        rms_values_size: int = 10,
        samplerate: int = 16000,
        chunk_size_s: float = 0.1,
    ) -> float:
    chunk_frames = int(chunk_size_s * samplerate)
    rms_values: list[float] = []
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', device=device.index) as stream:
        while len(rms_values) <= rms_values_size:
            chunk, _ = stream.read(chunk_frames)
            rms_values.append(np.sqrt(np.mean(chunk.astype(np.float32)**2)))
    threshold = 1.5 * np.mean(rms_values)
    return threshold


class InputAudioDevice:
    
    def __init__(self):
        self.logger = getLogger(self.__class__.__name__)
        try:
            devices = sd.query_devices(kind='input')
        except Exception as e:
            self.logger.critical(f"{e}")
            self.logger.info("Please check microphone settings and try again.")
            sys.exit(1)
        if devices and isinstance(devices, dict):
            devices = [devices]
        if len(devices) == 0:
            raise ValueError("No audio input devices found")
        else:
            self.logger.info(f"Found {len(devices)} audio input devices.")
        self.name = str(devices[0].get("name", "Unknown Device"))
        self.index = int(devices[0].get("index", 0))
        self.default_samplerate = int(devices[0].get("default_samplerate", 16000))
        self.max_channel = int(devices[0].get("max_input_channels", 1))


class SpeechDetector:
    def __init__(
            self, 
            model_path: Path, 
            language: str = "en", 
            beam_size: int = 5,
            silence_timeout_s: float = 1.5,
        ):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.logger = getLogger(self.__class__.__name__)
        self.device = InputAudioDevice()
        self.logger.info(f"⚙️  Using audio input device: {self.device.name} (index {self.device.index})")
        self.logger.info(f"🚀 Loading Audio to Text model from {model_path} ...")
        self.model = WhisperModel(str(model_path), device="cpu", compute_type="int16")
        self.language = language
        self.beam_size = beam_size
        self.silence_timeout = silence_timeout_s
        self.silence_threshold = compute_silence_threshold(self.device)
        self.logger.info(f"✅ Model loaded. Silence threshold set to: {self.silence_threshold:.3f}")
    
    def listen(
            self,
            duration_s: float,
            channels: int = 1,
            samplerate: int = 16000,
            chunk_size_s: float = 0.1,
            datatype: str = 'int16',
        ) -> np.ndarray:
        """
        Record audio until:
        - duration is reached, OR
        - RMS stays below silence_threshold for > silence_timeout seconds.
        Non-blocking, chunk-based recording.
        """
        self.logger.info(f"🎤 Recording up to {duration_s} seconds...")
        chunk_frames = int(chunk_size_s * samplerate)
        recorded_chunks: list[np.ndarray] = []
        silence_start: float | None = None
        
        start_time = time.time()
        with sd.InputStream(samplerate=samplerate, channels=channels, dtype=datatype, device=self.device.index) as stream:
            while True:
                chunk, _ = stream.read(chunk_frames)
                recorded_chunks.append(chunk)
                rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))
                print(f"\rIN-Audio RMS: {rms:.3f} | Silence threshold: {self.silence_threshold:.3f} ", end="")
                if rms < self.silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= self.silence_timeout:
                        self.logger.info(f" ⏹️  Stopping due to silence timeout: {self.silence_timeout} s")
                        break
                else:
                    silence_start = None
                if time.time() - start_time >= duration_s:
                    self.logger.info(f" ⏹️  Max duration reached: {duration_s} s")
                    break
        
        audio = np.concatenate(recorded_chunks, axis=0)
        self.logger.debug(f"Recorded {len(audio)} samples")
        if np.sqrt(np.mean(audio.astype(np.float32)**2)) < self.silence_threshold:
            audio = np.array([], dtype=audio.dtype)
        return audio
        
    def transcribe(self, wav_path: Path) -> str:
        self.logger.debug("🧠 Transcribing with Whisper...")
        segments, _ = self.model.transcribe(
            audio=str(wav_path),
            language=self.language,
            beam_size=self.beam_size
        )
        text = "".join(seg.text for seg in segments).strip()
        self.logger.debug(f"Text Length: {len(text)}")
        return text
    
    def listen_and_transcribe(self, duration_s: float) -> str:
        audio = self.listen(duration_s=duration_s)
        text = self.transcribe(audio_to_wav(audio))
        return text
