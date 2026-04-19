import sys
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from logging import getLogger
from faster_whisper import WhisperModel

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
        samplerate: int = 16000,
        chunk_time_s: float = 0.1,
        chunks: int = 20,
        guard_factor: float = 1.5
    ) -> float:
    chunk_rms_values: list[float] = []
    chunk_size = int(chunk_time_s * samplerate)
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', device=device.index) as stream:
        while len(chunk_rms_values) <= chunks:
            chunk, _ = stream.read(chunk_size)
            chunk_rms_values.append(np.sqrt(np.mean(chunk.astype(np.float32)**2)))
    mean_rms = np.mean(chunk_rms_values)
    std_rms = np.std(chunk_rms_values)
    threshold = guard_factor * (mean_rms + 2 * std_rms)
    return threshold, mean_rms, std_rms


class SpeechDetector:
    def __init__(
            self, 
            model_path: Path, 
            language: str = "en", 
            beam_size: int = 5,
        ):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.logger = getLogger(self.__class__.__name__)
        self.device = InputAudioDevice()
        self.logger.info(f"⚙️  Using audio input device: {self.device.name} (index {self.device.index})")
        self.logger.info(f"🚀 Loading Audio to Text model from {model_path} ...")
        self.model = WhisperModel(str(model_path), device="cpu", compute_type="int16")
        self.logger.info(f"✅ Model loaded.")
        self.language = language
        self.beam_size = beam_size
        self.logger.info("🎬 Calibrating silence threshold...")
        self.silence_threshold, mean_rms, std_rms = compute_silence_threshold(self.device)
        self.logger.info(f"✅ Silence threshold set to: {self.silence_threshold:.3f}, (Mean RMS: {mean_rms:.3f}, Std RMS: {std_rms:.3f})")
    
    def listen(
            self,
            silence_timeout_s: float = 1.5,
            channels: int = 1,
            samplerate: int = 16000,
            chunk_time_s: float = 0.1,
            datatype: str = 'int16',
            clear_timeout_s: float = 30.0
        ) -> np.ndarray:
        """
        Record audio from the microphone until silence timeout is detected based on RMS values.
        Non-blocking, chunk-based recording.
        """
        self.logger.info(f"🎤 Listening ... (Silence timeout: {silence_timeout_s} s). Press Ctrl+C to quit.")
        chunk_size = int(chunk_time_s * samplerate)
        timeout_chunks = int(silence_timeout_s / chunk_time_s)
        chunks: list[np.ndarray] = []
        chunk_rms_values: list[float] = []
        speech_start: float | None = None
        start_chunk: int = 0
        
        start_time = time.time()
        with sd.InputStream(samplerate=samplerate, channels=channels, dtype=datatype, device=self.device.index) as stream:
            while True:
                chunk, _ = stream.read(chunk_size)
                chunks.append(chunk)
                chunk_rms = np.sqrt(np.mean(chunk.astype(np.float32)**2))
                chunk_rms_values.append(chunk_rms)
                audio_rms = np.mean(chunk_rms_values)
                print(f"\rIN-Audio RMS: {audio_rms:.3f} | Silence threshold [{self.silence_threshold:.3f}] ", end="")
                if audio_rms > self.silence_threshold:
                    if speech_start is None:
                        speech_start = time.time()
                        start_chunk = int((speech_start - start_time) / chunk_time_s)
                    elif np.mean(chunk_rms_values[-timeout_chunks:]) < self.silence_threshold:
                        print()
                        self.logger.info(f" ⏹️  Silence timeout detected after speech. Stopping listening.")
                        break
                if time.time() - start_time > clear_timeout_s:
                    self.logger.debug(f"No speech detected for {clear_timeout_s} s. Clearing buffer.")
                    chunks.clear()
                    chunk_rms_values.clear()
                    speech_start = None
                    start_chunk = 0
                    start_time = time.time()
        
        tolerance = int(timeout_chunks / 2)
        start_chunk = start_chunk - tolerance if start_chunk > tolerance else 0
        end_chunk = len(chunks) - tolerance if len(chunks) > tolerance else len(chunks)
        if start_chunk >= end_chunk:
            audio = np.concatenate(chunks, axis=0)
        else:
            audio = np.concatenate(chunks[start_chunk:end_chunk], axis=0)
        if np.sqrt(np.mean(audio.astype(np.float32)**2)) < self.silence_threshold:
            audio = np.array([], dtype=audio.dtype)
        self.logger.debug(f"Recorded {len(audio)} samples")
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
    
    def listen_and_transcribe(self, silence_timeout_s: float) -> str:
        audio = self.listen(silence_timeout_s=silence_timeout_s)
        text = self.transcribe(audio_to_wav(audio))
        return text
