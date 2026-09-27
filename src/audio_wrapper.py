import io
import time
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from logging import getLogger
from faster_whisper import WhisperModel

from src.common import cprint


class AudioDeviceError(RuntimeError):
    """Raised when no usable audio input device is available."""


class InputAudioDevice:

    def __init__(self):
        self.logger = getLogger(self.__class__.__name__)
        try:
            devices = sd.query_devices(kind='input')
        except Exception as exc:
            self.logger.critical("%s", exc)
            raise AudioDeviceError(
                "No audio input device available. Check microphone settings."
            ) from exc
        if devices and isinstance(devices, dict):
            devices = [devices]
        if len(devices) == 0:
            raise AudioDeviceError("No audio input devices found")
        else:
            self.logger.info(f"Found {len(devices)} audio input devices.")
        self.name = str(devices[0].get("name", "Unknown Device"))
        self.index = int(devices[0].get("index", 0))
        self.default_samplerate = int(devices[0].get("default_samplerate", 16000))
        self.max_channel = int(devices[0].get("max_input_channels", 1))


def get_audio_tone(
        duration: float = 2.0,    # seconds
        samplerate: int = 16000,  # samples per second
        frequency: float = 440.0,  # Hz
        amplitude: float = 0.5,
    ) -> np.ndarray:
    """
    Generate an int16 sine tone. The peak sample is ``amplitude * 32767``.
    """
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return (wave * np.iinfo(np.int16).max).astype(np.int16)


def format_level(rms: float, threshold: float, show: bool) -> str:
    """Build the live RMS level line. Empty string when show is False."""
    if not show:
        return ""
    return f"\rIN-Audio RMS: {rms:.3f} | Silence threshold [{threshold:.3f}] "


def rms(samples: np.ndarray) -> float:
    """Root-mean-square level of an int16/float array. 0.0 for empty input."""
    arr = np.asarray(samples, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def audio_to_wav(
        audio: np.ndarray,
        out_file: str | Path = "input_audio.wav",
        samplerate: int = 16000
    ) -> Path:
    """
    Write audio to a WAV file and return its Path.

    Raises ValueError when there are no samples to write.
    """
    if np.asarray(audio).size == 0:
        raise ValueError("Cannot write WAV: no audio samples")
    path = Path(out_file)
    sf.write(str(path), audio, samplerate)
    return path


def audio_to_wav_buffer(
        audio: np.ndarray,
        samplerate: int = 16000
    ) -> io.BytesIO:
    """
    Return a seekable in-memory WAV. Raises ValueError for empty audio.
    """
    if np.asarray(audio).size == 0:
        raise ValueError("Cannot encode WAV: no audio samples")
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate, format="WAV")
    buf.seek(0)
    return buf


def compute_silence_threshold(
        device: InputAudioDevice,
        samplerate: int | None = None,
        chunk_time_s: float = 0.1,
        chunks: int = 20,
        guard_factor: float = 1.0,
        stream_factory=sd.InputStream
    ) -> tuple[float, float, float]:
    """
    Return ``(threshold, mean_rms, std_rms)`` sampled from ambient noise.
    """
    if samplerate is None:
        samplerate = device.default_samplerate
    chunk_size = int(chunk_time_s * samplerate)
    values: list[float] = []
    with stream_factory(
        samplerate=samplerate, channels=1, dtype='int16', device=device.index
    ) as stream:
        for _ in range(chunks):
            chunk, _ = stream.read(chunk_size)
            values.append(rms(chunk))
    mean_rms = float(np.mean(values))
    std_rms = float(np.std(values))
    return guard_factor * (mean_rms + 2 * std_rms), mean_rms, std_rms


def capture_speech(
        threshold: float,
        device_index: int,
        stream_factory=sd.InputStream,
        time_func=time.time,
        on_level=None,
        silence_timeout_s: float = 1.5,
        channels: int = 1,
        samplerate: int = 16000,
        chunk_time_s: float = 0.1,
        datatype: str = 'int16',
        clear_timeout_s: float = 30.0,
        max_record_chunks: int | None = None,
    ) -> np.ndarray:
    """
    Record until ``silence_timeout_s`` of consecutive sub-threshold chunks.

    Returns the trimmed speech samples, or an empty array if no speech was
    detected. ``max_record_chunks`` bounds the total number of chunks read.
    """
    chunk_size = int(chunk_time_s * samplerate)
    timeout_chunks = max(1, int(silence_timeout_s / chunk_time_s))
    chunks: list[np.ndarray] = []
    speech_started = False
    silent_chunks = 0
    start_chunk = 0
    start_time = time_func()
    read_count = 0

    with stream_factory(
        samplerate=samplerate, channels=channels, dtype=datatype,
        device=device_index,
    ) as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            read_count += 1
            chunks.append(chunk)
            level = rms(chunk)
            if on_level is not None:
                on_level(level)

            if level > threshold:
                if not speech_started:
                    speech_started = True
                    start_chunk = len(chunks) - 1
                silent_chunks = 0
            elif speech_started:
                silent_chunks += 1
                if silent_chunks >= timeout_chunks:
                    break

            if not speech_started and (time_func() - start_time) > clear_timeout_s:
                chunks.clear()
                silent_chunks = 0
                start_time = time_func()

            if max_record_chunks is not None and read_count >= max_record_chunks:
                break

    empty = np.empty(0, dtype=datatype)
    if not speech_started or not chunks:
        return empty
    tolerance = timeout_chunks // 2
    begin = start_chunk - tolerance if start_chunk > tolerance else 0
    end = len(chunks) - tolerance if len(chunks) > tolerance else len(chunks)
    if begin >= end:
        audio = np.concatenate(chunks, axis=0)
    else:
        audio = np.concatenate(chunks[begin:end], axis=0)
    if rms(audio) < threshold:
        return empty
    return audio


class SpeechDetector:
    def __init__(
            self,
            model_path: Path,
            language: str = "en",
            beam_size: int = 5,
            show_levels: bool = False,
            device=None,
            stream_factory=sd.InputStream,
            time_func=time.time,
            model=None,
            model_backend: str = "auto",
            compute_type: str = "int8",
            chunk_time_s: float = 0.1,
            calibration_chunks: int = 20,
            guard_factor: float = 1.0,
        ):
        self.show_levels = show_levels
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")
        self.logger = getLogger(self.__class__.__name__)
        self.device = device or InputAudioDevice()
        self.logger.info(f"ℹ️  Using audio input device: {self.device.name} (index {self.device.index})")
        self._stream_factory = stream_factory
        self._time_func = time_func
        self._model = model
        self._model_backend = model_backend
        self._compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.logger.info("⚙️  Calibrating silence threshold...")
        self.silence_threshold, mean_rms, std_rms = compute_silence_threshold(
            self.device, chunk_time_s=chunk_time_s, chunks=calibration_chunks,
            guard_factor=guard_factor, stream_factory=stream_factory,
        )
        self.logger.info(f"✅ Silence threshold set to: {self.silence_threshold:.3f}, (Mean RMS: {mean_rms:.3f}, Std RMS: {std_rms:.3f})")

    def _ensure_model(self):
        """Load the Whisper model on first use and return it."""
        if self._model is None:
            self.logger.info(f"🚀 Loading Audio to Text model from {self._model_path} ...")
            self._model = WhisperModel(
                str(self._model_path),
                device=self._model_backend,
                compute_type=self._compute_type,
            )
            self.logger.info("✅ Model loaded.")
        return self._model

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
        Record audio from the microphone until a silence timeout is detected.

        Non-blocking, chunk-based recording. Returns an empty array when no
        speech was detected.
        """
        self.logger.info(f"🎤 Listening ... (Silence timeout: {silence_timeout_s} s). Press Ctrl+C to quit.")

        def on_level(level: float) -> None:
            line = format_level(level, self.silence_threshold, self.show_levels)
            if line:
                cprint(line, "reset", end="", flush=True)

        return capture_speech(
            threshold=self.silence_threshold,
            device_index=self.device.index,
            stream_factory=self._stream_factory,
            time_func=self._time_func,
            on_level=on_level,
            silence_timeout_s=silence_timeout_s,
            channels=channels,
            samplerate=samplerate,
            chunk_time_s=chunk_time_s,
            datatype=datatype,
            clear_timeout_s=clear_timeout_s,
        )

    def transcribe(self, audio: str | Path | io.BufferedIOBase) -> str:
        """Transcribe a WAV path or binary file-like object with Whisper."""
        self.logger.debug("🧠 Transcribing with Whisper...")
        model = self._ensure_model()
        segments, _ = model.transcribe(
            audio=audio,
            language=self.language,
            beam_size=self.beam_size
        )
        text = "".join(seg.text for seg in segments).strip()
        self.logger.debug(f"Text Length: {len(text)}")
        return text

    def listen_and_transcribe(
            self,
            silence_timeout_s: float,
            samplerate: int = 16000
        ) -> str:
        """Record speech and return its transcription (empty string if silent)."""
        audio = self.listen(silence_timeout_s=silence_timeout_s, samplerate=samplerate)
        if np.asarray(audio).size == 0:
            return ""
        return self.transcribe(audio_to_wav_buffer(audio, samplerate=samplerate))
