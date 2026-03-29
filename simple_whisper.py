#!/usr/bin/env python3
"""Simple example: record audio and transcribe with faster-whisper."""

import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel


def list_devices():
    """List available audio input devices."""
    print("\n[🎤] Available input devices:")
    devices = sd.query_devices(kind='input')
    print(devices)
    return devices


def record_audio(duration: float = 3.0, samplerate: int = 16000) -> np.ndarray:
    """Record audio from microphone."""
    print(f"\n[🎙] Recording for {duration} seconds...")
    print("[⏹] Press Ctrl+C to stop early")
    
    audio = sd.rec(
        frames=int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype='float32',
        device=None  # Use default input device
    )
    
    sd.wait()  # Wait until recording is finished
    print(f"[✅] Recorded {len(audio)} samples")
    if audio.max() == 0 and audio.min() == 0:
        print("[!] Audio is completely silent - check microphone!")
    return audio


def get_audio_tone(duration: float = 2.0, samplerate: int = 16000, frequency: float = 440.0) -> np.ndarray:
    """Generate a test audio tone (sine wave)."""
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio.astype('float32')


def save_temp_wav(audio: np.ndarray, samplerate: int) -> str:
    """Save audio to a temporary WAV file."""
    audio_file = open(file="input_audio.wav", mode="wb")
    sf.write(audio_file, audio, samplerate)
    return audio_file.name


def transcribe(wav_path: str, model_path: str = "base") -> str:
    """Transcribe audio using faster-whisper."""
    print(f"\n[🧠] Loading Whisper model: {model_path}...")
    
    model = WhisperModel(model_path, device="cpu", compute_type="int16")
    
    print("[📝] Transcribing...")
    segments, info = model.transcribe(audio=wav_path, beam_size=5, language="en")
    
    print(f"[ℹ️] Detected language: {info.language} (probability: {info.language_probability:.2f})")
    
    text = "".join(seg.text for seg in segments).strip()
    return text


def main():
    # Settings
    DURATION = 3.0  # seconds
    SAMPLERATE = 16000  # Hz (lower = faster, 16k is fine for Whisper)
    
    # Step 1: List available devices
    # list_devices()
    
    # Step 2: Record audio
    # audio = record_audio(DURATION, SAMPLERATE)
    audio = get_audio_tone(duration=DURATION, samplerate=SAMPLERATE, frequency=440.0)  # For testing with tone
    
    # Step 3: Save to temp file
    wav_path = save_temp_wav(audio, SAMPLERATE)
    # wav_path = "test.wav"  # For testing with pre-recorded audio
    print(f"[💾] Saved to: {wav_path}")
    
    # Step 4: Transcribe
    text = transcribe(wav_path)
    
    print(f"\n{'='*50}")
    print(f"[📢] Transcription:")
    print(f"{text}")
    print(f"{'='*50}")
    
    # Cleanup
    # os.unlink(wav_path)


if __name__ == "__main__":
    main()