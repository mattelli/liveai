#!/usr/bin/env python3
"""
Simple example: detect speech and transcribe it.
"""
import logging
from pathlib import Path
from src.audio_wrapper import audio_to_wav, SpeechDetector


def main():
    # Settings
    DURATION = 3.0  # seconds
    SAMPLERATE = 16000 
    MODEL_PATH = "res/faster-whisper-medium"
    logging.basicConfig(level=logging.INFO)
    
    # Initialize audio wrapper
    detector = SpeechDetector(model_path=Path(MODEL_PATH))
    audio = detector.listen(duration=DURATION)
    wav_path = audio_to_wav(audio=audio, samplerate=SAMPLERATE)
    text = detector.transcribe(wav_path)

    print(f"\n{'='*50}")
    print(f"[📢] Transcription:")
    print(f"{text}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()