#!/usr/bin/env python3
"""
Simple example: record audio and transcribe with faster-whisper.
"""
import logging
from src.audio_wrapper import save_audio_wav, SpeechDetector


def main():
    # Settings
    DURATION = 3.0  # seconds
    SAMPLERATE = 16000 
    MODEL_PATH = "res/faster-whisper-medium"
    logging.basicConfig(level=logging.INFO)
    
    # Initialize audio wrapper
    sd = SpeechDetector(model_path=MODEL_PATH)
    audio = sd.record_audio(duration=DURATION)
    wav_path = save_audio_wav(audio=audio, samplerate=SAMPLERATE)
    text = sd.transcribe(wav_path)

    print(f"\n{'='*50}")
    print(f"[📢] Transcription:")
    print(f"{text}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()