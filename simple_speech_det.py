#!/usr/bin/env python3
"""
Simple example: detect speech and transcribe it.
"""
import logging
from pathlib import Path
from src.audio_wrapper import audio_to_wav, SpeechDetector


def main():
    DURATION = 20.0  # seconds
    MODEL_PATH = "res/faster-whisper-medium"
    logging.basicConfig(level=logging.INFO)
    detector = SpeechDetector(model_path=Path(MODEL_PATH), silence_timeout_s=2.0)
    
    print("Ready. Press Ctrl+C to quit.")
    while True:
        try:
            audio = detector.listen(duration_s=DURATION)
            text = detector.transcribe(audio_to_wav(audio))
            if not text:
                detector.logger.warning("⚠️  Empty transcription, skipping.")
                continue
            print(f"\n{'='*50}")
            print(f"[📢] Transcription:")
            print(f"{text}")
            print(f"{'='*50}")
        except KeyboardInterrupt, EOFError:
            detector.logger.info("👋 Exiting.")
            break


if __name__ == "__main__":
    main()