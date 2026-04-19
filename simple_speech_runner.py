#!/usr/bin/env python3
"""
Simple example: detect speech, transcribe it and generate speech.
"""
import logging
from pathlib import Path
from src.audio_wrapper import audio_to_wav, SpeechDetector
from src.audio_generator import SpeechGenerator


def main():
    # Settings
    TIMEOUT_S = 1.5
    MODEL_PATH = "res/faster-whisper-medium"
    VOICE_MODEL_PATH = "res/voices/en_US-hfc_female-medium.onnx"
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize audio wrapper
    sd = SpeechDetector(model_path=Path(MODEL_PATH))
    sg = SpeechGenerator(model_path=Path(VOICE_MODEL_PATH))
    
    print("Ready. Press Ctrl+C to quit.")
    while True:
        try:
            audio = sd.listen(silence_timeout_s=TIMEOUT_S)
            text = sd.transcribe(audio_to_wav(audio))
            if not text:
                sd.logger.warning("⚠️  Empty transcription, skipping.")
                continue
            print(f"\n{'='*50}")
            print(f"[📢] Transcription:")
            print(f"{text}")
            print(f"{'='*50}")
            sg.speak(text)
        except KeyboardInterrupt:
            sd.logger.info("👋 Exiting.")
            break


if __name__ == "__main__":
    main()