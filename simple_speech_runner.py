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
    DURATION = 3.0  # seconds
    SAMPLERATE = 16000 
    MODEL_PATH = "res/faster-whisper-medium"
    VOICE_MODEL_PATH = "res/voices/en_US-hfc_female-medium.onnx"
    logging.basicConfig(level=logging.INFO)
    
    # Initialize audio wrapper
    sd = SpeechDetector(model_path=Path(MODEL_PATH))
    sg = SpeechGenerator(model_path=Path(VOICE_MODEL_PATH))
    
    audio_in = sd.listen(duration=DURATION)
    wav_path = audio_to_wav(audio=audio_in, samplerate=SAMPLERATE)
    text = sd.transcribe(wav_path)
    sg.speak(text)

    print(f"\n{'='*50}")
    print(f"[📢] Transcription:")
    print(f"{text}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()