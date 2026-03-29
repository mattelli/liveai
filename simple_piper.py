#!/usr/bin/env python3
"""
Simple example: generate audio from text with piper.
"""
import logging
from pathlib import Path
from src.audio_generator import AudioGenerator
MODEL_PATH = "res/voices/en_US-hfc_female-medium.onnx"

text = "Hello Jenilyn you are original!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    audio_gen = AudioGenerator(model_path=Path(MODEL_PATH))
    audio_gen.save_audio(text, file_path=Path("output.wav"))
    audio_gen.streaming_audio(text)
