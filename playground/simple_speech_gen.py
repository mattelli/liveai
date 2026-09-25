#!/usr/bin/env python3
"""
Simple example: generate speech from text.
"""
import logging
from pathlib import Path

import sys
MAIN_PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAIN_PATH))

from src.audio_generator import SpeechGenerator

MODEL_PATH = MAIN_PATH / ".res" / "voices" / "en_US-hfc_female-medium.onnx"

text = "Hello Jenilyn you are original!"

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    speech = SpeechGenerator(model_path=Path(MODEL_PATH))
    speech.to_wav(text, file_path=Path("output.wav"))
    speech.speak(text)
