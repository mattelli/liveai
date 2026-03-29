import wave
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig
from res import VOICES_DIR

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=1.5,  # twice as slow
    noise_scale=1.0,  # more audio variation
    noise_w_scale=1.0,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)

voice = PiperVoice.load(VOICES_DIR / "en_US-hfc_female-medium.onnx")

stream = sd.RawOutputStream(
    samplerate=22050,
    channels=1,
    dtype="int16"
)


def save_audio(text: str):
    with wave.open("test.wav", "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=syn_config)

    
def streaming_audio(text: str):
    stream.start()
    for chunk in voice.synthesize(text, syn_config=syn_config):
        stream.write(chunk.audio_int16_bytes)
    stream.stop()
    stream.close()


if __name__ == "__main__":
    text = "Hello Jenilyn you are original!"
    save_audio(text)
    # streaming_audio(text)
