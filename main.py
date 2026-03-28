
import subprocess
import sys
import tempfile
import sounddevice as sd
import soundfile as sf
import numpy as np
from queue import Queue
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig
from openai import OpenAI
from res import VOICES_DIR, WHISPER_MODEL_DIR


# ---------- CONFIG ----------
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_MODEL = "qwen3.5-0.8b"  # e.g. "mistral"
WHISPER_MODEL_PATH = WHISPER_MODEL_DIR  # folder or ggml/gguf
PIPER_MODEL_PATH = VOICES_DIR / "en_US-hfc_female-medium.onnx"
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 6
# ----------------------------

audio_q = Queue()

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=1.0,  # twice as slow
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


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_q.put(indata.copy())


def record_chunk(seconds=RECORD_SECONDS):
    print(f"\n[🎙] Recording for {seconds} seconds...")
    audio_q.queue.clear()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        sd.sleep(int(seconds * 1000))
    frames = []
    while not audio_q.empty():
        frames.append(audio_q.get())
    if not frames:
        return None

    audio = np.concatenate(frames, axis=0)
    return audio


def save_wav(audio: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return tmp.name


def transcribe_whisper(model: WhisperModel, wav_path: str) -> str:
    print("[🧠] Transcribing with Whisper...")
    segments, info = model.transcribe(wav_path, beam_size=5)
    text = "".join(seg.text for seg in segments).strip()
    print(f"[STT] {text}")
    return text


def lmstudio_chat(client: OpenAI, prompt: str) -> str:
    print("[🤖] Sending to LM Studio...")
    resp = client.chat.completions.create(
        model=LMSTUDIO_MODEL,
        messages=[
            {"role": "system", "content": "You are a friendly AI assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    answer = resp.choices[0].message.content.strip()
    print(f"[LLM] {answer}")
    return answer


def speak_piper(text: str):
    print("[🔊] Speaking with Piper...")
    proc = subprocess.Popen(
        ["piper", "--model", PIPER_MODEL_PATH, "--sentence_silence", "0.3"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    out, _ = proc.communicate(input=text.encode("utf-8"))

    # Play raw audio from Piper (16kHz, 16-bit, mono)
    audio = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio, 16000)
    sd.wait()
    
def streaming_audio(text: str):
    stream.start()
    for chunk in voice.synthesize(text, syn_config=syn_config):
        stream.write(chunk.audio_int16_bytes)
    stream.stop()
    stream.close()


def main():
    print("[🚀] Loading Whisper model...")
    whisper_model = WhisperModel(str(WHISPER_MODEL_PATH), device="cpu", compute_type="int8")

    print("[🚀] Connecting to LM Studio...")
    client = OpenAI(
        base_url=LMSTUDIO_BASE_URL,
        api_key="lm-studio",  # dummy if auth disabled
    )

    print("\nReady. Press Enter to speak, or Ctrl+C to quit.")
    while True:
        try:
            input("\n[⏺] Press Enter to record...")
            audio = record_chunk()
            if audio is None:
                print("[!] No audio captured.")
                continue

            wav_path = save_wav(audio, SAMPLE_RATE)
            text = transcribe_whisper(whisper_model, wav_path)
            if not text:
                print("[!] Empty transcription, skipping.")
                continue

            reply = lmstudio_chat(client, text)
            streaming_audio(reply)

        except KeyboardInterrupt:
            print("\n[👋] Exiting.")
            break


if __name__ == "__main__":
    main()
