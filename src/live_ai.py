import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
from queue import Queue
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig
from openai import OpenAI
from types import SimpleNamespace
from logging import getLogger

syn_config = SynthesisConfig(
    volume=0.5,  # half as loud
    length_scale=1.0,  # twice as slow
    noise_scale=1.0,  # more audio variation
    noise_w_scale=1.0,  # more speaking variation
    normalize_audio=False, # use raw audio from voice
)

class LiveAi:
    
    def __init__(self, config: SimpleNamespace):
        self.logger = getLogger(self.__class__.__name__)
        self.config = config
        
        self.logger.info("[🚀] Loading Whisper model...")
        self.whisper_model = WhisperModel(
            str(config.whisper.model_path), 
            device="cpu", 
            compute_type=config.whisper.datatype
        )
        self.audio_q = Queue()
        
        self.logger.info("[🚀] Connecting to LLM Model ...")
        self.lm_client = OpenAI(
            base_url=config.lm_studio.url, 
            api_key=config.lm_studio.api_key
        )
        
        self.logger.info("[🚀] Loading Piper model...")
        self.voice = PiperVoice.load(config.piper.model_path)
        
        
        self.stream = sd.RawOutputStream(
            samplerate=config.piper.samplerate,
            channels=config.piper.channels,
            dtype=config.piper.datatype
        )
        
    def record_chunk(self) -> np.ndarray:
        seconds = self.config.whisper.record_seconds
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                 self.logger.warning(f"[⚠️] Audio callback status: {status}", file=sys.stderr)
            self.audio_q.put(indata.copy())
        
        self.logger.info(f"[🎙] Recording for {seconds} seconds...")
        self.audio_q.queue.clear()
        with sd.InputStream(
            device=None,
            samplerate=self.config.whisper.samplerate, 
            channels=self.config.whisper.channels, 
            callback=audio_callback
            ):
            sd.sleep(int(seconds * 1000))
        frames = []
        while not self.audio_q.empty():
            frames.append(self.audio_q.get())
        if not frames:
            self.logger.warning("[!] No audio frames captured!")
            return None
        audio = np.concatenate(frames, axis=0)
        if audio.max() == 0 and audio.min() == 0:
            self.logger.warning("[!] Audio is completely silent - check microphone!")
        return audio
        
    def save_wav(self, audio: np.ndarray) -> str:
        audio_file = open(file="input_audio.wav", mode="wb")
        sf.write(
            file=audio_file,
            data=audio,
            samplerate=self.config.whisper.samplerate
        )
        return audio_file.name
    
    def transcribe_whisper(self, wav_path: str) -> str:
        self.logger.info("[🧠] Transcribing with Whisper...")
        segments, info = self.whisper_model.transcribe(
            audio=wav_path,
            language=self.config.whisper.language,
            beam_size=self.config.whisper.beam_size)
        text = "".join(seg.text for seg in segments).strip()
        self.logger.info(f"Whisper info: {info}")
        return text
    
    def lm_studio_chat(self, prompt: str) -> str:
        self.logger.info("[🤖] Sending to LM Studio...")
        self.logger.info(f"Prompt: {prompt}")
        resp = self.lm_client.chat.completions.create(
            model=self.config.lm_studio.model,
            messages=[
                {"role": "system", "content": self.config.lm_studio.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.lm_studio.temperature,
        )
        answer = resp.choices[0].message.content.strip()
        self.logger.info(f"[LLM] {answer}")
        return answer

    def streaming_audio(self, text: str):
        self.stream.start()
        for chunk in self.voice.synthesize(text, syn_config=syn_config):
            self.stream.write(chunk.audio_int16_bytes)
        self.stream.stop()
        
    def run(self):
        self.logger.info("Ready. Press Enter to speak, or Ctrl+C to quit.")
        while True:
            try:
                self.logger.info("[⏺] Press Enter to record...")
                input()
                audio = self.record_chunk()
                if audio is None:
                    self.logger.warning("[!] No audio captured.")
                    continue
                text = self.transcribe_whisper(self.save_wav(audio))
                if not text:
                    self.logger.warning("[!] Empty transcription, skipping.")
                    continue
                self.streaming_audio(self.lm_studio_chat(text))
            except KeyboardInterrupt:
                self.logger.info("[👋] Exiting.")
                break
