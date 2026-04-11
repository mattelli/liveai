from openai import OpenAI
from logging import getLogger
from src.audio_wrapper import SpeechDetector, save_audio_wav
from src.audio_generator import AudioGenerator
from src.common import SimpleNamespace

class LiveAi:
    
    def __init__(self, config: SimpleNamespace):
        self.logger = getLogger(self.__class__.__name__)
        self.config = config
        
        self.audio_in = SpeechDetector(model_path=config.audio_input.model_path)
        self.audio_out = AudioGenerator(model_path=config.audio_output.model_path)
        self.logger.info("🚀 Connecting to LLM Model ...")
        self.lm_client = OpenAI(
            base_url=config.llm_model.url, 
            api_key=config.llm_model.api_key
        )
    
    def llm_chat(self, prompt: str) -> str:
        """
        Use OpenAI API to get a response from the LLM model.
        """
        self.logger.info(f"Sending to [🤖: {self.config.llm_model.model}] ...")
        self.logger.info(f"Prompt: {prompt}")
        resp = self.lm_client.chat.completions.create(
            model=self.config.llm_model.model,
            messages=[
                {"role": "system", "content": self.config.llm_model.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.llm_model.temperature,
        )
        answer = resp.choices[0].message.content.strip()
        self.logger.info(f"[LLM] {answer}")
        return answer
    
    def run(self):
        self.logger.info("Ready. Press Ctrl+C to quit.")
        while True:
            try:
                self.logger.info("▶️  Press Enter to record...")
                input()
                audio = self.audio_in.record_audio(
                    duration=self.config.audio_input.record_seconds,
                    silence_threshold=self.config.audio_input.silence_threshold,
                    silence_timeout=self.config.audio_input.silence_timeout
                )
                if audio is None:
                    self.logger.warning("⚠️ No audio captured.")
                    continue
                prompt = self.audio_in.transcribe(save_audio_wav(audio))
                if not prompt:
                    self.logger.warning("⚠️ Empty transcription, skipping.")
                    continue
                self.audio_out.streaming_audio(self.llm_chat(prompt))
            except KeyboardInterrupt, EOFError:
                self.logger.info("👋 Exiting.")
                break
