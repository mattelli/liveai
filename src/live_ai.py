from openai import OpenAI
from logging import getLogger
from src.audio_wrapper import SpeechDetector
from src.audio_generator import SpeechGenerator
from src.common import SimpleNamespace

class LiveAi:
    
    def __init__(self, config: SimpleNamespace):
        self.logger = getLogger(self.__class__.__name__)
        self.config = config
        
        self.detector = SpeechDetector(
            model_path=config.speech_det.model_path, 
            silence_timeout_s=config.speech_det.silence_timeout,
            language=config.speech_det.language
            )
        self.speech = SpeechGenerator(
            model_path=config.speech_gen.model_path,
            samplerate=config.speech_gen.samplerate
        )
        self.logger.info("🚀 Connecting to LLM Model ...")
        self.lm_client = OpenAI(
            base_url=config.brain.url, 
            api_key=config.brain.api_key
        )
    
    def llm_chat(self, prompt: str) -> str:
        """
        Use OpenAI API to get a response from the LLM model.
        """
        self.logger.info(f"Sending to [🤖: {self.config.brain.model}] ...")
        self.logger.info(f"Prompt: {prompt}")
        resp = self.lm_client.chat.completions.create(
            model=self.config.brain.model,
            messages=[
                {"role": "system", "content": self.config.brain.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.brain.temperature,
        )
        answer = resp.choices[0].message.content.strip()
        self.logger.info(f"[LLM] {answer}")
        return answer

    def run(self):
        self.logger.info("Ready. Press Ctrl+C to quit.")
        while True:
            try:
                prompt = self.detector.listen_and_transcribe(
                    duration_s=self.config.speech_det.listen_time_s
                )
                if not prompt:
                    self.logger.info("Empty transcription, skipping.")
                    continue
                self.speech.speak(self.llm_chat(prompt))
            except KeyboardInterrupt, EOFError:
                self.logger.info("👋 Exiting.")
                break
