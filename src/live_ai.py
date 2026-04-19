import lmstudio as lms
from lmstudio.sync_api import PredictionStream
from logging import getLogger
from src.audio_wrapper import SpeechDetector
from src.audio_generator import SpeechGenerator
from src.common import SimpleNamespace


class Brain:
    def __init__(self, config: SimpleNamespace):
        self.logger = getLogger(self.__class__.__name__)
        self.config = config
        self.logger.info(f"🚀 Connecting to LLM Model '{config.brain.model}' ...")
        if str(config.brain.api_key).lower() != "local":
            lms.configure_default_client(api_host=config.brain.api_key)
        else:
            llm_only = lms.list_downloaded_models("llm")
            for model in llm_only:
                if config.brain.model == model.model_key:
                    break
            else:
                raise ValueError(f"The configured Model '{config.brain.model}' is not available.")
        self.model = lms.llm(config.brain.model)
        self.chat = lms.Chat(initial_prompt=self.config.brain.system_prompt)
        self._chat_cfg = {
            "temperature": self.config.brain.temperature,
            "maxTokens": self.config.brain.max_tokens,
        }
        self.logger.info(f"✅ LLM Model '{self.config.brain.model}' connected.")
    
    def stream_response(self, prompt: str) -> PredictionStream:
        self.logger.info(f"Sending to [🤖: {self.config.brain.model}] ...")
        self.logger.info(f"Prompt:\n{prompt}")
        self.chat.add_user_message(prompt)
        resp_stream = self.model.respond_stream(
            self.chat,
            on_message=self.chat.append,
            config=self._chat_cfg
        )
        return resp_stream
    
    def get_response(self, prompt: str) -> str:
        resp_stream = self.stream_response(prompt)
        answer:list[str] = [fragment.content for fragment in resp_stream]
        response = " ".join(answer).strip()
        self.logger.info(f"Response:\n{response}")
        return response


class LiveAi:
    
    def __init__(self, config: SimpleNamespace):
        self.logger = getLogger(self.__class__.__name__)
        self.config = config
        
        self.detector = SpeechDetector(
            model_path=config.speech_det.model_path, 
            language=config.speech_det.language
        )
        self.speech = SpeechGenerator(
            model_path=config.speech_gen.model_path,
            samplerate=config.speech_gen.samplerate
        )
        self.brain = Brain(config=config)

    def run(self):
        self.logger.info("Ready. Press Ctrl+C to quit.")
        while True:
            try:
                prompt = self.detector.listen_and_transcribe(
                    silence_timeout_s=self.config.speech_det.silence_timeout_s,
                )
                if not prompt:
                    self.logger.info("Empty transcription, skipping.")
                    continue
                response = self.brain.get_response(prompt)
                self.speech.speak(response)
            except KeyboardInterrupt:
                self.logger.info("👋 Exiting.")
                break
