import lmstudio as lms
from lmstudio.sync_api import PredictionStream
from typing import Callable
from logging import getLogger
from src.audio_wrapper import SpeechDetector
from src.audio_generator import SpeechGenerator
from src.common import SimpleNamespace
from src.tools import (
    current_date_time, 
    extract_news, 
    capture_image, 
    save_financial_chart
)


def print_fragment(fragment, round_index=0):
    # .act() supplies the round index as the second parameter
    # Setting a default value means the callback is also
    # compatible with .complete() and .respond().
    print(fragment.content, end="", flush=True)


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
        try:
            current_model = lms.llm()
            current_model_info = current_model.get_info()
            if current_model_info.model_key != config.brain.model:
                self.logger.info(f"Unloading Model [{current_model_info.model_key}]")
                current_model.unload()
        except lms.LMStudioClientError:
            pass
            
        self.model = lms.llm(config.brain.model)
        self.chat = lms.Chat(initial_prompt=self.config.brain.system_prompt)
        self._chat_cfg = {
            "temperature": self.config.brain.temperature,
            "maxTokens": self.config.brain.max_tokens,
        }
        self.tools:list[Callable] = [
            self.model_handle_image, 
            current_date_time, 
            extract_news, 
            capture_image,
            save_financial_chart
        ]
        self.logger.info(f"✅ LLM Model '{self.config.brain.model}' connected.")
    
    def model_stream_response(self, prompt: str) -> PredictionStream:
        self.logger.info(f"Sending to Model [🤖: {self.config.brain.model}] ...")
        self.logger.debug(f"Prompt:\n{prompt}")
        self.chat.add_user_message(prompt)
        resp_stream = self.model.respond_stream(
            self.chat,
            on_message=self.chat.append,
            config=self._chat_cfg,
        )
        return resp_stream
    
    def model_handle_image(self, prompt:str, image:str) -> str:
        """Handle images to text, this allow to process an image such as describe it"""
        print("calling: model_handle_image")
        if not prompt:
            prompt = "describe the image"
        image_handle = lms.prepare_image(image)
        self.chat.add_user_message(prompt, images=[image_handle])
        prediction  = self.model.respond(self.chat)
        return prediction.content
    
    def model_response(self, prompt: str) -> str:
        resp_stream = self.model_stream_response(prompt)
        answer:list[str] = [fragment.content for fragment in resp_stream]
        response = " ".join(answer).strip()
        self.logger.debug(f"Response:\n{response}")
        return response
    
    def agent_response(self, prompt: str):
        self.logger.info(f"Sending to Agent [🤖: {self.config.brain.model}] ...")
        self.logger.debug(f"Prompt:\n{prompt}")
        self.chat.add_user_message(prompt)
        self.model.act(
            self.chat,
            self.tools,
            on_message=self.chat.append,
            config=self._chat_cfg,
            on_prediction_fragment=print_fragment,
        )


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
        print("\nReady. Press Ctrl+C to quit.")
        while True:
            try:
                prompt = self.detector.listen_and_transcribe(
                    silence_timeout_s=self.config.speech_det.silence_timeout_s,
                )
                if not prompt:
                    self.logger.info("Empty transcription, skipping.")
                    continue
                print(f"\nYou: {prompt}")
                response = self.brain.model_response(prompt)
                print(f"\n🤖 [{self.config.brain.model}]: {response}")
                self.speech.speak(response)
                print("\n---\n")
            except KeyboardInterrupt:
                print("\n👋 Exiting.")
                break
