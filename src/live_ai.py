import pathlib
import lmstudio as lms
from lmstudio.sync_api import PredictionStream
from typing import Callable, Literal
from logging import getLogger
from datetime import datetime, timezone
from src.audio_wrapper import SpeechDetector
from src.audio_generator import SpeechGenerator
from src.common import SimpleNamespace, dump_func_call
from src.tools import (
    exit_gracefully, system_info, current_date_time,
    read_text_file, write_text_file, get_working_directory, run_command,
    webpage_headlines, web_search, capture_image, save_financial_chart,
    search_pattern_in_files, replace_pattern_in_file, amend_text_file
)

Role = Literal['system', 'user', 'assistant']

class Brain:
    """
    Brain class that connects to the LLM model.
    """
    def __init__(self, config: SimpleNamespace):
        self.logger = getLogger(self.__class__.__name__)
        self.config = config
        self.logger.info("🚀 Connecting to LLM Model '%s' ...", config.brain.model)
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
                self.logger.info("Unloading Model [%s]", current_model_info.model_key)
                current_model.unload()
        except lms.LMStudioClientError:
            pass

        self.model = lms.llm(config.brain.model)
        self.info = self.model.get_info()
        self.context_window = int(float(self.config.brain.context_ratio)*self.info.context_length)
        self.chat = lms.Chat(initial_prompt=self.config.brain.system_prompt)
        self._chat_cfg = {
            "temperature": self.config.brain.temperature,
            "maxTokens": self.context_window,
        }
        self.logger.info("✅ LLM Model '%s' connected.", self.config.brain.model)

    def _add_user_prompt(self, prompt: str):
        self.chat.add_user_message(prompt)
        self.logger.debug("Prompt:\n%s", prompt)

    def model_fragments(self, prompt: str) -> PredictionStream:
        """
        Get response fragments as stream from the model for a given prompt.
        Tools are not supported in this method.
        """
        self.logger.info("Sending to Model [🤖: %s] ...", self.config.brain.model)
        self._add_user_prompt(prompt)
        resp_stream = self.model.respond_stream(
            self.chat,
            on_message=self.chat.append,
            config=self._chat_cfg,
        )
        return resp_stream

    @staticmethod
    def print_fragment(fragment):
        """
        Print a response fragment to the console.
        This can be used as a callback for streaming responses.
        """
        print(fragment.content, end="", flush=True)

    def _get_messages(self) -> list[dict]:
        try:
            return self.chat._get_history().get("messages", [])
        except:
            return []

    @staticmethod
    def _get_messages_from(role: Role, msg:list) -> list[str]:
        try:
            return [m["content"][0]["text"] for m in msg if m["role"] == role]
        except:
            return []

    def _clear_chat(self, all:bool=False) -> bool:
        try:
            msg_list = self._get_messages()
            self.chat._messages.clear()
            if not all:
                sys_msg = self._get_messages_from(role="system", msg=msg_list)
                usr_msg = self._get_messages_from(role="user", msg=msg_list)
                agt_msg = self._get_messages_from(role="assistant", msg=msg_list)
                self.chat.add_system_prompt(sys_msg[0])
                self.chat.add_user_message(self.recall_personal())
                self.chat.add_assistant_response("acknowledge")
                self.chat.add_user_message(usr_msg[-1])
                self.chat.add_assistant_response(agt_msg[-1])
            return True
        except:
            return False

    ## TOOLS ##

    @dump_func_call
    def model_info(self) -> str:
        """
        Returns the loaded model info.
        Useful when user ask for information about the loaded model.
        """
        return str(self.info)

    @dump_func_call
    def image_response(self, image:str, prompt:str = "describe the image") -> str:
        """
        Provide text response form an input image.
        Useful when user asks for getting a description of the input image content. 
        """
        if not self.info.vision:
            return "Cannot handle images"
        image_handle = lms.prepare_image(image)
        self.chat.add_user_message(prompt, images=[image_handle])
        prediction  = self.model.respond(self.chat)
        return prediction.content

    @dump_func_call
    def allow_speech(self, enable:bool) -> str:
        """
        Enable or disable speech for Agent responses.
        When enabled, the Agent will use the Speech Generator.
        It can be toggled on the fly during the conversation.
        """
        if enable and not hasattr(self, "speech"):
            self.logger.info("Initializing Speech Generator ...")
            self.speech = SpeechGenerator(
                model_path=self.config.speech_gen.model_path,
                samplerate=self.config.speech_gen.samplerate
            )
            return "Speech Generator initialized"
        elif not enable and hasattr(self, "speech"):
            self.logger.info("Releasing Speech Generator ...")
            del self.speech
            return "Speech Generator released"

    @dump_func_call
    def store_personal(self) ->  str:
        """
        Stores the user personal content to a text file.
        Useful when the users explicitly mention the prompted text is "personal". 
        """
        history = pathlib.Path(self.config.brain.history_path).resolve()
        now = datetime.now(tz=timezone.utc)
        timestamp = now.strftime("%d/%m/%Y, %H:%M")
        messages = self._get_messages()
        user_msgs = self._get_messages_from(role="user", msg=messages)
        if user_msgs and user_msgs[0] == self.config.agent.bootstrap_prompt:
            user_msgs = user_msgs[1:]
        else:
            user_msgs = user_msgs
        selected_msgs = [m for m in user_msgs if "personal" in m.lower()]
        user_prompt = f"At {timestamp} user wrote: {selected_msgs[-1]} \n"
        if not history.exists():
            history.write_text(user_prompt, encoding="utf-8")
        else:
            with open(file=history, mode="a", encoding="utf-8") as f:
                f.write(user_prompt)
        return f"Conversation written to {history}"

    @dump_func_call
    def recall_personal(self) -> str:
        """
        Read users personal data from file.
        Useful when the users ask to recall their personal data. 
        """
        history = pathlib.Path(self.config.brain.history_path).resolve()
        if not history.exists():
            return "No personal data found!"
        else:
            return history.read_text(encoding="utf-8")
        

class Agent(Brain):
    """
    Agent class that extends the Brain with tool usage capabilities and message history.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self._fragments_buffer:list[str] = []
        self.logger.info(
            "✅ Agent with Model '%s' initialized. Setting up tools ...", 
            self.config.brain.model
        )
        self.tools:list[Callable] = [
            self.model_info, self.image_response, self.allow_speech, 
            self.store_personal, self.recall_personal, 
            exit_gracefully, system_info, current_date_time,
            read_text_file, write_text_file, get_working_directory, run_command,
            webpage_headlines, web_search, capture_image, save_financial_chart,
            search_pattern_in_files, replace_pattern_in_file, amend_text_file
        ]
        self.logger.info(
            "✅ Agent with Model '%s' is ready with [%d] tools.", 
            self.config.brain.model, len(self.tools)
        )
        if self.config.agent.bootstrap_prompt:
            self.act(self.config.agent.bootstrap_prompt)

    def _handle_fragment(self, fragment, round_index=0):
        self._fragments_buffer.append(fragment.content)
        print(fragment.content, end="", flush=True)

    def _handle_round_end(self, round_index=0):
        formatted = self.model.apply_prompt_template(self.chat)
        token_count = len(self.model.tokenize(formatted))
        context_usage_percentage = float(token_count/self.context_window) * 100
        print(f"  [context: {context_usage_percentage:.2f} %]")
        response = " ".join(self._fragments_buffer).strip()
        if hasattr(self, "speech"):
            self.speech.speak(response)
        self._fragments_buffer.clear()
        if context_usage_percentage > float(self.config.agent.refresh_context) * 100:
            self._clear_chat()

    def act(self, prompt: str):
        """
        Perform an action based on the input prompt. 
        The response may involve using tools or just generating a text.
        """
        self.logger.info("Sending to Agent [🤖: %s] ...", self.config.brain.model)
        self._add_user_prompt(prompt)
        self.model.act(
            self.chat,
            self.tools,
            on_message=self.chat.append,
            config=self._chat_cfg,
            on_prediction_fragment=self._handle_fragment,
            on_round_end=self._handle_round_end
        )


class LiveAi:
    """
    Interactive AI agent that listens to user input via speech and tools.
    """
    def __init__(self, config: SimpleNamespace):
        self.logger = getLogger(self.__class__.__name__)
        self.config = config
        self.detector = SpeechDetector(
            model_path=config.speech_det.model_path, 
            language=config.speech_det.language
        )
        self.agent = Agent(config=config)

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
                print(f"\n🤖 [{self.config.brain.model}]: ", end="", flush=True)
                self.agent.act(prompt)
                print("\n---\n")
            except KeyboardInterrupt:
                print("\n👋 Exiting.")
                break
