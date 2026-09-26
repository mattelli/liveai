import pathlib
import lmstudio as lms
from lmstudio.sync_api import PredictionStream
from typing import Callable, Literal
from logging import getLogger
from datetime import datetime, timezone
from src.audio_wrapper import SpeechDetector
from src.audio_generator import SpeechGenerator
from src.common import SimpleNamespace, dump_func_call, cprint, StreamPrinter
from src.tools import (
    exit_gracefully, system_info, current_date_time, get_working_directory,
    read_text_file, write_text_file, run_command,
    web_search, webpage_headlines, webpage_content, capture_image, 
    fetch_financial_data, save_financial_chart, 
    search_pattern_in_files, replace_string_in_file, search_in_file
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

        self._model = lms.llm(config.brain.model)
        self._info = self._model.get_info()
        self._context_window = int(float(self.config.brain.context_ratio)*self._info.context_length)
        self._chat = lms.Chat(initial_prompt=self.config.brain.system_prompt)
        self._stream = StreamPrinter(text_color="cyan")
        self._chat_cfg = {
            "temperature": self.config.brain.temperature,
            "maxTokens": self._context_window,
        }
        self.logger.info("✅ LLM Model '%s' connected.", self.config.brain.model)

    def _add_user_prompt(self, prompt: str):
        self._chat.add_user_message(prompt)
        self.logger.debug("Prompt:\n%s", prompt)

    def model_fragments(self, prompt: str) -> PredictionStream:
        """
        Get response fragments as stream from the model for a given prompt.
        Tools are not supported in this method.
        """
        self.logger.info("Sending to Model [🤖: %s] ...", self.config.brain.model)
        self._add_user_prompt(prompt)
        resp_stream = self._model.respond_stream(
            self._chat,
            on_message=self._chat.append,
            config=self._chat_cfg,
        )
        return resp_stream

    def print_fragment(self, fragment):
        """
        Print a response fragment to the console.
        Fenced code blocks are syntax-highlighted; plain text is cyan.
        This can be used as a callback for streaming responses.
        """
        self._stream.feed(fragment.content)

    def flush_stream(self) -> None:
        """Flush any buffered text left after a streaming response ends."""
        self._stream.flush()

    def _get_messages(self) -> list[dict]:
        try:
            return self._chat._get_history().get("messages", [])
        except:
            return []

    @staticmethod
    def _get_messages_from(role: Role, msg:list) -> list[str]:
        try:
            return [m["content"][0]["text"] for m in msg if m["role"] == role]
        except:
            return []

    ## TOOLS ##

    @dump_func_call
    def clear_chat(self) -> bool:
        """
        Useful when user ask for cleaning conversation history.
        """
        try:
            msg_list = self._get_messages()
            self._chat._messages.clear()
            sys_msg = self._get_messages_from(role="system", msg=msg_list)
            usr_msg = self._get_messages_from(role="user", msg=msg_list)
            system_prompt = sys_msg[0] if sys_msg else self.config.brain.system_prompt
            self._chat.add_system_prompt(system_prompt)
            self._chat.add_user_message(self.recall_personal())
            if usr_msg:
                self._chat.add_user_message(usr_msg[-1])
            return "Chat cleared successfully"
        except Exception as e:
            return f"Error: {e}"

    @dump_func_call
    def model_info(self) -> str:
        """
        Returns the loaded model info.
        Useful when user ask for information about the loaded model.
        """
        return str(self._info)

    @dump_func_call
    def image_response(self, image:str, prompt:str = "describe the image") -> str:
        """
        Provide text response form an input image.
        Useful when user asks for getting a description of the input image content. 
        """
        if not self._info.vision:
            return "Cannot handle images"
        image_handle = lms.prepare_image(image)
        self._chat.add_user_message(prompt, images=[image_handle])
        prediction  = self._model.respond(self._chat)
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
            self.model_info, self.recall_personal, self.store_personal,
            self.image_response, self.allow_speech, self.clear_chat,
            capture_image, system_info, current_date_time, get_working_directory, 
            run_command, search_in_file, search_pattern_in_files, 
            read_text_file, replace_string_in_file, write_text_file,
            web_search, webpage_content, webpage_headlines,
            fetch_financial_data, save_financial_chart, 
            exit_gracefully
        ]
        self.logger.info(
            "✅ Agent with Model '%s' is ready with [%d] tools.", 
            self.config.brain.model, len(self.tools)
        )

    def bootstrap(self) -> None:
        """Run the configured bootstrap prompt (called by the session, not on init)."""
        if self.config.agent.bootstrap_prompt:
            self.act(self.config.agent.bootstrap_prompt)

    def _handle_fragment(self, fragment, round_index=0):
        self._fragments_buffer.append(fragment.content)
        self.print_fragment(fragment)

    def _handle_round_end(self, round_index=0):
        formatted = self._model.apply_prompt_template(self._chat)
        token_count = len(self._model.tokenize(formatted))
        context_usage_percentage = float(token_count/self._context_window) * 100
        self.flush_stream()
        print()
        cprint("-"*50 + f"\n[context: {context_usage_percentage:.2f} %]\n", "grey")
        response = " ".join(self._fragments_buffer).strip()
        if hasattr(self, "speech"):
            self.speech.speak(response)
        self._fragments_buffer.clear()

    def act(self, prompt: str):
        """
        Perform an action based on the input prompt. 
        The response may involve using tools or just generating a text.
        """
        cprint(f"🤖 [{self.config.brain.model}]", "grey")
        self.logger.info("Sending to Agent [🤖: %s] ...", self.config.brain.model)
        self._add_user_prompt(prompt)
        self._model.act(
            self._chat,
            self.tools,
            on_message=self._chat.append,
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
        iteration = 0
        while True:
            try:
                cprint("Press Ctrl+C to quit.", "grey")
                prompt = self.detector.listen_and_transcribe(
                    silence_timeout_s=self.config.speech_det.silence_timeout_s,
                )
                if not prompt:
                    self.logger.info("Empty transcription, skipping.")
                    continue
                iteration += 1
                cprint(f"\n[{iteration}] You: {prompt}", "green")
                self.agent.act(prompt)
            except KeyboardInterrupt:
                print("\n👋 Exiting.")
                break
