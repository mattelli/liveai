import pathlib
import lmstudio as lms
from lmstudio.sync_api import PredictionStream
from typing import Callable, Literal
from logging import getLogger
from datetime import datetime, timezone
from src.audio_generator import SpeechGenerator
from src.common import SimpleNamespace, dump_func_call, cprint, StreamPrinter
from src.tools import (
    exit_gracefully, system_info, current_date_time, get_working_directory,
    read_text_file, write_text_file, run_command,
    web_search, webpage_headlines, webpage_content, capture_image, 
    fetch_financial_data, save_financial_chart, 
    search_pattern_in_files, replace_string_in_file, search_in_file
)
from src.summarizer import TextSummarizer
from src.embedding_search import (
    DEFAULT_CHUNK_SIZE, DEFAULT_EXTENSIONS, DEFAULT_TOP_K, EmbeddingCorpus
)

Role = Literal['system', 'user', 'assistant']

class Brain:
    """
    Brain class that connects to the LLM model.
    Basic function calls like model response, streaming, and printing are available.
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
        self._context_window = self._info.context_length
        self._chat = lms.Chat(initial_prompt=self.config.brain.system_prompt)
        self._stream = StreamPrinter(text_color="cyan")
        self._chat_cfg = {
            "temperature": self.config.brain.temperature,
            "maxTokens": self._context_window,  # max tokens to generate per response
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

    ## TOOLS CALLS ##

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
    def summarize_text(self, file: str, max_sentences: int = 0) -> str:
        """
        Summarize the text content of a file into its most important sentences.
        Useful when the user asks to summarize, condense, shorten or outline a text file.
        ``file`` is the path of the text file to read; ``max_sentences`` = 0 uses the
        ratio configured in the ``[summarization]`` section.
        """
        file_path = pathlib.Path(file).resolve()
        if not file_path.exists() or not file_path.is_file():
            return f"Unable to read '{file_path}'"
        try:
            text = file_path.read_text(encoding="utf-8")
            if getattr(self, "_summarizer", None) is None:
                cfg = self.config.summarization
                self._summarizer = TextSummarizer(
                    model_path=cfg.model_path,
                    device=getattr(cfg, "device", "cpu"),
                    ratio=getattr(cfg, "ratio", 0.1),
                    threshold=getattr(cfg, "threshold", 0.5),
                    max_sentences=getattr(cfg, "max_sentences", 0),
                )
            return self._summarizer.summarize(text, max_sentences or None)
        except Exception as e:
            return f"Error: {e}"

    def _embedding_corpus(self):
        if getattr(self, "_emb_corpus", None) is None:
            cfg = self.config.embedding
            self._emb_corpus = EmbeddingCorpus(
                model_path=cfg.model_path,
                device=getattr(cfg, "device", "cpu"),
                extensions=getattr(cfg, "extensions", DEFAULT_EXTENSIONS),
                chunk_size=getattr(cfg, "chunk_size", DEFAULT_CHUNK_SIZE),
                top_k=getattr(cfg, "top_k", DEFAULT_TOP_K),
            )
        return self._emb_corpus

    @dump_func_call
    def store_corpus(self, directory: str, corpus_file: str = "", overwrite: bool = False) -> str:
        """
        Build a searchable corpus from all text files in a directory and store it to a file.
        Useful when the user asks to index, embed, or store documents for later semantic search.
        """
        try:
            output = corpus_file or str(self.config.embedding.corpus_path)
            return self._embedding_corpus().store(directory, output, overwrite)
        except Exception as e:
            return f"Error: {e}"

    @dump_func_call
    def query_corpus(self, query: str, corpus_file: str = "", top_k: int = 0) -> str:
        """
        Search a previously stored corpus for the content most relevant to a query text.
        Useful when the user asks a question that should be answered from indexed documents.
        """
        try:
            target = corpus_file or str(self.config.embedding.corpus_path)
            return self._embedding_corpus().query(target, query, top_k or None)
        except Exception as e:
            return f"Error: {e}"

    @dump_func_call
    def allow_speech(self, enable:bool) -> str:
        """
        Enable or disable speech for Agent responses.
        When enabled, the Agent will use the Speech Generator.
        It can be toggled on the fly during the conversation.
        """
        if enable and not hasattr(self, "speech"):
            self.logger.info("Initializing Speech Generator ...")
            gen_cfg = self.config.speech_gen
            self.speech = SpeechGenerator(
                model_path=gen_cfg.model_path,
                samplerate=gen_cfg.samplerate,
                volume=gen_cfg.volume,
                speech_speed=gen_cfg.speech_speed,
                speaker_id=getattr(gen_cfg, "speaker_id", None),
                noise_scale=getattr(gen_cfg, "noise_scale", 1.0),
                noise_w_scale=getattr(gen_cfg, "noise_w_scale", 1.0),
                normalize_audio=getattr(gen_cfg, "normalize_audio", False),
            )
            return "Speech Generator initialized"
        elif not enable and hasattr(self, "speech"):
            self.logger.info("Releasing Speech Generator ...")
            self.speech.close()
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
    Agent class that extends the Brain with tool usage capabilities.
    """
    def _build_tools(self) -> list[Callable]:
        return [
            self.model_info, self.recall_personal, self.store_personal,
            self.image_response, self.allow_speech, self.clear_chat,
            self.summarize_text,
            self.store_corpus, self.query_corpus,
            capture_image, system_info, current_date_time, get_working_directory, 
            run_command, search_in_file, search_pattern_in_files, 
            read_text_file, replace_string_in_file, write_text_file,
            web_search, webpage_content, webpage_headlines,
            fetch_financial_data, save_financial_chart, 
            exit_gracefully
        ]

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self._fragments_buffer:list[str] = []
        self.logger.info(
            "✅ Agent with Model '%s' initialized. Setting up tools ...", 
            self.config.brain.model
        )
        self.tools:list[Callable] = self._build_tools()
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
        cprint("-"*50 + f"\ncontext: {context_usage_percentage:.2f} % [{int(self._context_window / 1000)}k]\n", "grey")
        response = " ".join(self._fragments_buffer).strip()
        if hasattr(self, "speech"):
            self.speech.speak(response)
        self._fragments_buffer.clear()

    def act(self, prompt: str):
        """
        Perform an action based on the input prompt. 
        The response may involve using tools or just generating a text.
        """
        cprint(f"\n🤖 [{self.config.brain.model}]", "grey")
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
