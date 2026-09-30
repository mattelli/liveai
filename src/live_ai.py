import inspect
import pathlib
import sys
import lmstudio as lms
from lmstudio.sync_api import PredictionStream
from typing import Any, Callable, Literal, get_type_hints
from logging import getLogger
from datetime import datetime, timezone
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
    DEFAULT_CHUNK_SIZE, DEFAULT_CORPORA_DIR, DEFAULT_EXTENSIONS, DEFAULT_TOP_K,
    EmbeddingCorpus, list_corpora, resolve_corpus_dir
)
from src.document_loader import IMAGE_EXTENSIONS

Role = Literal['system', 'user', 'assistant']

# Padded into the history when a generation is cancelled before any text was
# produced: an empty assistant turn would be dropped by the model's prompt
# template and break user/assistant role alternation on the next request.
CANCELLED_PLACEHOLDER = "(generation cancelled by user)"


def __getattr__(name: str):
    """Lazily expose ``SpeechGenerator`` so importing this module does not pull
    in the heavy ``piper``/``sounddevice`` stack. The name stays patchable on
    the module for callers (and tests) that override it.
    """
    if name == "SpeechGenerator":
        from src.audio_generator import SpeechGenerator
        return SpeechGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _camelize(name: str) -> str:
    """``corpus_name`` -> ``corpusName`` (single words are unchanged)."""
    head, *rest = name.split("_")
    return head + "".join(part.capitalize() for part in rest)


def _camel_tool_definition(tool: Callable) -> dict:
    """Describe ``tool`` to the LLM with camelCase parameter names.

    LM Studio validates the arguments produced by the model against the names
    advertised in the tool schema *before* calling the Python implementation.
    Qwen-style models emit camelCase names (``corpusName``) even when the schema
    uses snake_case (``corpus_name``), so argument parsing fails with
    ``Object missing required field `corpus_name```.

    We advertise the camelCase spelling and translate the incoming arguments
    back to the real Python parameter names, so the model never sees a mismatch.
    """
    signature = inspect.signature(tool)
    try:
        hints = get_type_hints(tool, include_extras=True)
    except Exception:
        hints = getattr(tool, "__annotations__", {}) or {}

    parameters: dict = {}
    alias: dict[str, str] = {}
    for name, param in signature.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        camel = _camelize(name)
        alias[camel] = name
        annotation = hints.get(name, str)
        if param.default is inspect.Parameter.empty:
            parameters[camel] = annotation
        else:
            parameters[camel] = {"type": annotation, "default": param.default}

    def implementation(**kwargs):
        return tool(**{alias.get(key, key): value for key, value in kwargs.items()})

    implementation.__name__ = getattr(tool, "__name__", "tool")
    implementation.__doc__ = getattr(tool, "__doc__", "")
    return {
        "name": implementation.__name__,
        "description": (implementation.__doc__ or "").strip(),
        "parameters": parameters,
        "implementation": implementation,
    }


def _response_token_cap(max_response_tokens: float, context_window: int) -> int:
    """Resolve ``[brain] max_response_tokens`` into a per-generation cap.

    The config value is a ratio of the model's context window, so the cap
    self-adapts to whichever model is loaded: ``0.1`` on a 4k-window model
    caps a response at ~409 tokens. Valid range is 0 < ratio < 1; unset/0
    keeps the historical default of the full context window (no practical
    cap). The cap bounds any single generation - including one LM Studio
    cannot stop instantly after Ctrl+C (e.g. cancelled during prompt
    processing) - so a stale prediction can never monopolise the model.
    """
    if not max_response_tokens:
        return context_window
    try:
        ratio = float(max_response_tokens)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"brain.max_response_tokens must be a number (got {max_response_tokens!r})"
        ) from exc
    if not 0 < ratio < 1:
        raise ValueError(
            "brain.max_response_tokens must be a ratio with 0 < ratio < 1, "
            f"got {max_response_tokens!r}"
        )
    return max(1, int(context_window * ratio))


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

        def _match(model: Any) -> bool:
            key = model.model_key
            if config.brain.model == key:
                return True
            # Tolerate a missing/extra publisher prefix (e.g. "qwen3.5-0.8b"
            # vs "qwen/qwen3.5-0.8b") by comparing the bare model name.
            return config.brain.model.rsplit("/", 1)[-1] == key.rsplit("/", 1)[-1]

        for model in llm_only:
            if _match(model):
                break
        else:
            raise ValueError(f"The configured Model '{config.brain.model}' is not available.")
        # Use the exact key LM Studio reports for the matched model; a
        # publisher-prefixed config value may not resolve on its own.
        resolved_model_key = model.model_key
        try:
            current_model = lms.llm()
            current_model_info = current_model.get_info()
            if current_model_info.model_key != resolved_model_key:
                self.logger.info("Unloading Model [%s]", current_model_info.model_key)
                current_model.unload()
        except lms.LMStudioClientError:
            pass

        self._model = lms.llm(resolved_model_key)
        self._info = self._model.get_info()
        self._context_window = self._info.context_length
        self._chat = lms.Chat(initial_prompt=self.config.brain.system_prompt)
        self._stream = StreamPrinter(text_color="cyan")
        self._cancel_requested = False
        self._active_stream: PredictionStream | None = None
        self._chat_cfg = {
            "temperature": self.config.brain.temperature,
            "maxTokens": _response_token_cap(
                getattr(self.config.brain, "max_response_tokens", 0),
                self._context_window,
            ),
        }
        self.logger.info("✅ LLM Model '%s' connected.", self.config.brain.model)

    def _add_user_prompt(self, prompt: str):
        self._repair_history()
        self._chat.add_user_message(prompt)
        self.logger.debug("Prompt:\n%s", prompt)

    @staticmethod
    def _message_text(message) -> str:
        """Concatenated text parts of a chat message (tool parts ignored)."""
        texts = []
        for part in getattr(message, "content", None) or []:
            text = part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
            if isinstance(text, str):
                texts.append(text)
        return "".join(texts)

    def _repair_history(self) -> None:
        """Keep the chat history template-valid across cancelled/failed turns.

        LM Studio prompt templates require user/assistant roles to alternate
        (tool-call/result sequences excepted). Two failure modes break that:

        * a Ctrl+C-cancelled generation can end in an *empty* assistant
          message; templates drop empty turns, which leaves a tool result
          directly followed by a user message and fails rendering with
          "conversation roles must alternate ...";
        * a turn that errored or was cancelled before any response leaves the
          user prompt unanswered, so the next prompt would render as two
          consecutive user messages.

        Called before each new prompt: pad the empty assistant turn with a
        minimal placeholder and drop a trailing unanswered user prompt.
        """
        messages = self._chat._messages
        while messages:
            role = getattr(messages[-1], "role", None)
            if role == "assistant":
                if not self._message_text(messages[-1]).strip():
                    messages.pop()
                    self._chat.append(
                        {"role": "assistant", "content": {"text": CANCELLED_PLACEHOLDER}}
                    )
                break
            if role == "user":
                messages.pop()  # unanswered prompt from a failed/aborted turn
                continue
            break  # tool/system endings are left to the SDK layer

    @property
    def cancel_requested(self) -> bool:
        """True while a user cancel (Ctrl+C) is pending for the current turn."""
        return self._cancel_requested

    def cancel_generation(self) -> str:
        """
        Stop the in-flight model response as soon as possible (best effort).
        Not exposed to the LLM; invoked by the session's Ctrl+C handling.
        Cancels the active prediction stream server-side (stop reason
        ``userStopped``) for both streamed responses and ``act()`` rounds
        (via the SDK's ``on_prediction_stream`` callback); the pending flag
        covers a cancel raised while a tool is running (between rounds).
        Partial output is kept in the chat history.
        """
        self._cancel_requested = True
        stream, self._active_stream = self._active_stream, None
        if stream is not None:
            try:
                stream.cancel()
            except Exception as exc:
                self.logger.debug("Cancelling prediction stream failed: %s", exc)
        return "Cancel requested"

    def model_fragments(self, prompt: str) -> PredictionStream:
        """
        Get response fragments as stream from the model for a given prompt.
        Tools are not supported in this method.
        """
        self.logger.info("Sending to Model [🤖: %s] ...", self.config.brain.model)
        self._add_user_prompt(prompt)
        self._cancel_requested = False  # fresh turn
        self._stream.start_spinner("thinking")
        resp_stream = self._model.respond_stream(
            self._chat,
            on_message=self._chat.append,
            config=self._chat_cfg,
            on_prompt_processing_progress=lambda p: self._stream.note(
                f"processing prompt {min(100, int(p * 100))}%"
            ),
        )
        self._active_stream = resp_stream
        return resp_stream

    def print_fragment(self, fragment):
        """
        Print a response fragment to the console.
        Fenced code blocks are syntax-highlighted; plain text is cyan.
        This can be used as a callback for streaming responses.
        """
        self._stream.feed(fragment.content)

    def flush_stream(self) -> None:
        """Flush buffered text and stop the activity spinner (turn boundary)."""
        self._stream.flush()
        self._stream.stop_spinner()

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
                image_model_path=getattr(cfg, "image_model_path", None),
                multimodal=getattr(cfg, "multimodal", False),
                image_extensions=getattr(cfg, "image_extensions", IMAGE_EXTENSIONS),
            )
        return self._emb_corpus

    @dump_func_call
    def list_corpora(self) -> str:
        """
        List the available local corpora (name, mode, record counts, sources, built time).
        Useful before query_corpus when the user refers to a topic instead of a corpus name.
        """
        entries = list_corpora(self.config.embedding.corpora_dir)
        if not entries:
            return "No corpora available."
        lines = [f"Available corpora ({len(entries)}):"]
        for e in entries:
            lines.append(
                f"- {e['name']}: mode={e['mode']}, records={e['records']} "
                f"(text {e['text_records']} / image {e['image_records']}), "
                f"sources={e['sources']}, built={e['built_at']}"
            )
        return "\n".join(lines)

    @dump_func_call
    def store_corpus(self, directory: str, corpus_name: str, overwrite: bool = False) -> str:
        """
        Build a named corpus from all supported files in a directory.
        Useful when the user asks to index, embed, or store documents for later semantic search.
        """
        try:
            if not corpus_name or not corpus_name.strip():
                return "Error: a corpus name is required (call list_corpora to see available corpora)."
            corpus_dir = resolve_corpus_dir(corpus_name, self.config.embedding.corpora_dir)
            return self._embedding_corpus().store(directory, corpus_dir, overwrite)
        except Exception as e:
            return f"Error: {e}"

    @dump_func_call
    def query_corpus(self, query: str, corpus_name: str, kind: str = "", top_k: int = 0) -> str:
        """
        Search a named corpus for the content most relevant to a query text.
        Useful when the user asks a question that should be answered from indexed documents
        or when looking for a matching image.
        ``kind`` restricts results to 'text' or 'image'; leave it empty to search both.
        """
        try:
            if not corpus_name or not corpus_name.strip():
                return "Error: a corpus name is required."
            corpus_dir = resolve_corpus_dir(corpus_name, self.config.embedding.corpora_dir)
            return self._embedding_corpus().query(corpus_dir, query, top_k or None, kind or None)
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
            SpeechGenerator = getattr(sys.modules[__name__], "SpeechGenerator")
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
            self.list_corpora, self.store_corpus, self.query_corpus,
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
        self.tool_definitions:list[dict] = [_camel_tool_definition(tool) for tool in self.tools]
        self.logger.info(
            "✅ Agent with Model '%s' is ready with [%d] tools.", 
            self.config.brain.model, len(self.tools)
        )

    def bootstrap(self) -> None:
        """Run the configured bootstrap prompt (called by the session, not on init)."""
        if self.config.agent.bootstrap_prompt:
            self.act(self.config.agent.bootstrap_prompt)

    def _handle_prediction_stream(self, stream, round_index=0):
        """Track each ``act()`` round's prediction stream for Ctrl+C support.

        Called by the (patched) LM Studio SDK when each round's stream
        starts. A cancel requested while a tool was running (between
        rounds) is applied immediately here, so the new round stops
        before generating.
        """
        if self._cancel_requested:
            stream.cancel()
        self._active_stream = stream

    def _handle_fragment(self, fragment, round_index=0):
        self._fragments_buffer.append(fragment.content)
        self.print_fragment(fragment)

    def _handle_round_end(self, round_index=0):
        formatted = self._model.apply_prompt_template(self._chat)
        token_count = len(self._model.tokenize(formatted))
        context_usage_percentage = float(token_count/self._context_window) * 100
        self._stream.flush()  # keep the spinner alive across rounds
        cprint("")
        cprint("-"*50 + f"\ncontext: {context_usage_percentage:.2f} % [{int(self._context_window / 1000)}k]\n", "grey")
        self._stream.mark_line_clean()  # spinner may resume for the next round
        self._stream.note("working")
        response = " ".join(self._fragments_buffer).strip()
        if not self._cancel_requested and hasattr(self, "speech"):
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
        self._cancel_requested = False  # fresh turn
        self._stream.start_spinner("thinking")
        try:
            self._model.act(
                self._chat,
                self.tool_definitions,
                on_message=self._chat.append,
                config=self._chat_cfg,
                on_prediction_fragment=self._handle_fragment,
                on_prediction_stream=self._handle_prediction_stream,
                on_prompt_processing_progress=lambda p, _round=0: self._stream.note(
                    f"processing prompt {min(100, int(p * 100))}%"
                ),
                on_round_end=self._handle_round_end
            )
        finally:
            self._stream.stop_spinner()
            self._active_stream = None
