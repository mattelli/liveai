"""Personal-memory and conversation-history tools.

``store_personal`` appends personal notes from the conversation to a history
file, ``recall_personal`` reads them back, and ``clear_chat`` resets the live
conversation (re-seeded with the recalled personal data).

Follows the ``src.tools.image_gen`` pattern: ``main.py`` injects the
application config via :func:`configure` (``[brain]`` history path/system
prompt, ``[agent]`` bootstrap prompt). The one piece of non-config state - the
live message list of the running session - is bound by ``Brain`` via
:func:`bind_history`; tests override the provider seam directly.
"""

from __future__ import annotations

import pathlib
from datetime import datetime, timezone

from src.common import dump_func_call

__all__ = ["configure", "bind_history", "recall_personal", "store_personal", "clear_chat"]

_config_cache = None      # [brain] + [agent] sections injected via configure()
_history_provider = None  # seam: callable returning the live messages list


def configure(config) -> None:
    """
    Inject the application config into the tool (called by ``main.py`` after
    ``load_config``). Keeps the ``[brain]`` section (``history_path``,
    ``system_prompt``) and the ``[agent]`` section (``bootstrap_prompt``).
    """
    global _config_cache
    _config_cache = {
        "brain": getattr(config, "brain", None),
        "agent": getattr(config, "agent", None),
    }


def bind_history(provider) -> None:
    """
    Bind the live conversation history of the running session (called by
    ``Brain``). ``provider`` is a callable returning the current messages list
    (plain OpenAI-style dicts); it is re-read on every call, so clearing or
    reassigning the session history is always reflected.
    """
    global _history_provider
    _history_provider = provider


def _config(key, attr, default=None):
    section = (_config_cache or {}).get(key)
    return getattr(section, attr, default) if section is not None else default


def _history_file() -> pathlib.Path:
    return pathlib.Path(_config("brain", "history_path", "")).resolve()


def _shaped_user_messages() -> list[str]:
    """User messages of the live history (text at ``content[0]["text"]``)."""
    messages = _history_provider() if _history_provider else []
    shaped = [
        {"role": m["role"], "content": [{"text": m["content"]}]}
        for m in messages
        if isinstance(m.get("content"), str)
    ]
    try:
        return [m["content"][0]["text"] for m in shaped if m["role"] == "user"]
    except Exception:
        return []


@dump_func_call
def recall_personal() -> str:
    """
    Read users personal data from file.
    Useful when the users ask to recall their personal data.
    """
    history = _history_file()
    if not history.exists():
        return "No personal data found!"
    else:
        return history.read_text(encoding="utf-8")


@dump_func_call
def store_personal() -> str:
    """
    Stores the user personal content to a text file.
    Useful when the users explicitly mention the prompted text is "personal".
    """
    history = _history_file()
    now = datetime.now(tz=timezone.utc)
    timestamp = now.strftime("%d/%m/%Y, %H:%M")
    user_msgs = _shaped_user_messages()
    bootstrap_prompt = _config("agent", "bootstrap_prompt", None)
    if user_msgs and user_msgs[0] == bootstrap_prompt:
        user_msgs = user_msgs[1:]
    selected_msgs = [m for m in user_msgs if "personal" in m.lower()]
    user_prompt = f"At {timestamp} user wrote: {selected_msgs[-1]} \n"
    if not history.exists():
        history.write_text(user_prompt, encoding="utf-8")
    else:
        with open(file=history, mode="a", encoding="utf-8") as f:
            f.write(user_prompt)
    return f"Conversation written to {history}"


@dump_func_call
def clear_chat() -> str:
    """
    Useful when user ask for cleaning conversation history.
    """
    try:
        provider = _history_provider
        if provider is None:
            return "Error: no chat session is bound."
        messages = provider()
        system_prompt = _config("brain", "system_prompt", "")
        system = next(
            (m["content"] for m in messages if m["role"] == "system"),
            system_prompt,
        )
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        messages.clear()  # in place: the session keeps the same list object
        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": recall_personal()})
        if user_msgs:
            messages.append({"role": "user", "content": user_msgs[-1]})
        return "Chat cleared successfully"
    except Exception as e:
        return f"Error: {e}"
