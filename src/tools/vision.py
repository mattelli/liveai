"""Vision tool: describe an image through the multimodal chat model.

Unlike the other tool modules there is no config section here: everything it
needs (client, model, temperature, vision capability, live history) belongs to
the running chat session, so ``Brain`` binds itself via :func:`bind_session`
and tests inject a small fake session exposing the same read-only attributes.
"""

from __future__ import annotations

import base64
import mimetypes
import pathlib

from src.common import dump_func_call

__all__ = ["bind_session", "image_response"]

_session = None  # bound chat session (Brain): client, model_name, temperature,
#                 vision, vision_source, messages - see bind_session()


def bind_session(session) -> None:
    """
    Bind the running chat session (called by ``Brain``). The session must
    expose the read-only attributes ``client`` (OpenAI-compatible client),
    ``model_name``, ``temperature``, ``vision`` (bool), ``vision_source``
    ('server'/'config') and ``messages`` (the live history list, re-read on
    every access).
    """
    global _session
    _session = session


@dump_func_call
def image_response(image: str, prompt: str = "describe the image") -> str:
    """
    Provide text response form an input image.
    Useful when user asks for getting a description of the input image content.
    """
    if _session is None:
        return "Error: no chat session is bound."
    if not _session.vision:
        if _session.vision_source == "server":
            return (
                "Cannot handle images: the server reports the model has no "
                "'vision' capability"
            )
        return (
            "Cannot handle images: no 'vision' capability reported by the "
            "server and no [brain] vision = true override set"
        )
    if str(image).startswith(("http://", "https://")):
        image_url = {"url": str(image)}
    else:
        data = pathlib.Path(image).read_bytes()
        mime = mimetypes.guess_type(str(image))[0] or "image/png"
        image_url = {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}
    messages = _session.messages
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": image_url},
        ],
    })
    try:
        resp = _session.client.chat.completions.create(
            model=_session.model_name,
            messages=messages,
            temperature=_session.temperature,
        )
    except Exception as e:
        messages.pop()  # drop the image turn: provider rejected it
        return f"Error: {e}"
    answer = resp.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": answer})
    return answer
