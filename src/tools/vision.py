"""Vision tool: describe an image through the multimodal chat model.

Unlike the other tool modules there is no config section here: everything it
needs (client, model, temperature, vision capability, live history) belongs to
the running chat session, so ``Brain`` binds itself via :func:`bind_session`
and tests inject a small fake session exposing the same read-only attributes.

Multi-agent isolation (PLAN.md task 2): the tool logic lives in
:class:`VisionTools`, so every pooled agent can own its own instance
(:meth:`VisionTools.for_owner`) instead of sharing module-level state. The
module-level functions delegate to a module-default instance whose state is
backed by the module global ``_session``, keeping the historical
monkeypatch-able seam and call API working unchanged for the main agent,
existing callers and tests.
"""

from __future__ import annotations

import base64
import mimetypes
import pathlib

from src.common import dump_func_call

__all__ = ["VisionTools", "bind_session", "image_response"]

_session = None  # bound chat session (Brain or test fake) - see bind_session()


class VisionTools:
    """Per-agent vision tool (own session binding, no shared state).

    A pooled agent gets a fresh instance via :meth:`for_owner`, so parallel
    agents never fight over one module-level binding.
    """

    def __init__(self) -> None:
        self._session = None  # bound chat session (Brain or test fake)

    def for_owner(self, agent) -> "VisionTools":
        """Fresh instance for a specific agent (per-agent tool isolation)."""
        return VisionTools()

    def bind_session(self, session) -> None:
        """
        Bind the running chat session (called by ``Brain``). The session must
        expose the read-only attributes ``client`` (OpenAI-compatible client),
        ``model_name``, ``temperature``, ``vision`` (bool), ``vision_source``
        ('server'/'config') and ``messages`` (the live history list, re-read on
        every access).
        """
        self._session = session

    @dump_func_call
    def image_response(self, image: str, prompt: str = "describe the image") -> str:
        """
        Provide text response form an input image.
        Useful when user asks for getting a description of the input image content.
        """
        if self._session is None:
            return "Error: no chat session is bound."
        if not self._session.vision:
            if self._session.vision_source == "server":
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
        messages = self._session.messages
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_url},
            ],
        })
        try:
            kwargs: dict = {
                "model": self._session.model_name,
                "messages": messages,
                "temperature": self._session.temperature,
            }
            extra_body = getattr(self._session, "_provider_extra_body", lambda: None)()
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = self._session.client.chat.completions.create(**kwargs)
        except Exception as e:
            messages.pop()  # drop the image turn: provider rejected it
            return f"Error: {e}"
        answer = resp.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": answer})
        return answer


class _ModuleDefaultVisionTools(VisionTools):
    """Module-default instance backed by the module global ``_session``.

    Keeps the historical ``tools.vision._session`` seam (monkeypatched by
    tests) working while pooled agents use their own isolated instances.
    """

    @property
    def _session(self):
        return globals()["_session"]

    @_session.setter
    def _session(self, value) -> None:
        globals()["_session"] = value


# Module-default instance: keeps the historical module-level API working for
# the main agent, existing callers and tests. Pooled agents use their own
# instance instead (see Agent._build_tools / VisionTools.for_owner).
_default = _ModuleDefaultVisionTools()


def bind_session(session) -> None:
    """Bind the running chat session (module-default instance)."""
    _default.bind_session(session)


def image_response(image: str, prompt: str = "describe the image") -> str:
    """Provide text response form an input image.

    Useful when user asks for getting a description of the input image content.
    """
    return _default.image_response(image, prompt)
