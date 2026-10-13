"""Speech tool: toggle the Piper text-to-speech engine for agent responses.

Follows the ``src.tools.image_gen`` pattern: ``main.py`` injects the
application config via :func:`configure` (``[speech_gen]``), the heavy
``SpeechGenerator`` (piper + sounddevice) is imported lazily on first enable
and cached at module level, and tests override the ``_SPEECH_FACTORY`` seam.

While audio plays, Ctrl+C stops the speech only (never the agent turn): a
temporary SIGINT handler routes the interrupt to :func:`stop_speech`, and the
previous handler - the session's generation cancel or the default - is
restored afterwards.
"""

from __future__ import annotations

import contextlib
import signal

from src.common import cprint, dump_func_call

__all__ = ["configure", "allow_speech", "speak_if_enabled", "stop_speech"]

_config_cache = None       # [speech_gen] section injected via configure()
_speech_generator = None   # cached shared SpeechGenerator instance (None = disabled)
_SPEECH_FACTORY = None     # test seam: factory(section) -> SpeechGenerator
_speech_stopped = False    # set by the SIGINT handler while playback runs


def configure(config) -> None:
    """
    Inject the application config into the tool (called by ``main.py`` after
    ``load_config``). Only the ``[speech_gen]`` section is kept; a missing
    section means the SpeechGenerator defaults apply. Also releases a running
    generator, so a config change is honored on the next enable.
    """
    global _config_cache, _speech_generator
    _config_cache = getattr(config, "speech_gen", None)
    _speech_generator = None


def _speech_kwargs(cfg) -> dict:
    """Map the ``[speech_gen]`` section to SpeechGenerator constructor args."""
    return dict(
        model_path=cfg.model_path,
        samplerate=cfg.samplerate,
        volume=cfg.volume,
        speech_speed=cfg.speech_speed,
        speaker_id=getattr(cfg, "speaker_id", None),
        noise_scale=getattr(cfg, "noise_scale", 1.0),
        noise_w_scale=getattr(cfg, "noise_w_scale", 1.0),
        normalize_audio=getattr(cfg, "normalize_audio", False),
    )


@dump_func_call
def allow_speech(enable: bool) -> str:
    """
    Enable or disable speech for Agent responses.
    When enabled, the Agent will use the Speech Generator.
    It can be toggled on the fly during the conversation.
    """
    global _speech_generator
    if enable and _speech_generator is None:
        from src.audio_generator import SpeechGenerator  # heavy: piper + sounddevice

        factory = _SPEECH_FACTORY or SpeechGenerator
        _speech_generator = factory(**_speech_kwargs(_config_cache))
        return "Speech Generator initialized"
    elif not enable and _speech_generator is not None:
        _speech_generator.close()
        _speech_generator = None
        return "Speech Generator released"
    return ""


def stop_speech() -> None:
    """
    Request the running speech playback to stop (audio only).

    Never cancels the agent turn: remaining tool rounds may speak again.
    Safe to call while speech is disabled (no-op), from a signal handler or
    any other thread.
    """
    generator = _speech_generator
    if generator is not None:
        generator.request_stop()


def speak_if_enabled(text: str) -> None:
    """
    Speak ``text`` through the session speech generator, when enabled.

    While the audio plays, Ctrl+C is routed to :func:`stop_speech` - dropping
    buffered audio for immediate silence - instead of the agent's generation
    cancel; the previous SIGINT handler is restored afterwards.
    """
    global _speech_stopped
    if _speech_generator is None:
        return
    _speech_stopped = False
    with _speech_sigint_guard():
        _speech_generator.speak(text)
    if _speech_stopped:
        _speech_stopped = False  # consume: the message is printed once
        cprint("🔇 Speech stopped", "yellow")


@contextlib.contextmanager
def _speech_sigint_guard():
    """
    Route Ctrl+C to :func:`stop_speech` while speech audio is playing.

    Nested over the session's generation guard (``_generation_sigint_guard``):
    during playback the handler installed here wins, and on exit the previous
    handler - the generation cancel or the default - is restored. When
    installation is not possible (e.g. not the main thread), speech plays
    uninterruptible, as before this feature.
    """
    previous = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum, frame):
        global _speech_stopped
        _speech_stopped = True
        stop_speech()

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):  # e.g. not the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
