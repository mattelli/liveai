"""Speech tool: toggle the Piper text-to-speech engine for agent responses.

Follows the ``src.tools.image_gen`` pattern: ``main.py`` injects the
application config via :func:`configure` (``[speech_gen]``), the heavy
``SpeechGenerator`` (piper + sounddevice) is imported lazily on first enable
and cached at module level, and tests override the ``_SPEECH_FACTORY`` seam.
"""

from __future__ import annotations

from src.common import dump_func_call

__all__ = ["configure", "allow_speech", "speak_if_enabled"]

_config_cache = None       # [speech_gen] section injected via configure()
_speech_generator = None   # cached shared SpeechGenerator instance (None = disabled)
_SPEECH_FACTORY = None     # test seam: factory(section) -> SpeechGenerator


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


def speak_if_enabled(text: str) -> None:
    """Speak ``text`` through the session speech generator, when enabled."""
    if _speech_generator is not None:
        _speech_generator.speak(text)
