"""Image-generation tool: text-to-image via the local Stable-Diffusion pipeline.

Built on ``src.image_generator.ImageGenerator``. Heavy dependencies (``torch``,
``diffusers``) are imported lazily on first use, so importing ``src.tools``
stays cheap (``tests/test_tools.py::test_import_tools_is_lazy``). The
``ImageGenerator`` instance itself is cached at module level: loading a
diffusion pipeline takes a long time and lots of memory, so it must survive
across tool calls instead of being rebuilt for every image.

Generation defaults (models, size, steps, seed, output folder) come from the
``[image_gen]`` section of ``config.toml``; parameters the caller leaves unset
fall back to those configured defaults.
"""

from __future__ import annotations

import pathlib
from datetime import datetime

from src.common import dump_func_call
from src.image_generator import check_size

__all__ = ["GENERATED_IMAGE", "generate_image", "unload_image_generator"]

GENERATED_IMAGE: str = "generated_image.png"

_DEFAULT_CONFIG = pathlib.Path(__file__).resolve().parents[2] / "config.toml"

# Test seams (module-level so the public tool signatures stay LLM-friendly).
_GENERATOR_FACTORY = None        # builds the shared ImageGenerator; tests inject fakes
_CONFIG_PATH: str | None = None  # override for the config.toml location
_config_cache = None             # cached [image_gen] section (None when absent)
_generator = None                # cached shared ImageGenerator instance


def _config_section():
    """Load the ``[image_gen]`` section of config.toml once (None if missing)."""
    global _config_cache
    if _config_cache is None:
        from src.common import load_config
        config_path = pathlib.Path(_CONFIG_PATH) if _CONFIG_PATH else _DEFAULT_CONFIG
        try:
            _config_cache = getattr(load_config(config_path), "image_gen", None)
        except OSError:
            _config_cache = None
    return _config_cache


def _default_generator_factory():
    """Build the shared ``ImageGenerator`` from the ``[image_gen]`` config.

    Missing keys (or the whole section) fall back to the ``DEFAULT_*`` values
    from ``src.image_generator``.
    """
    from src.image_generator import (
        DEFAULT_DENOISE, DEFAULT_GUIDANCE, DEFAULT_I2I_MODEL,
        DEFAULT_NEGATIVE_PROMPT, DEFAULT_SEED, DEFAULT_STEPS, DEFAULT_T2I_MODEL,
        DEFAULT_WIDTH, DEFAULT_HEIGHT, ImageGenerator,
    )
    cfg = _config_section()
    if cfg is None:
        return ImageGenerator()

    def opt(key, default):
        return getattr(cfg, key, default)

    return ImageGenerator(
        t2i_model=opt("t2i_model", DEFAULT_T2I_MODEL),
        i2i_model=opt("i2i_model", DEFAULT_I2I_MODEL),
        device=opt("device", "auto"),
        negative_prompt=opt("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        width=opt("width", DEFAULT_WIDTH),
        height=opt("height", DEFAULT_HEIGHT),
        steps=opt("steps", DEFAULT_STEPS),
        seed=opt("seed", DEFAULT_SEED),
        guidance=opt("guidance", DEFAULT_GUIDANCE),
        denoise=opt("denoise", DEFAULT_DENOISE),
    )


def _shared_generator():
    """Build (once) and return the module-wide ``ImageGenerator``."""
    global _generator
    if _generator is None:
        factory = _GENERATOR_FACTORY
        if factory is None:
            factory = _default_generator_factory
        _generator = factory()
    return _generator


def _default_out_path() -> pathlib.Path:
    """Timestamped default output path under the configured ``out_dir``."""
    cfg = _config_section()
    out_dir = (
        pathlib.Path(cfg.out_dir)
        if cfg is not None and getattr(cfg, "out_dir", None)
        else pathlib.Path.cwd()
    )
    stem = pathlib.Path(GENERATED_IMAGE).stem
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return out_dir / f"{stem}_{stamp}.png"


@dump_func_call
def generate_image(
        prompt: str,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        steps: int | None = None,
        out_file: str | None = None,
    ) -> str:
    """
    Generate an image from a text description and save it as an image file.
    Useful when the user asks to create, draw, generate or imagine a
    picture/image/artwork/illustration from a prompt. The first call loads the
    diffusion model, which can take a while; later calls reuse it. Parameters
    left empty use the [image_gen] defaults from config.toml (512x512, 50
    steps). ``negative_prompt`` describes what to avoid, ``width``/``height``
    must be multiples of 8, ``steps`` trades quality for speed, and a ``seed``
    < 0 picks a random seed. Returns the absolute path of the saved image with
    the seed used, so the same image can be reproduced or varied with a new seed.
    """
    if not prompt or not prompt.strip():
        return "Error: a prompt is required."
    for name, value in (("width", width), ("height", height)):
        if value is None:
            continue
        try:
            check_size(value, name)
        except ValueError as exc:
            return f"Error: {exc}"

    out_path = pathlib.Path(out_file) if out_file else _default_out_path()
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".png")
    if out_path.is_dir():
        return f"Error: '{out_file or out_path}' is a directory, not a file name"

    try:
        generator = _shared_generator()
        if not generator.is_t2i_pipe_loaded():
            generator.load_t2i_pipe()
        image = generator.text2image(
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            steps=steps,
        )
        if not generator.save(image, out_path):
            return f"Unable to save image '{out_path}'"
        return f"{out_path.resolve()} (seed={generator.last_seed})"
    except ImportError as exc:
        return (
            f"Error: image generation needs the diffusers stack ({exc}). "
            "Install it into the Python environment running this assistant with: "
            "pip install diffusers transformers accelerate safetensors"
        )
    except Exception as e:
        return f"Error: {e}"


@dump_func_call
def unload_image_generator() -> str:
    """
    Free the memory (RAM/VRAM) used by the text-to-image model loaded by
    generate_image. Useful when the user asks to free memory after generating
    images, or before running heavy work. Returns a short status message.
    """
    global _generator
    generator, _generator = _generator, None
    if generator is None or not generator.is_t2i_pipe_loaded():
        return "No image generation model is loaded"
    generator.unload_t2i_pipe()
    return "Image generation model unloaded, memory freed"
