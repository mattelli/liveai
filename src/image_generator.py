"""Local image generation with Stable-Diffusion pipelines (diffusers).

All heavy dependencies (``torch``, ``diffusers``, ``PIL``) are imported lazily
on first use, so importing this module stays cheap for callers that never
generate images. Models and generation defaults are injected through the
constructor and mirror the ``[image_gen]`` section of ``config.toml``; every
generation parameter left as ``None`` resolves to those defaults.
"""

from __future__ import annotations

import pathlib
import random
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

__all__ = [
    "ImageGenerator", "check_size", "DEFAULT_T2I_MODEL", "DEFAULT_I2I_MODEL",
    "DEFAULT_NEGATIVE_PROMPT", "DEFAULT_WIDTH", "DEFAULT_HEIGHT",
    "DEFAULT_STEPS", "DEFAULT_SEED", "DEFAULT_GUIDANCE", "DEFAULT_DENOISE",
]

DEFAULT_T2I_MODEL = "UnfilteredAI/NSFW-GEN-ANIME"
DEFAULT_I2I_MODEL = "runwayml/stable-diffusion-v1-5"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, deformed, extra limbs, watermark, text"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STEPS = 50
DEFAULT_SEED = 42       # negative -> a random seed for every image
DEFAULT_GUIDANCE = 5.0  # img2img/inpaint: how closely to follow the prompt (3-12)
DEFAULT_DENOISE = 0.5   # img2img/inpaint: how much to change the input image (0-1)


def check_size(value: int | None, name: str) -> int:
    """
    Validate a generation dimension and return it as ``int``.

    Stable Diffusion needs positive multiples of 8; anything else raises
    ``ValueError`` with a message meant to be shown to the caller.
    """
    if value is None or value <= 0 or value % 8:
        raise ValueError(f"{name} must be a positive multiple of 8 (got {value})")
    return int(value)


class ImageGenerator:
    """
    Lazy wrapper around diffusers Stable-Diffusion pipelines.

    Pipelines load on demand (``load_t2i_pipe`` / ``load_i2i_pipe``) and are
    unloaded to free RAM/VRAM. Generation defaults come from the constructor;
    parameters left as ``None`` in ``text2image`` / ``image2image`` resolve to
    them. The seed actually used by the last generation is exposed as
    ``last_seed`` (a negative seed is randomized per image).
    """

    def __init__(self,
            t2i_model: str = DEFAULT_T2I_MODEL,
            i2i_model: str = DEFAULT_I2I_MODEL,
            device: str = "auto",
            negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
            width: int = DEFAULT_WIDTH,
            height: int = DEFAULT_HEIGHT,
            steps: int = DEFAULT_STEPS,
            seed: int = DEFAULT_SEED,
            guidance: float = DEFAULT_GUIDANCE,
            denoise: float = DEFAULT_DENOISE,
        ) -> None:
        self._t2i_model = t2i_model
        self._i2i_model = i2i_model
        self._device = None if device in (None, "", "auto") else device
        self._negative_prompt = negative_prompt
        self._width = width
        self._height = height
        self._steps = steps
        self._seed = seed
        self._guidance = guidance
        self._denoise = denoise
        self._t2i_pipe = None
        self._i2i_pipe = None
        self._in_paint = False
        self.last_seed: int | None = None

    ## configuration ##

    @property
    def t2i_model(self) -> str:
        """Model id (or local path) of the text-to-image pipeline."""
        return self._t2i_model

    @property
    def defaults(self) -> dict:
        """Configured generation defaults (mirrors the ``[image_gen]`` keys)."""
        return {
            "t2i_model": self._t2i_model, "i2i_model": self._i2i_model,
            "device": self._device or "auto",
            "negative_prompt": self._negative_prompt,
            "width": self._width, "height": self._height,
            "steps": self._steps, "seed": self._seed,
            "guidance": self._guidance, "denoise": self._denoise,
        }

    @property
    def device(self) -> str:
        """Torch device for the pipelines, resolved lazily ('cuda' or 'cpu')."""
        if self._device is None:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    @property
    def _torch_dtype(self):
        import torch
        return torch.float16 if self.device == "cuda" else torch.float32

    ## pipeline management ##

    def is_t2i_pipe_loaded(self) -> bool:
        return self._t2i_pipe is not None

    def is_i2i_pipe_loaded(self) -> bool:
        return self._i2i_pipe is not None

    def load_t2i_pipe(self) -> bool:
        """Load the text-to-image pipeline (downloads the model on first call)."""
        if self._t2i_pipe is not None:
            return True
        from diffusers import DiffusionPipeline
        print(f"Loading text2image model '{self._t2i_model}' on {self.device} ...")
        pipe = DiffusionPipeline.from_pretrained(
            self._t2i_model, use_safetensors=True, torch_dtype=self._torch_dtype
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)  # keep the console clean
        self._t2i_pipe = pipe
        return True

    def load_i2i_pipe(self, in_paint: bool = False) -> bool:
        """Load the image-to-image (or inpainting) pipeline."""
        if self._i2i_pipe is not None and self._in_paint == in_paint:
            return True
        from diffusers import (
            StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline,
        )
        self.unload_i2i_pipe()  # the two variants are mutually exclusive
        pipeline_cls = (
            StableDiffusionInpaintPipeline if in_paint
            else StableDiffusionImg2ImgPipeline
        )
        mode = "inpaint" if in_paint else "img2img"
        print(f"Loading {mode} model '{self._i2i_model}' on {self.device} ...")
        pipe = pipeline_cls.from_pretrained(
            self._i2i_model, use_safetensors=True, torch_dtype=self._torch_dtype
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._i2i_pipe = pipe
        self._in_paint = in_paint
        return True

    def unload_t2i_pipe(self) -> bool:
        if self._t2i_pipe is None:
            return False
        self._t2i_pipe = None
        self._free_memory()
        return True

    def unload_i2i_pipe(self) -> bool:
        if self._i2i_pipe is None:
            return False
        self._i2i_pipe = None
        self._free_memory()
        return True

    @staticmethod
    def _free_memory() -> None:
        """Best-effort release of Python/GPU memory after dropping a pipeline."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    ## generation ##

    def text2image(self, prompt: str,
            negative_prompt: str | None = None,
            width: int | None = None,
            height: int | None = None,
            seed: int | None = None,
            steps: int | None = None,
        ) -> Image.Image:
        """
        Generate an image from a text prompt. Any parameter left as ``None``
        resolves to the configured default (``[image_gen]`` in config.toml);
        a ``seed`` < 0 picks a random seed, exposed afterwards as ``last_seed``.
        """
        if not self._t2i_pipe:
            raise ValueError("Text2Image pipeline not initialized")
        import torch
        negative_prompt = (
            negative_prompt if negative_prompt is not None else self._negative_prompt
        )
        width = check_size(width if width is not None else self._width, "width")
        height = check_size(height if height is not None else self._height, "height")
        steps = steps if steps is not None and steps > 0 else self._steps
        if steps <= 0:
            raise ValueError(f"steps must be positive (got {steps})")
        seed = seed if seed is not None else self._seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
        self.last_seed = seed
        print(f"Using device: {self.device}")
        start = time.perf_counter()
        with torch.inference_mode():
            out_image = self._t2i_pipe(
                prompt,
                negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=torch.manual_seed(seed),
            ).images[0]
        print(f"Generation Time: {time.perf_counter() - start:.2f} seconds")
        return out_image

    def image2image(self,
            image: str | pathlib.Path,
            prompt: str,
            mask: str | pathlib.Path | None = None,
            negative_prompt: str | None = None,
            denoise: float | None = None,
            guidance: float | None = None,
        ) -> Image.Image:
        """
        Transform an input image following the prompt (img2img), or edit only
        the masked region when the pipeline was loaded with ``in_paint=True``.
        ``denoise`` 0-1: how much the model may change the input image.
        ``guidance`` 3-12: how closely the model follows the prompt.
        Any parameter left as ``None`` resolves to the configured default.
        """
        if not self._i2i_pipe:
            raise ValueError("Image2Image pipeline not initialized")
        import torch
        init_img = self._open_image(image)
        negative_prompt = (
            negative_prompt if negative_prompt is not None else self._negative_prompt
        )
        denoise = denoise if denoise is not None else self._denoise
        guidance = guidance if guidance is not None else self._guidance
        print(f"Using device: {self.device}")
        start = time.perf_counter()
        with torch.inference_mode():
            if self._in_paint:
                if not mask:
                    raise ValueError("Image-mask is required for inpainting")
                result = self._i2i_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_img,
                    mask_image=self._open_image(mask),
                    strength=denoise,
                    guidance_scale=guidance,
                )
            else:
                result = self._i2i_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_img,
                    strength=denoise,
                    guidance_scale=guidance,
                )
        print(f"Generation Time: {time.perf_counter() - start:.2f} seconds")
        return result.images[0]

    @staticmethod
    def _open_image(path: str | pathlib.Path) -> Image.Image:
        from PIL import Image
        file = pathlib.Path(path)
        if not file.is_file():
            raise FileNotFoundError(f"Image not found: '{file}'")
        return Image.open(file).convert("RGB")

    ## persistence ##

    @staticmethod
    def save(image: Image.Image, img_path: str | pathlib.Path) -> bool:
        """Store a generated image; returns False when there is nothing to save."""
        if not image:
            return False
        file = pathlib.Path(img_path)
        if file.is_dir():
            raise ValueError(f"'{file}' should not be a directory")
        file.parent.mkdir(parents=True, exist_ok=True)
        image.save(file)
        print(f"Image '{str(file)}' stored")
        return True
