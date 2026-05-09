# Python 3.9
# pip install diffusers
# pip install transformers
# pip install accelerate

# The default <CACHE_DIR> is ~/.cache/huggingface/hub. 
# It is customizable with the cache_dir argument on all methods, 
# or by specifying either HF_HOME or HF_HUB_CACHE environment variable.

# `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files.
# To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator.
# https://learn.microsoft.com/en-us/windows/advanced-settings/developer-mode


import PIL
import typing
import torch
from enum import Enum

class PipeType(str, Enum):
    STABLE_DIFFUSION = "runwayml/stable-diffusion-v1-5"
    FLUX = "black-forest-labs/FLUX.1-dev"

def get_device():
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    print(f"Using device: {device_name}")
    return torch.device(device_name)


def pipe_factory(pipe_type: PipeType) -> typing.Any:
    if pipe_type == PipeType.STABLE_DIFFUSION:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(pipe_type.value, use_safetensors=True).to(get_device())
    elif pipe_type == PipeType.FLUX:
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(pipe_type.value, dtype=torch.bfloat16).to(get_device())
        pipe.load_lora_weights("imagepipeline/flux_uncensored_nsfw_v1")
    else:
        raise ValueError(f"Unsupported pipe_type: {pipe_type}")
    return pipe


def text2image(prompt:str, pipe_type: PipeType, width:int=512, height:int=512, seed:int=42, steps:int=50) -> "PIL.Image.Image":
    pipe = pipe_factory(pipe_type)
    image = pipe(
        prompt, 
        width=width,
        height=height,
        num_inference_steps=steps,
        generator=torch.manual_seed(seed)
    ).images[0]
    return image


if __name__ == "__main__":
    prompt = "get an anime image of a blonde 20 year old girl with green eyes"
    image = text2image(prompt, PipeType.STABLE_DIFFUSION)
    image.save("test.png")
