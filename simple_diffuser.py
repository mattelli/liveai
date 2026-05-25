# Python 3.9
# pip install diffusers
# pip install transformers
# pip install accelerate
# pip install imageio-ffmpeg

# The default <CACHE_DIR> is ~/.cache/huggingface/hub. 
# It is customizable with the cache_dir argument on all methods, 
# or by specifying either HF_HOME or HF_HUB_CACHE environment variable.

# `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files.
# To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator.
# https://learn.microsoft.com/en-us/windows/advanced-settings/developer-mode

# From https://github.com/marcchicoine/runwayml-stable-diffusion-v1-5
# See https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py
# Patch 88 +,- [False for res in result]

import PIL
import typing
import torch
from enum import Enum
from diffusers.utils import (
    export_to_video,
    load_image,
    # export_to_gif,
    # load_video
)
from diffusers import (
    # AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    DiffusionPipeline,
)


class PipeType(str, Enum):
    # T2I = "runwayml/stable-diffusion-v1-5"
    # T2I = "UnfilteredAI/NSFW-gen-v2"
    T2I = "UnfilteredAI/NSFW-GEN-ANIME"
    I2I = "Estebandidoooi/realisticVisionV60B1_v51HyperVAE"
    T2V = ""
    I2V = ""


TYPE_CFG = {
    PipeType.T2I: {"use_safetensors": True},
}

LORA_WEIGHTS = {
}


def get_device():
    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    print(f"Using device: {device_name}")
    return torch.device(device_name)


def pipe_factory(pipe_type: PipeType) -> typing.Any:

    if pipe_type == PipeType.T2I:
        pipe = DiffusionPipeline.from_pretrained(pipe_type.value, **TYPE_CFG.get(pipe_type, {})).to(get_device())
    elif pipe_type == PipeType.T2V:
        pipe = DiffusionPipeline.from_pretrained(pipe_type.value, **TYPE_CFG.get(pipe_type, {})).to(get_device())
    elif pipe_type == PipeType.I2I:
        pipe = AutoPipelineForImage2Image.from_pretrained(pipe_type.value, **TYPE_CFG.get(pipe_type, {})).to(get_device())
    elif pipe_type == PipeType.I2V:
        pipe = DiffusionPipeline.from_pretrained(pipe_type.value, **TYPE_CFG.get(pipe_type, {})).to(get_device())
    else:
        raise ValueError(f"Unsupported pipe type: {pipe_type}")
    
    lora_weights = LORA_WEIGHTS.get(pipe_type)
    if lora_weights:
        pipe.load_lora_weights(lora_weights)
    return pipe


def text2image(prompt:str, width:int=512, height:int=512, seed:int=42, steps:int=50) -> "PIL.Image.Image":
    pipe = pipe_factory(PipeType.T2I)
    with torch.inference_mode():
        out_image = pipe(
            prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.manual_seed(seed)
        ).images[0]
    return out_image


def image2image(image: PIL.Image.Image, prompt:str, strength:float=0.8, seed:int=42, steps:int=50) -> "PIL.Image.Image":
    pipe = pipe_factory(PipeType.I2I)
    with torch.inference_mode():
        out_image = pipe(
            prompt,
            image=image,
            strength=strength,
            num_inference_steps=steps,
            generator=torch.manual_seed(seed)
        ).images[0]
    return out_image


def text2video(prompt:str, width:int=64, height:int=64, seed:int=42, steps:int=50) -> list["PIL.Image.Image"]:
    pipe = pipe_factory(PipeType.T2V)
    with torch.inference_mode():
        video_frames = pipe(
            prompt,
            width=width,
            height=height,
            num_frames=49,
            guidance_scale=6,
            num_inference_steps=steps,
            generator=torch.manual_seed(seed)
        ).frames[0]
    return video_frames


def image2video(image: PIL.Image.Image, prompt:str, strength:float=0.8, seed:int=42, steps:int=50) -> "PIL.Image.Image":
    pipe = pipe_factory(PipeType.I2V)
    with torch.inference_mode():
        out_image = pipe(
            prompt,
            image=image,
            strength=strength,
            num_inference_steps=steps,
            generator=torch.manual_seed(seed)
        ).frames[0]
    return out_image


## MAIN ###
def exampleT2I():
    out_image = text2image(
        prompt="get an anime image of a blonde girl with blue eyes, she is looking at the camera smiling, simple passport photo, no background, detailed face",
    )
    out_image.save("test.png")


def exampleI2I():
    input_image = load_image("res/test6.png")
    out_image = image2image(
        image=input_image,
        prompt="make the girl of the image smile",
    )
    out_image.save("test_i2i.png")


def exampleT2V():
    raise ValueError("Unable to run in mini hardware")
    video_frames = text2video(
        prompt="get a video of a girl with blue eyes talking, she is looking at the camera"
    )
    export_to_video(video_frames, "test.mp4", fps=8)


def exampleI2V():
    raise ValueError("Unable to run in mini hardware")
    video_frames = image2video(
        image= "res/test2.png",
        prompt="get a video of a girl with blue eyes talking, she is looking at the camera"
    )
    export_to_video(video_frames, "test.mp4", fps=8)


if __name__ == "__main__":
    exampleT2I()
    # exampleI2I()
    # exampleT2V()
    # exampleI2V()
