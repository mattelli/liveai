import time
import PIL
import torch
from enum import Enum
from diffusers import (
    # AutoPipelineForText2Image,
    DiffusionPipeline,
)


class PipeType(str, Enum):
    sd15 = "runwayml/stable-diffusion-v1-5"
    nsfw2 = "UnfilteredAI/NSFW-gen-v2"


TYPE_CFG = {
    PipeType.sd15: {"use_safetensors": True},
    PipeType.nsfw2: {"use_safetensors": True, "torch_dtype": torch.float32},
}


def get_device():
    device_name = "cpu"
    print(f"Using device: {device_name}")
    return torch.device(device_name)


def text2image(
        prompt:str,
        pipe_type: PipeType,
        width:int=512, height:int=512, seed:int=42, steps:int=40
    ) -> "PIL.Image.Image":
    pipe = DiffusionPipeline.from_pretrained(pipe_type.value, **TYPE_CFG.get(pipe_type, {})).to(get_device())
    with torch.inference_mode():
        out_image = pipe(
            prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=torch.manual_seed(seed)
        ).images[0]
    return out_image


def mainT2I(image_name: str, image_format: str="png"):
    start = time.perf_counter()
    out_image = text2image(
        prompt="get an anime image of a blonde girl with blue eyes, she is looking at the camera, no background, detailed face",
        pipe_type= PipeType.nsfw2
    )
    out_image.save(f"{image_name}.{image_format}")
    end = time.perf_counter()
    print(f"Execution Time: {end - start:.2f} seconds")


if __name__ == "__main__":
    mainT2I(image_name="test_b")
