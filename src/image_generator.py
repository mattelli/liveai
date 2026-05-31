import time
import PIL
import torch
import pathlib
from diffusers import DiffusionPipeline


class ImageGenerator:

    def __init__(self):
        self._pipe = DiffusionPipeline.from_pretrained(
            "UnfilteredAI/NSFW-gen-v2",
            use_safetensors=True,
            torch_dtype=torch.float32
        ).to(self.get_device())

    @staticmethod
    def get_device():
        device_name = "cpu"
        print(f"Using device: {device_name}")
        return torch.device(device_name)

    def text2image(self, prompt:str, negative_promt:str=" ",
            width:int=512, height:int=512, seed:int=42, steps:int=50
        ) -> "PIL.Image.Image":
        start = time.perf_counter()
        with torch.inference_mode():
            out_image = self._pipe(
                prompt,
                negative_promt,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=torch.manual_seed(seed)
            ).images[0]
        end = time.perf_counter()
        print(f"Generation Time: {end - start:.2f} seconds")
        return out_image
    
    @staticmethod
    def store(image, out_dir:str="./out_images", format:str="jpg") -> bool:
        if not image:
            return False
        out_dir_path = pathlib.Path(out_dir)
        if not out_dir_path.exists():
            out_dir_path.mkdir(parents=True)
        if out_dir_path.exists() and not out_dir_path.is_dir():
            out_dir_path = pathlib.Path.cwd()
        img_id = str(id(image))
        img_path = out_dir_path / f"img_{img_id}.{format}"
        image.save(img_path)
        print(f"Image stored in '{str(img_path)}'")
        return True
