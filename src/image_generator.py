import time
from PIL import Image
import torch
import pathlib
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline


class ImageGenerator:
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _torch_dtype =  torch.float16 if _device == "cuda" else torch.float32

    def __init__(self):
        self._t2i_pipe = None
        self._i2i_pipe = None
        self._in_paint = False

    def is_t2i_pipe_loaded(self) -> bool:
        return bool(self._t2i_pipe)
    
    def is_i2i_pipe_loaded(self) -> bool:
        return bool(self._i2i_pipe)

    def load_t2i_pipe(self) -> bool:
        model = "UnfilteredAI/NSFW-GEN-ANIME"
        self._t2i_pipe = DiffusionPipeline.from_pretrained(
            model, use_safetensors=True, torch_dtype=self._torch_dtype
        ).to(self._device)
        return True
    
    def load_i2i_pipe(self, in_paint:bool=False) -> bool:
        model = "runwayml/stable-diffusion-v1-5"
        if in_paint:
            self._i2i_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model, torch_dtype=self._torch_dtype
            ).to(self._device)
            self._in_paint = True
        else:
            self._i2i_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model, torch_dtype=self._torch_dtype
            ).to(self._device)
            self._in_paint = False
        return True
    
    def unload_t2i_pipe(self) -> bool:
        if self._t2i_pipe:
            del(self._t2i_pipe)
            self._t2i_pipe = None
            return True
        return False
    
    def unload_i2i_pipe(self) -> bool:
        if self._i2i_pipe:
            del(self._i2i_pipe)
            self._i2i_pipe = None
            return True
        return False

    def text2image(self, prompt:str, negative_prompt:str=" ",
            width:int=512, height:int=512, seed:int=42, steps:int=50
        ) -> Image:
        print(f"Using device: {self._device}")
        if not self._t2i_pipe:
            raise ValueError("Text2Image pipeline not initialized")
        start = time.perf_counter()
        with torch.inference_mode():
            out_image = self._t2i_pipe(
                prompt,
                negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=torch.manual_seed(seed)
            ).images[0]
        end = time.perf_counter()
        print(f"Generation Time: {end - start:.2f} seconds")
        return out_image
    
    def image2image(self, 
            image:Image, 
            prompt:str, 
            variants:int=2,
            mask_img: pathlib.Path|None = None, 
            negative_prompt:str=" ",
            denoise:float=0.5,
            guidance:float=5.0
        ) -> list:
        print(f"Using device: {self._device}")
        if not self._i2i_pipe:
            raise ValueError("Image2Image pipeline not initialized")
        if self._in_paint and not mask_img:
            raise ValueError("Image-mask is required")
        
        init_image = Image.open(image).convert("RGB")
        if mask_img:
            mask_image = Image.open(mask_img).convert("RGB")
        
        image_variants = []
        start = time.perf_counter()
        for i in range(variants):
            if self._in_paint:
                result = self._i2i_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    mask_image=mask_image,
                    strength=denoise,
                    guidance_scale=guidance,
                )
            else:
                result = self._i2i_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_image,
                    strength=denoise,
                    guidance_scale=guidance,
                )
            image_variants.append(result.images[0])
        end = time.perf_counter()
        print(f"Generation Time: {end - start:.2f} seconds")
        return image_variants
    
    @staticmethod
    def save(image, img_path:pathlib.Path) -> bool:
        if not image:
            return False
        if img_path.is_dir():
            raise ValueError("'%s' should not be a directory", str(img_path))
        image.save(img_path)
        print(f"Image '{str(img_path)}' stored")
        return True
