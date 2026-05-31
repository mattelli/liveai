import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from PIL import Image
import os

# ============================
# CONFIG
# ============================

MODEL = "runwayml/stable-diffusion-v1-5"   # Cambiabile con SDXL
INPUT_IMAGE = ".res/test_a.png"
MASK_IMAGE = ".res/mask.png"  # es. "mask.png" se vuoi inpainting
OUTPUT_DIR = "out_images"

PROMPT = "make the girl smile, same person, same pose, same lighting, same style, 4k resolution, masterpiece"
NEGATIVE = "distorted face, deformed eyes, asymmetry, extra teeth, bad anatomy"

NUM_VARIANTS = 2
DENOISE = 0.45
GUIDANCE = 8

# ============================
# LOAD PIPELINE
# ============================

device = "cuda" if torch.cuda.is_available() else "cpu"

if MASK_IMAGE:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
else:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

pipe = pipe.to(device)

# ============================
# LOAD IMAGES
# ============================

init_image = Image.open(INPUT_IMAGE).convert("RGB")

if MASK_IMAGE:
    mask_image = Image.open(MASK_IMAGE).convert("RGB")
else:
    mask_image = None

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# GENERATE VARIANTS
# ============================

for i in range(NUM_VARIANTS):
    if MASK_IMAGE:
        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            image=init_image,
            mask_image=mask_image,
            strength=DENOISE,
            guidance_scale=GUIDANCE,
        )
    else:
        result = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            image=init_image,
            strength=DENOISE,
            guidance_scale=GUIDANCE,
        )

    out = result.images[0]
    out.save(f"{OUTPUT_DIR}/variant_{i+1}.png")

print(f"Generated {NUM_VARIANTS} images in {OUTPUT_DIR}/")