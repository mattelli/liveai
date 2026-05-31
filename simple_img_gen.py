import pathlib
from src.image_generator import ImageGenerator

"""
Realistic Anime-Style Character Prompt

Prompt:
Ultra-detailed portrait of a young woman, realistic anime style, soft natural lighting, expressive eyes with intricate reflections, smooth skin texture, subtle blush, slightly tousled hair with individual strands visible, wearing a modern casual outfit, cinematic depth of field, 8k resolution, masterpiece, highly detailed, vibrant yet natural colors

Negative Prompt:
blurry, low quality, extra limbs, distorted face, overexposed, underexposed, cartoonish, grainy, watermark, text
"""


subject = "A young blond woman with expressive blue eyes, "
subject_details = "she is wearing a short red top,"
details = "realistic anime style, ultra-detailed half-length portrait, natural colors, soft natural lighting, cinematic depth of field, 4k resolution, masterpiece"

prompt = "Ultra-detailed portrait of a young blond woman, realistic anime style, soft natural lighting, expressive blue eyes with intricate reflections, smooth skin texture, subtle blush, slightly tousled hair with individual strands visible, wearing a modern casual outfit, cinematic depth of field, 8k resolution, masterpiece, highly detailed, vibrant yet natural colors"
negative = "blurry, low quality, extra limbs, distorted face, overexposed, underexposed, cartoonish, grainy, watermark, text, multiple faces"


if __name__ == "__main__":

    out_dir_path = pathlib.Path.cwd() / ".out_images"

    gen = ImageGenerator()
    if not gen.is_t2i_pipe_loaded():
        gen.load_t2i_pipe()
    image = gen.text2image(prompt, steps=50, seed=82)
    img_path = out_dir_path / f"img_{str(id(image))}.jpg"
    gen.save(image, img_path)
