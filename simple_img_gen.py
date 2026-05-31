from src.image_generator import ImageGenerator

"""
Realistic Anime-Style Character Prompt

Prompt:
Ultra-detailed portrait of a young woman, realistic anime style, soft natural lighting, expressive eyes with intricate reflections, smooth skin texture, subtle blush, slightly tousled hair with individual strands visible, wearing a modern casual outfit, cinematic depth of field, 8k resolution, masterpiece, highly detailed, vibrant yet natural colors

Negative Prompt:
blurry, low quality, extra limbs, distorted face, overexposed, underexposed, cartoonish, grainy, watermark, text
"""


subject = "Young blond woman with expressive blue eyes, "
subject_details = "she is wearing a short red top,"
details = "realistic anime style, ultra-detailed half-length portrait, natural colors, soft natural lighting, cinematic depth of field, 4k resolution, masterpiece"
negative = "blurry, low quality, extra limbs, distorted face, overexposed, underexposed, cartoonish, grainy, watermark, text"


if __name__ == "__main__":

    prompts = [
        subject + subject_details + details,
    ]
    for prompt in prompts:
        gen = ImageGenerator()
        image = gen.text2image(prompt, steps=25)
        gen.store(image)
