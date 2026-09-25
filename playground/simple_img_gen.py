import pathlib

import sys
MAIN_PATH = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MAIN_PATH))

from src.image_generator import ImageGenerator


def image_gen():
    """
    Realistic Anime-Style Character Prompt

    Prompt:
    Ultra-detailed portrait of a young woman, realistic anime style, soft natural lighting, 
    expressive eyes with intricate reflections, smooth skin texture, subtle blush, slightly tousled hair with individual strands visible, 
    wearing a modern casual outfit, cinematic depth of field, 8k resolution, masterpiece, highly detailed, vibrant yet natural colors

    Negative Prompt:
    blurry, low quality, extra limbs, distorted face, overexposed, underexposed, cartoonish, grainy, watermark, text
    """

    subject = "A young blond woman with expressive blue eyes, "
    subject_details = "she is wearing a short red top,"
    details = "realistic anime style, ultra-detailed half-length portrait, natural colors, soft natural lighting, cinematic depth of field, 4k resolution, masterpiece"

    prompt = subject + subject_details + details

    gen = ImageGenerator()
    if not gen.is_t2i_pipe_loaded():
        gen.load_t2i_pipe()
    image = gen.text2image(prompt, steps=50, seed=82)
    out_dir_path = pathlib.Path.cwd() / ".out_images"
    img_path = out_dir_path / f"img_{str(id(image))}.jpg"
    gen.save(image, img_path)


def image_change(variants:int=2):
    """
    Change facial expression based on image mask for to the same subject.
    """

    prompt = "make the girl smile with a wide smile, same person, same pose, same lighting, same style, 4k resolution, masterpiece"
    negative = "distorted face, deformed eyes, asymmetry, extra teeth, bad anatomy"
    input_image = pathlib.Path.cwd() / ".res" / "test_a.png"
    mask_image = pathlib.Path.cwd() / ".res" / "test_a_mask.png"

    gen = ImageGenerator()
    if not gen.is_i2i_pipe_loaded():
        gen.load_i2i_pipe(in_paint=True)
    
    out_dir_path = pathlib.Path.cwd() / ".out_images"
    for i in range(variants):
        print(f"Generating Image Variant {i+1}/{variants}")
        img_var = gen.image2image(
            image=input_image,
            prompt=prompt, 
            mask=mask_image, 
            negative_prompt=negative, 
            denoise=0.5,
            guidance=9
        )
        img_path = out_dir_path / f"img_{str(id(prompt))}_{i}.jpg"
        gen.save(img_var, img_path)


if __name__ == "__main__":
    # image_gen()
    image_change()
    
