from sentence_transformers import SentenceTransformer
from PIL import Image
from pathlib import Path

image_path = Path.cwd() / ".res" / "two_dogs_in_snow.jpg"

# Load CLIP model
model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")

# Encode an image:
img_emb = model.encode(Image.open(str(image_path)))

# Encode text descriptions
text_emb = model.encode(
    ["Two dogs in the snow", "A cat on a table", "A picture of London at night"]
)

# Compute similarities
similarity_scores = model.similarity(img_emb, text_emb)
print(similarity_scores)
