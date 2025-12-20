from compute_CLIP import CLIPEmbedder
import os
import torch
import json

DATA_PATH = "data/polyvore-outfit-dataset/polyvore_outfits/images"
TEXT_PATH = "data/polyvore-outfit-dataset/polyvore_outfits/polyvore_item_metadata.json"

with open(TEXT_PATH, "r") as f:
    text_data = json.load(f)

image_paths = []
texts = []

for item_id, meta in text_data.items():
    img_path = os.path.join(DATA_PATH, f"{item_id}.jpg")

    if not os.path.exists(img_path):
        continue  # skip missing images

    # choose what text you want CLIP to see
    text = f'{meta["url_name"]}'

    image_paths.append(img_path)
    texts.append(text)

clip = CLIPEmbedder()
clip.save("models/clip_polyvore_model")

image_embeddings = clip.encode_images_batched(
    image_paths,
    batch_size=32
)

text_embeddings = clip.encode_text(texts)

sims = image_embeddings @ text_embeddings.T
print(sims.diag().mean().item())

os.makedirs("embeddings", exist_ok=True)

torch.save(
    {
        "image_paths": image_paths,
        "texts": texts,
        "image_embeddings": image_embeddings,
        "text_embeddings": text_embeddings,
    },
    "embeddings/polyvore_clip_embeddings.pt"
)

