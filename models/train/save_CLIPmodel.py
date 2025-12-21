from compute_CLIP import CLIPEmbedder
import os
import torch
import json
from tqdm import tqdm

DATA_PATH = "data/polyvore-outfit-dataset/polyvore_outfits/images"
TEXT_PATH = "data/polyvore-outfit-dataset/polyvore_outfits/polyvore_item_metadata.json"

# ---------- Load metadata ----------
with open(TEXT_PATH, "r") as f:
    text_data = json.load(f)

image_paths = []
texts = []

for item_id, meta in text_data.items():
    img_path = os.path.join(DATA_PATH, f"{item_id}.jpg")

    if not os.path.exists(img_path):
        continue

    text = meta.get("url_name", "")
    image_paths.append(img_path)
    texts.append(text)

print(f"Matched {len(image_paths)} image–text pairs")

# ---------- CLIP ----------
clip = CLIPEmbedder()
clip.save("models/clip_polyvore_model")

# ---------- Image embeddings ----------
def encode_images_batched(embedder, image_paths, batch_size=32):
    all_feats = []
    num_batches = (len(image_paths) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(image_paths), batch_size),
        total=num_batches,
        desc="Encoding images",
    ):
        batch_paths = image_paths[i:i + batch_size]
        feats = embedder.encode_images(batch_paths)
        all_feats.append(feats)

    return torch.cat(all_feats, dim=0)

image_embeddings = encode_images_batched(
    clip,
    image_paths,
    batch_size=64
)

# ---------- Text embeddings ----------
def encode_text_batched(embedder, texts, batch_size=64):
    all_feats = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(texts), batch_size),
        total=num_batches,
        desc="Encoding text",
    ):
        batch_texts = texts[i:i + batch_size]
        feats = embedder.encode_text(batch_texts)
        all_feats.append(feats)

    return torch.cat(all_feats, dim=0)

text_embeddings = encode_text_batched(
    clip,
    texts,
    batch_size=64
)

# ---------- Sanity check ----------
pairwise_sims = (image_embeddings * text_embeddings).sum(dim=1)

print("Mean image-text similarity:", pairwise_sims.mean().item())


# ---------- Save ----------
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

print("✅ CLIP embeddings saved")