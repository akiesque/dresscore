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

def encode_text_batched(embedder, texts, batch_size=64):
    all_feats = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        feats = embedder.encode_text(batch_texts)
        all_feats.append(feats)

        if i % (batch_size * 20) == 0:
            print(f"Encoded {i}/{len(texts)} texts")

    return torch.cat(all_feats, dim=0)

text_embeddings = encode_text_batched(
    clip,
    texts,
    batch_size=64   
)

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

