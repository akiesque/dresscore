import json
import torch
import os

def load_outfits(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

disjoint_train = load_outfits(
    "data/polyvore-outfit-dataset/polyvore_outfits/disjoint/train.json"
)

data = torch.load("embeddings/polyvore_clip_embeddings.pt")

image_paths = data["image_paths"]
image_embeddings = data["image_embeddings"]

item_features = {}

for path, emb in zip(image_paths, image_embeddings):
    item_id = os.path.splitext(os.path.basename(path))[0]

    item_features[item_id] = {
        "clip": emb,
        # placeholders for now (we’ll fill later)
        "category_onehot": None,
        "attributes": None,
    }

def construct_outfit_from_json(outfit, item_features):
    outfit_items = []

    for item in outfit["items"]:
        item_id = item["item_id"]

        if item_id in item_features:
            outfit_items.append(item_features[item_id])

    return outfit_items

outfit = disjoint_train[0]
outfit_items = construct_outfit_from_json(outfit, item_features)

print(len(outfit_items))
print(outfit_items[0]["clip"].shape)
