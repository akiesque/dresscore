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

item_features = {}

for path, clip_emb in zip(data["image_paths"], data["image_embeddings"]):
    item_id = os.path.splitext(os.path.basename(path))[0]
    item_features[item_id] = {
        "clip": clip_emb
    }

print("Total CLIP items:", len(item_features))
print("Example keys:", list(item_features.keys())[:5])


# def construct_outfit_from_json(outfit, item_features):
#     outfit_items = []

#     for item in outfit["items"]:
#         item_id = item["item_id"]

#         if item_id in item_features:
#             outfit_items.append(item_features[item_id])

#     return outfit_items

# outfit = disjoint_train[0]
# outfit_items = construct_outfit_from_json(outfit, item_features)

# if len(outfit_items) == 0:
#     print("⚠️ Empty outfit, skipping")
# else:
#     print(outfit_items[0]["clip"].shape)

# print(len(outfit_items))
# print(outfit_items[0]["clip"].shape)
