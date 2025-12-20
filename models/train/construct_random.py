import json
import torch
from models.train.model_train import num_classes, attr_dim

def load_outfits(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

disjoint_train = load_outfits(
    "data/polyvore-outfit-dataset/polyvore_outfits/disjoint/train.json"
)

item_features = {
    item_id: {
        "clip": torch.Tensor([512]),
        "category_onehot": torch.Tensor([num_classes]),
        "attributes": torch.Tensor([attr_dim]),
    }
}

item_features = torch.load("embeddings/polyvore_clip_embeddings.pt")

def construct_outfit_from_json(outfit, item_features):
    features = []

    for item in outfit["items"]:
        item_id = item["item_id"]
        if item_id in item_features:
            features.append(item_features[item_id])

    return features
