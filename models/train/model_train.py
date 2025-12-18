import json
import os
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0
from tqdm.auto import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- Dataset ----------

class PolyvoreItemDataset(Dataset):
    
    def __init__(
        self,
        items: List[Tuple[str, int]],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, label = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_split_items(
    json_path: str,
    images_folder: str,
    metadata: dict,
) -> List[Tuple[str, str]]:
    with open(json_path, "r") as f:
        outfits = json.load(f)

    items: List[Tuple[str, str]] = []
    for outfit in outfits:
        for it in outfit.get("items", []):
            item_id = it.get("item_id")
            if not item_id:
                continue

            meta = metadata.get(item_id)
            if not meta:
                continue

            cat = meta.get("semantic_category")
            if not cat:
                continue

            img_path = os.path.join(images_folder, f"{item_id}.jpg")
            if not os.path.exists(img_path):
                continue

            items.append((img_path, cat.lower()))

    return items


def build_splits():
   
    base_dir = "data/polyvore-outfit-dataset/polyvore_outfits"
    images_folder = os.path.join(base_dir, "images")
    metadata_path = os.path.join(base_dir, "polyvore_item_metadata.json")
    nondisjoint_dir = os.path.join(base_dir, "nondisjoint")

    train_json = os.path.join(nondisjoint_dir, "train.json")
    valid_json = os.path.join(nondisjoint_dir, "valid.json")
    test_json = os.path.join(nondisjoint_dir, "test.json")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print("Building item lists from nondisjoint JSON splits...")
    train_items_raw = load_split_items(train_json, images_folder, metadata)
    valid_items_raw = load_split_items(valid_json, images_folder, metadata)
    test_items_raw = load_split_items(test_json, images_folder, metadata)

    print(f"Train items (raw): {len(train_items_raw)}")
    print(f"Valid items (raw): {len(valid_items_raw)}")
    print(f"Test items (raw):  {len(test_items_raw)}")

    # Decide on top 5 semantic categories by frequency in train split
    counter = Counter(cat for _, cat in train_items_raw)
    top5 = [c for c, _ in counter.most_common(5)]
    print("Top 5 categories:", top5)

    cat_to_idx = {cat: i for i, cat in enumerate(top5)}

    def filter_and_index(items_raw: List[Tuple[str, str]]) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        for img_path, cat in items_raw:
            if cat in cat_to_idx:
                out.append((img_path, cat_to_idx[cat]))
        return out

    train_items = filter_and_index(train_items_raw)
    valid_items = filter_and_index(valid_items_raw)
    test_items = filter_and_index(test_items_raw)

    print(f"Train items (filtered): {len(train_items)}")
    print(f"Valid items (filtered): {len(valid_items)}")
    print(f"Test items (filtered):  {len(test_items)}")

    return train_items, valid_items, test_items, cat_to_idx


def main():
    # ----- Data -----
    train_items, valid_items, test_items, cat_to_idx = build_splits()
    num_classes = len(cat_to_idx)
    print(f"Using {num_classes} classes.")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_ds = PolyvoreItemDataset(train_items, transform=transform)
    valid_ds = PolyvoreItemDataset(valid_items, transform=transform)
    test_ds = PolyvoreItemDataset(test_items, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # ----- Model -----
    model = efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    patience = 5           
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # ----- Training loop -----
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [train]",
            leave=False,
        )

        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_so_far = total_loss / len(progress_bar)
            progress_bar.set_postfix(loss=f"{avg_so_far:.4f}")

        avg_train_loss = total_loss / max(len(train_loader), 1)
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

        # Simple validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= max(len(valid_loader), 1)
        val_acc = correct / total if total > 0 else 0.0


        # ----- Early stopping check -----

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Optional: save best model
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_to_idx": cat_to_idx,
                },
                "models/checkpoints/efficientnet_b0_best.pth",
            )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"Best Val Loss: {best_val_loss:.4f}"
            )
            break


    # ----- Final test evaluation -----
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {test_acc:.4f}")

    # ----- Save model -----
    os.makedirs("models/checkpoints", exist_ok=True)
    ckpt_path = "models/checkpoints/efficientnet_b0_polyvore_categories.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": cat_to_idx,
        },
        ckpt_path,
    )
    print(f"Saved model checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

