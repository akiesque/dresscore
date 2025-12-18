import json
import os
import cv2
import numpy as np


def extract_color(img: np.ndarray) -> np.ndarray:
    """Return dominant HSV color for an RGB image as a 3‑element vector."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    pixels = hsv.reshape(-1, 3).astype(np.float32)
    _compactness, _labels, centers = cv2.kmeans(
        pixels,
        1,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    return centers[0]


def extract_brightness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(gray.mean())


def extract_texture(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.mean())


def collect_item_ids(json_path: str) -> set[str]:
    """Collect all item_ids present in a Polyvore split JSON."""
    with open(json_path, "r") as f:
        outfits = json.load(f)

    ids: set[str] = set()
    for outfit in outfits:
        for it in outfit.get("items", []):
            item_id = it.get("item_id")
            if item_id:
                ids.add(item_id)
    return ids


def main() -> None:
    """
    Run attribute extraction (color, brightness, texture) for all items
    in the nondisjoint Polyvore splits and save to a JSON file.
    """
    base_dir = "data/polyvore-outfit-dataset/polyvore_outfits"
    images_folder = os.path.join(base_dir, "images")
    nondisjoint_dir = os.path.join(base_dir, "nondisjoint")

    train_json = os.path.join(nondisjoint_dir, "train.json")
    valid_json = os.path.join(nondisjoint_dir, "valid.json")
    test_json = os.path.join(nondisjoint_dir, "test.json")

    print("Collecting item IDs from nondisjoint splits...")
    item_ids = set()
    for path in (train_json, valid_json, test_json):
        if os.path.exists(path):
            item_ids |= collect_item_ids(path)

    item_ids = sorted(item_ids)
    print(f"Total unique item_ids found: {len(item_ids)}")

    attributes: dict[str, dict] = {}

    for idx, item_id in enumerate(item_ids, start=1):
        img_path = os.path.join(images_folder, f"{item_id}.jpg")
        if not os.path.exists(img_path):
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # Convert BGR (OpenCV default) to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            dom_hsv = extract_color(img_rgb)
            brightness = extract_brightness(img_rgb)
            texture = extract_texture(img_rgb)
        except Exception as e:  # robustness for any weird images
            print(f"Skipping {item_id} due to error: {e}")
            continue

        attributes[item_id] = {
            "color_h": float(dom_hsv[0]),
            "color_s": float(dom_hsv[1]),
            "color_v": float(dom_hsv[2]),
            "brightness": float(brightness),
            "texture": float(texture),
        }

        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(item_ids)} items...")

    out_path = os.path.join(nondisjoint_dir, "item_attributes.json")
    os.makedirs(nondisjoint_dir, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(attributes, f)

    print(f"Saved attributes for {len(attributes)} items to {out_path}")


if __name__ == "__main__":
    main()
