import os
import json
import torch
from typing import Dict, List, Tuple

from models.embedder import CLIPEmbedder


# Same style prompts used in outfit construction
STYLE_PROMPTS = [
    "This outfit is casual and stylish",
    "This outfit has matching colors",
    "This outfit is bold and fashionable",
    "This outfit looks elegant",
    "This outfit is perfect for going out",
]

# Essential categories, to detect classic outfit structure
ESSENTIAL_CATEGORIES = ["tops", "bottoms", "shoes"]


def analyze_color_similarity(embeddings: torch.Tensor) -> Tuple[str | None, float]:
    """
    Recompute color-based heuristic with slightly stricter thresholds.
    Returns (reason, score).
    """
    if embeddings.size(0) < 2:
        return None, 0.0

    sims: List[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(torch.dot(embeddings[i], embeddings[j]).item())

    avg_sim = sum(sims) / len(sims) if sims else 0.0

    # Updated threshold from your note
    if avg_sim > 0.65:
        return "colors match nicely", avg_sim
    elif avg_sim > 0.55:
        return "colors complement each other", avg_sim
    else:
        return None, avg_sim


def analyze_category_combination(categories: List[str]) -> Tuple[str | None, float]:
    """
    Recompute category-based heuristic, making it 'compete' with color based
    by lowering its score a bit when there are many categories.
    Returns (reason, score).
    """
    has_essentials = all(cat in categories for cat in ESSENTIAL_CATEGORIES)

    if has_essentials and len(categories) == 3:
        # Classic 3-piece outfit
        return "classic outfit combination", 0.7
    elif has_essentials and len(categories) >= 4:
        # Your requested change: lower weight vs. before
        return "complete and well-balanced outfit", 0.6
    elif "outerwear" in categories:
        return "layered outfit with outerwear", 0.5
    else:
        return None, 0.0


def analyze_style_consistency(
    embeddings: torch.Tensor, style_embeddings: torch.Tensor
) -> Tuple[str | None, float]:
    """
    Zero-shot match between average outfit embedding and style prompts.
    Uses a slightly lower best_sim threshold to allow more matches.
    """
    if embeddings.size(0) < 2:
        return None, 0.0

    outfit_emb = embeddings.mean(dim=0)
    outfit_emb = outfit_emb / outfit_emb.norm()

    sims: List[float] = []
    for style_emb in style_embeddings:
        sims.append(torch.dot(outfit_emb, style_emb).item())

    best_idx = max(range(len(sims)), key=lambda i: sims[i])
    best_sim = sims[best_idx]

    # Slightly lower than before so more outfits get a style explanation
    if best_sim > 0.25:
        return STYLE_PROMPTS[best_idx].lower(), best_sim
    return None, best_sim


def build_outfit_from_folder(outfit_dir: str) -> Tuple[Dict[str, Dict], List[str]]:
    """
    Reconstruct a minimal 'outfit' dict + list of image paths from folder contents.
    Assumes filenames like 'category_itemid.jpg'.
    """
    outfit: Dict[str, Dict] = {}
    image_paths: List[str] = []

    for fname in os.listdir(outfit_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        category = fname.split("_")[0].lower()
        item_id = os.path.splitext("_".join(fname.split("_")[1:]))[0]
        path = os.path.join(outfit_dir, fname)

        outfit[category] = {
            "item_id": item_id,
            "image_path": path,
        }
        image_paths.append(path)

    # Keep a stable ordering
    image_paths.sort()
    return outfit, image_paths


def main() -> None:
    outfits_root = "outfits"
    if not os.path.exists(outfits_root):
        print("No 'outfits' folder found. Run the generator first.")
        return

    print("Loading CLIP model (for 10 outfits only)...")
    embedder = CLIPEmbedder()

    print("Encoding style prompts...")
    style_embeddings = embedder.encode_text(STYLE_PROMPTS)

    # Process only the existing top outfit folders
    outfit_dirs = [
        os.path.join(outfits_root, d)
        for d in os.listdir(outfits_root)
        if os.path.isdir(os.path.join(outfits_root, d))
    ]
    outfit_dirs.sort()

    if not outfit_dirs:
        print("No outfits found to re-evaluate.")
        return

    print(f"Found {len(outfit_dirs)} outfit folders to re-evaluate.")

    for outfit_dir in outfit_dirs:
        print(f"\nRe-evaluating {os.path.basename(outfit_dir)}...")

        outfit, image_paths = build_outfit_from_folder(outfit_dir)
        if len(image_paths) < 2:
            print("  Skipping (not enough images).")
            continue

        # Recompute embeddings just for these images
        embeddings = embedder.encode_images(image_paths)

        # Heuristic reasons
        categories = list(outfit.keys())
        color_reason, _ = analyze_color_similarity(embeddings)
        cat_reason, _ = analyze_category_combination(categories)

        # Let heuristics "compete": pick non-null one with higher implicit score
        candidate_reasons: List[Tuple[str, float, str]] = []
        if color_reason:
            candidate_reasons.append((color_reason, 1.0, "color"))
        if cat_reason:
            # Use the same scores as in analyze_category_combination
            _, cat_score = analyze_category_combination(categories)
            candidate_reasons.append((cat_reason, cat_score, "category"))

        heuristic_reason = None
        heuristic_type = None
        if candidate_reasons:
            candidate_reasons.sort(key=lambda x: x[1], reverse=True)
            heuristic_reason, _, heuristic_type = candidate_reasons[0]

        # Text-based style reason
        text_reason, _ = analyze_style_consistency(embeddings, style_embeddings)

        # Combined note
        final_note: List[str] = []
        if heuristic_reason:
            final_note.append(heuristic_reason)
        if text_reason:
            final_note.append(text_reason)

        if final_note:
            print("  📝 Note:", " and ".join(final_note))

        # Load and update reasoning.json if present
        reasoning_path = os.path.join(outfit_dir, "reasoning.json")
        reasoning_data = {}
        if os.path.exists(reasoning_path):
            try:
                with open(reasoning_path, "r") as f:
                    reasoning_data = json.load(f)
            except Exception:
                reasoning_data = {}

        reasoning_data["heuristic_reason"] = heuristic_reason
        reasoning_data["text_reason"] = text_reason
        reasoning_data["note"] = " and ".join(final_note) if final_note else None

        with open(reasoning_path, "w") as f:
            json.dump(reasoning_data, f, indent=2)

        print("  Updated reasoning.json")


if __name__ == "__main__":
    main()

