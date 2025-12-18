import json
import os
import shutil
import random
import torch
from collections import defaultdict
from models.embedder import CLIPEmbedder
from models.scorer import compatibility_score

seed = 42
random.seed(seed)
torch.manual_seed(seed)

# --- Paths ---
metadata_path = "data/polyvore-outfit-dataset/polyvore_outfits/polyvore_item_metadata.json"
images_folder = "data/polyvore-outfit-dataset/polyvore_outfits/images"
output_folder = "outfits"

# Essential categories for a complete outfit
ESSENTIAL_CATEGORIES = ["tops", "bottoms", "shoes"]
OPTIONAL_CATEGORIES = ["bags", "outerwear", "accessories", "jewellery", "hats", "scarves"]

# Style prompts for zero-shot reasoning
STYLE_PROMPTS = [
    "This outfit is casual and stylish",
    "This outfit has matching colors",
    "This outfit is bold and fashionable",
    "This outfit looks elegant",
    "This outfit is perfect for going out",
]

# Clear and create output folder
if os.path.exists(output_folder):
    print(f"Clearing existing outfits folder...")
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)
print(f"✅ Output folder '{output_folder}' ready")

# Load embedder
print("Loading CLIP embedder...")
embedder = CLIPEmbedder()

# Load metadata
print("Loading item metadata...")
with open(metadata_path, "r") as f:
    metadata = json.load(f)

print(f"Found {len(metadata)} items in metadata")

# Group items by semantic category
print("Grouping items by category...")
items_by_category = defaultdict(list)

for item_id, item_data in metadata.items():
    semantic_category = item_data.get("semantic_category", "").lower()
    if semantic_category:
        image_path = os.path.join(images_folder, f"{item_id}.jpg")
        if os.path.exists(image_path):
            items_by_category[semantic_category].append({
                "item_id": item_id,
                "image_path": image_path,
                "category_id": item_data.get("category_id", ""),
                "title": item_data.get("title", "")
            })

# Print category statistics
print("\nItems per category:")
for category in sorted(items_by_category.keys()):
    print(f"  {category}: {len(items_by_category[category])} items")

# Check if we have enough items in essential categories
for cat in ESSENTIAL_CATEGORIES:
    if cat not in items_by_category or len(items_by_category[cat]) == 0:
        print(f"\n⚠️  WARNING: No items found in essential category '{cat}'!")
        print("Cannot create complete outfits without this category.")
        exit(1)

print(f"\n✅ All essential categories have items available")

# Precompute embeddings for all items
print("Precomputing item embeddings...")
item_embeddings_cache = {}
all_image_paths = []
all_items = []

for category_items in items_by_category.values():
    for item in category_items:
        all_image_paths.append(item["image_path"])
        all_items.append(item)

# Batch encode all items for efficiency
print(f"Encoding {len(all_image_paths)} items...")
batch_size = 100
for i in range(0, len(all_image_paths), batch_size):
    batch_paths = all_image_paths[i:i+batch_size]
    batch_embeddings = embedder.encode_images(batch_paths)
    for j, path in enumerate(batch_paths):
        item_embeddings_cache[path] = batch_embeddings[j]
    if (i + batch_size) % 500 == 0 or i + batch_size >= len(all_image_paths):
        print(f"  Encoded {min(i + batch_size, len(all_image_paths))}/{len(all_image_paths)} items...")

print("✅ All item embeddings precomputed")

print("Encoding style prompts...")
style_embeddings = embedder.encode_text(STYLE_PROMPTS)

# Heuristic reasoning functions
def analyze_color_similarity(embeddings):
    if len(embeddings) < 2:
        return None, 0.0
    
    # Compute pairwise similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = torch.dot(embeddings[i], embeddings[j]).item()
            similarities.append(sim)
    
    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
    
    if avg_sim > 0.7:
        return "colors match nicely", avg_sim
    elif avg_sim > 0.5:
        return "colors complement each other", avg_sim
    else:
        return None, avg_sim

def analyze_category_combination(outfit_categories):
    has_essentials = all(cat in outfit_categories for cat in ESSENTIAL_CATEGORIES)
    
    if has_essentials and len(outfit_categories) == 3:
        return "classic outfit combination", 1.0
    elif has_essentials and len(outfit_categories) >= 4:
        return "complete and well-balanced outfit", 1.0
    elif "outerwear" in outfit_categories:
        return "layered outfit with outerwear", 0.8
    else:
        return None, 0.0

def analyze_style_consistency(embeddings, style_embeddings, prompts):
    if len(embeddings) < 2:
        return None, 0.0
    
    # Average outfit embedding
    outfit_embedding = embeddings.mean(dim=0)
    outfit_embedding = outfit_embedding / outfit_embedding.norm()
    
    # Compute similarity with each style prompt
    similarities = []
    for style_emb in style_embeddings:
        sim = torch.dot(outfit_embedding, style_emb).item()
        similarities.append(sim)
    
    # Find best matching prompt
    best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    best_sim = similarities[best_idx]
    
    if best_sim > 0.3:  # Threshold for meaningful match
        return prompts[best_idx].lower(), best_sim
    else:
        return None, best_sim

def get_heuristic_reason(outfit, embeddings):
    reasons = []
    
    # Color similarity
    color_reason, color_score = analyze_color_similarity(embeddings)
    if color_reason:
        reasons.append((color_reason, color_score, "color"))
    
    # Category combination
    cat_reason, cat_score = analyze_category_combination(list(outfit.keys()))
    if cat_reason:
        reasons.append((cat_reason, cat_score, "category"))
    
    # Return best heuristic reason
    if reasons:
        reasons.sort(key=lambda x: x[1], reverse=True)
        return reasons[0][0], reasons[0][2]
    return None, None

def get_text_reason(embeddings, style_embeddings, prompts):
    reason, score = analyze_style_consistency(embeddings, style_embeddings, prompts)
    return reason, score

# Function to construct an outfit
def construct_outfit(items_by_category, include_optional=True):
    outfit = {}
    image_paths = []
    
    # Add essential items
    for category in ESSENTIAL_CATEGORIES:
        if category in items_by_category and items_by_category[category]:
            item = random.choice(items_by_category[category])
            outfit[category] = item
            image_paths.append(item["image_path"])
    
    # Add optional items (randomly select 1-3 optional categories)
    if include_optional:
        available_optional = [cat for cat in OPTIONAL_CATEGORIES 
                           if cat in items_by_category and items_by_category[cat]]
        if available_optional:
            num_optional = random.randint(1, min(3, len(available_optional)))
            selected_optional = random.sample(available_optional, num_optional)
            for category in selected_optional:
                item = random.choice(items_by_category[category])
                outfit[category] = item
                image_paths.append(item["image_path"])
    
    return outfit, image_paths

# Generate and test outfits
print(f"\nGenerating and testing outfits...")
print("=" * 60)

outfit_scores = []
num_outfits_to_test = 1000  # Test 1000 random outfit combinations

for i in range(num_outfits_to_test):
    if (i + 1) % 100 == 0:
        print(f"Testing outfit {i + 1}/{num_outfits_to_test}...")
    
    try:
        outfit, image_paths = construct_outfit(items_by_category, include_optional=True)
        
        if len(image_paths) < 2:
            continue
        
        # Get precomputed embeddings for this outfit
        outfit_embeddings = torch.stack([item_embeddings_cache[path] for path in image_paths])
        
        # Compute compatibility score using precomputed embeddings
        score = compatibility_score(outfit_embeddings)
        
        # Get reasoning
        heuristic_reason, heuristic_type = get_heuristic_reason(outfit, outfit_embeddings)
        text_reason, text_score = get_text_reason(outfit_embeddings, style_embeddings, STYLE_PROMPTS)
        
        outfit_scores.append({
            "outfit": outfit,
            "score": score,
            "image_paths": image_paths,
            "item_count": len(image_paths),
            "heuristic_reason": heuristic_reason,
            "heuristic_type": heuristic_type,
            "text_reason": text_reason,
            "text_score": text_score
        })
    except Exception as e:
        print(f"Error processing outfit {i + 1}: {e}")
        continue

# Sort by compatibility score (descending)
outfit_scores.sort(key=lambda x: x["score"], reverse=True)

# Get top 10
top_10 = outfit_scores[:10]

print(f"\n{'=' * 60}")
print(f"Top 10 most compatible constructed outfits:")
print(f"{'=' * 60}\n")

# Copy images to output folder
for rank, outfit_data in enumerate(top_10, 1):
    outfit = outfit_data["outfit"]
    score = outfit_data["score"]
    image_paths = outfit_data["image_paths"]
    heuristic_reason = outfit_data.get("heuristic_reason")
    text_reason = outfit_data.get("text_reason")
    
    print(f"#{rank} - Score: {score:.2f}/100 ({len(image_paths)} items)")
    print(f"  Categories: {', '.join(sorted(outfit.keys()))}")
    
    # Print combined reasoning note
    final_note = []
    if heuristic_reason:
        final_note.append(heuristic_reason)
    if text_reason:
        final_note.append(text_reason)
    if final_note:
        print(f"  📝 Note: {' and '.join(final_note)}")
    
    # Create folder for this outfit
    outfit_dir = os.path.join(output_folder, f"outfit_{rank:02d}_score_{score:.1f}")
    os.makedirs(outfit_dir, exist_ok=True)
    
    # Copy images with category labels
    for category, item in outfit.items():
        item_id = item["item_id"]
        src_path = item["image_path"]
        dest_path = os.path.join(outfit_dir, f"{category}_{item_id}.jpg")
        shutil.copy2(src_path, dest_path)
        print(f"    {category}: {item_id}")
    
    # Save reasoning to a JSON file
    reasoning_data = {
        "score": score,
        "item_count": len(image_paths),
        "categories": list(outfit.keys()),
        "heuristic_reason": heuristic_reason,
        "text_reason": text_reason
    }
    reasoning_path = os.path.join(outfit_dir, "reasoning.json")
    with open(reasoning_path, "w") as f:
        json.dump(reasoning_data, f, indent=2)
    
    print()

print(f"✅ Top 10 constructed outfits saved to '{output_folder}/' folder")
