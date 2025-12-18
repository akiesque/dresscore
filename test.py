import json
from models.embedder import CLIPEmbedder
from models.scorer import compatibility_score

# Load the JSON file to get outfit data
with open("data/polyvore-outfit-dataset/polyvore_outfits/disjoint/valid.json", "r") as f:
    outfits = json.load(f)

# Get the first outfit and extract image paths from item_ids
if outfits:
    first_outfit = outfits[0]
    base_image_dir = "data/polyvore-outfit-dataset/polyvore_outfits/images"
    
    # Extract item_ids and construct image paths
    image_paths = [
        f"{base_image_dir}/{item['item_id']}.jpg" 
        for item in first_outfit.get("items", [])[:2]  # Get first 2 items for testing
    ]
    
    print(f"Using image paths: {image_paths}")
    
    embedder = CLIPEmbedder()
    embeddings = embedder.encode_images(image_paths)
    score = compatibility_score(embeddings)
    print("Compatibility Score:", score)
else:
    print("No outfits found in the JSON file")
