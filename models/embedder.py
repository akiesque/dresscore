import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_safetensors=True
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        self.model.eval()  # IMPORTANT: inference mode

    @torch.no_grad()
    def encode_images(self, image_paths):
        images = [Image.open(p).convert("RGB") for p in image_paths]

        inputs = self.processor(
            images=images,
            return_tensors="pt"
        ).to(self.device)

        image_features = self.model.get_image_features(**inputs)

        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )

        return image_features.cpu()
    
    @torch.no_grad()
    def encode_text(self, texts):
        """Encode text prompts into embeddings."""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        text_features = self.model.get_text_features(**inputs)
        
        # Normalize for cosine similarity
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )
        
        return text_features.cpu()