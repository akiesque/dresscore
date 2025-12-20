import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIPEmbedder:
    def __init__(self, device=None, model_name="openai/clip-vit-base-patch32"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.model = CLIPModel.from_pretrained(
            model_name,
            use_safetensors=True
        ).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()  # inference mode

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
    def encode_images_batched(self, image_paths, batch_size=32):
        all_feats = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            feats = self.encode_images(batch_paths)
            all_feats.append(feats)

        return torch.cat(all_feats, dim=0)

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32,
        ).to(self.device)

        text_features = self.model.get_text_features(**inputs)

        # Normalize for cosine similarity
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

        return text_features.cpu()

    # ---------- SAVE / LOAD ----------
    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        print(f"CLIP model + processor saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: str, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        embedder = cls.__new__(cls)
        embedder.device = device
        embedder.model_name = save_dir

        embedder.model = CLIPModel.from_pretrained(
            save_dir,
            use_safetensors=True
        ).to(device)

        embedder.processor = CLIPProcessor.from_pretrained(save_dir)
        embedder.model.eval()

        return embedder
