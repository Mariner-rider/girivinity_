"""
VisionEncoder — projects CLIP ViT patch embeddings into the LLM's token embedding space.

Architecture:
  PIL Image
    → CLIPProcessor (resize, normalize, patchify)
    → CLIPVisionModel (frozen ViT-L/14)
    → patch embeddings: (N_patches=256, encoder_dim=1024)
    → LayerNorm
    → nn.Linear(encoder_dim, projection_dim)   ← ONLY trainable component
    → (N_patches, projection_dim=3072)

These N_patches tokens are prepended to the text token sequence before
the LLM's first transformer layer.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")
_transformers = importlib.import_module("transformers")
CLIPVisionModel = _transformers.CLIPVisionModel
CLIPProcessor = _transformers.CLIPProcessor
Image = importlib.import_module("PIL.Image")


class VisionEncoder(nn.Module):
    def __init__(
        self,
        clip_model_id: str,
        encoder_dim: int,
        projection_dim: int,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(clip_model_id)
        self.encoder = CLIPVisionModel.from_pretrained(clip_model_id).to(device)
        self.encoder.eval()
        # Freeze ALL encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        # Trainable projection
        self.norm = nn.LayerNorm(encoder_dim).to(device)
        self.projection = nn.Linear(encoder_dim, projection_dim, bias=False).to(device)
        nn.init.xavier_uniform_(self.projection.weight)

    def encode(self, image: Any) -> torch.Tensor:
        """image: PIL.Image or path string. Returns (N_patches, projection_dim)."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        # outputs.last_hidden_state: (1, N_patches+1, encoder_dim) — drop [CLS]
        patch_embeds = outputs.last_hidden_state[:, 1:, :]  # (1, N_patches, encoder_dim)
        patch_embeds = self.norm(patch_embeds)
        projected = self.projection(patch_embeds)  # (1, N_patches, projection_dim)
        return projected.squeeze(0)  # (N_patches, projection_dim)

    def encode_batch(self, images: list[Any]) -> torch.Tensor:
        """Returns (B, N_patches, projection_dim)."""
        return torch.stack([self.encode(img) for img in images], dim=0)

    def save_projection(self, path: str) -> None:
        torch.save(
            {"norm": self.norm.state_dict(), "projection": self.projection.state_dict()},
            path,
        )

    def load_projection(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.norm.load_state_dict(ckpt["norm"])
        self.projection.load_state_dict(ckpt["projection"])
