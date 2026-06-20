"""
AudioEncoder — projects Whisper mel-spectrogram features into LLM embedding space.

Architecture:
  Audio file
    → whisper.load_audio() + pad_or_trim()
    → log_mel_spectrogram()
    → Whisper encoder (frozen)
    → encoder output: (T, whisper_dim)   T ≈ 1500 for 30s audio
    → mean pooling over T → (whisper_dim,)
    → LayerNorm
    → nn.Linear(whisper_dim, projection_dim)  ← trainable
    → (1, projection_dim) — a single "audio token"

For longer audio: split into 30s chunks, encode each, mean-pool across chunks.
"""

from __future__ import annotations

import importlib


torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")
whisper = importlib.import_module("whisper")
np = importlib.import_module("numpy")


class AudioEncoder(nn.Module):
    def __init__(
        self,
        whisper_size: str,
        encoder_dim: int,
        projection_dim: int,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.whisper_dim = encoder_dim
        model = whisper.load_model(whisper_size, device=device)
        self.encoder = model.encoder  # Only encoder, not decoder
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.norm = nn.LayerNorm(encoder_dim).to(device)
        self.projection = nn.Linear(encoder_dim, projection_dim, bias=False).to(device)
        nn.init.xavier_uniform_(self.projection.weight)

    def encode(self, audio_path: str) -> torch.Tensor:
        """Returns (1, projection_dim) — single audio token."""
        audio = whisper.load_audio(audio_path)
        audio = np.asarray(audio)
        # Handle audio longer than 30s: chunk and mean-pool
        chunk_size = whisper.audio.SAMPLE_RATE * 30
        chunks = [audio[i : i + chunk_size] for i in range(0, len(audio), chunk_size)]
        if not chunks:
            chunks = [audio]
        chunk_encodings = []
        for chunk in chunks:
            padded = whisper.pad_or_trim(chunk)
            mel = whisper.log_mel_spectrogram(padded).to(self.device)
            with torch.no_grad():
                encoded = self.encoder(mel.unsqueeze(0))  # (1, T, dim)
            pooled = encoded.mean(dim=1)  # (1, dim)
            chunk_encodings.append(pooled)
        audio_feat = torch.stack(chunk_encodings, dim=0).mean(dim=0)  # (1, dim)
        audio_feat = self.norm(audio_feat)
        projected = self.projection(audio_feat)  # (1, projection_dim)
        return projected  # (1, projection_dim)

    def encode_batch(self, audio_paths: list[str]) -> torch.Tensor:
        """Returns (B, 1, projection_dim)."""
        return torch.stack([self.encode(p) for p in audio_paths], dim=0)

    def save_projection(self, path: str) -> None:
        torch.save(
            {"norm": self.norm.state_dict(), "projection": self.projection.state_dict()},
            path,
        )

    def load_projection(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.norm.load_state_dict(ckpt["norm"])
        self.projection.load_state_dict(ckpt["projection"])
