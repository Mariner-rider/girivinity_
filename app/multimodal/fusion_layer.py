"""
MultimodalFusionLayer — combines text, image, and audio token sequences.

Three fusion modes:

CONCAT (simplest):
  [audio_tokens | image_tokens | text_tokens] → single sequence
  Pro: no extra parameters. Con: sequence grows large.

CROSS_ATTENTION (recommended):
  Text tokens = Query
  Image + audio tokens = Key/Value
  2-layer cross-attention block
  Output shape same as text_tokens
  Pro: text tokens selectively attend to relevant image/audio regions

GATED (most expressive):
  Per-modality learned gate: sigmoid(W_gate @ modal_mean) in [0,1]
  Weighted sum: text + gate_img * image_mean + gate_audio * audio_mean
  All projected to same dim, then concat with original text
"""

from __future__ import annotations

import importlib
from typing import Literal


torch = importlib.import_module("torch")
nn = importlib.import_module("torch.nn")


class MultimodalFusionLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        fusion_mode: Literal["concat", "cross_attention", "gated"] = "cross_attention",
        n_heads: int = 12,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.fusion_mode = fusion_mode

        if fusion_mode == "cross_attention":
            self.cross_attn_layers = nn.ModuleList(
                [
                    nn.MultiheadAttention(
                        model_dim,
                        n_heads,
                        dropout=dropout,
                        batch_first=True,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.layer_norms = nn.ModuleList(
                [nn.LayerNorm(model_dim) for _ in range(n_layers)]
            )
            self.ff_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(model_dim, model_dim * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(model_dim * 4, model_dim),
                    )
                    for _ in range(n_layers)
                ]
            )
            self.ff_norms = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(n_layers)])

        elif fusion_mode == "gated":
            self.gate_image = nn.Linear(model_dim, 1)
            self.gate_audio = nn.Linear(model_dim, 1)
            self.image_proj = nn.Linear(model_dim, model_dim)
            self.audio_proj = nn.Linear(model_dim, model_dim)
            self.output_norm = nn.LayerNorm(model_dim)
        elif fusion_mode != "concat":
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    def forward(
        self,
        text_embeds: torch.Tensor,  # (B, T_text, model_dim)
        image_embeds: torch.Tensor | None,  # (B, N_patches, model_dim) or None
        audio_embeds: torch.Tensor | None,  # (B, 1, model_dim) or None
    ) -> torch.Tensor:
        """Returns fused sequence: (B, seq_len, model_dim)."""

        if image_embeds is None and audio_embeds is None:
            return text_embeds

        if self.fusion_mode == "concat":
            parts = []
            if audio_embeds is not None:
                parts.append(audio_embeds)
            if image_embeds is not None:
                parts.append(image_embeds)
            parts.append(text_embeds)
            return torch.cat(parts, dim=1)

        if self.fusion_mode == "cross_attention":
            # Build key/value from all visual/audio tokens
            kv_parts = []
            if audio_embeds is not None:
                kv_parts.append(audio_embeds)
            if image_embeds is not None:
                kv_parts.append(image_embeds)
            kv = torch.cat(kv_parts, dim=1)  # (B, N_kv, dim)

            x = text_embeds
            for attn, ln, ff, ff_ln in zip(
                self.cross_attn_layers,
                self.layer_norms,
                self.ff_layers,
                self.ff_norms,
                strict=True,
            ):
                # Cross-attention: text queries attend to image/audio keys
                attn_out, _ = attn(query=x, key=kv, value=kv)
                x = ln(x + attn_out)
                # Feed-forward
                x = ff_ln(x + ff(x))
            return x  # (B, T_text, dim) — same shape as input text_embeds

        if self.fusion_mode == "gated":
            x = text_embeds
            if image_embeds is not None:
                img_mean = image_embeds.mean(dim=1, keepdim=True)  # (B, 1, dim)
                gate = torch.sigmoid(self.gate_image(img_mean))  # (B, 1, 1)
                img_contrib = gate * self.image_proj(img_mean).expand_as(x)
                x = x + img_contrib
            if audio_embeds is not None:
                aud_mean = audio_embeds  # (B, 1, dim)
                gate = torch.sigmoid(self.gate_audio(aud_mean))  # (B, 1, 1)
                aud_contrib = gate * self.audio_proj(aud_mean).expand_as(x)
                x = x + aud_contrib
            return self.output_norm(x)

        raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))
