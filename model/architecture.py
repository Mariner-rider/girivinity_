from __future__ import annotations

import logging
import math as _math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GirivinityConfig:
    dim: int = 3072
    n_layers: int = 28
    n_heads: int = 24
    n_kv_heads: int = 8
    vocab_size: int = 32000
    max_seq_len: int = 4096
    ffn_multiplier: float = 2.667
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    dropout_rate: float = 0.0
    use_checkpoint: bool = False

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def ffn_dim(self) -> int:
        raw = int(self.dim * self.ffn_multiplier)
        return (raw + 255) // 256 * 256

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "GirivinityConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text()).get("architecture", {})
        filtered = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        dropped = [k for k in raw if k not in cls.__dataclass_fields__]
        if dropped:
            logger.warning("GirivinityConfig: unknown config keys ignored: %s", dropped)
        return cls(**filtered)

    @classmethod
    def small(cls) -> "GirivinityConfig":
        return cls(
            dim=1024,
            n_layers=16,
            n_heads=16,
            n_kv_heads=4,
            vocab_size=32000,
            max_seq_len=4096,
            ffn_multiplier=2.667,
            rope_theta=500000.0,
        )

    @classmethod
    def v2_enhanced(cls) -> "GirivinityConfig":
        return cls(
            dim=3072,
            n_layers=28,
            n_heads=24,
            n_kv_heads=8,
            vocab_size=32000,
            max_seq_len=4096,
            ffn_multiplier=2.667,
            rope_theta=500000.0,
        )

    @classmethod
    def v2_kv_only(cls) -> "GirivinityConfig":
        return cls(
            dim=3072,
            n_layers=28,
            n_heads=24,
            n_kv_heads=8,
            vocab_size=32000,
            max_seq_len=4096,
            ffn_multiplier=2.667,
            rope_theta=500000.0,
        )

    @classmethod
    def v2_mhc_only(cls) -> "GirivinityConfig":
        return cls(
            dim=3072,
            n_layers=28,
            n_heads=24,
            n_kv_heads=8,
            vocab_size=32000,
            max_seq_len=4096,
            ffn_multiplier=2.667,
            rope_theta=500000.0,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        norm = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.to(x.dtype) * self.weight


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    _, _, T, D = x.shape
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    cos = cos[:T, : D // 2].unsqueeze(0).unsqueeze(0)
    sin = sin[:T, : D // 2].unsqueeze(0).unsqueeze(0)
    x_rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    return x * cos + x_rotated * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim**-0.5
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.o_proj._is_output_proj = True

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        past_len = kv_cache[0].shape[2] if kv_cache is not None else 0

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v)

        T_kv = k.shape[2]
        k = (
            k.unsqueeze(2)
            .expand(B, self.n_kv_heads, self.n_rep, T_kv, self.head_dim)
            .reshape(B, self.n_heads, T_kv, self.head_dim)
        )
        v = (
            v.unsqueeze(2)
            .expand(B, self.n_kv_heads, self.n_rep, T_kv, self.head_dim)
            .reshape(B, self.n_heads, T_kv, self.head_dim)
        )

        attn_mask = None
        if mask is not None and (past_len > 0 or pad_mask is not None):
            if past_len > 0:
                corrected_mask = torch.full(
                    (T, T_kv), float("-inf"), device=x.device, dtype=x.dtype
                )
                corrected_mask = torch.triu(corrected_mask, diagonal=past_len + 1)
                attn_mask = corrected_mask.unsqueeze(0).unsqueeze(0)
            else:
                attn_mask = mask.to(dtype=x.dtype, device=x.device)
        if pad_mask is not None:
            if pad_mask.shape[-1] != T_kv:
                pad_delta = T_kv - pad_mask.shape[-1]
                if pad_delta > 0:
                    prefix = torch.zeros(
                        (*pad_mask.shape[:-1], pad_delta),
                        device=pad_mask.device,
                        dtype=pad_mask.dtype,
                    )
                    pad_mask = torch.cat([prefix, pad_mask], dim=-1)
                else:
                    pad_mask = pad_mask[..., -T_kv:]
            pad_mask = pad_mask.to(dtype=q.dtype, device=q.device)
            attn_mask = pad_mask if attn_mask is None else attn_mask + pad_mask

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=(past_len == 0 and attn_mask is None),
            dropout_p=0.0,
            scale=self.scale,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)
        self.down._is_output_proj = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple]:
        attn_out, new_cache = self.attn(
            self.attn_norm(x), cos, sin, mask, kv_cache, pad_mask
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, new_cache


class GirivinityModel(nn.Module):
    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self._n_layers = cfg.n_layers
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        causal = torch.full((cfg.max_seq_len, cfg.max_seq_len), float("-inf")).triu(1)
        self.register_buffer("causal_mask", causal, persistent=False)

        cos, sin = build_rope_cache(
            cfg.max_seq_len, cfg.head_dim, cfg.rope_theta, torch.device("cpu")
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        self.lm_head.weight = self.embed.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "_is_output_proj", False):
                std = 0.02 / _math.sqrt(2 * self.cfg.n_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list[tuple]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[tuple]]:
        _, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, (
            f"Input sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )
        x = self.embed(input_ids)

        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)

        causal_mask = self.causal_mask[:T, :T].to(x.device).unsqueeze(0).unsqueeze(0)
        pad_mask = None
        if attention_mask is not None:
            pad_mask = torch.where(
                attention_mask.bool(),
                torch.zeros_like(attention_mask, dtype=x.dtype),
                torch.full_like(attention_mask, float("-inf"), dtype=x.dtype),
            ).unsqueeze(1).unsqueeze(1)

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if (kv_caches and i < len(kv_caches)) else None
            if self.cfg.use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint

                x, new_cache = checkpoint(
                    layer,
                    x,
                    cos,
                    sin,
                    causal_mask,
                    cache,
                    pad_mask,
                    use_reentrant=False,
                )
            else:
                x, new_cache = layer(x, cos, sin, causal_mask, cache, pad_mask)
            new_caches.append(new_cache)

        logits = self.lm_head(self.norm(x))
        return logits, new_caches

    def param_count(self) -> str:
        n = sum(p.numel() for p in set(self.parameters()))
        return f"{n / 1e6:.1f}M parameters"

    def param_count_detailed(self) -> dict:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in set(module.parameters()))

        embed = count(self.embed)
        layers = count(self.layers)
        norm = count(self.norm)
        total = sum(p.numel() for p in set(self.parameters()))
        return {
            "total": f"{total / 1e9:.3f}B",
            "embedding": f"{embed / 1e6:.1f}M",
            "layers": f"{layers / 1e9:.3f}B",
            "norm": f"{norm / 1e3:.1f}K",
            "per_layer": f"{layers / self.cfg.n_layers / 1e6:.1f}M",
        }

    def __repr__(self) -> str:
        cfg = self.cfg
        n = sum(p.numel() for p in set(self.parameters()))
        return (
            "GirivinityModel(\n"
            f"  params={n / 1e9:.3f}B\n"
            f"  dim={cfg.dim}, layers={cfg.n_layers}, heads={cfg.n_heads} "
            f"(kv={cfg.n_kv_heads})\n"
            f"  ffn_dim={cfg.ffn_dim}, vocab={cfg.vocab_size}, ctx={cfg.max_seq_len}\n"
            f"  rope_theta={cfg.rope_theta}, dropout={cfg.dropout_rate}\n"
            f"  use_checkpoint={cfg.use_checkpoint}\n"
            ")"
        )

    def memory_estimate(self, batch_size: int = 1, seq_len: int = 512) -> dict:
        cfg = self.cfg
        param_bytes = sum(p.numel() * p.element_size() for p in set(self.parameters()))
        act_bytes_per_layer = batch_size * seq_len * cfg.dim * 4 * 4
        total_act = act_bytes_per_layer * cfg.n_layers
        kv_bytes = 2 * batch_size * cfg.n_kv_heads * seq_len * cfg.head_dim * 4 * cfg.n_layers
        total = param_bytes + total_act + kv_bytes
        return {
            "params_mb": round(param_bytes / 1e6, 1),
            "activations_mb": round(total_act / 1e6, 1),
            "kv_cache_mb": round(kv_bytes / 1e6, 1),
            "total_mb": round(total / 1e6, 1),
            "total_gb": round(total / 1e9, 2),
        }
