from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GirivinityConfig:
    # Girivinity 3B — ~2.92B parameters
    dim: int = 3072
    n_layers: int = 28
    n_heads: int = 24
    n_kv_heads: int = 8
    vocab_size: int = 32000
    max_seq_len: int = 4096
    ffn_multiplier: float = 2.667
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0   # Extended RoPE for longer context
    # Cross-layer KV sharing (Gemma 4 technique)
    # Layers >= kv_sharing_start_layer reuse KV from
    # the most recent anchor layer before them.
    # Set to n_layers to disable (default off).
    kv_sharing_start_layer: int = 999  # disabled by default
    # Per-Layer Embeddings (Gemma 4 technique)
    # Adds cheap token-specific capacity to each layer.
    # ple_dim=0 disables PLE entirely.
    ple_dim: int = 0   # disabled by default; set 64 to enable
    # Manifold-Constrained Hyper-Connections (DeepSeek V4)
    # n_residual_streams=1 = standard single residual (disabled)
    # n_residual_streams=4 = DeepSeek V4 setting
    n_residual_streams: int = 1  # disabled by default

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads   # 128

    @property
    def ffn_dim(self) -> int:
        # Round to nearest multiple of 256 for hardware efficiency
        raw = int(self.dim * self.ffn_multiplier)
        return (raw + 255) // 256 * 256  # 8192

    @property
    def kv_sharing_enabled(self) -> bool:
        return self.kv_sharing_start_layer < self.n_layers

    @property
    def ple_enabled(self) -> bool:
        return self.ple_dim > 0

    @property
    def mhc_enabled(self) -> bool:
        return self.n_residual_streams > 1

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "GirivinityConfig":
        import yaml
        from pathlib import Path
        raw = yaml.safe_load(Path(path).read_text()).get("architecture", {})
        return cls(**{k: v for k, v in raw.items()
                     if k in cls.__dataclass_fields__})

    @classmethod
    def small(cls) -> "GirivinityConfig":
        """360M config for edge/testing use."""
        return cls(
            dim=1024, n_layers=16, n_heads=16, n_kv_heads=4,
            vocab_size=32000, max_seq_len=4096, ffn_multiplier=2.667,
        )

    @classmethod
    def v2_enhanced(cls) -> "GirivinityConfig":
        return cls(
            dim=3072, n_layers=28, n_heads=24, n_kv_heads=8,
            vocab_size=32000, max_seq_len=4096,
            ffn_multiplier=2.667, rope_theta=500000.0,
            kv_sharing_start_layer=14,
            ple_dim=64,
            n_residual_streams=4,
        )

    @classmethod
    def v2_kv_only(cls) -> "GirivinityConfig":
        return cls(
            dim=3072, n_layers=28, n_heads=24, n_kv_heads=8,
            vocab_size=32000, max_seq_len=4096,
            kv_sharing_start_layer=14,
        )

    @classmethod
    def v2_mhc_only(cls) -> "GirivinityConfig":
        return cls(
            dim=3072, n_layers=28, n_heads=24, n_kv_heads=8,
            vocab_size=32000, max_seq_len=4096,
            n_residual_streams=4,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


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
    B, H, T, D = x.shape
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[:T, :D // 2].unsqueeze(0).unsqueeze(0)
    sin = sin[:T, :D // 2].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.head_dim
        self.scale      = self.head_dim ** -0.5
        self.n_rep      = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        shared_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        if shared_kv is not None:
            k, v = shared_kv
        else:
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v)

        # Expand KV heads to match Q heads
        k_exp = k.repeat_interleave(self.n_rep, dim=1)
        v_exp = v.repeat_interleave(self.n_rep, dim=1)

        attn = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)

        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up   = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class PerLayerEmbedding(nn.Module):
    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.ple_table = nn.Embedding(cfg.vocab_size, cfg.n_layers * cfg.ple_dim)
        self.ple_proj = nn.Linear(cfg.ple_dim, cfg.dim, bias=False)
        self.ple_norm = nn.LayerNorm(cfg.dim)
        self.n_layers = cfg.n_layers
        self.ple_dim = cfg.ple_dim

    def get_layer_slice(self, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        packed = self.ple_table(input_ids)
        start = layer_idx * self.ple_dim
        end = start + self.ple_dim
        return packed[:, :, start:end]


class DecoderBlock(nn.Module):
    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm  = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn  = SwiGLU(cfg)
        if cfg.ple_enabled:
            self.ple_gate = nn.Linear(cfg.dim, cfg.ple_dim, bias=False)
        else:
            self.ple_gate = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
        shared_kv: Optional[tuple] = None,
        ple_slice: Optional[torch.Tensor] = None,
        ple_proj: Optional[nn.Linear] = None,
        ple_norm: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, tuple]:
        attn_out, new_cache = self.attn(
            self.attn_norm(x), cos, sin, mask, kv_cache, shared_kv
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        if (
            self.ple_gate is not None
            and ple_slice is not None
            and ple_proj is not None
            and ple_norm is not None
        ):
            gate = torch.sigmoid(self.ple_gate(x))
            gated = gate * ple_slice
            ple_up = ple_norm(ple_proj(gated))
            x = x + ple_up
        return x, new_cache


class ManifoldHyperConnection(nn.Module):
    def __init__(self, dim: int, n: int = 4) -> None:
        super().__init__()
        self.n = n
        self.dim = dim
        self.pre_map = nn.Parameter(torch.ones(n, 1) / n)
        self.post_map = nn.Parameter(torch.ones(1, n) / n)
        self.res_map = nn.Parameter(torch.eye(n))

    def get_doubly_stochastic(self) -> torch.Tensor:
        w = self.res_map.abs()
        for _ in range(5):
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            w = w / (w.sum(dim=0, keepdim=True) + 1e-8)
        return w

    def pre_combine(self, streams: torch.Tensor) -> torch.Tensor:
        w = F.softmax(self.pre_map, dim=0)
        return (streams * w.unsqueeze(0).unsqueeze(0)).sum(dim=2)

    def post_distribute(self, streams: torch.Tensor, layer_out: torch.Tensor) -> torch.Tensor:
        w = F.softplus(self.post_map)
        w = w / (w.sum() + 1e-8)
        update = layer_out.unsqueeze(2) * w.unsqueeze(0).unsqueeze(0)
        return streams + update

    def mix_streams(self, streams: torch.Tensor) -> torch.Tensor:
        ds = self.get_doubly_stochastic()
        B, T, n, d = streams.shape
        flat = streams.view(B * T, n, d)
        mixed = torch.einsum("ij,bjd->bid", ds, flat)
        return mixed.view(B, T, n, d)


class GirivinityModel(nn.Module):
    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers  = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        if cfg.ple_enabled:
            self.ple = PerLayerEmbedding(cfg)
        else:
            self.ple = None
        if cfg.mhc_enabled:
            self.mhc = ManifoldHyperConnection(cfg.dim, cfg.n_residual_streams)
        else:
            self.mhc = None

        cos, sin = build_rope_cache(
            cfg.max_seq_len, cfg.head_dim, cfg.rope_theta, torch.device("cpu")
        )
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list[tuple]] = None,
    ) -> tuple[torch.Tensor, list[tuple]]:
        B, T = input_ids.shape
        x = self.embed(input_ids)

        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)

        causal_mask = torch.full(
            (T, T), float("-inf"), device=x.device
        ).triu(1).unsqueeze(0).unsqueeze(0)

        ple_slices = None
        if self.ple is not None:
            ple_slices = [self.ple.get_layer_slice(input_ids, i) for i in range(self.cfg.n_layers)]

        if self.mhc is not None:
            n = self.cfg.n_residual_streams
            streams = x.unsqueeze(2).expand(-1, -1, n, -1).contiguous()

        new_caches = []
        last_anchor_kv: Optional[tuple] = None
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None

            use_shared_kv = (
                self.cfg.kv_sharing_enabled
                and i >= self.cfg.kv_sharing_start_layer
                and last_anchor_kv is not None
            )
            shared_kv = last_anchor_kv if use_shared_kv else None
            ple_slice = ple_slices[i] if ple_slices else None

            if self.mhc is not None:
                streams = self.mhc.mix_streams(streams)
                x = self.mhc.pre_combine(streams)

            attn_out, new_cache = layer.attn(
                layer.attn_norm(x), cos, sin, causal_mask, cache, shared_kv
            )
            attn_residual = x + attn_out
            ffn_out = layer.ffn(layer.ffn_norm(attn_residual))
            layer_out = attn_residual + ffn_out

            if layer.ple_gate is not None and ple_slice is not None and self.ple is not None:
                gate = torch.sigmoid(layer.ple_gate(layer_out))
                gated = gate * ple_slice
                ple_up = self.ple.ple_norm(self.ple.ple_proj(gated))
                layer_out = layer_out + ple_up

            if self.mhc is not None:
                streams = self.mhc.post_distribute(streams, layer_out)
                x = streams.mean(dim=2)
            else:
                x = layer_out

            new_caches.append(new_cache)
            if not use_shared_kv:
                last_anchor_kv = new_cache

        logits = self.lm_head(self.norm(x))
        return logits, new_caches

    def param_count(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{n / 1e6:.1f}M parameters"

    def param_count_detailed(self) -> dict:
        embed = sum(p.numel() for p in self.embed.parameters())
        layers = sum(p.numel() for p in self.layers.parameters())
        norm = sum(p.numel() for p in self.norm.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "total": f"{total/1e9:.3f}B",
            "embedding": f"{embed/1e6:.1f}M",
            "layers": f"{layers/1e9:.3f}B",
            "norm": f"{norm/1e3:.1f}K",
            "per_layer": f"{layers/self.cfg.n_layers/1e6:.1f}M",
        }
