from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import yaml


@dataclass(slots=True)
class GirivinityConfig:
    dim: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 4096
    ffn_multiplier: float = 2.667
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        return self.dim // self.n_heads

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config.yaml") -> GirivinityConfig:
        path = Path(config_path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
        raw = raw or {}
        model_section = raw.get("model") or {}
        architecture = model_section.get("architecture") or raw.get("architecture") or {}
        values = {**model_section, **architecture}
        field_names = cls.__dataclass_fields__.keys()
        kwargs = {name: values[name] for name in field_names if name in values}
        return cls(**kwargs)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq_len = x.size(1)
        cos = self.cos[start_pos : start_pos + seq_len].to(device=x.device, dtype=x.dtype)
        sin = self.sin[start_pos : start_pos + seq_len].to(device=x.device, dtype=x.dtype)
        cos = cos.view(1, seq_len, 1, -1)
        sin = sin.view(1, seq_len, 1, -1)
        return (x * cos) + (_rotate_half(x) * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: GirivinityConfig) -> None:
        super().__init__()
        if config.n_heads % config.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for grouped query attention")
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_heads // config.n_kv_heads
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        *,
        start_pos: int = 0,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        q = rope(q, start_pos=start_pos)
        k = rope(k, start_pos=start_pos)

        if cache is not None:
            end_pos = start_pos + seq_len
            if "k" not in cache or cache["k"].size(0) != batch_size:
                max_seq_len = rope.cos.size(0)
                cache["k"] = torch.zeros(
                    batch_size,
                    max_seq_len,
                    self.n_kv_heads,
                    self.head_dim,
                    device=x.device,
                    dtype=k.dtype,
                )
                cache["v"] = torch.zeros_like(cache["k"])
            cache["k"][:, start_pos:end_pos] = k
            cache["v"][:, start_pos:end_pos] = v
            k = cache["k"][:, :end_pos]
            v = cache["v"][:, :end_pos]

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self._causal_mask(seq_len, k.size(-2), start_pos, x.device, scores.dtype)
        weights = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.n_heads * self.head_dim)
        return self.wo(out)

    def _causal_mask(
        self,
        seq_len: int,
        key_len: int,
        start_pos: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        query_positions = torch.arange(start_pos, start_pos + seq_len, device=device).view(seq_len, 1)
        key_positions = torch.arange(key_len, device=device).view(1, key_len)
        mask = torch.zeros(seq_len, key_len, device=device, dtype=dtype)
        mask = mask.masked_fill(key_positions > query_positions, torch.finfo(dtype).min)
        return mask.view(1, 1, seq_len, key_len)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: GirivinityConfig) -> None:
        super().__init__()
        hidden_dim = int(config.ffn_multiplier * config.dim)
        hidden_dim = ((hidden_dim + 255) // 256) * 256
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DecoderBlock(nn.Module):
    def __init__(self, config: GirivinityConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        *,
        start_pos: int = 0,
        cache: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), rope, start_pos=start_pos, cache=cache)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class GirivinityModel(nn.Module):
    """Decoder-only transformer with RoPE, GQA, KV cache, RMSNorm, and SwiGLU."""

    def __init__(self, config: GirivinityConfig | None = None) -> None:
        super().__init__()
        self.config = config or GirivinityConfig.from_yaml()
        self.token_embeddings = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.rope = RotaryEmbedding(self.config.head_dim, self.config.max_seq_len)
        self.layers = nn.ModuleList([DecoderBlock(self.config) for _ in range(self.config.n_layers)])
        self.norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)
        self.output = nn.Linear(self.config.dim, self.config.vocab_size, bias=False)
        self.output.weight = self.token_embeddings.weight

    @classmethod
    def from_yaml(cls, config_path: str | Path = "config.yaml") -> GirivinityModel:
        return cls(GirivinityConfig.from_yaml(config_path))

    def init_kv_cache(self, batch_size: int, device: torch.device | str | None = None) -> list[dict[str, torch.Tensor]]:
        parameter = next(self.parameters())
        device = device or parameter.device
        dtype = parameter.dtype
        return [
            {
                "k": torch.zeros(
                    batch_size,
                    self.config.max_seq_len,
                    self.config.n_kv_heads,
                    self.config.head_dim,
                    device=device,
                    dtype=dtype,
                ),
                "v": torch.zeros(
                    batch_size,
                    self.config.max_seq_len,
                    self.config.n_kv_heads,
                    self.config.head_dim,
                    device=device,
                    dtype=dtype,
                ),
            }
            for _ in self.layers
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        start_pos: int = 0,
        kv_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq]")
        if start_pos + input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("sequence exceeds configured max_seq_len")

        x = self.token_embeddings(input_ids)
        for index, layer in enumerate(self.layers):
            cache = kv_cache[index] if kv_cache is not None else None
            x = layer(x, self.rope, start_pos=start_pos, cache=cache)
        x = self.norm(x)
        return self.output(x)
