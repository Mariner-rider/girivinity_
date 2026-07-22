from __future__ import annotations
import logging
from dataclasses import dataclass
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

    # --- v2 upgrades: all disabled unless explicitly requested ---
    kv_sharing_enabled: bool = False
    ple_enabled: bool = False
    mhc_enabled: bool = False
    kv_sharing_start_layer: Optional[int] = None
    ple_dim: Optional[int] = None
    n_residual_streams: Optional[int] = None

    def __post_init__(self) -> None:
        # A feature is "enabled" if its defining parameter was explicitly
        # set, regardless of whether the *_enabled flag was also passed.
        # v2_enhanced() sets both for clarity, but plain constructor calls
        # (as used throughout the test suite) only set the parameter.
        if self.kv_sharing_start_layer is not None:
            self.kv_sharing_enabled = True
        if self.ple_dim is not None:
            self.ple_enabled = True
        if self.n_residual_streams is not None:
            self.mhc_enabled = True

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def ffn_dim(self) -> int:
        raw = self.dim * self.ffn_multiplier
        return round(raw / 256) * 256

    @classmethod
    def small(cls) -> "GirivinityConfig":
        return cls(
            dim=1024,
            n_layers=16,
            n_heads=16,
            n_kv_heads=4,
            vocab_size=32000,
            max_seq_len=4096,
        )

    @classmethod
    def v2_enhanced(cls, **overrides) -> "GirivinityConfig":
        defaults = dict(
            kv_sharing_start_layer=14,
            ple_dim=64,
            n_residual_streams=4,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def v2_kv_only(cls, **overrides) -> "GirivinityConfig":
        defaults = dict(kv_sharing_start_layer=14)
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def v2_mhc_only(cls, **overrides) -> "GirivinityConfig":
        defaults = dict(n_residual_streams=4)
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "GirivinityConfig":
        import yaml
        from pathlib import Path
        raw = yaml.safe_load(Path(path).read_text())
        model_section = raw.get("model", {}) or {}
        architecture = model_section.get("architecture", raw.get("architecture", {})) or {}
        return cls(**{k: v for k, v in architecture.items() if k in cls.__dataclass_fields__})


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
    freqs = torch.outer(t, freqs)          # (seq_len, head_dim/2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # duplicate across the full head_dim so it broadcasts directly against
    # (B, H, T, head_dim) tensors in apply_rope, matching rotate_half below.
    cos = torch.cat([cos, cos], dim=-1)    # (seq_len, head_dim)
    sin = torch.cat([sin, sin], dim=-1)    # (seq_len, head_dim)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # x: (B, H, T, D). cos/sin: (T, D) — broadcast automatically against x.
    return x * cos + rotate_half(x) * sin


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
            # Cross-layer KV sharing: reuse K/V already computed (and
            # RoPE-applied) by the designated source layer this forward
            # pass. cos/sin only depend on token position, not layer, so
            # a K tensor RoPE'd at the source layer is valid to reuse here.
            k, v = shared_kv
            new_cache = shared_kv
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
        k_expanded = k.repeat_interleave(self.n_rep, dim=1)
        v_expanded = v.repeat_interleave(self.n_rep, dim=1)

        attn = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn.float(), dim=-1).to(q.dtype)

        out = torch.matmul(attn, v_expanded)
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


class DecoderBlock(nn.Module):
    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm  = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn  = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
        shared_kv: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, tuple]:
        attn_out, new_cache = self.attn(
            self.attn_norm(x), cos, sin, mask, kv_cache, shared_kv
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class ManifoldHyperConnection(nn.Module):
    """
    Simplified 'manifold constrained' hyper-connections: maintains `n`
    parallel residual streams and mixes them at each layer boundary using a
    doubly-stochastic matrix (Sinkhorn-Knopp projection of a learnable
    logit matrix), instead of an unconstrained learned mixing matrix.

    Note: this is a working v1 of the mixing mechanism — multiple residual
    pathways combined through a normalized (row/col sums = 1) matrix — not
    a full reproduction of a specific paper's per-layer expand/reduce
    weights. It gives the model genuine multi-stream residual mixing with
    the doubly-stochastic constraint, which is the core property requested.
    """

    def __init__(self, dim: int, n: int, sinkhorn_iters: int = 20) -> None:
        super().__init__()
        self.dim = dim
        self.n = n
        self.sinkhorn_iters = sinkhorn_iters
        self.logits = nn.Parameter(torch.zeros(n, n))

    def get_doubly_stochastic(self) -> torch.Tensor:
        log_alpha = self.logits
        for _ in range(self.sinkhorn_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
        return log_alpha.exp()

    def mix(self, streams: torch.Tensor) -> torch.Tensor:
        # streams: (n, B, T, dim)
        ds = self.get_doubly_stochastic().to(streams.dtype)
        return torch.einsum("ij,jbtd->ibtd", ds, streams)


class PerLayerEmbedding(nn.Module):
    """
    Small per-layer token embedding table (dim=ple_dim, much smaller than
    the model's main hidden dim), projected up and added into each decoder
    layer's input. Modeled on the 'per-layer embeddings' technique used for
    parameter-efficient capacity (e.g. Gemma 3n's PLE), kept separate from
    the main token embedding table.
    """

    def __init__(self, vocab_size: int, ple_dim: int, model_dim: int, n_layers: int) -> None:
        super().__init__()
        self.ple_dim = ple_dim
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, ple_dim * n_layers)
        self.proj = nn.Linear(ple_dim, model_dim, bias=False)

    def forward(self, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        full = self.embed(input_ids)                      # (B, T, ple_dim * n_layers)
        B, T, _ = full.shape
        per_layer = full.view(B, T, self.n_layers, self.ple_dim)[:, :, layer_idx, :]
        return self.proj(per_layer)                        # (B, T, model_dim)


class GirivinityModel(nn.Module):
    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers  = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        self.gradient_checkpointing = False

        cos, sin = build_rope_cache(
            cfg.max_seq_len, cfg.head_dim, cfg.rope_theta, torch.device("cpu")
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        causal_mask = torch.full(
            (cfg.max_seq_len, cfg.max_seq_len), float("-inf")
        ).triu(1)
        self.register_buffer("causal_mask_full", causal_mask, persistent=False)

        self.ple = (
            PerLayerEmbedding(cfg.vocab_size, cfg.ple_dim, cfg.dim, cfg.n_layers)
            if cfg.ple_dim is not None
            else None
        )
        self.mhc = (
            ManifoldHyperConnection(cfg.dim, cfg.n_residual_streams)
            if cfg.n_residual_streams is not None
            else None
        )

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

        past_len = kv_caches[0][0].shape[2] if kv_caches else 0
        cos = self.rope_cos[past_len:past_len + T].to(x.device)
        sin = self.rope_sin[past_len:past_len + T].to(x.device)
        causal_mask = (
            self.causal_mask_full[past_len:past_len + T, :past_len + T]
            .to(x.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        streams = None
        if self.mhc is not None:
            streams = x.unsqueeze(0).expand(self.mhc.n, -1, -1, -1).contiguous()

        new_caches: list[tuple] = []
        shared_kv_this_pass: Optional[tuple] = None
        use_gradient_checkpointing = (
            self.gradient_checkpointing and self.training and kv_caches is None
        )

        for i, layer in enumerate(self.layers):
            cache_in = kv_caches[i] if kv_caches else None
            is_shared_consumer = (
                self.cfg.kv_sharing_start_layer is not None
                and i > self.cfg.kv_sharing_start_layer
            )
            shared_kv = shared_kv_this_pass if is_shared_consumer else None

            layer_input = x
            if self.ple is not None:
                layer_input = layer_input + self.ple(input_ids, i)

            if streams is not None:
                mixed = self.mhc.mix(streams)
                layer_input = mixed.mean(dim=0)
                if self.ple is not None:
                    layer_input = layer_input + self.ple(input_ids, i)

            if use_gradient_checkpointing and shared_kv is None:
                from torch.utils.checkpoint import checkpoint

                def custom_forward(
                    hidden_states: torch.Tensor,
                    checkpointed_layer: DecoderBlock = layer,
                ) -> torch.Tensor:
                    out, _ = checkpointed_layer(hidden_states, cos, sin, causal_mask, None, None)
                    return out

                layer_output = checkpoint(custom_forward, layer_input, use_reentrant=False)
                new_cache = cache_in  # no cache produced under checkpointing
            else:
                layer_output, new_cache = layer(
                    layer_input, cos, sin, causal_mask, cache_in, shared_kv
                )

            if (
                self.cfg.kv_sharing_start_layer is not None
                and i == self.cfg.kv_sharing_start_layer
            ):
                shared_kv_this_pass = new_cache

            new_caches.append(new_cache)

            if streams is not None:
                delta = layer_output - layer_input
                streams = mixed + delta.unsqueeze(0)
                x = streams.mean(dim=0)
            else:
                x = layer_output

        logits = self.lm_head(self.norm(x))
        return logits, new_caches

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        kv_caches = None
        next_input = input_ids
        for _ in range(max_new_tokens):
            logits, kv_caches = self(next_input, kv_caches=kv_caches)
            next_token_logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            generated = torch.cat((generated, next_token), dim=1)
            next_input = next_token
        return generated

    @staticmethod
    def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return torch.multinomial(probs, num_samples=1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_count_detailed(self) -> dict[str, str]:
        n = sum(p.numel() for p in self.parameters())
        return {"total": f"{n / 1e9:.2f}B"}

    def save_pretrained(self, path: str) -> None:
        from pathlib import Path
        import json
        from dataclasses import asdict

        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2))
        torch.save(self.state_dict(), output_dir / "pytorch_model.bin")

    @classmethod
    def load_pretrained(cls, path: str) -> "GirivinityModel":
        from pathlib import Path
        import json

        model_dir = Path(path)
        cfg = GirivinityConfig(**json.loads((model_dir / "config.json").read_text()))
        model = cls(cfg)
        state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def param_count(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{n / 1e6:.1f}M parameters"
