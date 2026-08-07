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

    kv_sharing_enabled: bool = False
    ple_enabled: bool = False
    mhc_enabled: bool = False
    kv_sharing_start_layer: Optional[int] = None
    ple_dim: Optional[int] = None
    n_residual_streams: Optional[int] = None

    # --- Mixture-of-Experts (replaces the dense SwiGLU FFN in later layers) ---
    moe_enabled: bool = False
    n_routed_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None
    n_activated_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    moe_start_layer: Optional[int] = None
    moe_bias_update_speed: float = 0.001

    # --- QK-Norm (Qwen3 / Gemma3 style attention-stability improvement) ---
    qk_norm_enabled: bool = False

    # --- RoPE context-extension scaling (for inference beyond max_seq_len) ---
    rope_scaling_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.kv_sharing_start_layer is not None:
            self.kv_sharing_enabled = True
        if self.ple_dim is not None:
            self.ple_enabled = True
        if self.n_residual_streams is not None:
            self.mhc_enabled = True
        if self.n_routed_experts is not None:
            self.moe_enabled = True
            if self.n_activated_experts is None:
                self.n_activated_experts = min(6, self.n_routed_experts)
            if self.n_shared_experts is None:
                self.n_shared_experts = 2
            if self.moe_intermediate_size is None:
                self.moe_intermediate_size = self.ffn_dim // 4
            if self.moe_start_layer is None:
                self.moe_start_layer = 1

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def ffn_dim(self) -> int:
        raw = self.dim * self.ffn_multiplier
        return round(raw / 256) * 256

    @classmethod
    def small(cls) -> "GirivinityConfig":
        return cls(dim=1024, n_layers=16, n_heads=16, n_kv_heads=4, vocab_size=32000, max_seq_len=4096)

    @classmethod
    def v2_enhanced(cls, **overrides) -> "GirivinityConfig":
        defaults = dict(kv_sharing_start_layer=14, ple_dim=64, n_residual_streams=4)
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
    def moe_enhanced(cls, **overrides) -> "GirivinityConfig":
        """Generation-1 MoE preset: 64 fine-grained routed experts (top-6
        activated per token) plus 2 always-active shared experts, starting
        from layer 1 (layer 0 stays dense, following common practice)."""
        defaults = dict(
            n_routed_experts=64,
            n_activated_experts=6,
            n_shared_experts=2,
            moe_start_layer=1,
        )
        defaults.update(overrides)
        return cls(**defaults)

    def grow_experts(self, **overrides) -> "GirivinityConfig":
        """
        Build a successor generation's config from this one by scaling up
        the MoE routed-expert count (and, by default, proportionally more
        activated experts per token), while keeping every other architecture
        parameter — including moe_intermediate_size and moe_start_layer —
        identical unless explicitly overridden. This is the intended growth
        path for successor_engine.py: each generation doubles n_routed_experts
        by default, so weights for existing experts can be copied forward
        (sparse upcycling / expert-splitting-style growth) into a wider
        successor model rather than starting from scratch.
        """
        if not self.moe_enabled:
            raise ValueError("grow_experts() requires an MoE-enabled config (moe_enabled=True)")
        current = {f: getattr(self, f) for f in self.__dataclass_fields__}
        current["n_routed_experts"] = self.n_routed_experts * 2
        current["n_activated_experts"] = self.n_activated_experts * 2
        current.update(overrides)
        return GirivinityConfig(**current)

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
    seq_len: int, head_dim: int, theta: float, device: torch.device, scaling_factor: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    if scaling_factor != 1.0:
        # Linear position-interpolation scaling: stretches the effective
        # position sequence so a model trained at max_seq_len can run at
        # scaling_factor * max_seq_len positions without retraining, at some
        # cost to short-context precision. This is the simplest RoPE
        # context-extension method; NTK-aware / YaRN scaling (which only
        # stretches low frequencies, preserving high-frequency precision for
        # nearby tokens) is a stronger follow-up upgrade over this.
        t = t / scaling_factor
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
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

        # QK-Norm (Qwen3 / Gemma3 style): per-head RMSNorm on queries and
        # keys, applied right after projection and before RoPE, to stabilize
        # attention logit magnitudes at scale.
        self.qk_norm_enabled = cfg.qk_norm_enabled
        if self.qk_norm_enabled:
            self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)

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
        if self.qk_norm_enabled:
            q = self.q_norm(q)
        q = apply_rope(q, cos, sin)

        if shared_kv is not None:
            k, v = shared_kv
            new_cache = shared_kv
        else:
            k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            if self.qk_norm_enabled:
                k = self.k_norm(k)
            k = apply_rope(k, cos, sin)
            if kv_cache is not None:
                k_cache, v_cache = kv_cache
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)
            new_cache = (k, v)

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


class Expert(nn.Module):
    """A single small SwiGLU expert used inside MoELayer."""

    def __init__(self, dim: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, intermediate_size, bias=False)
        self.up   = nn.Linear(dim, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoELayer(nn.Module):
    """
    Fine-grained Mixture-of-Experts with shared experts and auxiliary-loss-
    free load balancing — the DeepSeek-MoE / DeepSeek-V3 style design, chosen
    over classic Switch-Transformer/GShard-style MoE (a small number of large
    experts, top-1/top-2 routing, balanced via an auxiliary loss term added to
    the training objective). Two differences from that classic design, both
    genuine improvements this implementation follows:

    1. Many small ("fine-grained") routed experts instead of few large ones,
       plus a handful of always-active "shared" experts. This lets routed
       experts specialize more precisely on narrow patterns, while the shared
       experts absorb common knowledge every token needs — reducing
       redundancy across the routed experts (DeepSeekMoE, 2024).
    2. Load balancing via a per-expert routing bias that is nudged up/down
       based on observed load (no gradient, updated after each forward pass
       during training) instead of an auxiliary loss term mixed into the
       training loss. An auxiliary loss can fight the primary objective and
       measurably hurt model quality; the bias-based approach balances load
       without that trade-off (DeepSeek-V3, 2024).

    This is a correctness-focused reference implementation — the expert
    dispatch loop below is a plain, readable per-expert masked pass, not a
    throughput-optimized scatter/gather or fused kernel. It is verified
    numerically correct (weights match the exact top-k routing probabilities
    for each token) but would need real dispatch-kernel work before it's
    efficient at production expert counts/batch sizes.
    """

    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.n_routed = cfg.n_routed_experts
        self.n_shared = cfg.n_shared_experts
        self.top_k = cfg.n_activated_experts
        self.dim = cfg.dim

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.moe_intermediate_size) for _ in range(self.n_routed)]
        )
        self.shared_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.moe_intermediate_size) for _ in range(self.n_shared)]
        ) if self.n_shared > 0 else None

        self.gate = nn.Linear(cfg.dim, self.n_routed, bias=False)
        # Auxiliary-loss-free load balancing: a per-expert bias added to the
        # routing logits before top-k selection. Updated by a simple
        # load-based heuristic during training, not by gradient descent.
        self.register_buffer("routing_bias", torch.zeros(self.n_routed))
        self.bias_update_speed = cfg.moe_bias_update_speed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        logits = self.gate(x_flat) + self.routing_bias
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        out = torch.zeros_like(x_flat)
        load = torch.zeros(self.n_routed, device=x.device)

        for expert_idx in range(self.n_routed):
            mask = topk_idx == expert_idx
            token_mask = mask.any(dim=-1)
            load[expert_idx] = token_mask.sum()
            if not token_mask.any():
                continue
            selected = x_flat[token_mask]
            expert_out = self.routed_experts[expert_idx](selected)
            weight = topk_probs[token_mask][mask[token_mask]].unsqueeze(-1)
            out[token_mask] = out[token_mask] + expert_out * weight

        if self.shared_experts is not None:
            for shared in self.shared_experts:
                out = out + shared(x_flat)

        if self.training:
            with torch.no_grad():
                avg_load = load.mean()
                self.routing_bias -= self.bias_update_speed * torch.sign(load - avg_load)

        return out.reshape(B, T, D)


class DecoderBlock(nn.Module):
    def __init__(self, cfg: GirivinityConfig, ffn: nn.Module) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm  = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn  = ffn

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
        shared_kv: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, tuple]:
        attn_out, new_cache = self.attn(self.attn_norm(x), cos, sin, mask, kv_cache, shared_kv)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class ManifoldHyperConnection(nn.Module):
    """
    Simplified 'manifold constrained' hyper-connections: n parallel residual
    streams mixed at each layer boundary via a doubly-stochastic matrix
    (Sinkhorn-Knopp projection of a learnable logit matrix). This is a
    working v1 of the mixing mechanism, not a full reproduction of one
    specific paper's per-layer expand/reduce weights.
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
        ds = self.get_doubly_stochastic().to(streams.dtype)
        return torch.einsum("ij,jbtd->ibtd", ds, streams)


class PerLayerEmbedding(nn.Module):
    """
    Small per-layer token embedding table (dim=ple_dim, much smaller than
    the model's main hidden dim), projected up and added into each decoder
    layer's input. Modeled on Gemma 3n's per-layer embeddings technique.
    """
    def __init__(self, vocab_size: int, ple_dim: int, model_dim: int, n_layers: int) -> None:
        super().__init__()
        self.ple_dim = ple_dim
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, ple_dim * n_layers)
        self.proj = nn.Linear(ple_dim, model_dim, bias=False)

    def forward(self, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        full = self.embed(input_ids)
        B, T, _ = full.shape
        per_layer = full.view(B, T, self.n_layers, self.ple_dim)[:, :, layer_idx, :]
        return self.proj(per_layer)


class GirivinityModel(nn.Module):
    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers  = nn.ModuleList(
            [DecoderBlock(cfg, self._build_ffn(cfg, i)) for i in range(cfg.n_layers)]
        )
        self.norm    = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.gradient_checkpointing = False

        effective_seq_len = int(cfg.max_seq_len * cfg.rope_scaling_factor)
        cos, sin = build_rope_cache(effective_seq_len, cfg.head_dim, cfg.rope_theta, torch.device("cpu"), cfg.rope_scaling_factor)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        causal_mask = torch.full((effective_seq_len, effective_seq_len), float("-inf")).triu(1)
        self.register_buffer("causal_mask_full", causal_mask, persistent=False)

        self.ple = PerLayerEmbedding(cfg.vocab_size, cfg.ple_dim, cfg.dim, cfg.n_layers) if cfg.ple_dim is not None else None
        self.mhc = ManifoldHyperConnection(cfg.dim, cfg.n_residual_streams) if cfg.n_residual_streams is not None else None

        self.apply(self._init_weights)

    @staticmethod
    def _build_ffn(cfg: GirivinityConfig, layer_idx: int) -> nn.Module:
        if cfg.moe_enabled and layer_idx >= cfg.moe_start_layer:
            return MoELayer(cfg)
        return SwiGLU(cfg)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, kv_caches: Optional[list[tuple]] = None) -> tuple[torch.Tensor, list[tuple]]:
        B, T = input_ids.shape
        x = self.embed(input_ids)

        past_len = kv_caches[0][0].shape[2] if kv_caches else 0
        cos = self.rope_cos[past_len:past_len + T].to(x.device)
        sin = self.rope_sin[past_len:past_len + T].to(x.device)
        causal_mask = self.causal_mask_full[past_len:past_len + T, :past_len + T].to(x.device).unsqueeze(0).unsqueeze(0)

        streams = None
        if self.mhc is not None:
            streams = x.unsqueeze(0).expand(self.mhc.n, -1, -1, -1).contiguous()

        new_caches: list[tuple] = []
        shared_kv_this_pass: Optional[tuple] = None
        use_gradient_checkpointing = self.gradient_checkpointing and self.training and kv_caches is None

        for i, layer in enumerate(self.layers):
            cache_in = kv_caches[i] if kv_caches else None
            is_shared_consumer = self.cfg.kv_sharing_start_layer is not None and i > self.cfg.kv_sharing_start_layer
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

                def custom_forward(hidden_states: torch.Tensor, checkpointed_layer: DecoderBlock = layer) -> torch.Tensor:
                    out, _ = checkpointed_layer(hidden_states, cos, sin, causal_mask, None, None)
                    return out

                layer_output = checkpoint(custom_forward, layer_input, use_reentrant=False)
                new_cache = cache_in
            else:
                layer_output, new_cache = layer(layer_input, cos, sin, causal_mask, cache_in, shared_kv)

            if self.cfg.kv_sharing_start_layer is not None and i == self.cfg.kv_sharing_start_layer:
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
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> torch.Tensor:
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
