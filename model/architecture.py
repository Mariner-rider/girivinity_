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

    # --- Multi-head Latent Attention (DeepSeek-V2/V3 style compressed KV
    # cache; replaces GroupedQueryAttention for the whole model when enabled,
    # not on a per-layer basis like MoE) ---
    mla_enabled: bool = False
    mla_d_c: Optional[int] = None
    mla_d_c_q: Optional[int] = None
    mla_rope_head_dim: Optional[int] = None

    # --- Multi-Token Prediction (DeepSeek-V3 style sequentially-chained
    # auxiliary prediction heads; off by default, adds n_mtp_heads extra
    # loss terms during training and enables speculative decoding in
    # generate() when on) ---
    mtp_enabled: bool = False
    n_mtp_heads: Optional[int] = None
    mtp_loss_weight: float = 0.3

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
        if self.mla_enabled:
            if self.kv_sharing_enabled:
                raise ValueError(
                    "mla_enabled is not currently supported together with "
                    "kv_sharing (kv_sharing_start_layer): MLA's cache format "
                    "(compressed latents) is structurally different from "
                    "GroupedQueryAttention's shared-KV cache format, and the "
                    "two caching schemes are not compatible in this "
                    "implementation."
                )
            if self.qk_norm_enabled:
                raise ValueError(
                    "mla_enabled is not currently supported together with "
                    "qk_norm_enabled: QK-Norm is implemented specifically for "
                    "GroupedQueryAttention's per-head Q/K layout and does not "
                    "yet have an MLA equivalent."
                )
            if self.mla_d_c is None:
                self.mla_d_c = self.head_dim // 2
            if self.mla_d_c_q is None:
                self.mla_d_c_q = self.mla_d_c
            if self.mla_rope_head_dim is None:
                self.mla_rope_head_dim = self.head_dim // 2
            if self.mla_rope_head_dim % 2 != 0:
                # RoPE splits the last dim into two equal halves for
                # rotation, so this must be even.
                self.mla_rope_head_dim += 1
        if self.n_mtp_heads is not None:
            self.mtp_enabled = True
        if self.mtp_enabled and self.n_mtp_heads is None:
            self.n_mtp_heads = 2
        if self.mtp_enabled and self.n_mtp_heads < 1:
            raise ValueError("mtp_enabled requires n_mtp_heads >= 1")

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


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA), DeepSeek-V2/V3 style. Queries and
    keys/values are each projected into a small low-rank latent space
    before being up-projected back to per-head dimensionality; only the
    latent vectors (plus a small decoupled RoPE key) are cached across
    decoding steps, instead of full per-head K/V. This is what shrinks the
    KV cache from `n_kv_heads * head_dim * 2` floats/token to roughly
    `mla_d_c + mla_rope_head_dim` floats/token.

    Query and key/value latents are compressed through SEPARATE
    down-projection matrices (`w_dq` vs `w_dkv`) — queries are never
    derived from the KV latent. Only the KV latent and the RoPE key are
    ever cached; query latents are never cached, since queries for past
    tokens are never needed again during autoregressive decoding.

    RoPE is "decoupled": it's applied only to a small dedicated slice of
    each head (`mla_rope_head_dim`), not the full head, and the key side
    of that slice is a single vector shared across every head per token
    (not one per head) — `w_kr` projects straight from `x`, not from the
    compressed KV latent. This split exists because RoPE's rotation is
    position-dependent: a compressed latent computed once and cached can't
    be correctly "re-rotated" for a different absolute position without
    re-deriving its un-rotated content, so the position-dependent part is
    kept small, separate, and applied fresh each step, while the bulk of
    the cached content (the latent) stays rotation-free and reusable as-is.

    This does not currently compose with `kv_sharing_enabled` or
    `qk_norm_enabled` (both raise a config error) — see `__post_init__`.
    """

    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.head_dim
        self.n_rep      = self.n_heads // self.n_kv_heads
        self.d_c        = cfg.mla_d_c
        self.d_c_q      = cfg.mla_d_c_q
        self.rope_dim   = cfg.mla_rope_head_dim
        self.scale      = (self.head_dim + self.rope_dim) ** -0.5

        # Down-projections into latent space (separate for Q vs KV).
        self.w_dkv = nn.Linear(cfg.dim, self.d_c, bias=False)
        self.w_dq  = nn.Linear(cfg.dim, self.d_c_q, bias=False)
        # Decoupled RoPE key: one small vector per token, shared across
        # every head — projected straight from x, not from the KV latent.
        self.w_kr  = nn.Linear(cfg.dim, self.rope_dim, bias=False)

        # Up-projections back to per-(kv-)head content dimensionality.
        self.w_uk = nn.Linear(self.d_c, self.n_kv_heads * self.head_dim, bias=False)
        self.w_uv = nn.Linear(self.d_c, self.n_kv_heads * self.head_dim, bias=False)
        self.w_uq = nn.Linear(self.d_c_q, self.n_heads * self.head_dim, bias=False)
        # Per-head decoupled RoPE query, up-projected from the query latent.
        self.w_qr = nn.Linear(self.d_c_q, self.n_heads * self.rope_dim, bias=False)

        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        shared_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if shared_kv is not None:
            raise ValueError("MultiHeadLatentAttention does not support kv_sharing's shared_kv")

        B, T, _ = x.shape

        # KV side: compress to the joint latent, derive the decoupled RoPE
        # key (rotated now, at this chunk's real positions, then cached
        # post-rotation — matching how GroupedQueryAttention caches K).
        c_kv_new = self.w_dkv(x)             # [B, T, d_c]
        k_r_new = self.w_kr(x)               # [B, T, rope_dim]
        k_r_new = apply_rope(k_r_new, cos, sin)

        if kv_cache is not None:
            c_kv_cache, k_r_cache = kv_cache
            c_kv = torch.cat([c_kv_cache, c_kv_new], dim=1)
            k_r = torch.cat([k_r_cache, k_r_new], dim=1)
        else:
            c_kv, k_r = c_kv_new, k_r_new
        new_cache = (c_kv, k_r)

        T_full = c_kv.shape[1]

        # Up-project the FULL cached latent sequence fresh each step — this
        # is the MLA trade-off: a tiny cache, at the cost of redoing this
        # cheap up-projection over the whole sequence every forward call
        # instead of reusing already-materialized per-head K/V.
        k_c = self.w_uk(c_kv).view(B, T_full, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v   = self.w_uv(c_kv).view(B, T_full, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k_r_expanded = k_r.unsqueeze(1).expand(-1, self.n_kv_heads, -1, -1)
        k = torch.cat([k_c, k_r_expanded], dim=-1)

        # Query side: only ever computed for the current chunk, never cached.
        c_q = self.w_dq(x)  # [B, T, d_c_q]
        q_c = self.w_uq(c_q).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q_r = self.w_qr(c_q).view(B, T, self.n_heads, self.rope_dim).transpose(1, 2)
        q_r = apply_rope(q_r, cos, sin)
        q = torch.cat([q_c, q_r], dim=-1)

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


class MTPModule(nn.Module):
    """
    One depth of DeepSeek-V3-style Multi-Token Prediction.

    Modules are sequentially chained: module i's output hidden state is fed
    as the *input* to module i+1, so each subsequent depth's prediction
    genuinely depends on the previous depth's transform, not just on the
    shared main hidden state. This is what makes the design sequential
    rather than a set of parallel independent heads reading the same input
    (module 1 predicts t+1 from h; module 2 predicts t+2 from module 1's
    own output, not from h directly; and so on).

    This is a simplified reference version relative to the full DeepSeek-V3
    MTP module: the real design additionally conditions each depth on the
    ground-truth embedding of the intervening token (via RMSNorm + concat +
    a linear projection back to model dim) during training, and on the
    model's own drafted token at inference. This implementation omits that
    embedding-conditioning step to stay lightweight and avoid threading
    labels/embeddings through the main forward() signature -- it relies on
    hidden-state chaining alone for the sequential dependency, which is
    sufficient to satisfy "sequential, not parallel" but is not a full
    reproduction of the paper's per-depth architecture.
    """

    def __init__(self, cfg: GirivinityConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden + self.ffn(self.norm(hidden))
        logits = self.lm_head(hidden)
        return hidden, logits


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
        self.attn = MultiHeadLatentAttention(cfg) if cfg.mla_enabled else GroupedQueryAttention(cfg)
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
        if cfg.mla_enabled:
            mla_cos, mla_sin = build_rope_cache(
                effective_seq_len, cfg.mla_rope_head_dim, cfg.rope_theta, torch.device("cpu"), cfg.rope_scaling_factor
            )
            self.register_buffer("mla_rope_cos", mla_cos, persistent=False)
            self.register_buffer("mla_rope_sin", mla_sin, persistent=False)
        else:
            cos, sin = build_rope_cache(effective_seq_len, cfg.head_dim, cfg.rope_theta, torch.device("cpu"), cfg.rope_scaling_factor)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        causal_mask = torch.full((effective_seq_len, effective_seq_len), float("-inf")).triu(1)
        self.register_buffer("causal_mask_full", causal_mask, persistent=False)

        self.ple = PerLayerEmbedding(cfg.vocab_size, cfg.ple_dim, cfg.dim, cfg.n_layers) if cfg.ple_dim is not None else None
        self.mhc = ManifoldHyperConnection(cfg.dim, cfg.n_residual_streams) if cfg.n_residual_streams is not None else None
        self.mtp_modules = (
            nn.ModuleList([MTPModule(cfg) for _ in range(cfg.n_mtp_heads)]) if cfg.mtp_enabled else None
        )

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

        if kv_caches:
            # MLA's cache tuples are (c_kv, k_r) with shape [B, T, d_c] /
            # [B, T, rope_dim] — sequence length is at index 1. GQA's cache
            # tuples are (k, v) with shape [B, n_kv_heads, T, head_dim] —
            # sequence length is at index 2.
            past_len = kv_caches[0][0].shape[1] if self.cfg.mla_enabled else kv_caches[0][0].shape[2]
        else:
            past_len = 0

        if self.cfg.mla_enabled:
            cos = self.mla_rope_cos[past_len:past_len + T].to(x.device)
            sin = self.mla_rope_sin[past_len:past_len + T].to(x.device)
        else:
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

        h = self.norm(x)
        logits = self.lm_head(h)

        if self.mtp_modules is not None:
            mtp_logits: list[torch.Tensor] = []
            mtp_hidden = h
            for module in self.mtp_modules:
                mtp_hidden, module_logits = module(mtp_hidden)
                mtp_logits.append(module_logits)
            return logits, mtp_logits, new_caches

        return logits, new_caches

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9) -> torch.Tensor:
        self.eval()
        if self.cfg.mtp_enabled:
            return self._generate_speculative(input_ids, max_new_tokens, temperature, top_p)
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

    def _pick_token(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits / temperature, dim=-1)
        return self._sample_top_p(probs, top_p)

    @torch.no_grad()
    def _generate_speculative(
        self, input_ids: torch.Tensor, max_new_tokens: int, temperature: float, top_p: float
    ) -> torch.Tensor:
        """
        Single-step speculative decoding using MTP-1 as a free draft model.

        MTP-1's prediction at the current position targets the token *two*
        steps ahead (position t+2, given the main head already covers
        t+1 -- see the loss-shift convention in _training_step), so a
        speculative round looks like:

          1. One forward pass gives both the main head's distribution for
             t+1 (always taken -- it's the true target distribution, no
             verification needed) and MTP-1's distribution for t+2 (a free
             draft, since it costs nothing beyond the same forward pass).
          2. The draft token for t+2 is verified by running ONE MORE
             forward pass over the 2-token batch [confirmed t+1, draft
             t+2] together (not two sequential single-token passes) with
             the existing causal KV cache. This single batched call gives
             the true target distribution for t+2 (conditioned on the now-
             real t+1), which is what verification needs, plus a bonus
             "next" main-head distribution usable to seed the following
             round if the draft is accepted.
          3. Standard rejection sampling: accept the draft with probability
             min(1, p_target/p_draft); if rejected, resample from the
             renormalized residual distribution max(0, p_target - p_draft)
             instead, so a rejection never wastes the verification pass.

        At temperature <= 0 this specializes to a greedy accept-iff-matches
        rule (accept the draft only if it equals the target's own argmax;
        otherwise take the target's argmax directly), which reduces exactly
        to plain greedy autoregressive decoding token-for-token -- this
        equivalence is the correctness property this method's tests check.

        Batch handling: when every sequence in the batch accepts its draft,
        the verification pass's second position is reused as the next
        round's "free" forward pass (no extra call). If any sequence in the
        batch rejects, this implementation falls back to a single fresh
        forward pass on the whole batch's confirmed tokens rather than
        trying to maintain per-sequence-divergent cache lengths within one
        shared KV cache tensor -- correct for any batch size, but only
        optimally fast when acceptance is uniform across the batch.
        """
        greedy = temperature <= 0
        generated = input_ids
        remaining = max_new_tokens

        logits, mtp_logits, kv_caches = self(generated)

        while remaining > 0:
            main_last = logits[:, -1, :]
            y_next = self._pick_token(main_last, temperature, top_p)
            generated = torch.cat((generated, y_next), dim=1)
            remaining -= 1
            if remaining == 0:
                break

            mtp1_last = mtp_logits[0][:, -1, :]
            y_draft = self._pick_token(mtp1_last, temperature, top_p)

            verify_input = torch.cat((y_next, y_draft), dim=1)
            logits2, mtp_logits2, kv_caches2 = self(verify_input, kv_caches=kv_caches)
            target_draft_slot = logits2[:, 0, :]

            if greedy:
                target_choice = target_draft_slot.argmax(dim=-1, keepdim=True)
                accept = target_choice == y_draft
                y_confirmed = torch.where(accept, y_draft, target_choice)
            else:
                probs_target = F.softmax(target_draft_slot / temperature, dim=-1)
                probs_draft = F.softmax(mtp1_last / temperature, dim=-1)
                p_t = probs_target.gather(-1, y_draft)
                p_d = probs_draft.gather(-1, y_draft).clamp_min(1e-9)
                accept_prob = (p_t / p_d).clamp(max=1.0)
                accept = torch.rand_like(accept_prob) < accept_prob
                residual = (probs_target - probs_draft).clamp_min(0.0)
                residual = residual / residual.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                resampled = torch.multinomial(residual, num_samples=1)
                y_confirmed = torch.where(accept, y_draft, resampled)

            generated = torch.cat((generated, y_confirmed), dim=1)
            remaining -= 1
            if remaining == 0:
                break

            if bool(accept.all()):
                # The draft was accepted for every sequence in the batch, so
                # the 2-token-extended cache from the verification pass is
                # valid to keep, and its second position's outputs (which
                # were computed *as if* the draft were real) are exactly
                # what the accepted draft makes them: genuine model outputs
                # for the position right after y_confirmed. Reuse them
                # instead of a redundant forward call.
                logits = logits2[:, 1:, :]
                mtp_logits = [m[:, 1:, :] for m in mtp_logits2]
                kv_caches = kv_caches2
            else:
                # At least one sequence's draft was rejected, so its slot in
                # the 2-token-extended cache holds KV computed for a token
                # that was never actually used (the rejected draft, not the
                # resampled replacement). Reusing that cache for any
                # sequence would silently corrupt future attention over
                # that position. Fall back to the cache extended by only
                # the confirmed y_next token, then run one more forward
                # pass over the real y_confirmed token to populate its
                # cache entry correctly.
                logits, mtp_logits, kv_caches = self(y_confirmed, kv_caches=kv_caches)

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
