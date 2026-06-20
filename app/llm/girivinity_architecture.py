from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class GirivinityConfig:
    vocab_size: int = 65536
    hidden_dim: int = 2048
    num_heads: int = 16
    num_layers: int = 24
    ffn_dim: int = 8192
    max_seq_len: int = 32768
    dropout: float = 0.1
    use_rope: bool = True
    use_gqa: bool = True
    num_kv_heads: int = 4
    use_rmsnorm: bool = True

    @property
    def head_dim(self) -> int:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        return self.hidden_dim // self.num_heads

    @property
    def effective_num_kv_heads(self) -> int:
        return self.num_kv_heads if self.use_gqa else self.num_heads


class GirivinityRMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: Tensor) -> Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class GirivinityRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 32768,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, seq_len: int, offset: int = 0) -> Tensor:
        positions = torch.arange(offset, offset + seq_len, device=x.device, dtype=self.inv_freq.dtype)
        if offset + seq_len > self.max_seq_len:
            positions = positions / self._dynamic_scale(offset + seq_len)
        elif self.scaling_factor != 1.0:
            positions = positions / self.scaling_factor

        freqs = torch.outer(positions, self.inv_freq.to(x.device))
        cos = freqs.cos().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
        return self.apply_rotary(x, cos, sin)

    def _dynamic_scale(self, seq_len: int) -> float:
        return max(self.scaling_factor, seq_len / self.max_seq_len)

    @staticmethod
    def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
        repeated_cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
        repeated_sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
        return (x * repeated_cos) + (rotated * repeated_sin)


class GirivinityGroupedQueryAttention(nn.Module):
    def __init__(self, config: GirivinityConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.effective_num_kv_heads
        self.head_dim = config.head_dim
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.rotary = GirivinityRotaryEmbedding(self.head_dim, config.max_seq_len) if config.use_rope else None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        past_len = past_key_value[0].size(2) if past_key_value is not None else 0

        query = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        if self.rotary is not None:
            query = self.rotary(query, seq_len=seq_len, offset=past_len)
            key = self.rotary(key, seq_len=seq_len, offset=past_len)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        present_key_value = (key, value)
        if self.num_kv_groups > 1:
            key = key.repeat_interleave(self.num_kv_groups, dim=1)
            value = value.repeat_interleave(self.num_kv_groups, dim=1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self._causal_mask(seq_len, key.size(2), past_len, hidden_states.device)
        if attention_mask is not None:
            scores = scores + self._expand_attention_mask(attention_mask, scores.dtype, seq_len, key.size(2))

        attn_weights = F.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output), present_key_value

    @staticmethod
    def _causal_mask(seq_len: int, total_len: int, past_len: int, device: torch.device) -> Tensor:
        query_positions = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(-1)
        key_positions = torch.arange(total_len, device=device).unsqueeze(0)
        mask = key_positions > query_positions
        return torch.zeros((1, 1, seq_len, total_len), device=device).masked_fill(mask, float("-inf"))

    @staticmethod
    def _expand_attention_mask(mask: Tensor, dtype: torch.dtype, seq_len: int, total_len: int) -> Tensor:
        if mask.dim() == 2:
            expanded = mask[:, None, None, :].to(dtype=dtype)
            if expanded.size(-1) != total_len:
                pad_len = total_len - expanded.size(-1)
                if pad_len > 0:
                    pad = torch.ones((*expanded.shape[:-1], pad_len), device=mask.device, dtype=dtype)
                    expanded = torch.cat((pad, expanded), dim=-1)
            return (1.0 - expanded[:, :, :, -total_len:]) * torch.finfo(dtype).min
        if mask.dim() == 4:
            return mask.to(dtype=dtype)
        if mask.dim() == 3:
            return mask[:, None, :, :].to(dtype=dtype)
        raise ValueError("attention_mask must have shape [B, S], [B, T, S], or [B, 1, T, S]")


class GirivinityFFN(nn.Module):
    def __init__(self, config: GirivinityConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.dropout(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class GirivinityDecoderLayer(nn.Module):
    def __init__(self, config: GirivinityConfig) -> None:
        super().__init__()
        norm_cls = GirivinityRMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.input_norm = norm_cls(config.hidden_dim)
        self.post_attention_norm = norm_cls(config.hidden_dim)
        self.self_attn = GirivinityGroupedQueryAttention(config)
        self.ffn = GirivinityFFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_value: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attn_input = self.input_norm(hidden_states)
        attn_output, present_key_value = self.self_attn(attn_input, attention_mask, past_key_value)
        hidden_states = hidden_states + self.dropout(attn_output)
        ffn_input = self.post_attention_norm(hidden_states)
        hidden_states = hidden_states + self.dropout(self.ffn(ffn_input))
        return hidden_states, present_key_value


class GirivinityModel(nn.Module):
    def __init__(self, config: GirivinityConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([GirivinityDecoderLayer(config) for _ in range(config.num_layers)])
        norm_cls = GirivinityRMSNorm if config.use_rmsnorm else nn.LayerNorm
        self.final_norm = norm_cls(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    @classmethod
    def from_config(cls, config: GirivinityConfig) -> "GirivinityModel":
        return cls(config)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        hidden_states = self.embed_tokens(input_ids)
        next_key_values = []
        use_gradient_checkpointing = (
            self.gradient_checkpointing
            and self.training
            and past_key_values is None
        )
        for idx, layer in enumerate(self.layers):
            past = past_key_values[idx] if past_key_values is not None else None
            if use_gradient_checkpointing:
                from torch.utils.checkpoint import checkpoint

                def custom_forward(
                    states: Tensor,
                    checkpointed_layer: GirivinityDecoderLayer = layer,
                ) -> Tensor:
                    return checkpointed_layer(states, attention_mask, None)[0]

                hidden_states = checkpoint(custom_forward, hidden_states, use_reentrant=False)
                continue
            hidden_states, present = layer(hidden_states, attention_mask, past)
            next_key_values.append(present)
        logits = self.lm_head(self.final_norm(hidden_states))
        return logits, next_key_values

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tensor:
        self.eval()
        generated = input_ids
        past_key_values = None
        next_input = input_ids
        for _ in range(max_new_tokens):
            logits, past_key_values = self(next_input, past_key_values=past_key_values)
            next_token_logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            generated = torch.cat((generated, next_token), dim=1)
            next_input = next_token
        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_pretrained(self, path: str) -> None:
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(asdict(self.config), indent=2))
        torch.save(self.state_dict(), output_dir / "pytorch_model.bin")

    @classmethod
    def load_pretrained(cls, path: str) -> "GirivinityModel":
        model_dir = Path(path)
        config = GirivinityConfig(**json.loads((model_dir / "config.json").read_text()))
        model = cls(config)
        state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _sample_top_p(probs: Tensor, top_p: float) -> Tensor:
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

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
