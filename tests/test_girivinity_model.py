from __future__ import annotations

import torch

from model.architecture import GirivinityConfig, GirivinityModel


def test_girivinity_model_forward_returns_logits_and_supports_cache():
    config = GirivinityConfig(
        dim=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=128,
        max_seq_len=16,
        ffn_multiplier=2.0,
    )
    model = GirivinityModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits = model(input_ids)

    assert logits.shape == (2, 5, config.vocab_size)

    cache = model.init_kv_cache(batch_size=2, device="cpu")
    first = model(input_ids[:, :3], start_pos=0, kv_cache=cache)
    second = model(input_ids[:, 3:], start_pos=3, kv_cache=cache)

    assert first.shape == (2, 3, config.vocab_size)
    assert second.shape == (2, 2, config.vocab_size)
    assert cache[0]["k"].shape == (2, config.max_seq_len, config.n_kv_heads, config.head_dim)
