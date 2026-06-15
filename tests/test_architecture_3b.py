import pytest
import torch
from model.architecture import GirivinityConfig, GirivinityModel


def test_3b_config_defaults():
    cfg = GirivinityConfig()
    assert cfg.dim == 3072
    assert cfg.n_layers == 28
    assert cfg.n_heads == 24
    assert cfg.n_kv_heads == 8
    assert cfg.head_dim == 128
    assert cfg.ffn_dim == 8192


def test_3b_param_count():
    cfg = GirivinityConfig()
    model = GirivinityModel(cfg)
    d = model.param_count_detailed()
    total_b = float(d["total"].replace("B", ""))
    assert 2.5 <= total_b <= 3.5, f"Expected ~3B, got {d['total']}"


def test_small_config_still_works():
    cfg = GirivinityConfig.small()
    assert cfg.dim == 1024
    assert cfg.n_layers == 16
    model = GirivinityModel(cfg)
    ids = torch.randint(0, 1000, (1, 8))
    logits, _ = model(ids)
    assert logits.shape == (1, 8, cfg.vocab_size)


def test_rope_theta_extended():
    cfg = GirivinityConfig()
    assert cfg.rope_theta == 500000.0


def test_ffn_dim_multiple_of_256():
    cfg = GirivinityConfig()
    assert cfg.ffn_dim % 256 == 0
