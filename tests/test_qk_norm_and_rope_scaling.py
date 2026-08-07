import torch

from model.architecture import GirivinityConfig, GirivinityModel


def test_qk_norm_disabled_by_default():
    cfg = GirivinityConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=32, ffn_multiplier=2.0)
    assert cfg.qk_norm_enabled is False
    model = GirivinityModel(cfg)
    assert not hasattr(model.layers[0].attn, "q_norm")


def test_qk_norm_enabled_adds_norm_modules_and_forward_works():
    cfg = GirivinityConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=32, ffn_multiplier=2.0, qk_norm_enabled=True)
    model = GirivinityModel(cfg)
    assert hasattr(model.layers[0].attn, "q_norm")
    assert hasattr(model.layers[0].attn, "k_norm")
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        logits, caches = model(ids)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert len(caches) == cfg.n_layers


def test_qk_norm_actually_normalizes_magnitude():
    from model.architecture import GroupedQueryAttention

    cfg = GirivinityConfig(dim=64, n_heads=4, n_kv_heads=2, qk_norm_enabled=True)
    attn = GroupedQueryAttention(cfg)
    x = torch.randn(2, 5, 64) * 10
    q_raw = attn.q_proj(x).view(2, 5, 4, 16).transpose(1, 2)
    q_normed = attn.q_norm(q_raw)
    rms = q_normed.pow(2).mean(-1).sqrt().mean().item()
    assert abs(rms - 1.0) < 0.15


def test_rope_scaling_defaults_to_no_op():
    cfg = GirivinityConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=16, ffn_multiplier=2.0)
    assert cfg.rope_scaling_factor == 1.0
    model = GirivinityModel(cfg)
    assert model.rope_cos.shape[0] == 16


def test_rope_scaling_extends_effective_context():
    cfg = GirivinityConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=16, ffn_multiplier=2.0, rope_scaling_factor=2.0)
    model = GirivinityModel(cfg)
    assert model.rope_cos.shape[0] == 32
    ids_long = torch.randint(0, cfg.vocab_size, (1, 24))
    with torch.no_grad():
        logits, caches = model(ids_long)
    assert logits.shape == (1, 24, cfg.vocab_size)


def test_without_rope_scaling_exceeding_max_seq_len_fails():
    cfg = GirivinityConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=16, ffn_multiplier=2.0)
    model = GirivinityModel(cfg)
    ids_long = torch.randint(0, cfg.vocab_size, (1, 24))
    try:
        with torch.no_grad():
            model(ids_long)
        assert False, "should have failed without rope scaling"
    except RuntimeError:
        pass


def test_qk_norm_composes_with_moe_and_kv_sharing():
    cfg = GirivinityConfig(
        dim=64, n_layers=4, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=32, ffn_multiplier=2.0,
        qk_norm_enabled=True, rope_scaling_factor=1.5,
        kv_sharing_start_layer=2,
        n_routed_experts=8, n_activated_experts=2, n_shared_experts=1, moe_intermediate_size=32, moe_start_layer=1,
    )
    model = GirivinityModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits, caches = model(ids)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert len(caches) == cfg.n_layers
