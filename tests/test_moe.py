import torch

from model.architecture import Expert, GirivinityConfig, GirivinityModel, MoELayer, SwiGLU


def _small_moe_cfg(**overrides):
    defaults = dict(
        dim=64, n_layers=4, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=32,
        ffn_multiplier=2.0,
        n_routed_experts=8, n_activated_experts=2, n_shared_experts=1,
        moe_intermediate_size=32, moe_start_layer=1,
    )
    defaults.update(overrides)
    return GirivinityConfig(**defaults)


def test_moe_disabled_by_default():
    cfg = GirivinityConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=32)
    assert cfg.moe_enabled is False
    model = GirivinityModel(cfg)
    assert all(isinstance(layer.ffn, SwiGLU) for layer in model.layers)


def test_moe_enhanced_preset_enables_expected_defaults():
    cfg = GirivinityConfig.moe_enhanced(dim=64, n_layers=4, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=32, moe_intermediate_size=32)
    assert cfg.moe_enabled is True
    assert cfg.n_routed_experts == 64
    assert cfg.n_activated_experts == 6
    assert cfg.n_shared_experts == 2
    assert cfg.moe_start_layer == 1


def test_moe_layer_assignment_respects_moe_start_layer():
    cfg = _small_moe_cfg(moe_start_layer=2)
    model = GirivinityModel(cfg)
    assert isinstance(model.layers[0].ffn, SwiGLU)
    assert isinstance(model.layers[1].ffn, SwiGLU)
    assert isinstance(model.layers[2].ffn, MoELayer)
    assert isinstance(model.layers[3].ffn, MoELayer)


def test_moe_forward_shape_and_cache_length():
    cfg = _small_moe_cfg()
    model = GirivinityModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    with torch.no_grad():
        logits, caches = model(ids)
    assert logits.shape == (2, 10, cfg.vocab_size)
    assert len(caches) == cfg.n_layers


def test_moe_incremental_decode_with_kv_cache():
    cfg = _small_moe_cfg()
    model = GirivinityModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        _, caches = model(ids)
        next_tok = torch.randint(0, cfg.vocab_size, (1, 1))
        logits2, caches2 = model(next_tok, kv_caches=caches)
    assert logits2.shape == (1, 1, cfg.vocab_size)
    assert len(caches2) == cfg.n_layers


def test_moe_gradients_reach_experts_and_shared_expert():
    cfg = _small_moe_cfg()
    model = GirivinityModel(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (4, 20))
    logits, _ = model(ids)
    logits.sum().backward()

    moe_layer = model.layers[1].ffn
    assert isinstance(moe_layer, MoELayer)
    for expert in moe_layer.routed_experts:
        assert expert.gate.weight.grad is not None
    for shared in moe_layer.shared_experts:
        assert shared.gate.weight.grad is not None
    assert moe_layer.gate.weight.grad is not None


def test_moe_load_balancing_bias_moves_and_stays_centered():
    cfg = _small_moe_cfg()
    model = GirivinityModel(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (4, 20))
    model(ids)
    moe_layer = model.layers[1].ffn
    assert (moe_layer.routing_bias != 0).any()
    assert abs(moe_layer.routing_bias.sum().item()) < 1e-3 * cfg.n_routed_experts


def test_moe_save_and_load_round_trip():
    import tempfile
    cfg = _small_moe_cfg()
    model = GirivinityModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        reloaded = GirivinityModel.load_pretrained(tmp)
        reloaded.eval()
        with torch.no_grad():
            out1, _ = model(ids)
            out2, _ = reloaded(ids)
        assert torch.allclose(out1, out2)


def test_grow_experts_doubles_routed_and_activated_experts():
    gen1 = GirivinityConfig.moe_enhanced(dim=64, n_layers=4, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=32, moe_intermediate_size=32)
    gen2 = gen1.grow_experts()
    assert gen2.n_routed_experts == gen1.n_routed_experts * 2
    assert gen2.n_activated_experts == gen1.n_activated_experts * 2
    assert gen2.moe_intermediate_size == gen1.moe_intermediate_size
    assert gen2.moe_start_layer == gen1.moe_start_layer

    model = GirivinityModel(gen2)
    ids = torch.randint(0, gen2.vocab_size, (1, 6))
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape == (1, 6, gen2.vocab_size)


def test_grow_experts_rejects_non_moe_config():
    cfg = GirivinityConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=32)
    try:
        cfg.grow_experts()
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_moe_composes_with_kv_sharing_ple_and_mhc():
    cfg = GirivinityConfig(
        dim=64, n_layers=6, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=32,
        kv_sharing_start_layer=3, ple_dim=16, n_residual_streams=2,
        n_routed_experts=8, n_activated_experts=2, n_shared_experts=1,
        moe_intermediate_size=32, moe_start_layer=2,
    )
    model = GirivinityModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits, caches = model(ids)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert len(caches) == cfg.n_layers
