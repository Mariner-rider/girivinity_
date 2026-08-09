import torch

from model.architecture import (
    GirivinityConfig,
    GirivinityModel,
    GroupedQueryAttention,
    MultiHeadLatentAttention,
)


def _small_mla_cfg(**overrides):
    defaults = dict(
        dim=64, n_layers=3, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=64,
        ffn_multiplier=2.0, mla_enabled=True,
    )
    defaults.update(overrides)
    return GirivinityConfig(**defaults)


def test_mla_disabled_by_default():
    cfg = GirivinityConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=32, ffn_multiplier=2.0)
    assert cfg.mla_enabled is False
    assert cfg.mla_d_c is None
    model = GirivinityModel(cfg)
    assert all(isinstance(layer.attn, GroupedQueryAttention) for layer in model.layers)
    assert hasattr(model, "rope_cos")
    assert not hasattr(model, "mla_rope_cos")


def test_mla_enabled_resolves_defaults_and_uses_mla_attention():
    cfg = _small_mla_cfg()
    assert cfg.mla_enabled is True
    assert cfg.mla_d_c == cfg.head_dim // 2
    assert cfg.mla_d_c_q == cfg.mla_d_c
    assert cfg.mla_rope_head_dim == cfg.head_dim // 2
    model = GirivinityModel(cfg)
    assert all(isinstance(layer.attn, MultiHeadLatentAttention) for layer in model.layers)
    assert hasattr(model, "mla_rope_cos")
    assert not hasattr(model, "rope_cos")


def test_mla_rope_head_dim_rounds_odd_up_to_even():
    # head_dim=6 -> head_dim//2=3 (odd); RoPE needs an even dim to split
    # into rotation pairs, so this must round up to 4, not stay at 3.
    cfg = GirivinityConfig(dim=24, n_heads=4, n_kv_heads=2, mla_enabled=True)
    assert cfg.head_dim == 6
    assert cfg.mla_rope_head_dim == 4
    model = GirivinityModel(GirivinityConfig(dim=24, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=50, max_seq_len=16, mla_enabled=True))
    model.eval()
    ids = torch.randint(0, 50, (1, 6))
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape == (1, 6, 50)


def test_mla_forward_shape_and_cache_length():
    cfg = _small_mla_cfg()
    model = GirivinityModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    with torch.no_grad():
        logits, caches = model(ids)
    assert logits.shape == (2, 10, cfg.vocab_size)
    assert len(caches) == cfg.n_layers


def test_mla_kv_cache_is_compressed_not_full_per_head():
    """Verification (b): the cache must store the compressed latent
    (d_c), not full per-head K/V (n_kv_heads * head_dim)."""
    cfg = _small_mla_cfg()
    model = GirivinityModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        _, caches = model(ids)
    c_kv, k_r = caches[0]
    assert c_kv.shape[-1] == cfg.mla_d_c
    assert c_kv.shape[-1] != cfg.n_kv_heads * cfg.head_dim
    assert k_r.shape[-1] == cfg.mla_rope_head_dim


def test_mla_incremental_decode_matches_full_context_forward():
    """The critical correctness check: since the up-projection is redone
    fresh from the cached latent on every step, a bug in cache
    concatenation, position offsetting, or RoPE re-application would
    show up as a numerical mismatch here even though shapes would still
    look fine — this is a much stronger check than a shape-only test."""
    torch.manual_seed(0)
    cfg = _small_mla_cfg(n_layers=3)
    model = GirivinityModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 20))

    with torch.no_grad():
        full_logits, _ = model(ids)

        prefix = ids[:, :5]
        logits_inc, caches = model(prefix)
        chunks = [logits_inc]
        for t in range(5, 20):
            next_tok = ids[:, t:t + 1]
            logits_step, caches = model(next_tok, kv_caches=caches)
            chunks.append(logits_step)
        inc_logits = torch.cat(chunks, dim=1)

    assert full_logits.shape == inc_logits.shape
    assert torch.allclose(full_logits, inc_logits, atol=1e-4)


def test_mla_long_prefill_then_one_more_token_reuses_cache():
    """Verification (a): generate a long context via kv_cache, then one
    more token, and confirm the cache was actually extended (reused and
    grown) rather than recomputed from scratch."""
    torch.manual_seed(0)
    cfg = GirivinityConfig(
        dim=32, n_layers=2, n_heads=2, n_kv_heads=1, vocab_size=100, max_seq_len=2048,
        ffn_multiplier=2.0, mla_enabled=True,
    )
    model = GirivinityModel(cfg)
    model.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 1024))
    with torch.no_grad():
        logits, caches = model(ids)
        assert logits.shape == (1, 1024, cfg.vocab_size)
        assert caches[0][0].shape[1] == 1024

        next_tok = torch.randint(0, cfg.vocab_size, (1, 1))
        logits2, caches2 = model(next_tok, kv_caches=caches)

    assert logits2.shape == (1, 1, cfg.vocab_size)
    # Cache must have grown by exactly one position, and the first 1024
    # cached positions must be untouched (proving reuse, not recomputation).
    assert caches2[0][0].shape[1] == 1025
    assert torch.equal(caches2[0][0][:, :1024, :], caches[0][0])
    assert torch.equal(caches2[0][1][:, :1024, :], caches[0][1])


def test_mla_cache_memory_ratio_vs_gqa_at_real_3b_scale():
    """Verification (c): MLA cache < 15% of the equivalent GQA cache.

    This ratio is per-token and context-length-independent (both cache
    formats grow linearly in the sequence dimension), so it's checked
    analytically against the real production config's resolved values
    rather than by materializing an actual 8192-token cache — multiplying
    both sides by 8192 doesn't change the ratio, only the absolute sizes.
    Also checked empirically below at the same n_kv_heads=8 the real
    config uses, since this ratio is n_kv_heads-dependent (not universal
    across all configs) and worth confirming against a real forward pass,
    not just formula arithmetic.
    """
    cfg = GirivinityConfig(mla_enabled=True)  # real production defaults: dim=3072, n_heads=24, n_kv_heads=8
    mla_floats_per_token = cfg.mla_d_c + cfg.mla_rope_head_dim
    gqa_floats_per_token = 2 * cfg.n_kv_heads * cfg.head_dim
    ratio = mla_floats_per_token / gqa_floats_per_token
    assert ratio < 0.15

    context = 8192
    mla_cache_size_at_context = mla_floats_per_token * context
    gqa_cache_size_at_context = gqa_floats_per_token * context
    assert mla_cache_size_at_context < 0.15 * gqa_cache_size_at_context


def test_mla_cache_memory_ratio_empirical_at_matching_n_kv_heads():
    """Same ratio, confirmed empirically via real tensors from an actual
    forward pass, at a small-but-proportional config (same n_kv_heads=8
    as the real target, so the ratio matches — it depends on n_kv_heads,
    not on the absolute model size)."""
    torch.manual_seed(0)
    shared = dict(dim=256, n_layers=2, n_heads=16, n_kv_heads=8, vocab_size=200, max_seq_len=64, ffn_multiplier=2.0)
    model_mla = GirivinityModel(GirivinityConfig(mla_enabled=True, **shared))
    model_gqa = GirivinityModel(GirivinityConfig(mla_enabled=False, **shared))
    model_mla.eval()
    model_gqa.eval()

    ids = torch.randint(0, 200, (1, 16))
    with torch.no_grad():
        _, caches_mla = model_mla(ids)
        _, caches_gqa = model_gqa(ids)

    mla_per_token = caches_mla[0][0].shape[-1] + caches_mla[0][1].shape[-1]
    gqa_per_token = (caches_gqa[0][0].numel() + caches_gqa[0][1].numel()) // 16
    assert mla_per_token / gqa_per_token < 0.15


def test_mla_gradients_flow_to_all_projection_matrices():
    cfg = _small_mla_cfg()
    model = GirivinityModel(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    logits, _ = model(ids)
    logits.sum().backward()

    attn = model.layers[0].attn
    assert isinstance(attn, MultiHeadLatentAttention)
    for name, p in attn.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"


def test_mla_save_and_load_round_trip():
    import tempfile

    cfg = _small_mla_cfg()
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


def test_mla_composes_with_moe_and_rope_scaling():
    cfg = GirivinityConfig(
        dim=64, n_layers=4, n_heads=4, n_kv_heads=2, vocab_size=200, max_seq_len=32, ffn_multiplier=2.0,
        mla_enabled=True, rope_scaling_factor=2.0,
        n_routed_experts=8, n_activated_experts=2, n_shared_experts=1, moe_intermediate_size=32, moe_start_layer=1,
    )
    model = GirivinityModel(cfg)
    model.eval()
    assert model.mla_rope_cos.shape[0] == int(cfg.max_seq_len * cfg.rope_scaling_factor)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits, caches = model(ids)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert len(caches) == cfg.n_layers


def test_mla_rejects_kv_sharing():
    try:
        GirivinityConfig(mla_enabled=True, kv_sharing_start_layer=2)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_mla_rejects_qk_norm():
    try:
        GirivinityConfig(mla_enabled=True, qk_norm_enabled=True)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_mla_generate_end_to_end():
    torch.manual_seed(0)
    cfg = _small_mla_cfg(n_layers=2)
    model = GirivinityModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(ids, max_new_tokens=6, temperature=0.0)
    assert out.shape == (1, 10)
    assert torch.equal(out[:, :4], ids)
