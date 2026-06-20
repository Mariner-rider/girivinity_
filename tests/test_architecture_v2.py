import pytest


def _import_model():
    """Import with real torch if available, skip if not."""
    pytest.importorskip("torch")
    from model.architecture import (
        GirivinityConfig,
        GirivinityModel,
        ManifoldHyperConnection,
        PerLayerEmbedding,
    )
    return GirivinityConfig, GirivinityModel, ManifoldHyperConnection, PerLayerEmbedding


def test_default_config_has_features_disabled():
    pytest.importorskip("torch")
    from model.architecture import GirivinityConfig

    cfg = GirivinityConfig()
    assert cfg.kv_sharing_enabled is False
    assert cfg.ple_enabled is False
    assert cfg.mhc_enabled is False


def test_v2_enhanced_config_has_all_features():
    pytest.importorskip("torch")
    from model.architecture import GirivinityConfig

    cfg = GirivinityConfig.v2_enhanced()
    assert cfg.kv_sharing_enabled is True
    assert cfg.ple_enabled is True
    assert cfg.mhc_enabled is True
    assert cfg.kv_sharing_start_layer == 14
    assert cfg.ple_dim == 64
    assert cfg.n_residual_streams == 4


def test_standard_forward_still_works():
    pytest.importorskip("torch")
    import torch
    from model.architecture import GirivinityConfig, GirivinityModel

    cfg = GirivinityConfig.small()
    model = GirivinityModel(cfg)
    ids = torch.randint(0, 100, (1, 8))
    logits, caches = model(ids)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert len(caches) == cfg.n_layers


def test_kv_sharing_forward():
    pytest.importorskip("torch")
    import torch
    from model.architecture import GirivinityConfig, GirivinityModel

    cfg = GirivinityConfig(
        dim=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=200,
        max_seq_len=64,
        ffn_multiplier=2.0,
        kv_sharing_start_layer=2,
    )
    model = GirivinityModel(cfg)
    ids = torch.randint(0, 200, (1, 8))
    logits, caches = model(ids)
    assert logits.shape == (1, 8, 200)
    assert len(caches) == 4


def test_ple_forward():
    pytest.importorskip("torch")
    import torch
    from model.architecture import GirivinityConfig, GirivinityModel

    cfg = GirivinityConfig(
        dim=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=200,
        max_seq_len=64,
        ffn_multiplier=2.0,
        ple_dim=16,
    )
    model = GirivinityModel(cfg)
    assert model.ple is not None
    ids = torch.randint(0, 200, (1, 8))
    logits, _ = model(ids)
    assert logits.shape == (1, 8, 200)


def test_mhc_forward():
    pytest.importorskip("torch")
    import torch
    from model.architecture import GirivinityConfig, GirivinityModel

    cfg = GirivinityConfig(
        dim=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=200,
        max_seq_len=64,
        ffn_multiplier=2.0,
        n_residual_streams=4,
    )
    model = GirivinityModel(cfg)
    assert model.mhc is not None
    ids = torch.randint(0, 200, (1, 8))
    logits, _ = model(ids)
    assert logits.shape == (1, 8, 200)


def test_v2_enhanced_full_forward():
    pytest.importorskip("torch")
    import torch
    from model.architecture import GirivinityConfig, GirivinityModel

    cfg = GirivinityConfig(
        dim=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=200,
        max_seq_len=64,
        ffn_multiplier=2.0,
        kv_sharing_start_layer=2,
        ple_dim=8,
        n_residual_streams=4,
    )
    model = GirivinityModel(cfg)
    ids = torch.randint(0, 200, (1, 8))
    logits, caches = model(ids)
    assert logits.shape == (1, 8, 200)
    assert len(caches) == 4


def test_mhc_doubly_stochastic():
    pytest.importorskip("torch")
    import torch
    from model.architecture import ManifoldHyperConnection

    mhc = ManifoldHyperConnection(dim=64, n=4)
    ds = mhc.get_doubly_stochastic()
    assert (ds >= 0).all()
    row_sums = ds.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(4), atol=0.01)
    col_sums = ds.sum(dim=0)
    assert torch.allclose(col_sums, torch.ones(4), atol=0.01)


def test_kv_sharing_reduces_unique_caches():
    pytest.importorskip("torch")
    import torch
    from model.architecture import GirivinityConfig, GirivinityModel

    cfg = GirivinityConfig(
        dim=64,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=200,
        max_seq_len=64,
        ffn_multiplier=2.0,
        kv_sharing_start_layer=3,
    )
    model = GirivinityModel(cfg)
    ids = torch.randint(0, 200, (1, 4))
    logits, caches = model(ids)
    assert logits.shape == (1, 4, 200)
    assert len(caches) == 6
