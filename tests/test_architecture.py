import torch
from model.architecture import GirivinityConfig, GirivinityModel


def test_model_output_shape():
    cfg = GirivinityConfig(
        dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
        vocab_size=1000, max_seq_len=128, ffn_multiplier=2.0
    )
    model = GirivinityModel(cfg)
    input_ids = torch.randint(0, 1000, (1, 16))
    logits, caches = model(input_ids)
    assert logits.shape == (1, 16, 1000)
    assert len(caches) == 2


def test_param_count_string():
    cfg = GirivinityConfig(
        dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
        vocab_size=1000, max_seq_len=128, ffn_multiplier=2.0
    )
    model = GirivinityModel(cfg)
    s = model.param_count()
    assert "M parameters" in s


def test_kv_cache_incremental():
    cfg = GirivinityConfig(
        dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
        vocab_size=1000, max_seq_len=128, ffn_multiplier=2.0
    )
    model = GirivinityModel(cfg)
    model.eval()
    with torch.no_grad():
        input_ids = torch.randint(0, 1000, (1, 8))
        logits1, caches = model(input_ids)
        next_token = torch.randint(0, 1000, (1, 1))
        logits2, _ = model(next_token, kv_caches=caches)
    assert logits2.shape == (1, 1, 1000)
