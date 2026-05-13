from __future__ import annotations
import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from model.architecture import GirivinityConfig, GirivinityModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def export_to_gguf(
    weights_path: str,
    output_dir: str,
    quant_type: str = "Q4_K_M",
) -> Path:
    weights = Path(weights_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = GirivinityConfig.from_yaml()
    model = GirivinityModel(cfg)
    state = torch.load(weights / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    hf_path = out_dir / "hf_export"
    hf_path.mkdir(exist_ok=True)

    try:
        from safetensors.torch import save_file
        save_file(model.state_dict(), hf_path / "model.safetensors")
        logger.info("Weights exported to safetensors")
    except ImportError:
        torch.save(model.state_dict(), hf_path / "pytorch_model.bin")
        logger.info("safetensors not available — saved as pytorch_model.bin")

    hf_config = {
        "architectures": ["GirivinityForCausalLM"],
        "hidden_size": cfg.dim,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "intermediate_size": cfg.ffn_dim,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_seq_len,
        "rms_norm_eps": cfg.norm_eps,
        "rope_theta": cfg.rope_theta,
        "model_type": "girivinity",
        "torch_dtype": "float32",
    }
    import json
    (hf_path / "config.json").write_text(json.dumps(hf_config, indent=2))

    tok_src = Path("models/tokeniser/tokeniser.json")
    if tok_src.exists():
        shutil.copy(tok_src, hf_path / "tokeniser.json")

    gguf_path = out_dir / "model.gguf"
    convert_script = Path("llama.cpp/convert_hf_to_gguf.py")

    if not convert_script.exists():
        logger.warning(
            "llama.cpp not found at %s. "
            "Clone it: git clone https://github.com/ggerganov/llama.cpp",
            convert_script,
        )
        logger.info(
            "Manual conversion command when llama.cpp is available:\n"
            "  python llama.cpp/convert_hf_to_gguf.py %s "
            "--outfile %s --outtype %s",
            hf_path, gguf_path, quant_type.lower(),
        )
        return hf_path

    result = subprocess.run(
        [
            sys.executable, str(convert_script),
            str(hf_path),
            "--outfile", str(gguf_path),
            "--outtype", quant_type.lower(),
        ],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        logger.error("GGUF conversion failed:\n%s", result.stderr)
        raise RuntimeError("GGUF conversion failed")

    logger.info("GGUF model saved to %s", gguf_path)
    return gguf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/base/final")
    parser.add_argument("--output",  default="models/girivinity_quantised")
    parser.add_argument("--quant",   default="Q4_K_M")
    args = parser.parse_args()
    export_to_gguf(args.weights, args.output, args.quant)
