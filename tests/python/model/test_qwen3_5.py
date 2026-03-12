# pylint: disable=invalid-name,missing-docstring
"""Unit tests for Qwen3.5 hybrid model architecture (DeltaNet + full attention)."""

import pytest

from mlc_llm.model import MODELS


# Minimal config dict with small dimensions for fast testing.
# Mirrors the structure of a real HuggingFace Qwen3.5 config.json.
SMALL_QWEN35_CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "vocab_size": 1000,
    "rms_norm_eps": 1e-6,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ],
    "linear_num_key_heads": 4,
    "linear_key_head_dim": 16,
    "linear_num_value_heads": 4,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "context_window_size": 512,
    "prefill_chunk_size": 256,
    "rope_parameters": {
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
    },
}


def test_qwen35_model_registered():
    """Verify Qwen3.5 model is in the registry."""
    assert "qwen3_5" in MODELS, "qwen3_5 should be registered in MODELS"


def test_qwen35_creation():
    """Test Qwen3.5 model creation and export to TVM IR.

    Verifies:
    - Config can be loaded from dict
    - Model instance can be created
    - Model exports to TVM IR successfully
    - Named parameters include both linear and full attention components
    - All expected functions are exported (including create_rnn_state for hybrid state)
    """
    model_info = MODELS["qwen3_5"]
    config = model_info.config.from_dict(SMALL_QWEN35_CONFIG)
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )

    # Verify export succeeded
    assert mod is not None
    assert len(named_params) > 0

    # Verify parameters include both layer types
    param_names = [name for name, _ in named_params]
    has_linear_attn = any("linear_attn" in n for n in param_names)
    has_full_attn = any("self_attn" in n for n in param_names)
    assert has_linear_attn, "Should have DeltaNet linear attention parameters"
    assert has_full_attn, "Should have full attention parameters"

    # Verify all expected functions are exported (hybrid model needs both KV cache and RNN state)
    expected_funcs = [
        "embed",
        "prefill",
        "decode",
        "batch_prefill",
        "batch_decode",
        "batch_verify",
        "create_paged_kv_cache",
        "create_rnn_state",
    ]
    for func_name in expected_funcs:
        assert func_name in mod, f"Module should contain '{func_name}' function"

    mod.show(black_format=False)
    for name, param in named_params:
        print(name, param.shape, param.dtype)


def test_qwen35_config_validation():
    """Test Qwen3.5 configuration computed fields."""
    model_info = MODELS["qwen3_5"]
    config = model_info.config.from_dict(SMALL_QWEN35_CONFIG)

    # Verify layer type counts
    assert config.num_linear_attn_layers == 3
    assert config.num_full_attn_layers == 1

    # Verify rotary dim computed from partial_rotary_factor
    assert config.rotary_dim == int(config.head_dim * 0.25)

    # Verify context/prefill propagated
    assert config.context_window_size == 512
    assert config.prefill_chunk_size == 256

    print(
        f"Qwen3.5 Config: hidden={config.hidden_size}, "
        f"layers={config.num_hidden_layers} "
        f"(linear={config.num_linear_attn_layers}, full={config.num_full_attn_layers}), "
        f"rotary_dim={config.rotary_dim}"
    )


if __name__ == "__main__":
    test_qwen35_model_registered()
    test_qwen35_creation()
    test_qwen35_config_validation()
