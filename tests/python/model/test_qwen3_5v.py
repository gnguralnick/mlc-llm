# pylint: disable=invalid-name,missing-docstring
"""Unit tests for Qwen3.5 Vision-Language model architecture."""

import pytest

from mlc_llm.model import MODELS


# Minimal VLM config with small dimensions for fast testing.
# image_size=56, patch_size=14 -> 4x4 grid, spatial_merge_size=2 -> 2x2 = 4 tokens per image
SMALL_QWEN35V_CONFIG = {
    "text_config": {
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
    },
    "vision_config": {
        "hidden_size": 64,
        "num_heads": 2,
        "depth": 2,
        "intermediate_size": 128,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "out_hidden_size": 128,
        "in_channels": 3,
    },
    "image_size": 56,
    "image_token_id": 248056,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
}


def test_qwen35v_model_registered():
    """Verify Qwen3.5V model is in the registry."""
    assert "qwen3_5_v" in MODELS, "qwen3_5_v should be registered in MODELS"


def test_qwen35v_creation():
    """Test Qwen3.5V model creation and export to TVM IR.

    Verifies:
    - Config can be loaded from dict
    - Model instance can be created
    - Model exports to TVM IR successfully
    - Named parameters include visual and language_model components
    - All expected functions are exported (including image_embed)
    """
    model_info = MODELS["qwen3_5_v"]
    config = model_info.config.from_dict(SMALL_QWEN35V_CONFIG)
    model = model_info.model(config)
    mod, named_params = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
    )

    # Verify export succeeded
    assert mod is not None
    assert len(named_params) > 0

    # Verify VLM composition: params from both components
    param_names = [name for name, _ in named_params]
    has_visual = any(n.startswith("visual.") for n in param_names)
    has_language = any(n.startswith("language_model.") for n in param_names)
    assert has_visual, "Should have visual encoder parameters"
    assert has_language, "Should have language_model parameters"

    # Verify all expected functions are exported
    expected_funcs = [
        "embed",
        "image_embed",
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


def test_qwen35v_config_validation():
    """Test Qwen3.5V configuration computed properties."""
    model_info = MODELS["qwen3_5_v"]
    config = model_info.config.from_dict(SMALL_QWEN35V_CONFIG)

    # image_size=56, patch_size=14 -> grid 4x4
    assert config.grid_h == 56 // 14  # 4
    assert config.grid_w == 56 // 14  # 4

    # spatial_merge_size=2 -> (4/2) * (4/2) = 4 tokens per image
    assert config.tokens_per_image == (config.grid_h // 2) * (config.grid_w // 2)

    # Verify text config propagated
    assert config.vocab_size == 1000
    assert config.context_window_size == 512

    print(
        f"Qwen3.5V Config: grid={config.grid_h}x{config.grid_w}, "
        f"tokens_per_image={config.tokens_per_image}, "
        f"vocab_size={config.vocab_size}"
    )


if __name__ == "__main__":
    test_qwen35v_model_registered()
    test_qwen35v_creation()
    test_qwen35v_config_validation()
