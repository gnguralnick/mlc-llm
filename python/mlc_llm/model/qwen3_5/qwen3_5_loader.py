"""
This file specifies how MLC's Qwen3.5 parameter maps from HuggingFace format.

HF uses model.language_model.* prefix; MLC uses model.* prefix.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .qwen3_5_model import Qwen35Config, Qwen35LMHeadModel


def _mlc_to_hf(mlc_name: str) -> str:
    """Convert MLC parameter name to HuggingFace parameter name."""
    # model.layers.X.Y -> model.language_model.layers.X.Y
    # model.embed_tokens.weight -> model.language_model.embed_tokens.weight
    # model.norm.weight -> model.language_model.norm.weight
    # lm_head.weight -> model.language_model.lm_head.weight (if not tied)
    if mlc_name.startswith("model."):
        return "model.language_model." + mlc_name[len("model."):]
    if mlc_name.startswith("lm_head."):
        return "model.language_model." + mlc_name
    return mlc_name


def huggingface(model_config: Qwen35Config, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping from MLC LLM parameters to HuggingFace parameters."""
    model = Qwen35LMHeadModel(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)

    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)
    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
        layer_type = model_config.layer_types[i]

        if layer_type == "full_attention":
            # MLP: concat gate_proj + up_proj -> gate_up_proj
            mlc_mlp = f"model.layers.{i}.mlp"
            hf_mlp = f"model.language_model.layers.{i}.mlp"
            mapping.add_mapping(
                f"{mlc_mlp}.gate_up_proj.weight",
                [
                    f"{hf_mlp}.gate_proj.weight",
                    f"{hf_mlp}.up_proj.weight",
                ],
                functools.partial(
                    lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                    dtype=named_parameters[f"{mlc_mlp}.gate_up_proj.weight"].dtype,
                ),
            )
        elif layer_type == "linear_attention":
            # norm.weight -> g_norm.weight
            mlc_attn = f"model.layers.{i}.linear_attn"
            hf_attn = f"model.language_model.layers.{i}.linear_attn"
            mlc_norm_name = f"{mlc_attn}.g_norm.weight"
            hf_norm_name = f"{hf_attn}.norm.weight"
            if mlc_norm_name in named_parameters:
                mapping.add_mapping(
                    mlc_norm_name,
                    [hf_norm_name],
                    functools.partial(
                        lambda x, dtype: x.astype(dtype),
                        dtype=named_parameters[mlc_norm_name].dtype,
                    ),
                )

            # MLP: concat gate_proj + up_proj -> gate_up_proj
            mlc_mlp = f"model.layers.{i}.mlp"
            hf_mlp = f"model.language_model.layers.{i}.mlp"
            mapping.add_mapping(
                f"{mlc_mlp}.gate_up_proj.weight",
                [
                    f"{hf_mlp}.gate_proj.weight",
                    f"{hf_mlp}.up_proj.weight",
                ],
                functools.partial(
                    lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                    dtype=named_parameters[f"{mlc_mlp}.gate_up_proj.weight"].dtype,
                ),
            )

    # All remaining parameters: 1:1 mapping with HF prefix translation
    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            hf_name = _mlc_to_hf(mlc_name)
            mapping.add_mapping(
                mlc_name,
                [hf_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    return mapping
