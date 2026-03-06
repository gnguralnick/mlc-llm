"""
This file specifies how MLC's InternVL parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .internvl_model import InternVLConfig, InternVLForCausalLM


def huggingface(
    model_config: InternVLConfig, quantization: Quantization
) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : InternVLConfig
        The configuration of the InternVL model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = InternVLForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    # ========== Language model weights ==========
    # HF prefix: "language_model." → MLC prefix: "language_model."
    llm_prefix = "language_model."

    for i in range(model_config.llm_config.num_hidden_layers):
        # Qwen3 fuses q/k/v into c_attn; HF has separate q_proj/k_proj/v_proj
        attn = f"model.layers.{i}.self_attn"
        mlc_name = f"{llm_prefix}{attn}.c_attn.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [
                f"{llm_prefix}{attn}.q_proj.weight",
                f"{llm_prefix}{attn}.k_proj.weight",
                f"{llm_prefix}{attn}.v_proj.weight",
            ],
            functools.partial(
                lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )
        if model_config.llm_config.attention_bias:
            mlc_name = f"{llm_prefix}{attn}.c_attn.bias"
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [
                    f"{llm_prefix}{attn}.q_proj.bias",
                    f"{llm_prefix}{attn}.k_proj.bias",
                    f"{llm_prefix}{attn}.v_proj.bias",
                ],
                functools.partial(
                    lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

        # Fuse gate_proj + up_proj → gate_up_proj
        mlp = f"model.layers.{i}.mlp"
        mlc_name = f"{llm_prefix}{mlp}.gate_up_proj.weight"
        mlc_param = named_parameters[mlc_name]
        mapping.add_mapping(
            mlc_name,
            [
                f"{llm_prefix}{mlp}.gate_proj.weight",
                f"{llm_prefix}{mlp}.up_proj.weight",
            ],
            functools.partial(
                lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

    # ========== Vision model weights ==========
    # Position embedding: HF stores as Parameter (1, N, D) → reshape to (N, D) for nn.Embedding
    pos_emb_mlc = "vision_model.vision_model.embeddings.position_embedding.weight"
    if pos_emb_mlc in named_parameters:
        mlc_param = named_parameters[pos_emb_mlc]
        mapping.add_mapping(
            pos_emb_mlc,
            ["vision_model.embeddings.position_embedding"],
            functools.partial(
                lambda x, dtype: x.reshape(-1, x.shape[-1]).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

    # Class embedding: HF stores as (1, 1, D) → reshape to (D,) for nn.Parameter
    cls_emb_mlc = "vision_model.vision_model.embeddings.class_embedding"
    if cls_emb_mlc in named_parameters:
        mlc_param = named_parameters[cls_emb_mlc]
        mapping.add_mapping(
            cls_emb_mlc,
            ["vision_model.embeddings.class_embedding"],
            functools.partial(
                lambda x, dtype: x.reshape(-1).astype(dtype),
                dtype=mlc_param.dtype,
            ),
        )

    # ========== MLP projector weights ==========
    # HF Sequential: 0=LayerNorm, 1=Linear, 2=GELU(no params), 3=Linear
    # MLC: mlp1.norm, mlp1.fc1, mlp1.act(GELU), mlp1.fc2
    mlp1_map = {
        "mlp1.norm.weight": "mlp1.0.weight",
        "mlp1.norm.bias": "mlp1.0.bias",
        "mlp1.fc1.weight": "mlp1.1.weight",
        "mlp1.fc1.bias": "mlp1.1.bias",
        "mlp1.fc2.weight": "mlp1.3.weight",
        "mlp1.fc2.bias": "mlp1.3.bias",
    }
    for mlc_name, hf_name in mlp1_map.items():
        if mlc_name in named_parameters:
            mlc_param = named_parameters[mlc_name]
            mapping.add_mapping(
                mlc_name,
                [hf_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    # ========== Remaining weights (1:1 mapping) ==========
    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            # For language_model.* params, HF name matches MLC name
            if mlc_name.startswith("language_model."):
                hf_name = mlc_name
            elif mlc_name.startswith("vision_model.vision_model."):
                # MLC: vision_model.vision_model.* → HF: vision_model.*
                hf_name = mlc_name.replace("vision_model.vision_model.", "vision_model.", 1)
            else:
                hf_name = mlc_name
            mapping.add_mapping(
                mlc_name,
                [hf_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )
    return mapping
