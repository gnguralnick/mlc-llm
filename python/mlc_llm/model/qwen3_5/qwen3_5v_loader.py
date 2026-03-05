"""Weight mapping from HuggingFace to MLC for Qwen3.5 Vision-Language model."""

import functools

import numpy as np

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .qwen3_5v_model import Qwen35VConfig, Qwen35VForCausalLM


def _interpolate_pos_embed(pos_weight, src_grid, tgt_h, tgt_w):
    """Bilinear interpolation from src_grid x src_grid to tgt_h x tgt_w, raster order.

    pos_weight: (src_grid*src_grid, hidden) numpy array
    Returns: (tgt_h*tgt_w, hidden) numpy array
    """
    hidden = pos_weight.shape[1]
    pos_2d = pos_weight.reshape(src_grid, src_grid, hidden)

    h_coords = np.linspace(0, src_grid - 1, tgt_h)
    w_coords = np.linspace(0, src_grid - 1, tgt_w)

    # Build interpolation indices and weights for all target positions at once
    h_floor = np.floor(h_coords).astype(np.int64)
    w_floor = np.floor(w_coords).astype(np.int64)
    h_ceil = np.minimum(h_floor + 1, src_grid - 1)
    w_ceil = np.minimum(w_floor + 1, src_grid - 1)
    dh = (h_coords - h_floor).astype(np.float32)
    dw = (w_coords - w_floor).astype(np.float32)

    result = np.zeros((tgt_h, tgt_w, hidden), dtype=pos_weight.dtype)
    for i in range(tgt_h):
        for j in range(tgt_w):
            result[i, j] = (
                (1 - dh[i]) * (1 - dw[j]) * pos_2d[h_floor[i], w_floor[j]]
                + (1 - dh[i]) * dw[j] * pos_2d[h_floor[i], w_ceil[j]]
                + dh[i] * (1 - dw[j]) * pos_2d[h_ceil[i], w_floor[j]]
                + dh[i] * dw[j] * pos_2d[h_ceil[i], w_ceil[j]]
            )

    return result.reshape(tgt_h * tgt_w, hidden)


def huggingface(  # pylint: disable=too-many-locals
    model_config: Qwen35VConfig, quantization: Quantization
) -> ExternMapping:
    """Returns parameter mapping from MLC LLM parameters to HuggingFace parameters."""
    model = Qwen35VForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)
    mapping = ExternMapping()

    text_config = model_config.text_config

    # ========== Language model weights ==========
    for i in range(text_config.num_hidden_layers):
        layer_type = text_config.layer_types[i]

        if layer_type == "full_attention":
            # MLP: concat gate_proj + up_proj -> gate_up_proj
            mlc_mlp = f"language_model.model.layers.{i}.mlp"
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
            mlc_attn = f"language_model.model.layers.{i}.linear_attn"
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
            mlc_mlp = f"language_model.model.layers.{i}.mlp"
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

    # ========== Vision encoder weights ==========
    vision_config = model_config.vision_config

    # Conv3D -> Conv2D: sum over temporal dimension.
    # HF uses nn.Conv3d(3, hidden, (2, 16, 16)) for video support (2 consecutive frames).
    # For single images, HF duplicates the image into identical temporal frames, so
    # Conv3D(w) @ [img, img] == Conv2D(w.sum(dim=temporal)) @ img. We sum at conversion
    # time to use a standard Conv2D at runtime, avoiding the need for a Conv3D operator
    # (which would require a custom TIR kernel for Metal).
    # HF shape: (out_channels, in_channels, temporal=2, patch, patch)
    # MLC shape: (out_channels, in_channels, patch, patch)
    conv_mlc = "visual.patch_embed.proj.weight"
    conv_hf = "model.visual.patch_embed.proj.weight"
    if conv_mlc in named_parameters:
        mapping.add_mapping(
            conv_mlc,
            [conv_hf],
            functools.partial(
                lambda w, dtype: w.sum(axis=2).astype(dtype),
                dtype=named_parameters[conv_mlc].dtype,
            ),
        )

    # Position embedding: bilinear interpolation from 48x48 to grid_h x grid_w.
    # HF stores learned embeddings for a 48x48 grid (2304 positions), but our fixed
    # resolution (e.g. 448px) only needs a 28x28 grid (784 positions). We interpolate
    # in pure numpy at conversion time rather than at runtime, which avoids needing a
    # scipy dependency or a custom TVM interpolation op.
    pos_mlc = "visual.pos_embed"
    pos_hf = "model.visual.pos_embed.weight"
    if pos_mlc in named_parameters:
        src_grid = int(vision_config.num_position_embeddings ** 0.5)  # 48
        tgt_h = model_config.grid_h
        tgt_w = model_config.grid_w
        mapping.add_mapping(
            pos_mlc,
            [pos_hf],
            functools.partial(
                lambda w, dtype, sg, th, tw: _interpolate_pos_embed(
                    w, sg, th, tw
                ).astype(dtype),
                dtype=named_parameters[pos_mlc].dtype,
                sg=src_grid,
                th=tgt_h,
                tw=tgt_w,
            ),
        )

    # Vision blocks: rename HF linear_fc1/linear_fc2 -> MLC fc1/fc2
    for i in range(vision_config.depth):
        # MLP renaming
        for suffix in ["weight", "bias"]:
            for fc_hf, fc_mlc in [("linear_fc1", "fc1"), ("linear_fc2", "fc2")]:
                mlc_name = f"visual.blocks.{i}.mlp.{fc_mlc}.{suffix}"
                hf_name = f"model.visual.blocks.{i}.mlp.{fc_hf}.{suffix}"
                if mlc_name in named_parameters:
                    mapping.add_mapping(
                        mlc_name,
                        [hf_name],
                        functools.partial(
                            lambda x, dtype: x.astype(dtype),
                            dtype=named_parameters[mlc_name].dtype,
                        ),
                    )

    # Merger: rename HF linear_fc1/linear_fc2 -> MLC fc1/fc2
    for suffix in ["weight", "bias"]:
        for fc_hf, fc_mlc in [("linear_fc1", "fc1"), ("linear_fc2", "fc2")]:
            mlc_name = f"visual.merger.{fc_mlc}.{suffix}"
            hf_name = f"model.visual.merger.{fc_hf}.{suffix}"
            if mlc_name in named_parameters:
                mapping.add_mapping(
                    mlc_name,
                    [hf_name],
                    functools.partial(
                        lambda x, dtype: x.astype(dtype),
                        dtype=named_parameters[mlc_name].dtype,
                    ),
                )

    # ========== Remaining weights: 1:1 mapping with prefix translation ==========
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


def _mlc_to_hf(mlc_name: str) -> str:
    """Convert MLC parameter name to HuggingFace parameter name."""
    # Language model: language_model.model.X -> model.language_model.X
    if mlc_name.startswith("language_model.model."):
        return "model.language_model." + mlc_name[len("language_model.model."):]
    if mlc_name.startswith("language_model.lm_head."):
        return "model.language_model." + mlc_name[len("language_model."):]
    # Vision: visual.X -> model.visual.X
    if mlc_name.startswith("visual."):
        return "model." + mlc_name
    return mlc_name
