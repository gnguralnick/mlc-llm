"""
Implements the InternViT Vision Encoder (InternViT-300M).

Key differences from SigLIP (siglip_vision.py):
- CLS token: learnable CLS token prepended to patch embeddings
- Fused QKV: single qkv Linear projection, split into Q, K, V
- Layer scaling: ls1, ls2 per-layer scalars that scale attention/MLP outputs
- Attention output projection named 'proj' (not 'out_proj')
- Position embeddings size = num_patches + 1 (for CLS)
"""

import dataclasses
import logging
from typing import Any, Dict

from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Module, Tensor
from tvm.relax.frontend.nn.op import (
    add,
    broadcast_to,
    concat,
    permute_dims,
    reshape,
    split,
    wrap_nested,
)
from tvm.relax.op import arange

from mlc_llm import op as op_ext
from mlc_llm.support.config import ConfigBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class InternVisionConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Config for the InternViT vision encoder."""

    hidden_size: int = 1024
    image_size: int = 448
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    qk_normalization: bool = False
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


# pylint: disable=invalid-name,missing-docstring


class InternVisionEmbeddings(Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter((config.hidden_size,))

        self.patch_embedding = nn.modules.Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1  # +1 for CLS
        self.position_embedding = nn.Embedding(num=self.num_positions, dim=self.embed_dim)

    def forward(self, pixel_values: Tensor) -> Tensor:
        batch_size = pixel_values.shape[0]
        # pixel_values: (batch, channels, height, width)
        patch_embeds = self.patch_embedding(pixel_values)  # (batch, embed_dim, grid, grid)
        patch_embeds = reshape(patch_embeds, shape=(batch_size, self.embed_dim, -1))
        patch_embeds = permute_dims(patch_embeds, axes=(0, 2, 1))  # (batch, num_patches, embed_dim)

        # Prepend CLS token
        cls_token = reshape(self.class_embedding, shape=(1, 1, self.embed_dim))
        cls_tokens = broadcast_to(cls_token, shape=(batch_size, 1, self.embed_dim))
        embeddings = concat([cls_tokens, patch_embeds], dim=1)  # (batch, num_patches+1, embed_dim)

        # Add position embeddings
        posi_ids = reshape(
            wrap_nested(arange(0, self.num_positions, dtype="int32"), name="arange"),
            shape=(1, -1),
        )
        batch_position_embedding = broadcast_to(
            self.position_embedding(posi_ids),
            shape=(batch_size, self.num_positions, self.embed_dim),
        )
        embeddings = add(embeddings, batch_position_embedding)
        return embeddings


class InternVisionMLP(Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionAttention(Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.qkv = nn.Linear(
            self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias
        )
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        d, h = self.head_dim, self.num_heads
        b, s, _ = hidden_states.shape

        qkv = self.qkv(hidden_states)  # (b, s, 3*embed_dim)
        qkv = reshape(qkv, (b, s, 3 * h, d))
        # Split into Q, K, V along head dimension
        q, k, v = split(qkv, indices_or_sections=[h, 2 * h], axis=2)

        attn_output = op_ext.attention(q, k, v, None)
        attn_output = self.proj(attn_output)
        return attn_output


class InternVisionEncoderLayer(Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.attn = InternVisionAttention(config)
        self.norm1 = nn.LayerNorm(normalized_shape=self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = InternVisionMLP(config)
        self.norm2 = nn.LayerNorm(normalized_shape=self.embed_dim, eps=config.layer_norm_eps)
        self.ls1 = nn.Parameter((self.embed_dim,))
        self.ls2 = nn.Parameter((self.embed_dim,))

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states=hidden_states)
        hidden_states = hidden_states * self.ls1
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states * self.ls2
        hidden_states = residual + hidden_states
        return hidden_states


class InternVisionEncoder(Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [InternVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class InternVisionTransformer(Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)
        # InternViT: no post_layernorm; output is raw last hidden state

    def forward(self, pixel_values: Tensor) -> Tensor:
        hidden_states = self.embeddings(pixel_values)
        return self.encoder(inputs_embeds=hidden_states)


class InternVisionModel(Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.vision_model = InternVisionTransformer(config)

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.vision_model(pixel_values)
