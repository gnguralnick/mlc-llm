"""Implementation for InternVL3.5 Vision-Language architecture.

InternVL3.5 = InternViT-300M (vision) + pixel unshuffle + 2-layer MLP + Qwen3 (LLM).
"""

import dataclasses
from typing import Any, Dict, Optional

from tvm import relax, s_tir, target, te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.relax.frontend.nn.op import wrap_nested
from tvm.relax.op import strided_slice
from tvm.script import tir as T

from mlc_llm import op as op_ext
from mlc_llm.model.vision import ImageProcessor
from mlc_llm.model.vision.intern_vit import InternVisionConfig, InternVisionModel
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase

from ..qwen3.qwen3_model import Qwen3Config, Qwen3LMHeadModel

logger = logging.getLogger(__name__)


def _create_split_and_pad_tiles_func(max_patches, image_size, dtype):
    """Split a tiled image into individual tiles and pad to a constant batch size.

    Metal codegen requires constant-size allocations, so downstream vision encoder
    operations need a fixed batch dimension. This function splits the composite image
    into tiles and pads with zeros to max_patches.

    Input:  (1, C, crop_h*image_size, crop_w*image_size)
    Output: (max_patches, C, image_size, image_size) — constant shape
    """

    @T.prim_func
    def split_and_pad_func(
        image: T.handle,
        output: T.handle,
        crop_h: T.int64(),
        crop_w: T.int64(),
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
        n, c, big_h, big_w = T.int64(), T.int64(), T.int64(), T.int64()
        image_buf = T.match_buffer(image, (n, c, big_h, big_w), dtype=dtype)
        out_buf = T.match_buffer(
            output, (T.int64(max_patches), c, image_size, image_size), dtype=dtype
        )

        for tile_idx in T.thread_binding(T.int64(max_patches), thread="blockIdx.x"):
            for c_idx in T.thread_binding(c, thread="blockIdx.y"):
                for h_idx, w_idx in T.grid(T.int64(image_size), T.int64(image_size)):
                    with T.sblock("split_pad"):
                        T.reads(image_buf[T.int64(0), c_idx, h_idx, w_idx])
                        T.writes(out_buf[tile_idx, c_idx, h_idx, w_idx])
                        if tile_idx < crop_h * crop_w:
                            out_buf[tile_idx, c_idx, h_idx, w_idx] = image_buf[
                                T.int64(0),
                                c_idx,
                                (tile_idx // crop_w) * T.int64(image_size) + h_idx,
                                (tile_idx % crop_w) * T.int64(image_size) + w_idx,
                            ]
                        else:
                            out_buf[tile_idx, c_idx, h_idx, w_idx] = T.float32(0)

    sch = s_tir.Schedule(split_and_pad_func)
    block = sch.get_sblock("split_pad")
    loop_x, loop_y = sch.get_loops(block)[-2:]
    xo, xi = sch.split(loop_x, factors=[32, None])
    yo, yi = sch.split(loop_y, factors=[32, None])
    sch.reorder(xo, yo, xi, yi)
    t = sch.fuse(xo, yo)
    ty, tx = sch.split(t, factors=[None, 32])
    sch.bind(ty, "threadIdx.y")
    sch.bind(tx, "threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


def _create_slice_flatten_func(max_tiles, tokens_per_tile, hidden_size, dtype):
    """Slice valid tiles from padded output and flatten to 2D.

    The vision encoder operates on a constant batch (max_tiles) due to Metal
    codegen constraints. This function removes padding tiles and produces
    the correct dynamic-length token sequence.

    Input:  (max_tiles, tokens_per_tile, hidden_size) — constant shape
    Output: (num_valid_tiles * tokens_per_tile, hidden_size) — dynamic first dim
    """

    @T.prim_func
    def slice_flatten_func(
        inp: T.handle,
        out: T.handle,
        num_valid_tiles: T.int64(),
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True})
        inp_buf = T.match_buffer(
            inp,
            (T.int64(max_tiles), T.int64(tokens_per_tile), T.int64(hidden_size)),
            dtype=dtype,
        )
        out_buf = T.match_buffer(
            out,
            (num_valid_tiles * T.int64(tokens_per_tile), T.int64(hidden_size)),
            dtype=dtype,
        )

        for token_idx, h_idx in T.grid(
            num_valid_tiles * T.int64(tokens_per_tile), T.int64(hidden_size)
        ):
            with T.sblock("copy"):
                T.reads(
                    inp_buf[
                        token_idx // T.int64(tokens_per_tile),
                        token_idx % T.int64(tokens_per_tile),
                        h_idx,
                    ]
                )
                T.writes(out_buf[token_idx, h_idx])
                out_buf[token_idx, h_idx] = inp_buf[
                    token_idx // T.int64(tokens_per_tile),
                    token_idx % T.int64(tokens_per_tile),
                    h_idx,
                ]

    sch = s_tir.Schedule(slice_flatten_func)
    block = sch.get_sblock("copy")
    loops = sch.get_loops(block)  # [token_idx, h_idx]
    sch.bind(loops[0], "blockIdx.x")
    ho, hi = sch.split(loops[1], factors=[None, 256])
    sch.bind(hi, "threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


INTERN_VIT_DEFAULT_CONFIG = {
    "hidden_size": 1024,
    "image_size": 448,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "num_channels": 3,
    "layer_norm_eps": 1e-6,
    "qkv_bias": True,
    "qk_normalization": False,
}


@dataclasses.dataclass
class InternVLConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the InternVL Chat model."""

    vision_config: InternVisionConfig = None
    llm_config: Qwen3Config = None
    downsample_ratio: float = 0.5
    force_image_size: int = 448
    ps_version: str = "v2"
    select_layer: int = -1
    max_dynamic_patch: int = 12
    use_thumbnail: bool = True
    vocab_size: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    context_window_size: int = -1
    prefill_chunk_size: int = -1
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # Parse vision_config
        vision_config_dict: Dict[str, Any]
        if isinstance(self.vision_config, InternVisionConfig):
            vision_config_dict = dataclasses.asdict(self.vision_config)
        elif self.vision_config is not None:
            vision_config_dict = dict(self.vision_config)
        else:
            vision_config_dict = dict(INTERN_VIT_DEFAULT_CONFIG)

        for k, v in vision_config_dict.pop("kwargs", {}).items():
            vision_config_dict[k] = v

        self.vision_config = InternVisionConfig.from_dict(vision_config_dict)

        # Parse llm_config (HF uses "llm_config", not "text_config")
        if self.llm_config is None:
            raise ValueError("InternVLConfig requires llm_config")

        llm_config_dict: Dict[str, Any]
        if isinstance(self.llm_config, Qwen3Config):
            llm_config_dict = dataclasses.asdict(self.llm_config)
        else:
            llm_config_dict = dict(self.llm_config)

        for k, v in llm_config_dict.pop("kwargs", {}).items():
            llm_config_dict[k] = v

        # Remove model_type if present (nested config may have it)
        llm_config_dict.pop("model_type", None)

        self.llm_config = Qwen3Config.from_dict(llm_config_dict)

        # Set vocab_size from llm_config if not explicitly provided
        if self.vocab_size == 0:
            self.vocab_size = self.llm_config.vocab_size

        # Propagate sizes from llm_config
        if self.context_window_size <= 0:
            self.context_window_size = self.llm_config.context_window_size
        if self.prefill_chunk_size <= 0:
            self.prefill_chunk_size = self.llm_config.prefill_chunk_size
        # Ensure prefill_chunk_size can fit the maximum image tokens in one chunk,
        # since the runtime cannot split ImageData across prefill steps.
        tokens_per_tile = (self.vision_config.image_size // self.vision_config.patch_size) ** 2 // 4
        include_thumbnail = 1 if self.max_dynamic_patch > 1 else 0
        max_image_tokens = (self.max_dynamic_patch + include_thumbnail) * tokens_per_tile
        if self.prefill_chunk_size < max_image_tokens:
            self.prefill_chunk_size = max_image_tokens



# pylint: disable=invalid-name,missing-docstring


class InternVLMLPProjector(nn.Module):
    """MLP projector: LayerNorm → Linear → GELU → Linear.

    HF Sequential indices: 0=LayerNorm, 1=Linear, 2=GELU, 3=Linear
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_features, out_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


class InternVLForCausalLM(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.config = config

        # Vision encoder
        self.vision_model = InternVisionModel(config.vision_config)

        # MLP projector: vit_hidden * 4 (pixel unshuffle) → llm_hidden
        vit_hidden = config.vision_config.hidden_size
        llm_hidden = config.llm_config.hidden_size
        mlp_in = int(vit_hidden * (1.0 / config.downsample_ratio) ** 2)  # 1024 * 4 = 4096
        self.mlp1 = InternVLMLPProjector(mlp_in, llm_hidden)

        # Language model (Qwen3)
        self.language_model = Qwen3LMHeadModel(config.llm_config)

        # Image processor
        self.image_processor = ImageProcessor()

        # Cache config values for inference methods
        self.num_hidden_layers = config.llm_config.num_hidden_layers
        self.num_attention_heads = config.llm_config.num_attention_heads
        self.num_key_value_heads = config.llm_config.num_key_value_heads
        self.head_dim = config.llm_config.head_dim
        self.hidden_size = config.llm_config.hidden_size
        self.vocab_size = config.vocab_size
        self.rope_theta = config.llm_config.rope_theta
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.dtype = "float32"
        self.image_dtype = (
            "uint32"
            if target.Target.current() and target.Target.current().kind.name == "webgpu"
            else "uint8"
        )

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def pixel_unshuffle(self, x: Tensor) -> Tensor:
        """Pixel unshuffle (ps_version="v2"): reduces spatial resolution by 2x, increases channels 4x.

        Replicates the HF pixel_shuffle(x, scale_factor=0.5) method exactly:
          view(n, w, h*0.5, c/0.5) → permute(0,2,1,3) → view(n, h*0.5, w*0.5, c/0.25) → permute(0,2,1,3)

        Input:  (batch, h*w, hidden) where h=w=image_size/patch_size (e.g. 32 for 448/14)
        Output: (batch, h*w/4, hidden*4)
        """
        vit_hidden = self.config.vision_config.hidden_size
        grid = self.config.vision_config.image_size // self.config.vision_config.patch_size
        half_grid = grid // 2
        b = x.shape[0]

        # (batch, h*w, hidden) → (batch, w, h, c) — spatial layout
        x = op.reshape(x, (b, grid, grid, vit_hidden))

        # Step 1: (n, w, h, c) → (n, w, h/2, c*2) — merge pairs along h into channels
        x = op.reshape(x, (b, grid, half_grid, vit_hidden * 2))

        # Step 2: permute(0, 2, 1, 3) → (n, h/2, w, c*2)
        x = op.permute_dims(x, (0, 2, 1, 3))

        # Step 3: (n, h/2, w, c*2) → (n, h/2, w/2, c*4) — merge pairs along w into channels
        x = op.reshape(x, (b, half_grid, half_grid, vit_hidden * 4))

        # Step 4 (v2): permute(0, 2, 1, 3) — swap h and w back
        x = op.permute_dims(x, (0, 2, 1, 3))

        # Flatten to sequence: (batch, h/2 * w/2, c*4)
        x = op.reshape(x, (b, half_grid * half_grid, vit_hidden * 4))
        return x

    def image_embed(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        pixel_values: Tensor,
        resized_height,
        resized_width,
        crop_height,
        crop_width,
    ) -> Tensor:
        image_size = self.config.force_image_size
        vit_hidden = self.config.vision_config.hidden_size
        num_patches = (image_size // self.config.vision_config.patch_size) ** 2  # 1024

        # Step 1: NHWC → NCHW
        pixel_values = op.permute_dims(pixel_values, (0, 3, 1, 2))

        # Step 2: Resize to target tile grid: (1, 3, crop_h*448, crop_w*448)
        pixel_values = self.image_processor.resize(
            pixel_values, {"height": resized_height, "width": resized_width}
        )

        # Step 3: Rescale uint8 → float32, normalize with ImageNet mean/std
        pixel_values = self.image_processor.rescale(pixel_values)
        pixel_values = self.image_processor.normalize_imagenet(pixel_values)

        # Step 4: Create thumbnail by resizing to (1, 3, 448, 448)
        # When max_dynamic_patch == 1, the thumbnail is identical to the single tile
        # (both are the image resized to 448×448), so skip it to halve compute.
        include_thumbnail = self.config.max_dynamic_patch > 1
        if include_thumbnail:
            thumbnail = self.image_processor.resize(
                pixel_values, {"height": image_size, "width": image_size}
            )
            thumbnail = op.wrap_nested(
                relax.BlockBuilder()
                .current()
                .match_cast(
                    thumbnail._expr,  # pylint: disable=protected-access
                    relax.TensorStructInfo(
                        [1, 3, image_size, image_size], thumbnail.dtype
                    ),
                ),
                "thumbnail",
            )

        # Step 5: Split into tiles AND pad to constant max batch.
        # Metal codegen requires constant-size allocations, so the vision encoder
        # must see a fixed batch dimension. We pad to max_dynamic_patch here;
        # the thumbnail (if included) is concatenated next.
        max_patches = self.config.max_dynamic_patch  # compile-time constant
        max_tiles = max_patches + (1 if include_thumbnail else 0)
        pixel_values = op.tensor_ir_op(
            _create_split_and_pad_tiles_func(max_patches, image_size, pixel_values.dtype),
            "split_and_pad_tiles",
            [pixel_values, crop_height, crop_width],
            [
                Tensor.placeholder(
                    (max_patches, 3, image_size, image_size),
                    pixel_values.dtype,
                )
            ],
        )

        # Step 6: Concatenate padded tiles + thumbnail → constant shape
        if include_thumbnail:
            all_tiles = op.concat([pixel_values, thumbnail], dim=0)
        else:
            all_tiles = pixel_values

        # Step 7: Cast to model dtype
        all_tiles = all_tiles.astype(self.dtype)

        # Step 8: Vision encoder → (max_tiles, num_patches+1, vit_hidden)
        vit_output = self.vision_model(all_tiles)

        # Step 9: Remove CLS token → (max_tiles, num_patches, vit_hidden)
        vit_output = wrap_nested(
            strided_slice(
                vit_output._expr,  # pylint: disable=protected-access
                axes=[1],
                begin=[1],
                end=[num_patches + 1],
            ),
            name="remove_cls",
        )

        # Step 10: Pixel unshuffle → (max_tiles, num_patches/4, vit_hidden*4)
        vit_output = self.pixel_unshuffle(vit_output)

        # Step 11: MLP projector → (max_tiles, num_patches/4, llm_hidden)
        projected = self.mlp1(vit_output)

        # Step 12: Slice valid tiles and flatten to 2D.
        # The vision encoder ran on a constant batch (max_tiles) due to Metal
        # codegen constraints. Remove padding tiles and flatten to (N, hidden).
        # C++ runtime requires ndim == 2 and copies shape[0] tokens.
        tokens_per_tile = num_patches // 4  # 256 after pixel unshuffle
        num_valid_tiles = crop_height * crop_width + (1 if include_thumbnail else 0)
        projected = op.tensor_ir_op(
            _create_slice_flatten_func(
                max_tiles, tokens_per_tile, self.hidden_size, projected.dtype
            ),
            "slice_flatten_tiles",
            [projected, num_valid_tiles],
            [
                Tensor.placeholder(
                    (num_valid_tiles * tokens_per_tile, self.hidden_size),
                    projected.dtype,
                )
            ],
        )

        return projected

    def get_logits(self, hidden_states: Tensor):
        if self.language_model.tie_word_embeddings:
            logits = self.language_model.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.language_model.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def batch_forward(
        self,
        input_embeds: Tensor,
        paged_kv_cache: PagedKVCache,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()

        hidden_states = self.language_model.model(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self.get_logits(hidden_states)
        return logits

    def embed(self, input_ids: Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = op.ccl_broadcast_from_worker0(input_ids)
        return self.language_model.model.embed_tokens(input_ids)

    def prefill(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        def _index(x: te.Tensor):  # x[:-1,:]
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states = self.language_model.model(input_embed, paged_kv_cache)
        hidden_states = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def decode(self, input_embed: Tensor, paged_kv_cache: PagedKVCache):
        op_ext.configure()

        hidden_states = self.language_model.model(input_embed, paged_kv_cache)
        logits = self.get_logits(hidden_states)
        return logits, paged_kv_cache

    def batch_prefill(
        self,
        input_embeds: Tensor,
        logit_positions: Tensor,
        paged_kv_cache: PagedKVCache,
    ):
        if self.tensor_parallel_shards > 1:
            logit_positions = op.ccl_broadcast_from_worker0(logit_positions)
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_verify(self, input_embeds: Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def create_paged_kv_cache(  # pylint: disable=too-many-arguments
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> PagedKVCache:
        return PagedKVCache.create_generic(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads // self.tensor_parallel_shards,
            num_key_value_heads=self.num_key_value_heads // self.tensor_parallel_shards,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "image_embed": {
                "pixel_values": nn.spec.Tensor(
                    [1, "image_height", "image_width", 3], self.image_dtype
                ),
                "resized_height": nn.spec.Int(),
                "resized_width": nn.spec.Int(),
                "crop_height": nn.spec.Int(),
                "crop_width": nn.spec.Int(),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
