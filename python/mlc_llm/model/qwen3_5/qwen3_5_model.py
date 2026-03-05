"""Implementation for Qwen3.5 architecture (hybrid linear + full attention)."""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from tvm import te, tir
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Object, Tensor, op
from tvm.relax.frontend.nn.llm.kv_cache import RopeMode
from tvm.script import tir as T

from mlc_llm import op as op_ext
from mlc_llm.nn.hybrid_state import HybridState
from mlc_llm.nn.kv_cache import PagedKVCache
from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.support import logging
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name,missing-docstring,too-many-locals,too-many-arguments
# pylint: disable=too-many-instance-attributes,too-many-statements


@dataclasses.dataclass
class Qwen35Config(ConfigBase):
    """Configuration of the Qwen3.5 model."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    layer_types: List[str]
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_value_head_dim: int = 128
    tie_word_embeddings: bool = True
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    dtype: str = "float32"
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, source: Dict[str, Any]) -> "Qwen35Config":
        """Flatten text_config before parsing."""
        source = dict(source)  # shallow copy
        if "text_config" in source:
            tc = source.pop("text_config")
            if isinstance(tc, dict):
                for k, v in tc.items():
                    if k not in source:
                        source[k] = v
        return super().from_dict(source)

    def __post_init__(self):
        # rope params may be nested
        rope_params = self.kwargs.pop("rope_parameters", {})
        self.rope_theta = rope_params.get("rope_theta", 10_000_000)
        self.partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.25)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    logger.info(
                        "%s not found in config.json. Falling back to %s (%d)",
                        bold("context_window_size"),
                        bold(name),
                        self.context_window_size,
                    )
                    break
            else:
                raise ValueError(
                    "Unable to determine the maximum sequence length, because none of "
                    "`context_window_size`, `max_position_embeddings` or "
                    "`max_sequence_length` is provided in `config.json`."
                )
        if self.prefill_chunk_size == 0:
            self.prefill_chunk_size = min(self.context_window_size, 2048)
        elif self.prefill_chunk_size > self.context_window_size:
            self.prefill_chunk_size = min(self.context_window_size, 2048)

        # Computed
        self.num_full_attn_layers = sum(
            1 for t in self.layer_types if t == "full_attention"
        )
        self.num_linear_attn_layers = sum(
            1 for t in self.layer_types if t == "linear_attention"
        )
        # linear attention dims
        self.linear_conv_dim = (
            self.linear_key_head_dim * self.linear_num_key_heads * 2
            + self.linear_value_head_dim * self.linear_num_value_heads
        )


# --------------------------------------------------------------------------
# Utility ops
# --------------------------------------------------------------------------


def l2_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """L2 normalize along the last dimension."""
    norm = op.sqrt(op.sum(x * x, axis=-1, keepdims=True) + eps)
    return x / norm


def token_shift(state: Tensor, x: Tensor):
    """Shift tokens by 1 position, filling position 0 from state."""

    def _te_token_shift(state: te.Tensor, x: te.Tensor):
        return te.compute(
            x.shape,
            lambda b, i, j: tir.if_then_else(i == 0, state[b, j], x[b, i - 1, j]),
        )

    return op.tensor_expr_op(_te_token_shift, "token_shift", [state, x])


def _stable_softplus(x: Tensor, threshold: float = 20.0) -> Tensor:
    """Numerically stable softplus: log(1 + exp(x)).

    For x > threshold, returns x directly (avoids exp overflow).
    Matches PyTorch F.softplus behavior.
    """

    def _te_stable_softplus(x: te.Tensor):
        return te.compute(
            x.shape,
            lambda *indices: tir.if_then_else(
                x[indices] > tir.const(threshold, "float32"),
                x[indices],
                tir.log(tir.exp(x[indices]) + tir.const(1.0, "float32")),
            ),
        )

    return op.tensor_expr_op(_te_stable_softplus, "stable_softplus", [x])


def last_token(x: Tensor):
    """Extract the last token along the sequence dimension."""
    batch, seq_len, hidden_size = x.shape

    def _te_last_token(x: te.Tensor):
        return te.compute(
            (batch, 1, hidden_size), lambda b, _, j: x[b, x.shape[1] - 1, j]
        )

    return x if seq_len == 1 else op.tensor_expr_op(_te_last_token, "last_token", [x])


# --------------------------------------------------------------------------
# DeltaNet TIR kernel
# --------------------------------------------------------------------------


def create_deltanet_func(
    num_heads: int,
    key_head_dim: int,
    value_head_dim: int,
):
    """Create a TIR function for the recurrent gated delta rule.

    State shape: (batch, num_heads, key_head_dim, value_head_dim)
    Threading: blockIdx.y=batch, blockIdx.x=num_heads, threadIdx.x=value_head_dim
    Sequential over time steps.
    """

    @T.prim_func
    def deltanet_func(
        var_q: T.handle,
        var_k: T.handle,
        var_v: T.handle,
        var_g: T.handle,
        var_beta: T.handle,
        var_state: T.handle,
        var_out: T.handle,
        var_out_state: T.handle,
    ):
        T.func_attr({"op_pattern": 8, "tir.noalias": True, "tir.is_scheduled": 1})
        batch_size, seq_len = T.int64(), T.int64()
        q_buf = T.match_buffer(
            var_q, (batch_size, seq_len, num_heads, key_head_dim), dtype="float32"
        )
        k_buf = T.match_buffer(
            var_k, (batch_size, seq_len, num_heads, key_head_dim), dtype="float32"
        )
        v_buf = T.match_buffer(
            var_v, (batch_size, seq_len, num_heads, value_head_dim), dtype="float32"
        )
        g_buf = T.match_buffer(
            var_g, (batch_size, seq_len, num_heads), dtype="float32"
        )
        beta_buf = T.match_buffer(
            var_beta, (batch_size, seq_len, num_heads), dtype="float32"
        )
        state_buf = T.match_buffer(
            var_state,
            (batch_size, num_heads, key_head_dim, value_head_dim),
            dtype="float32",
        )
        out_buf = T.match_buffer(
            var_out, (batch_size, seq_len, num_heads, value_head_dim), dtype="float32"
        )
        out_state_buf = T.match_buffer(
            var_out_state,
            (batch_size, num_heads, key_head_dim, value_head_dim),
            dtype="float32",
        )

        for b in T.thread_binding(batch_size, thread="blockIdx.y"):
            for h in T.thread_binding(num_heads, thread="blockIdx.x"):
                for j in T.thread_binding(value_head_dim, thread="threadIdx.x"):
                    # Initialize state
                    for i in range(key_head_dim):
                        with T.sblock("init_state"):
                            vb, vh, vi, vj = T.axis.remap("SSSS", [b, h, i, j])
                            out_state_buf[vb, vh, vi, vj] = state_buf[vb, vh, vi, vj]

                    for t in range(seq_len):
                        with T.sblock("compute"):
                            vb = T.axis.spatial(batch_size, b)
                            vt = T.axis.opaque(seq_len, t)
                            vh = T.axis.spatial(num_heads, h)
                            vj = T.axis.spatial(value_head_dim, j)

                            # NOTE: All intermediate accumulations use out_buf as
                            # scratch space (not local vars) because Metal codegen
                            # silently drops local-variable writes inside T.sblock.

                            # 1) Decay the state column j by exp(g_t) for all rows
                            for i in range(key_head_dim):
                                out_state_buf[vb, vh, i, vj] = (
                                    out_state_buf[vb, vh, i, vj]
                                    * T.exp(g_buf[vb, vt, vh])
                                )

                            # 2) Read: kv_mem_j = sum_i S[i,j] * k_t[i]
                            # Use out_buf[vb,vt,vh,vj] as scratch
                            out_buf[vb, vt, vh, vj] = T.float32(0)
                            for i in range(key_head_dim):
                                out_buf[vb, vt, vh, vj] += (
                                    out_state_buf[vb, vh, i, vj]
                                    * k_buf[vb, vt, vh, i]
                                )

                            # 3) Delta = (v_t[j] - kv_mem_j) * beta_t
                            # Overwrite scratch with delta_j
                            out_buf[vb, vt, vh, vj] = (
                                v_buf[vb, vt, vh, vj] - out_buf[vb, vt, vh, vj]
                            ) * beta_buf[vb, vt, vh]

                            # 4) Rank-1 update: S[i,j] += k_t[i] * delta_j
                            for i in range(key_head_dim):
                                out_state_buf[vb, vh, i, vj] += (
                                    k_buf[vb, vt, vh, i]
                                    * out_buf[vb, vt, vh, vj]
                                )

                            # 5) Output: out[j] = sum_i S[i,j] * q_t[i]
                            out_buf[vb, vt, vh, vj] = T.float32(0)
                            for i in range(key_head_dim):
                                out_buf[vb, vt, vh, vj] += (
                                    out_state_buf[vb, vh, i, vj]
                                    * q_buf[vb, vt, vh, i]
                                )

    return deltanet_func


# --------------------------------------------------------------------------
# Gated RMSNorm: (1+weight) * rms_norm(x) * silu(gate)
# --------------------------------------------------------------------------


class GatedRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = nn.Parameter((hidden_size,))
        self.eps = eps

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        # HF Qwen3_5RMSNormGated initializes weight to ones (not zeros like Qwen3_5RMSNorm),
        # and applies weight * rms_norm(x) directly (no +1 offset).
        normed = op.rms_norm(x, self.weight, axes=[-1], epsilon=self.eps)
        return normed * op.silu(gate)


# --------------------------------------------------------------------------
# GatedDeltaNet (linear attention layer)
# --------------------------------------------------------------------------

DELTANET_CONV_STATE_ID = 0
DELTANET_RECURRENT_STATE_ID = 1


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35Config, rnn_layer_id: int):
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.rnn_layer_id = rnn_layer_id
        self.dtype = "float32"

        self.in_proj_qkv = nn.Linear(
            config.hidden_size, self.conv_dim, bias=False
        )
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)

        self.conv1d = nn.Conv1D(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )

        self.A_log = nn.Parameter((self.num_v_heads,))
        self.dt_bias = nn.Parameter((self.num_v_heads,))

        self.g_norm = GatedRMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

    def forward(
        self, hidden_states: Tensor, state: HybridState
    ) -> Tuple[Tensor, HybridState]:
        b, seq_len, _ = hidden_states.shape

        # Projections
        mixed_qkv = self.in_proj_qkv(hidden_states)  # (B, T, conv_dim)
        z = self.in_proj_z(hidden_states)  # (B, T, value_dim)
        z = op.reshape(z, (b, seq_len, self.num_v_heads, self.head_v_dim))

        b_gate = op.sigmoid(self.in_proj_b(hidden_states))  # (B, T, num_v_heads)
        a_val = self.in_proj_a(hidden_states)  # (B, T, num_v_heads)

        # Causal conv1d with state management
        conv_state = state.rnn_get(
            self.rnn_layer_id,
            DELTANET_CONV_STATE_ID,
            (b, self.conv_dim, self.conv_kernel_size - 1),
            self.dtype,
        )

        # Transpose for conv: (B, T, C) -> (B, C, T)
        mixed_qkv_t = op.permute_dims(mixed_qkv, [0, 2, 1])

        # Concatenate conv state with input: (B, C, kernel-1+T)
        mixed_conv = op.concat([conv_state, mixed_qkv_t], dim=2)

        # Save new conv state (last kernel_size-1 timesteps)
        new_conv_state = _slice_last(mixed_conv, self.conv_kernel_size - 1)
        state = state.rnn_set(
            self.rnn_layer_id, DELTANET_CONV_STATE_ID, new_conv_state
        )

        # Apply depthwise conv1d + silu activation
        conv_out = op.conv1d(
            mixed_conv,
            self.conv1d.weight,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.conv_dim,
        )
        conv_out = op.silu(conv_out)  # (B, C, T)

        # Transpose back: (B, C, T) -> (B, T, C)
        conv_out = op.permute_dims(conv_out, [0, 2, 1])

        # Split into Q, K, V
        q, k, v = op.split(
            conv_out, [self.key_dim, self.key_dim * 2], axis=-1
        )
        q = op.reshape(q, (b, seq_len, self.num_k_heads, self.head_k_dim))
        k = op.reshape(k, (b, seq_len, self.num_k_heads, self.head_k_dim))
        v = op.reshape(v, (b, seq_len, self.num_v_heads, self.head_v_dim))

        # L2 normalize Q and K
        q = l2_norm(q)
        k = l2_norm(k)

        # GQA expand Q, K from num_k_heads to num_v_heads
        if self.num_v_heads != self.num_k_heads:
            repeat_factor = self.num_v_heads // self.num_k_heads
            q = op.repeat(q, repeat_factor, axis=2)
            k = op.repeat(k, repeat_factor, axis=2)

        # Compute decay g = -exp(A_log) * softplus(a + dt_bias)
        # All in float32 — cast from model dtype since params are stored in model dtype
        a_float = a_val.astype("float32")
        dt_bias_f32 = self.dt_bias.astype("float32")
        a_plus_bias = a_float + op.reshape(dt_bias_f32, (1, 1, self.num_v_heads))
        softplus_val = _stable_softplus(a_plus_bias)
        neg_A = op.negative(op.exp(self.A_log.astype("float32")))
        g = op.reshape(neg_A, (1, 1, self.num_v_heads)) * softplus_val
        # g shape: (B, T, num_v_heads)

        # Get recurrent state
        recurrent_state = state.rnn_get(
            self.rnn_layer_id,
            DELTANET_RECURRENT_STATE_ID,
            (b, self.num_v_heads, self.head_k_dim, self.head_v_dim),
            "float32",
        )

        # DeltaNet TIR kernel
        # Scale Q by 1/sqrt(key_head_dim) as in HF reference
        scale = 1.0 / (self.head_k_dim**0.5)
        q_f32 = (q * scale).astype("float32")
        k_f32 = k.astype("float32")
        v_f32 = v.astype("float32")

        out, new_recurrent_state = op.tensor_ir_op(
            create_deltanet_func(
                num_heads=self.num_v_heads,
                key_head_dim=self.head_k_dim,
                value_head_dim=self.head_v_dim,
            ),
            "deltanet",
            [q_f32, k_f32, v_f32, g, b_gate.astype("float32"), recurrent_state],
            [
                Tensor.placeholder(
                    [b, seq_len, self.num_v_heads, self.head_v_dim], "float32"
                ),
                Tensor.placeholder(
                    [b, self.num_v_heads, self.head_k_dim, self.head_v_dim],
                    "float32",
                ),
            ],
        )

        state = state.rnn_set(
            self.rnn_layer_id, DELTANET_RECURRENT_STATE_ID, new_recurrent_state
        )

        # Cast float32 kernel output back to model dtype
        out = out.astype(self.dtype)

        # Gated RMSNorm: (1+weight) * rms_norm(out) * silu(z)
        # Reshape to (B*T*num_v_heads, head_v_dim) for norm
        out_flat = op.reshape(out, (b * seq_len * self.num_v_heads, self.head_v_dim))
        z_flat = op.reshape(z, (b * seq_len * self.num_v_heads, self.head_v_dim))
        normed = self.g_norm(out_flat, z_flat)
        normed = op.reshape(normed, (b, seq_len, self.value_dim))

        output = self.out_proj(normed)
        return output, state

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype
        # NOTE: Do NOT override A_log/dt_bias to float32 here.
        # The weight converter stores all params as bfloat16 (f32-to-bf16 format).
        # If these params are declared float32 at compile time but stored as bf16,
        # the runtime misinterprets the bf16 bytes as fp16, corrupting the values.
        # Instead, keep them in model dtype and cast to float32 at computation time.


def _slice_last(x: Tensor, n: int) -> Tensor:
    """Slice the last n elements along dim=2."""
    b, c, total = x.shape

    def _te_slice(x: te.Tensor):
        return te.compute(
            (b, c, n),
            lambda bi, ci, ni: x[bi, ci, x.shape[2] - n + ni],
        )

    return op.tensor_expr_op(_te_slice, "slice_last", [x])


# --------------------------------------------------------------------------
# GatedAttention (full attention layer)
# --------------------------------------------------------------------------


class Qwen35GatedAttention(nn.Module):
    def __init__(self, config: Qwen35Config, kv_layer_id: int):
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.kv_layer_id = kv_layer_id
        self.rotary_dim = config.rotary_dim

        # q_proj outputs query + gate concatenated: (hidden -> num_heads * head_dim * 2)
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim * 2,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        # RMSNorm per head with (1+weight) formulation
        self.q_norm = Qwen35RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self, hidden_states: Tensor, state: HybridState
    ) -> Tensor:
        d = self.head_dim
        h_q = self.num_attention_heads
        h_kv = self.num_key_value_heads
        b, s, _ = hidden_states.shape

        # Project Q (includes gate), K, V
        q_gate = self.q_proj(hidden_states)  # (B, S, h_q * d * 2)
        q_gate = op.reshape(q_gate, (b, s, h_q, d * 2))
        q, gate = op.split(q_gate, 2, axis=-1)  # each (B, S, h_q, d)

        k = op.reshape(self.k_proj(hidden_states), (b, s, h_kv, d))
        v = op.reshape(self.v_proj(hidden_states), (b, s, h_kv, d))

        # Apply QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Fuse QKV for PagedKVCache attention (partial RoPE handled by rotary_dim param)
        qkv = op.concat([q, k, v], dim=2)  # (B, S, h_q + 2*h_kv, d)

        # TODO(M-RoPE): PagedKVCache currently applies 1D RoPE to all tokens. For full
        # M-RoPE support, image tokens need 3D position IDs (t, h, w) where each dimension
        # gets its own slice of head_dim for rotary encoding. This requires:
        # 1. Passing per-token 3D position IDs from image_embed through the prefill call
        # 2. Modifying PagedKVCache (or using a custom RoPE application before caching) to
        #    apply separate rotary embeddings per dimension instead of sequential 1D positions
        # 3. During decode, all three dimensions increment equally so no special handling needed
        # Only the 8 full-attention layers are affected; DeltaNet layers have no position encoding.
        attn_output = state.attention_with_fused_qkv(
            self.kv_layer_id,
            qkv,
            self.num_attention_heads,
            sm_scale=self.head_dim**-0.5,
        )
        attn_output = op.reshape(attn_output, (b, s, h_q * d))

        # Apply output gate: output * sigmoid(gate)
        gate_flat = op.reshape(gate, (b, s, h_q * d))
        attn_output = attn_output * op.sigmoid(gate_flat)

        return self.o_proj(attn_output)


# --------------------------------------------------------------------------
# MLP
# --------------------------------------------------------------------------


class Qwen35MLP(nn.Module):
    def __init__(self, config: Qwen35Config):
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(op.silu(x1) * x2)


# --------------------------------------------------------------------------
# RMSNorm with (1+weight) formulation
# --------------------------------------------------------------------------


class Qwen35RMSNorm(nn.Module):
    """RMSNorm with (1+weight) scaling, matching HF Qwen3_5RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = nn.Parameter((hidden_size,))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return op.rms_norm(x, self.weight + 1, axes=[-1], epsilon=self.eps)


# --------------------------------------------------------------------------
# Decoder layer
# --------------------------------------------------------------------------


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config: Qwen35Config, layer_idx: int, rnn_layer_id: int, kv_layer_id: int):
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(config, rnn_layer_id)
        else:
            self.self_attn = Qwen35GatedAttention(config, kv_layer_id)
        self.mlp = Qwen35MLP(config)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self, hidden_states: Tensor, state: HybridState
    ) -> Tuple[Tensor, HybridState]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states, state = self.linear_attn(hidden_states, state)
        else:
            hidden_states = self.self_attn(hidden_states, state)

        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states, state


# --------------------------------------------------------------------------
# Embedding (shared with lm_head when tie_word_embeddings=True)
# --------------------------------------------------------------------------


class Qwen35Embedding(nn.Embedding):
    def lm_head_forward(self, x: Tensor):
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight, out_dtype="float32")


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------


class Qwen35Model(nn.Module):
    def __init__(self, config: Qwen35Config):
        self.embed_tokens = Qwen35Embedding(config.vocab_size, config.hidden_size)
        rnn_id = 0
        kv_id = 0
        layers = []
        for i in range(config.num_hidden_layers):
            if config.layer_types[i] == "linear_attention":
                layers.append(Qwen35DecoderLayer(config, i, rnn_id, -1))
                rnn_id += 1
            else:
                layers.append(Qwen35DecoderLayer(config, i, -1, kv_id))
                kv_id += 1
        self.layers = nn.ModuleList(layers)
        self.norm = Qwen35RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, inputs: Tensor, state: HybridState):
        hidden_states = inputs
        # TODO(DeepStack): HF's Qwen3.5 supports injecting vision features at intermediate
        # layers specified by vision_config.deepstack_visual_indexes. Currently disabled in
        # released models (deepstack_visual_indexes=[]):
        #   https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json
        #   https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json
        # If enabled, this loop would need to:
        # 1. Accept vision_embeds and image_token_mask as additional arguments
        # 2. At each layer index in deepstack_visual_indexes, replace the hidden states at
        #    image token positions with a projection of the vision encoder output for that layer
        # 3. Each DeepStack injection layer would have its own projection (Linear → GELU → Linear)
        for layer in self.layers:
            hidden_states, state = layer(hidden_states, state)
        hidden_states = self.norm(hidden_states)
        return hidden_states, state


class Qwen35LMHeadModel(nn.Module):
    def __init__(self, config: Qwen35Config):
        self.model = Qwen35Model(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.rope_theta = config.rope_theta
        self.rotary_dim = config.rotary_dim
        self.num_full_attn_layers = config.num_full_attn_layers
        self.num_linear_attn_layers = config.num_linear_attn_layers
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.linear_conv_dim = config.linear_conv_dim
        self.linear_conv_kernel_dim = config.linear_conv_kernel_dim

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def _get_logits(self, hidden_states: Tensor) -> Tensor:
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.lm_head_forward(hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits

    def embed(self, input_ids: Tensor):
        return self.model.embed_tokens(input_ids)

    def prefill(self, input_embed: Tensor, state: HybridState):
        op_ext.configure()

        def _index(x: te.Tensor):
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        hidden_states, state = self.model(input_embed, state)
        hidden_states = op.tensor_expr_op(
            _index, name_hint="index", args=[hidden_states]
        )
        logits = self._get_logits(hidden_states)
        return logits, state

    def decode(self, input_embed: Tensor, state: HybridState):
        op_ext.configure()
        hidden_states, state = self.model(input_embed, state)
        logits = self._get_logits(hidden_states)
        return logits, state

    def batch_forward(
        self,
        input_embeds: Tensor,
        state: HybridState,
        logit_positions: Optional[Tensor] = None,
    ):
        op_ext.configure()
        hidden_states, state = self.model(input_embeds, state)
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
        logits = self._get_logits(hidden_states)
        return logits, state

    def batch_prefill(
        self, input_embeds: Tensor, logit_positions: Tensor, state: HybridState
    ):
        logits, state = self.batch_forward(input_embeds, state, logit_positions)
        return logits, state

    def batch_decode(self, input_embeds: Tensor, state: HybridState):
        logits, state = self.batch_forward(input_embeds, state)
        return logits, state

    def batch_verify(self, input_embeds: Tensor, state: HybridState):
        logits, state = self.batch_forward(input_embeds, state)
        return logits, state

    def create_paged_kv_cache(
        self,
        max_batch_size: tir.Var,
        max_total_seq_len: tir.Var,
        prefill_chunk_size: tir.Var,
        page_size: tir.Var,
        support_sliding_window: tir.Var,
    ) -> Object:
        """Create the PagedKVCache for the full attention layers."""
        return PagedKVCache.create_generic(
            attn_kind="mha",
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.num_full_attn_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            qk_head_dim=self.head_dim,
            v_head_dim=self.head_dim,
            rope_mode=RopeMode.NORMAL,
            rope_scale=1,
            rope_theta=self.rope_theta,
            dtype=self.dtype,
            rotary_dim=self.rotary_dim,
        )

    def create_rnn_state(
        self,
        max_batch_size: tir.Var,
        max_history: tir.Var,
    ) -> Object:
        """Create the RNNState for the linear attention layers."""
        init_values = [
            op.zeros(
                (self.linear_conv_dim, self.linear_conv_kernel_dim - 1),
                dtype=self.dtype,
            ),
            op.zeros(
                (self.num_v_heads, self.head_k_dim, self.head_v_dim),
                dtype="float32",
            ),
        ]
        return RNNState.create(
            max_batch_size=max_batch_size,
            num_hidden_layers=self.num_linear_attn_layers,
            max_history=max_history,
            init_values=init_values,
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
            "prefill": {
                "input_embed": nn.spec.Tensor(
                    [1, "seq_len", self.hidden_size], self.dtype
                ),
                "state": nn.spec.Object(object_type=HybridState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor(
                    [1, 1, self.hidden_size], self.dtype
                ),
                "state": nn.spec.Object(object_type=HybridState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": nn.spec.Tensor(
                    [1, "seq_len", self.hidden_size], self.dtype
                ),
                "logit_positions": nn.spec.Tensor(["batch_size"], "int32"),
                "state": nn.spec.Object(object_type=HybridState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": nn.spec.Tensor(
                    ["batch_size", 1, self.hidden_size], self.dtype
                ),
                "state": nn.spec.Object(object_type=HybridState),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": nn.spec.Tensor(
                    [1, "seq_len", self.hidden_size], self.dtype
                ),
                "state": nn.spec.Object(object_type=HybridState),
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
            "create_rnn_state": {
                "max_batch_size": int,
                "max_history": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
