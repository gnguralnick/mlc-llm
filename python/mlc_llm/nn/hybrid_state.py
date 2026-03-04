"""HybridState: wraps PagedKVCache + RNNState for hybrid linear/full attention models.

At compilation time, the model exports separate `create_paged_kv_cache` and
`create_rnn_state` functions. At runtime, the C++ serving layer calls both and
wraps them with `vm.builtin.hybrid_state_create`.

This class provides the Python-side methods that emit the correct packed function
calls for attention, RNN get/set, and query positions.
"""

from typing import Sequence

from tvm import relax as rx
from tvm import tir
from tvm.relax.frontend.nn import Object, Tensor


class HybridState(Object):
    """State object wrapping PagedKVCache (for full attention layers)
    and RNNState (for linear attention layers)."""

    def attention_with_fused_qkv(
        self,
        layer_id: int,
        qkv: Tensor,
        num_qo_heads: int,
        sm_scale: float,
    ) -> Tensor:
        """Compute attention via the inner PagedKVCache.

        Works because HybridState IS-A AttentionKVCache in C++, so
        vm.builtin.attention_kv_cache_attention_with_fused_qkv dispatches
        correctly via vtable.
        """
        # pylint: disable=protected-access
        b, s, _, d = qkv._expr.struct_info.shape
        qkv = qkv.reshape(b * s, qkv.shape[2], d)
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.call_dps_packed(
                    "vm.builtin.attention_kv_cache_attention_with_fused_qkv",
                    [self._expr, layer_id, sm_scale, qkv._expr],
                    out_sinfo=rx.TensorStructInfo((b * s, num_qo_heads, d), qkv.dtype),
                )
            )
        ).reshape(b, s, num_qo_heads, d)
        # pylint: enable=protected-access

    def get_query_positions(self, total_length: tir.PrimExpr) -> Tensor:
        """Get in-sequence query positions from the inner PagedKVCache."""
        return Tensor(
            _expr=rx.BlockBuilder.current().emit(
                rx.call_pure_packed(
                    "vm.builtin.attention_kv_cache_get_query_positions",
                    self._expr,
                    sinfo_args=rx.TensorStructInfo((total_length,), "int32"),
                )
            )
        )

    def rnn_get(
        self,
        layer_id: int,
        state_id: int,
        shape: Sequence[tir.PrimExpr],
        dtype: str,
    ) -> Tensor:
        """Get RNN state from the inner RNNState."""
        bb = rx.BlockBuilder.current()
        return Tensor(
            _expr=bb.emit(
                rx.call_dps_packed(
                    "vm.builtin.hybrid_state_rnn_get",
                    [self._expr, layer_id, state_id],
                    out_sinfo=rx.TensorStructInfo(shape, dtype),
                )
            )
        )

    def rnn_set(self, layer_id: int, state_id: int, value: Tensor) -> "HybridState":
        """Set RNN state in the inner RNNState."""
        bb = rx.BlockBuilder.current()
        return HybridState(
            _expr=bb.emit(
                rx.call_pure_packed(
                    "vm.builtin.hybrid_state_rnn_set",
                    self._expr,
                    rx.PrimValue(layer_id),
                    rx.PrimValue(state_id),
                    value._expr,  # pylint: disable=protected-access
                    sinfo_args=[rx.ObjectStructInfo()],
                )
            ),
            _name="hybrid_state_rnn_set",
        )
