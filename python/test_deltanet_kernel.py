"""Test DeltaNet TIR kernel in isolation on Metal."""
import numpy as np
import tvm
from tvm.runtime import tensor as nd
from tvm.script import tir as T

num_heads = 2
key_head_dim = 4
value_head_dim = 4
dtype = "float32"


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
    q_buf = T.match_buffer(var_q, (batch_size, seq_len, num_heads, key_head_dim), dtype="float32")
    k_buf = T.match_buffer(var_k, (batch_size, seq_len, num_heads, key_head_dim), dtype="float32")
    v_buf = T.match_buffer(var_v, (batch_size, seq_len, num_heads, value_head_dim), dtype="float32")
    g_buf = T.match_buffer(var_g, (batch_size, seq_len, num_heads), dtype="float32")
    beta_buf = T.match_buffer(var_beta, (batch_size, seq_len, num_heads), dtype="float32")
    state_buf = T.match_buffer(
        var_state, (batch_size, num_heads, key_head_dim, value_head_dim), dtype="float32"
    )
    out_buf = T.match_buffer(
        var_out, (batch_size, seq_len, num_heads, value_head_dim), dtype=dtype
    )
    out_state_buf = T.match_buffer(
        var_out_state, (batch_size, num_heads, key_head_dim, value_head_dim), dtype="float32"
    )

    for b in T.thread_binding(batch_size, thread="blockIdx.y"):
        for h in T.thread_binding(num_heads, thread="blockIdx.x"):
            for j in T.thread_binding(value_head_dim, thread="threadIdx.x"):
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

                        g_t: T.float32 = T.exp(g_buf[vb, vt, vh])
                        for i in range(key_head_dim):
                            out_state_buf[vb, vh, i, vj] = out_state_buf[vb, vh, i, vj] * g_t

                        kv_mem_j: T.float32 = T.float32(0)
                        for i in range(key_head_dim):
                            kv_mem_j += out_state_buf[vb, vh, i, vj] * k_buf[vb, vt, vh, i]

                        delta_j: T.float32 = (
                            v_buf[vb, vt, vh, vj] - kv_mem_j
                        ) * beta_buf[vb, vt, vh]

                        for i in range(key_head_dim):
                            out_state_buf[vb, vh, i, vj] += k_buf[vb, vt, vh, i] * delta_j

                        out_buf[vb, vt, vh, vj] = T.cast(T.float32(0), dtype)
                        for i in range(key_head_dim):
                            out_buf[vb, vt, vh, vj] += T.cast(
                                out_state_buf[vb, vh, i, vj] * q_buf[vb, vt, vh, i], dtype
                            )


def main():
    target = tvm.target.Target("metal")
    with tvm.transform.PassContext(opt_level=0):
        mod = tvm.IRModule({"deltanet": deltanet_func})
        lib = tvm.build(mod, target=target)

    dev = tvm.metal(0)

    np.random.seed(42)
    B, T = 1, 3
    q = np.random.randn(B, T, num_heads, key_head_dim).astype("float32") * 0.1
    k = np.random.randn(B, T, num_heads, key_head_dim).astype("float32") * 0.1
    v = np.random.randn(B, T, num_heads, value_head_dim).astype("float32") * 0.1
    g = -np.abs(np.random.randn(B, T, num_heads).astype("float32") * 0.5)
    beta = np.random.rand(B, T, num_heads).astype("float32") * 0.5
    state = np.zeros((B, num_heads, key_head_dim, value_head_dim), dtype="float32")

    # Numpy reference
    state_ref = state.copy()
    out_ref = np.zeros((B, T, num_heads, value_head_dim), dtype="float32")
    for t in range(T):
        for h in range(num_heads):
            g_t = np.exp(g[0, t, h])
            state_ref[0, h] *= g_t
            kv_mem = (state_ref[0, h] * k[0, t, h, :, None]).sum(axis=0)
            delta = (v[0, t, h] - kv_mem) * beta[0, t, h]
            state_ref[0, h] += k[0, t, h, :, None] * delta[None, :]
            out_ref[0, t, h] = (state_ref[0, h] * q[0, t, h, :, None]).sum(axis=0)

    # Run on Metal
    def to_tvm(arr, dev):
        t = tvm.runtime.empty(arr.shape, dtype=str(arr.dtype), device=dev)
        t.copyfrom(arr)
        return t

    tvm_q = to_tvm(q, dev)
    tvm_k = to_tvm(k, dev)
    tvm_v = to_tvm(v, dev)
    tvm_g = to_tvm(g, dev)
    tvm_beta = to_tvm(beta, dev)
    tvm_state = to_tvm(state, dev)
    tvm_out = to_tvm(np.zeros((B, T, num_heads, value_head_dim), dtype="float32"), dev)
    tvm_out_state = to_tvm(
        np.zeros((B, num_heads, key_head_dim, value_head_dim), dtype="float32"), dev
    )

    lib["deltanet_func"](tvm_q, tvm_k, tvm_v, tvm_g, tvm_beta, tvm_state, tvm_out, tvm_out_state)

    metal_out = tvm_out.numpy()
    metal_state = tvm_out_state.numpy()

    print("=== Output comparison ===")
    for t in range(T):
        for h in range(num_heads):
            print(f"  t={t} h={h} ref: {out_ref[0, t, h]}")
            print(f"  t={t} h={h} mtl: {metal_out[0, t, h]}")
    print(
        f"Output match: {np.allclose(out_ref, metal_out, atol=1e-5)}, "
        f"max diff: {np.abs(out_ref - metal_out).max():.8f}"
    )
    print("\n=== State comparison ===")
    for h in range(num_heads):
        print(f"  h={h} ref:\n{state_ref[0, h]}")
        print(f"  h={h} mtl:\n{metal_state[0, h]}")
    print(
        f"State match: {np.allclose(state_ref, metal_state, atol=1e-5)}, "
        f"max diff: {np.abs(state_ref - metal_state).max():.8f}"
    )


if __name__ == "__main__":
    main()
