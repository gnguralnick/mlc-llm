/*!
 * \file serve/hybrid_state.h
 * \brief HybridState wrapping PagedAttentionKVCache + RNNState for hybrid
 *        linear-attention / full-attention models (e.g. Qwen3.5).
 */
#ifndef MLC_LLM_CPP_SERVE_HYBRID_STATE_H_
#define MLC_LLM_CPP_SERVE_HYBRID_STATE_H_

// kv_state.h is a private TVM header; use relative path from cpp/serve/
#include "../../3rdparty/tvm/src/runtime/vm/kv_state.h"

namespace mlc {
namespace llm {
namespace serve {

namespace ffi = tvm::ffi;
using namespace tvm::runtime;
using namespace tvm::runtime::vm;

/*!
 * \brief A state object that wraps both a PagedAttentionKVCache (for full
 *        attention layers) and an RNNState (for linear attention layers).
 *
 *  - KVStateObj methods (Clear, AddSequence, etc.) are forwarded to BOTH.
 *  - AttentionKVCacheObj methods (GetNumAvailablePages, AttentionWithFusedQKV,
 *    GetQueryPositions, etc.) are forwarded to the inner PagedAttentionKVCache.
 *  - RNN get/set are forwarded to the inner RNNState via custom builtins.
 */
class HybridStateObj : public AttentionKVCacheObj {
 public:
  explicit HybridStateObj(AttentionKVCache kv_cache, RNNState rnn_state)
      : kv_cache_(std::move(kv_cache)), rnn_state_(std::move(rnn_state)) {}

  /************** KVStateObj (forwarded to both) **************/
  void Clear() final;
  void AddSequence(int64_t seq_id) final;
  void RemoveSequence(int64_t seq_id) final;
  void ForkSequence(int64_t parent_seq_id, int64_t child_seq_id, int64_t fork_pos) final;
  void PopN(int64_t seq_id, int32_t n) final;
  void BeginForward(const ffi::Shape& seq_ids, const ffi::Shape& append_lengths,
                    const ffi::Optional<ffi::Shape>& token_tree_parent_ptr) final;
  void EndForward() final;

  /************** AttentionKVCacheObj (forwarded to inner KV cache) **************/
  bool Empty() const final;
  int32_t GetNumAvailablePages() const final;
  int32_t GetTotalSequenceLength() const final;
  void EnableSlidingWindowForSeq(int64_t seq_id, int32_t sliding_window_size,
                                 int32_t attn_sink_size) final;
  void CommitAcceptedTokenTreeNodes(const ffi::Shape& seq_ids,
                                    const ffi::Shape& leaf_indices) final;
  ffi::Shape DisaggPrepareRecv(int64_t seq_id, int length) final;
  void DisaggMarkSend(int64_t seq_id, int64_t begin,
                      const ffi::Shape& compressed_remote_position_map,
                      int32_t recver_pe_offset) final;
  void AttentionWithFusedQKV(int64_t layer_id, Tensor qkv_data, ffi::Optional<Tensor> mask,
                              Tensor o_data, double sm_scale) final;
  void SelfAttention(int64_t layer_id, Tensor q_data, Tensor k_data, Tensor v_data,
                     Tensor o_data, Tensor lse_data, double sm_scale) final;
  void CrossAttention(int64_t layer_id, Tensor q_data, Tensor o_data, Tensor lse_data,
                      double sm_scale) final;
  void AppendMLAKV(int64_t layer_id, Tensor kv_data) final;
  ffi::Array<Tensor> MergeAttnOutputInplace(Tensor o_self_attn, Tensor lse_self_attn,
                                             Tensor o_cross_attn, Tensor lse_cross_attn) final;
  void LinearAttention(int64_t layer_id, Tensor q_data, Tensor k_data, Tensor v_data,
                       double sm_scale) final;
  Tensor GetQueryPositions() final;
  void DebugGetKV(int64_t seq_id, int64_t start_pos, int64_t end_pos, Tensor k_data,
                  Tensor v_data) final;
  void DebugGetKVMLA(int64_t seq_id, int64_t start_pos, int64_t end_pos, Tensor kv_data) final;
  void DebugSetKV(int64_t seq_id, int64_t start_pos, Tensor k_data, Tensor v_data) final;

  /************** RNN access (forwarded to inner RNNState) **************/
  void RNNGet(int64_t layer_id, int64_t state_id, Tensor o_data);
  void RNNSet(int64_t layer_id, int64_t state_id, Tensor data);

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO("mlc.serve.HybridState", HybridStateObj, AttentionKVCacheObj);

 private:
  AttentionKVCache kv_cache_;
  RNNState rnn_state_;
};

class HybridState : public AttentionKVCache {
 public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(HybridState, AttentionKVCache, HybridStateObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_CPP_SERVE_HYBRID_STATE_H_
