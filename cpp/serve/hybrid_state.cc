/*!
 * \file serve/hybrid_state.cc
 * \brief HybridState implementation — delegates to inner PagedAttentionKVCache
 *        and RNNState for hybrid linear/full attention models.
 */
#include "hybrid_state.h"

#include <tvm/ffi/reflection/registry.h>

namespace mlc {
namespace llm {
namespace serve {

/************** KVStateObj — forwarded to both **************/

void HybridStateObj::Clear() {
  kv_cache_->Clear();
  rnn_state_->Clear();
}

void HybridStateObj::AddSequence(int64_t seq_id) {
  kv_cache_->AddSequence(seq_id);
  rnn_state_->AddSequence(seq_id);
}

void HybridStateObj::RemoveSequence(int64_t seq_id) {
  kv_cache_->RemoveSequence(seq_id);
  rnn_state_->RemoveSequence(seq_id);
}

void HybridStateObj::ForkSequence(int64_t parent_seq_id, int64_t child_seq_id, int64_t fork_pos) {
  kv_cache_->ForkSequence(parent_seq_id, child_seq_id, fork_pos);
  rnn_state_->ForkSequence(parent_seq_id, child_seq_id, fork_pos);
}

void HybridStateObj::PopN(int64_t seq_id, int32_t n) {
  kv_cache_->PopN(seq_id, n);
  rnn_state_->PopN(seq_id, n);
}

void HybridStateObj::BeginForward(const ffi::Shape& seq_ids, const ffi::Shape& append_lengths,
                                   const ffi::Optional<ffi::Shape>& token_tree_parent_ptr) {
  kv_cache_->BeginForward(seq_ids, append_lengths, token_tree_parent_ptr);
  rnn_state_->BeginForward(seq_ids, append_lengths, token_tree_parent_ptr);
}

void HybridStateObj::EndForward() {
  kv_cache_->EndForward();
  rnn_state_->EndForward();
}

/************** AttentionKVCacheObj — forwarded to inner KV cache **************/

bool HybridStateObj::Empty() const { return kv_cache_->Empty(); }

int32_t HybridStateObj::GetNumAvailablePages() const { return kv_cache_->GetNumAvailablePages(); }

int32_t HybridStateObj::GetTotalSequenceLength() const {
  return kv_cache_->GetTotalSequenceLength();
}

void HybridStateObj::EnableSlidingWindowForSeq(int64_t seq_id, int32_t sliding_window_size,
                                                int32_t attn_sink_size) {
  kv_cache_->EnableSlidingWindowForSeq(seq_id, sliding_window_size, attn_sink_size);
}

void HybridStateObj::CommitAcceptedTokenTreeNodes(const ffi::Shape& seq_ids,
                                                   const ffi::Shape& leaf_indices) {
  kv_cache_->CommitAcceptedTokenTreeNodes(seq_ids, leaf_indices);
}

ffi::Shape HybridStateObj::DisaggPrepareRecv(int64_t seq_id, int length) {
  return kv_cache_->DisaggPrepareRecv(seq_id, length);
}

void HybridStateObj::DisaggMarkSend(int64_t seq_id, int64_t begin,
                                     const ffi::Shape& compressed_remote_position_map,
                                     int32_t recver_pe_offset) {
  kv_cache_->DisaggMarkSend(seq_id, begin, compressed_remote_position_map, recver_pe_offset);
}

void HybridStateObj::AttentionWithFusedQKV(int64_t layer_id, Tensor qkv_data,
                                            ffi::Optional<Tensor> mask, Tensor o_data,
                                            double sm_scale) {
  kv_cache_->AttentionWithFusedQKV(layer_id, std::move(qkv_data), std::move(mask),
                                    std::move(o_data), sm_scale);
}

void HybridStateObj::SelfAttention(int64_t layer_id, Tensor q_data, Tensor k_data, Tensor v_data,
                                    Tensor o_data, Tensor lse_data, double sm_scale) {
  kv_cache_->SelfAttention(layer_id, std::move(q_data), std::move(k_data), std::move(v_data),
                            std::move(o_data), std::move(lse_data), sm_scale);
}

void HybridStateObj::CrossAttention(int64_t layer_id, Tensor q_data, Tensor o_data,
                                     Tensor lse_data, double sm_scale) {
  kv_cache_->CrossAttention(layer_id, std::move(q_data), std::move(o_data), std::move(lse_data),
                             sm_scale);
}

void HybridStateObj::AppendMLAKV(int64_t layer_id, Tensor kv_data) {
  kv_cache_->AppendMLAKV(layer_id, std::move(kv_data));
}

ffi::Array<Tensor> HybridStateObj::MergeAttnOutputInplace(Tensor o_self_attn, Tensor lse_self_attn,
                                                           Tensor o_cross_attn,
                                                           Tensor lse_cross_attn) {
  return kv_cache_->MergeAttnOutputInplace(std::move(o_self_attn), std::move(lse_self_attn),
                                            std::move(o_cross_attn), std::move(lse_cross_attn));
}

void HybridStateObj::LinearAttention(int64_t layer_id, Tensor q_data, Tensor k_data,
                                      Tensor v_data, double sm_scale) {
  kv_cache_->LinearAttention(layer_id, std::move(q_data), std::move(k_data), std::move(v_data),
                              sm_scale);
}

Tensor HybridStateObj::GetQueryPositions() { return kv_cache_->GetQueryPositions(); }

void HybridStateObj::DebugGetKV(int64_t seq_id, int64_t start_pos, int64_t end_pos, Tensor k_data,
                                 Tensor v_data) {
  kv_cache_->DebugGetKV(seq_id, start_pos, end_pos, std::move(k_data), std::move(v_data));
}

void HybridStateObj::DebugGetKVMLA(int64_t seq_id, int64_t start_pos, int64_t end_pos,
                                    Tensor kv_data) {
  kv_cache_->DebugGetKVMLA(seq_id, start_pos, end_pos, std::move(kv_data));
}

void HybridStateObj::DebugSetKV(int64_t seq_id, int64_t start_pos, Tensor k_data, Tensor v_data) {
  kv_cache_->DebugSetKV(seq_id, start_pos, std::move(k_data), std::move(v_data));
}

/************** RNN access — forwarded to inner RNNState **************/

void HybridStateObj::RNNGet(int64_t layer_id, int64_t state_id, Tensor o_data) {
  rnn_state_->Get(layer_id, state_id, std::move(o_data));
}

void HybridStateObj::RNNSet(int64_t layer_id, int64_t state_id, Tensor data) {
  rnn_state_->Set(layer_id, state_id, std::move(data));
}

//-------------------------------------------------
//  Registration
//-------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Factory: create HybridState wrapping a PagedAttentionKVCache and an RNNState.
  refl::GlobalDef().def(
      "vm.builtin.hybrid_state_create",
      [](AttentionKVCache kv_cache, RNNState rnn_state) {
        auto n = tvm::ffi::make_object<HybridStateObj>(std::move(kv_cache), std::move(rnn_state));
        return HybridState(std::move(n));
      });
  // Custom builtins for accessing the RNN part of the hybrid state.
  refl::GlobalDef()
      .def_method("vm.builtin.hybrid_state_rnn_get", &HybridStateObj::RNNGet)
      .def("vm.builtin.hybrid_state_rnn_set",
           [](HybridState state, int64_t layer_id, int64_t state_id, Tensor data) {
             state->RNNSet(layer_id, state_id, std::move(data));
             return state;
           });
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
