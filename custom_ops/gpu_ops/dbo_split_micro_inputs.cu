// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "helper.h"
#include "paddle/extension.h"

namespace {

struct DboMicroOut {
  int64_t *ids;
  int *batch_id;
  int *cu_seqlens_q;
  int *cu_seqlens_k;
  int *seq_lens_this_time;
  int *seq_lens_decoder;
};

// Split a decode batch into two micro-batches.
//
// Every input is a full-batch tensor pre_process already built, so there is
// nothing to recompute:
//   slot i is A's   iff  cu_seqlens_q[i] + seq_lens_this_time[i] <= split_tok
//   cu_a[i] = min(cu[i], split_tok)      cu_b[i] = max(cu[i] - split_tok, 0)
//   token t < split_tok -> a.ids[t]      else -> b.ids[t - split_tok]
__global__ void BuildDboMicroInputsKernel(
    const int64_t *__restrict__ ids,
    const int *__restrict__ batch_id,
    const int *__restrict__ cu_seqlens_q,
    const int *__restrict__ seq_lens_this_time,
    const int *__restrict__ seq_lens_decoder,
    DboMicroOut a,
    DboMicroOut b,
    const int bsz,
    const int token_num,
    const int split_tok) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;

  // Slot side: bsz + 1 cu_seqlens entries, bsz length entries.
  for (int i = idx; i <= bsz; i += stride) {
    // get_padding_offset returns cu_seqlens uninitialized when the batch holds
    // no token at all, so keep it inside the token range we can address.
    const int cu = min(max(cu_seqlens_q[i], 0), token_num);
    const int cu_a = min(cu, split_tok);
    const int cu_b = max(cu - split_tok, 0);
    // get_padding_offset writes cu_seqlens_k == cu_seqlens_q; keep that.
    a.cu_seqlens_q[i] = cu_a;
    a.cu_seqlens_k[i] = cu_a;
    b.cu_seqlens_q[i] = cu_b;
    b.cu_seqlens_k[i] = cu_b;

    if (i < bsz) {
      const int len = seq_lens_this_time[i];
      const int dec = seq_lens_decoder[i];
      // A takes a slot only if its whole token range fits before the split.
      const int len_a = (cu + len <= split_tok) ? len : 0;
      const int len_b = len - len_a;
      a.seq_lens_this_time[i] = len_a;
      b.seq_lens_this_time[i] = len_b;
      // A slot a micro-batch does not own must look idle to the attn backend.
      a.seq_lens_decoder[i] = len_a > 0 ? dec : 0;
      b.seq_lens_decoder[i] = len_b > 0 ? dec : 0;
    }
  }

  // Token side: one contiguous cut, so at most one warp diverges.
  for (int t = idx; t < token_num; t += stride) {
    if (t < split_tok) {
      a.ids[t] = ids[t];
      a.batch_id[t] = batch_id[t];
    } else {
      const int j = t - split_tok;
      b.ids[j] = ids[t];
      b.batch_id[j] = batch_id[t];
    }
  }
}

inline DboMicroOut MakeMicroOut(const paddle::Tensor &ids,
                                const paddle::Tensor &batch_id,
                                const paddle::Tensor &cu_seqlens_q,
                                const paddle::Tensor &cu_seqlens_k,
                                const paddle::Tensor &seq_lens_this_time,
                                const paddle::Tensor &seq_lens_decoder) {
  return DboMicroOut{const_cast<int64_t *>(ids.data<int64_t>()),
                     const_cast<int *>(batch_id.data<int>()),
                     const_cast<int *>(cu_seqlens_q.data<int>()),
                     const_cast<int *>(cu_seqlens_k.data<int>()),
                     const_cast<int *>(seq_lens_this_time.data<int>()),
                     const_cast<int *>(seq_lens_decoder.data<int>())};
}

}  // namespace

void BuildDboMicroInputs(const paddle::Tensor &ids_remove_padding,
                         const paddle::Tensor &batch_id_per_token,
                         const paddle::Tensor &cu_seqlens_q,
                         const paddle::Tensor &seq_lens_this_time,
                         const paddle::Tensor &seq_lens_decoder,
                         const paddle::Tensor &a_ids,
                         const paddle::Tensor &a_batch_id,
                         const paddle::Tensor &a_cu_seqlens_q,
                         const paddle::Tensor &a_cu_seqlens_k,
                         const paddle::Tensor &a_seq_lens_this_time,
                         const paddle::Tensor &a_seq_lens_decoder,
                         const paddle::Tensor &b_ids,
                         const paddle::Tensor &b_batch_id,
                         const paddle::Tensor &b_cu_seqlens_q,
                         const paddle::Tensor &b_cu_seqlens_k,
                         const paddle::Tensor &b_seq_lens_this_time,
                         const paddle::Tensor &b_seq_lens_decoder,
                         const int token_num,
                         const int split_token_num) {
  const int bsz = seq_lens_this_time.shape()[0];
  constexpr int kBlockSize = 256;
  const int work = (bsz + 1) > token_num ? (bsz + 1) : token_num;
  const int grid = (work + kBlockSize - 1) / kBlockSize;

  BuildDboMicroInputsKernel<<<grid, kBlockSize, 0, seq_lens_this_time.stream()>>>(
      ids_remove_padding.data<int64_t>(),
      batch_id_per_token.data<int>(),
      cu_seqlens_q.data<int>(),
      seq_lens_this_time.data<int>(),
      seq_lens_decoder.data<int>(),
      MakeMicroOut(a_ids,
                   a_batch_id,
                   a_cu_seqlens_q,
                   a_cu_seqlens_k,
                   a_seq_lens_this_time,
                   a_seq_lens_decoder),
      MakeMicroOut(b_ids,
                   b_batch_id,
                   b_cu_seqlens_q,
                   b_cu_seqlens_k,
                   b_seq_lens_this_time,
                   b_seq_lens_decoder),
      bsz,
      token_num,
      split_token_num);
}
