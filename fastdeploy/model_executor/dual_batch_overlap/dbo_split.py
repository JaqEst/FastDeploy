# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Split a decode batch into two micro-batches for DBO.

The split is by *token*, not by slot: a slot whose ``seq_lens_this_time`` is 0
(frozen by a step retry, block-stepped, or just finished) owns no token, so slot
index and token index do not line up.  Slot-side tensors therefore keep their
full length and are masked down to the slots the micro-batch owns.
"""

from __future__ import annotations

import dataclasses

import paddle

from fastdeploy.model_executor.ops.gpu import (
    build_dbo_micro_inputs as build_dbo_micro_inputs_kernel,
)


def allocate_dbo_token_buffer(max_num_seqs, decoder_step_token_num):
    """Allocate one micro-batch's persistent input buffers."""
    max_tokens = (max_num_seqs * decoder_step_token_num + 1) // 2
    return {
        "seq_lens_this_time": paddle.full([max_num_seqs], 0, dtype="int32"),
        "seq_lens_decoder": paddle.full([max_num_seqs], 0, dtype="int32"),
        "ids_remove_padding": paddle.full([max_tokens], 0, dtype="int64"),
        "batch_id_per_token": paddle.full([max_tokens], 0, dtype="int32"),
        "cu_seqlens_q": paddle.full([max_num_seqs + 1], 0, dtype="int32"),
        "cu_seqlens_k": paddle.full([max_num_seqs + 1], 0, dtype="int32"),
    }


def build_dbo_micro_inputs(
    ids_remove_padding,
    batch_id_per_token,
    seq_lens_this_time,
    seq_lens_decoder,
    cu_seqlens_q,
    token_num,
    attn_buffers,
    token_buffers,
):
    """Build the two per-micro-batch ``ForwardMeta`` override dicts."""
    bsz = seq_lens_this_time.shape[0]
    split_tok = (token_num + 1) // 2
    micro_a, micro_b = token_buffers

    build_dbo_micro_inputs_kernel(
        ids_remove_padding,
        batch_id_per_token,
        cu_seqlens_q,
        seq_lens_this_time,
        seq_lens_decoder,
        micro_a["ids_remove_padding"],
        micro_a["batch_id_per_token"],
        micro_a["cu_seqlens_q"],
        micro_a["cu_seqlens_k"],
        micro_a["seq_lens_this_time"],
        micro_a["seq_lens_decoder"],
        micro_b["ids_remove_padding"],
        micro_b["batch_id_per_token"],
        micro_b["cu_seqlens_q"],
        micro_b["cu_seqlens_k"],
        micro_b["seq_lens_this_time"],
        micro_b["seq_lens_decoder"],
        token_num,
        split_tok,
    )

    # The buffers stay max-sized so their addresses stay captured; hand out
    # views carrying this step's lengths.
    return [
        {
            "ids_remove_padding": buffers["ids_remove_padding"][:num_tokens],
            "batch_id_per_token": buffers["batch_id_per_token"][:num_tokens],
            "cu_seqlens_q": buffers["cu_seqlens_q"][: bsz + 1],
            "cu_seqlens_k": buffers["cu_seqlens_k"][: bsz + 1],
            "seq_lens_this_time": buffers["seq_lens_this_time"][:bsz],
            "seq_lens_decoder": buffers["seq_lens_decoder"][:bsz],
            **attn_buffers[mb_id],
        }
        for mb_id, (buffers, num_tokens) in enumerate(zip(token_buffers, (split_tok, token_num - split_tok)))
    ]


def split_decode_forward_meta(forward_meta):
    """Return two ForwardMeta views over disjoint halves of the decode batch."""
    return tuple(dataclasses.replace(forward_meta, **overrides) for overrides in forward_meta.dbo_micro_inputs)
