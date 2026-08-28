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

"""Model-agnostic dual-batch overlap (DBO) driver for AFD decode.

Each MoE layer is expressed as six stages.  Two micro-batches walk the same
stage stream, with B one op behind A, A first in every step:

    A attn -> A dsend -> B attn -> A drecv -> B dsend -> A local -> B drecv
           -> A csend -> B local -> A crecv -> B csend -> A attn(L+1) -> B crecv -> ...

Four communication phases are each covered by one compute phase.

Control flow depends only on the layer count, so the unrolled kernel stream is
identical for a given captured shape and stays CUDA Graph safe.
"""

from __future__ import annotations

DBO_STAGES = (
    "attn",
    "dispatch_send",
    "dispatch_recv",
    "local",
    "combine_send",
    "combine_recv",
)

NUM_DBO_STAGES = len(DBO_STAGES)


class DBOMicroState:
    """Per-micro-batch state carried through the pipeline.

    ``stash`` holds the intermediates a stage hands to later stages of the same
    layer (routing output, comm handle, recv hook, shared-expert output).

    Every micro-batch owns at least one token: the batch size handed to the split
    is always even (capture sizes are filtered under CUDA Graph, and the eager
    path rounds up).  The op count is what keeps the ATTN and FFN sides in
    lock-step, so it must never depend on how many tokens a micro-batch got.
    """

    __slots__ = ("hidden_states", "residual", "forward_meta", "microbatch_id", "stash")

    def __init__(self, hidden_states, residual, forward_meta, microbatch_id):
        self.hidden_states = hidden_states
        self.residual = residual
        self.forward_meta = forward_meta
        self.microbatch_id = microbatch_id
        self.stash = {}


def run_dbo_pipeline(dbo_layers, state_a, state_b):
    """Drive two micro-batches through ``dbo_layers`` with B one op behind A."""
    num_ops = len(dbo_layers) * NUM_DBO_STAGES
    for t in range(num_ops + 1):
        if t < num_ops:
            run_dbo_op(dbo_layers, state_a, t)
        if t >= 1:
            run_dbo_op(dbo_layers, state_b, t - 1)


def run_dbo_op(dbo_layers, state, op_idx):
    """Execute op ``op_idx`` of the flattened (layer, stage) stream."""
    layer_off, stage_id = divmod(op_idx, NUM_DBO_STAGES)
    layer = dbo_layers[layer_off]
    getattr(layer, f"dbo_{DBO_STAGES[stage_id]}")(state)


def assert_supports_dbo(layers):
    """Fail loudly instead of silently degrading when a model has no DBO stages."""
    for layer in layers:
        if not getattr(layer, "supports_dbo", False):
            raise NotImplementedError(
                f"{type(layer).__name__} does not implement the DBO stages. "
                "Implement dbo_attn/dbo_dispatch_send/dbo_dispatch_recv/dbo_local/"
                "dbo_combine_send/dbo_combine_recv and set supports_dbo = True, "
                "or disable afd_config.enable_dbo."
            )
        for stage in DBO_STAGES:
            if not hasattr(layer, f"dbo_{stage}"):
                raise NotImplementedError(
                    f"{type(layer).__name__} claims supports_dbo but is missing dbo_{stage}."
                )


def assert_backend_supports_dbo(attn_backend):
    """Fail at startup instead of producing wrong output on an unvalidated backend."""
    if not getattr(attn_backend, "supports_dbo", False):
        raise NotImplementedError(
            f"{type(attn_backend).__name__} is not validated for AFD DBO. The backend must "
            "plan every ForwardMeta it is handed (AttentionBackend.plan_split_kv_block) and "
            "must not cache per-step values on itself. Set supports_dbo = True once verified, "
            "or disable afd_config.enable_dbo."
        )
