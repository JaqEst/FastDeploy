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

"""
AFD-aware EPLB table helpers.

AFD uses one communication world for ATTN and FFN ranks, but only FFN ranks
hold routed expert weights. These helpers build EPLB tables in that global
physical expert id space while keeping ATTN rank slots as phantom slots.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from fastdeploy.eplb.eplb import rebalance_experts


def get_afd_expert_layout_sizes(
    num_logical_experts: int,
    redundant_experts_num: int,
    world_size: int,
    ffn_ranks: Sequence[int],
) -> Tuple[int, int, int]:
    """Return ``(ffn_replicas, local_physical_experts, global_physical_experts)``."""
    if num_logical_experts <= 0:
        raise ValueError(f"num_logical_experts must be positive, got {num_logical_experts}")
    if redundant_experts_num < 0:
        raise ValueError(f"redundant_experts_num must be non-negative, got {redundant_experts_num}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not ffn_ranks:
        raise ValueError("AFD EPLB requires at least one FFN rank.")

    ffn_replicas = num_logical_experts + redundant_experts_num
    if ffn_replicas % len(ffn_ranks) != 0:
        raise ValueError(
            "AFD EPLB requires logical + redundant experts to be divisible by FFN ranks: "
            f"num_logical_experts={num_logical_experts}, "
            f"redundant_experts_num={redundant_experts_num}, "
            f"ffn_ranks={list(ffn_ranks)}"
        )
    local_physical_experts = ffn_replicas // len(ffn_ranks)
    global_physical_experts = local_physical_experts * world_size
    return ffn_replicas, local_physical_experts, global_physical_experts


def _map_ffn_physical_to_global(
    ffn_physical_ids: np.ndarray,
    local_physical_experts: int,
    ffn_ranks: Sequence[int],
) -> np.ndarray:
    mapped = np.full_like(ffn_physical_ids, -1, dtype=np.int32)
    valid = ffn_physical_ids >= 0
    if not np.any(valid):
        return mapped

    ffn_rank_indexes = ffn_physical_ids[valid] // local_physical_experts
    local_offsets = ffn_physical_ids[valid] % local_physical_experts
    ffn_rank_array = np.asarray(ffn_ranks, dtype=np.int32)
    mapped[valid] = ffn_rank_array[ffn_rank_indexes] * local_physical_experts + local_offsets
    return mapped


def build_afd_redundant_expert_tables(
    weight: np.ndarray,
    num_replicas: int,
    world_size: int,
    ffn_ranks: Sequence[int],
    redundant_experts_num: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    eplb_strategy: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build AFD global physical expert tables.

    ``rebalance_experts`` is first run over FFN ranks only.  The resulting
    FFN-local physical ids are then mapped into the global AFD physical id
    space, where each ATTN rank owns the same number of phantom slots.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be [layers, logical_experts], got shape={weight.shape}")

    num_layers, num_logical_experts = weight.shape
    ffn_ranks = sorted(int(rank) for rank in ffn_ranks)
    ffn_replicas, local_physical_experts, global_physical_experts = get_afd_expert_layout_sizes(
        num_logical_experts=num_logical_experts,
        redundant_experts_num=redundant_experts_num,
        world_size=world_size,
        ffn_ranks=ffn_ranks,
    )
    if num_replicas != global_physical_experts:
        raise ValueError(
            "AFD num_replicas must match global physical expert slots: "
            f"num_replicas={num_replicas}, expected={global_physical_experts}"
        )

    ffn_phy2log, ffn_log2phy, expert_count = rebalance_experts(
        weight,
        ffn_replicas,
        num_groups,
        num_nodes,
        num_gpus,
        eplb_strategy,
    )

    global_phy2log = np.full((num_layers, global_physical_experts), -1, dtype=np.int32)
    for ffn_rank_index, global_rank in enumerate(ffn_ranks):
        ffn_start = ffn_rank_index * local_physical_experts
        ffn_end = ffn_start + local_physical_experts
        global_start = global_rank * local_physical_experts
        global_end = global_start + local_physical_experts
        global_phy2log[:, global_start:global_end] = ffn_phy2log[:, ffn_start:ffn_end]

    max_replicas = redundant_experts_num + 1
    global_log2phy = np.full((num_layers, num_logical_experts, max_replicas), -1, dtype=np.int32)
    mapped_log2phy = _map_ffn_physical_to_global(ffn_log2phy, local_physical_experts, ffn_ranks)
    if mapped_log2phy.shape[-1] > max_replicas:
        raise ValueError(
            "AFD EPLB generated more replicas per logical expert than expected: "
            f"actual={mapped_log2phy.shape[-1]}, max={max_replicas}"
        )
    global_log2phy[:, :, : mapped_log2phy.shape[-1]] = mapped_log2phy
    return global_phy2log, global_log2phy, expert_count


__all__ = [
    "build_afd_redundant_expert_tables",
    "get_afd_expert_layout_sizes",
]
