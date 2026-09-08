"""
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

import numpy as np

__all__ = ["evict_order", "keep_from_order"]


def evict_order(importance: np.ndarray, similarity: np.ndarray):
    """
    Rank logical experts by how expendable they are, per layer, minimising

        Loss(K) = 1/D * sum_d sum_u P[d, u] * (1 - max_{j in K} S[d, u, j])

    The greedy gives up the cheapest expert each round and recomputes; ranking
    once on the full set would give up both halves of a mutually similar pair,
    each replaceable only by the other. The length-k prefix of the result is
    exactly what the same greedy would drop if asked for k, so one order serves
    every surviving-slot count.

    Args:
        importance: [D, L, E] importance score per domain, layer and expert.
        similarity: [D, L, E, E], how well expert j stands in for expert u.
            The diagonal must be 1 so that a kept expert costs nothing.

    Returns:
        order: [L, E] int16, experts in the order they should be given up.
        cost: [L, E] float32, Loss increase per step, non-decreasing along a
            row. The last entry is +inf.
    """
    if importance.ndim != 3:
        raise ValueError(f"importance must be [D, L, E], got shape {importance.shape}")
    num_domains, num_layers, num_experts = importance.shape
    if similarity.shape != (num_domains, num_layers, num_experts, num_experts):
        raise ValueError(
            f"similarity must be {(num_domains, num_layers, num_experts, num_experts)}, "
            f"got shape {similarity.shape}"
        )
    if num_experts < 2:
        raise ValueError(f"need at least 2 experts per layer, got {num_experts}")

    diagonal = np.einsum("dluu->dlu", similarity)
    if not np.allclose(diagonal, 1.0):
        raise ValueError(
            f"similarity needs a unit diagonal, observed range "
            f"[{diagonal.min()}, {diagonal.max()}]"
        )

    order = np.empty((num_layers, num_experts), dtype=np.int16)
    cost = np.empty((num_layers, num_experts), dtype=np.float32)
    for layer in range(num_layers):
        order[layer], cost[layer] = _order_one_layer(importance[:, layer], similarity[:, layer])
    return order, cost


def _order_one_layer(importance: np.ndarray, similarity: np.ndarray):
    """importance [D, E], similarity [D, E, E] (not modified) -> (order, cost)."""
    num_domains, num_experts = importance.shape
    flat_importance = importance.reshape(-1).astype(np.float64)
    # Rows are (domain, expert) pairs, columns are candidate stand-ins.
    work = np.array(similarity, dtype=np.float64).reshape(-1, num_experts)
    rows = np.arange(num_domains * num_experts)

    alive = np.ones(num_experts, dtype=bool)
    order = np.empty(num_experts, dtype=np.int16)
    cost = np.empty(num_experts, dtype=np.float32)

    for step in range(num_experts - 1):
        best_col = work.argmax(axis=1)
        best = work[rows, best_col]
        work[rows, best_col] = -np.inf
        second = work.max(axis=1)
        work[rows, best_col] = best

        # A kept expert scores its own diagonal, so its entry is the cost of
        # giving it up; an already-dropped one scores its current stand-in.
        gain = flat_importance * (best - second)
        delta = np.zeros(num_experts)
        np.add.at(delta, best_col, gain)  # add.at: best_col repeats

        victim = int(np.where(alive, delta, np.inf).argmin())
        order[step] = victim
        cost[step] = delta[victim] / num_domains
        alive[victim] = False
        # Masking the column also clears the diagonal, so a dropped expert stops
        # covering itself.
        work[:, victim] = -np.inf

    order[num_experts - 1] = int(np.flatnonzero(alive)[0])
    cost[num_experts - 1] = np.inf
    return order, cost


def keep_from_order(order: np.ndarray, capacity: int) -> np.ndarray:
    """
    Args:
        order: [L, E] int, from `evict_order`.
        capacity: physical slots per layer, i.e. num_local_physical_experts
            times the number of surviving FFN ranks.

    Returns:
        keep: [L, E] bool, all True when every expert fits.
    """
    if capacity < 1:
        raise ValueError(f"capacity must be at least 1, got {capacity}")
    num_layers, num_experts = order.shape
    keep = np.ones((num_layers, num_experts), dtype=bool)
    num_evict = num_experts - capacity
    if num_evict <= 0:
        return keep
    keep[np.arange(num_layers)[:, None], order[:, :num_evict]] = False
    return keep
