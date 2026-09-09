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

__all__ = ["evict_order", "keep_from_order", "card_imbalance", "coverage_repair"]


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


def card_imbalance(slots: np.ndarray, load: np.ndarray, num_cards: int) -> float:
    """
    Peak over mean card load for one layer. A logical expert's load splits evenly
    over its replicas, matching the kernel's uniform pick among them.

    Args:
        slots: [C] logical expert per surviving slot, ordered by (card, slot).
        load: [E] recent token count per logical expert.
        num_cards: surviving FFN cards; len(slots) must divide by it.
    """
    count = np.bincount(slots, minlength=load.shape[0])
    per_card = (load[slots] / count[slots]).reshape(num_cards, -1).sum(axis=1)
    mean = per_card.mean()
    return float(per_card.max() / mean) if mean > 0 else 1.0


def coverage_repair(
    keep: np.ndarray,
    slots: np.ndarray,
    importance: np.ndarray,
    load: np.ndarray,
    num_cards: int,
    target_imbalance: float = 1.05,
    max_swap_rounds: int = 400,
):
    """
    Turn a target keep set into the H2D writes that realise it.

    Two phases per layer. First refill coverage, taking slots from experts with a
    spare replica before slots holding the last copy of an expert `keep` drops.
    Then trade slots between the hottest and coldest card until the load spread
    is acceptable; a trade costs two writes but no coverage.

    Args:
        keep: [L, E] bool, from `keep_from_order`.
        slots: [L, C] logical expert per surviving slot, ordered by (card, slot).
            Not modified.
        importance: [L, E], orders which expert to bring back first and which to
            give up when no spare replica is left.
        load: [L, E] recent token count per logical expert.
        num_cards: surviving FFN cards; C must divide by it.
        target_imbalance: stop trading once peak/mean card load is at most this.
        max_swap_rounds: cap on trades per layer.

    Returns:
        moves: list of (layer, slot, expert), in the order they should be
            applied. Coverage comes first, so truncating still leaves a valid
            placement.
        new_slots: [L, C] the placement after every move.
    """
    num_layers, num_experts = keep.shape
    capacity = slots.shape[1]
    if slots.shape[0] != num_layers:
        raise ValueError(f"keep has {num_layers} layers, slots has {slots.shape[0]}")
    if capacity % num_cards:
        raise ValueError(f"capacity {capacity} is not a multiple of num_cards {num_cards}")
    over = keep.sum(axis=1) > capacity
    if over.any():
        raise ValueError(
            f"layer {int(np.flatnonzero(over)[0])} keeps "
            f"{int(keep.sum(axis=1).max())} experts but only has {capacity} slots"
        )

    new_slots = np.array(slots)
    moves = []
    for layer in range(num_layers):
        row = new_slots[layer]
        for slot, expert in _refill_layer(keep[layer], row, importance[layer]):
            moves.append((layer, slot, expert))
        for slot, expert in _balance_layer(
            row, load[layer], num_cards, target_imbalance, max_swap_rounds
        ):
            moves.append((layer, slot, expert))
    return moves, new_slots


def _refill_layer(keep: np.ndarray, slots: np.ndarray, importance: np.ndarray):
    """Bring every kept expert back into `slots` (modified in place)."""
    count = np.bincount(slots, minlength=keep.shape[0])
    incoming = np.flatnonzero(keep & (count == 0))
    incoming = incoming[np.argsort(-importance[incoming])]

    moves = []
    for expert in incoming:
        spare = np.flatnonzero(count >= 2)
        if spare.size:
            donor = int(spare[np.argmax(count[spare])])
        else:
            doomed = np.flatnonzero((count >= 1) & ~keep)
            donor = int(doomed[np.argmin(importance[doomed])])
        slot = int(np.flatnonzero(slots == donor)[-1])
        count[donor] -= 1
        slots[slot] = expert
        count[expert] += 1
        moves.append((slot, int(expert)))
    return moves


def _balance_layer(
    slots: np.ndarray, load: np.ndarray, num_cards: int, target: float, max_rounds: int
):
    """Trade slots between the hottest and coldest card (`slots` modified in place)."""
    count = np.bincount(slots, minlength=load.shape[0])
    per_slot = load[slots] / count[slots]
    slots_per_card = slots.shape[0] // num_cards

    moves = []
    for _ in range(max_rounds):
        per_card = per_slot.reshape(num_cards, slots_per_card).sum(axis=1)
        mean = per_card.mean()
        if mean <= 0 or per_card.max() / mean <= target:
            break

        hot, cold = int(per_card.argmax()), int(per_card.argmin())
        hot_slots = slice(hot * slots_per_card, (hot + 1) * slots_per_card)
        cold_slots = slice(cold * slots_per_card, (cold + 1) * slots_per_card)
        # Trading slot i of the hot card for slot j of the cold one moves this
        # much load. Refuse trades that overshoot, or the pair oscillates.
        gain = per_slot[hot_slots][:, None] - per_slot[cold_slots][None, :]
        allowed = (gain > 0) & (per_card[hot] - gain > per_card[cold] + gain - 1e-9)
        if not allowed.any():
            break

        i, j = divmod(int(np.where(allowed, gain, -np.inf).argmax()), slots_per_card)
        left, right = hot * slots_per_card + i, cold * slots_per_card + j
        slots[left], slots[right] = slots[right], slots[left]
        per_slot[left], per_slot[right] = per_slot[right], per_slot[left]
        moves.append((left, int(slots[left])))
        moves.append((right, int(slots[right])))
    return moves
