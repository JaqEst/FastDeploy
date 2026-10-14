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

__all__ = ["evict_order", "keep_from_order", "gpu_imbalance", "coverage_repair"]


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
    num_domains, num_layers, num_experts = importance.shape

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

    best_col = work.argmax(axis=1)
    best = work[rows, best_col]
    work[rows, best_col] = -np.inf
    second_col = work.argmax(axis=1)
    second = work[rows, second_col]
    work[rows, best_col] = best
    gain = flat_importance * (best - second)
    delta = np.bincount(best_col, weights=gain, minlength=num_experts)

    for step in range(num_experts - 1):
        victim = int(np.where(alive, delta, np.inf).argmin())
        order[step] = victim
        cost[step] = delta[victim] / num_domains
        alive[victim] = False
        if step == num_experts - 2:
            break
        # Masking the column also clears the diagonal, so a dropped expert stops
        # covering itself.
        work[:, victim] = -np.inf

        affected = np.flatnonzero((best_col == victim) | (second_col == victim))
        if affected.size:
            delta -= np.bincount(best_col[affected], weights=gain[affected], minlength=num_experts)
            sub = work[affected]
            local = np.arange(affected.size)
            best_col[affected] = sub.argmax(axis=1)
            best[affected] = sub[local, best_col[affected]]
            sub[local, best_col[affected]] = -np.inf
            second_col[affected] = sub.argmax(axis=1)
            second[affected] = sub[local, second_col[affected]]
            gain[affected] = flat_importance[affected] * (best[affected] - second[affected])
            delta += np.bincount(best_col[affected], weights=gain[affected], minlength=num_experts)

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


def gpu_imbalance(slots: np.ndarray, load: np.ndarray, num_gpus: int) -> float:
    """
    Peak over mean GPU load for one layer. A logical expert's load splits evenly
    over its replicas, matching the kernel's uniform pick among them.

    Args:
        slots: [C] logical expert per surviving slot, ordered by (gpu, slot).
        load: [E] recent token count per logical expert.
        num_gpus: surviving GPUs; len(slots) must divide by it.
    """
    count = np.bincount(slots, minlength=load.shape[0])
    per_gpu = (load[slots] / count[slots]).reshape(num_gpus, -1).sum(axis=1)
    mean = per_gpu.mean()
    return float(per_gpu.max() / mean) if mean > 0 else 1.0


def coverage_repair(
    keep: np.ndarray,
    slots: np.ndarray,
    importance: np.ndarray,
    load: np.ndarray,
    num_gpus: int,
    target_imbalance: float = 1.05,
    max_swap_rounds: int = 400,
):
    """
    Turn a target keep set into the H2D writes that realise it.

    Two phases per layer. First refill coverage, taking slots from experts with a
    spare replica before slots holding the last copy of an expert `keep` drops.
    Then trade slots between the hottest and coldest GPU until the load spread
    is acceptable; a trade costs two writes but no coverage.

    Args:
        keep: [L, E] bool, from `keep_from_order`.
        slots: [L, C] logical expert per surviving slot, ordered by (gpu, slot).
            Not modified.
        importance: [L, E], orders which expert to bring back first and which to
            give up when no spare replica is left.
        load: [L, E] recent token count per logical expert.
        num_gpus: surviving FFN GPUs; C must divide by it.
        target_imbalance: stop trading once peak/mean GPU load is at most this.
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
    if capacity % num_gpus:
        raise ValueError(f"capacity {capacity} is not a multiple of num_gpus {num_gpus}")
    over = keep.sum(axis=1) > capacity
    if over.any():
        raise ValueError(
            f"layer {int(np.flatnonzero(over)[0])} keeps "
            f"{int(keep.sum(axis=1).max())} experts but only has {capacity} slots"
        )

    new_slots = np.array(slots)
    slots_per_gpu = capacity // num_gpus
    gpu_writes = np.zeros(num_gpus, dtype=np.int64)
    for layer in range(num_layers):
        row = new_slots[layer]
        _refill_layer(keep[layer], row, importance[layer], num_gpus, gpu_writes)
        _balance_layer(row, load[layer], num_gpus, target_imbalance, max_swap_rounds)
        np.add.at(gpu_writes, np.flatnonzero(slots[layer] != row) // slots_per_gpu, 1)

    changed = np.nonzero(slots != new_slots)
    moves = [(int(layer), int(slot), int(new_slots[layer, slot])) for layer, slot in zip(*changed)]
    return moves, new_slots


def _refill_layer(
    keep: np.ndarray,
    slots: np.ndarray,
    importance: np.ndarray,
    num_gpus: int,
    gpu_writes: np.ndarray,
) -> None:
    """Bring every kept expert back into `slots` (modified in place)."""
    count = np.bincount(slots, minlength=keep.shape[0])
    incoming = np.flatnonzero(keep & (count == 0))
    incoming = incoming[np.argsort(-importance[incoming])]
    slots_per_gpu = slots.shape[0] // num_gpus
    load = gpu_writes.astype(np.int64).copy()

    for expert in incoming:
        spare = count[slots] >= 2
        donors = np.flatnonzero(spare) if spare.any() else np.flatnonzero(~keep[slots])
        gpus = donors // slots_per_gpu
        # Least loaded GPU, then lowest slot: donors is ascending and argmin takes the
        # first minimum.
        pick = int(np.argmin(load[gpus]))
        slot, gpu = int(donors[pick]), int(gpus[pick])

        count[slots[slot]] -= 1
        slots[slot] = expert
        count[expert] += 1
        load[gpu] += 1


def _balance_layer(
    slots: np.ndarray,
    load: np.ndarray,
    num_gpus: int,
    target: float,
    max_rounds: int,
) -> None:
    """Trade slots between the hottest and coldest GPU (`slots` modified in place)."""
    count = np.bincount(slots, minlength=load.shape[0])
    per_slot = load[slots] / count[slots]
    slots_per_gpu = slots.shape[0] // num_gpus

    for _ in range(max_rounds):
        per_gpu = per_slot.reshape(num_gpus, slots_per_gpu).sum(axis=1)
        mean = per_gpu.mean()
        if mean <= 0 or per_gpu.max() / mean <= target:
            break

        hot, cold = int(per_gpu.argmax()), int(per_gpu.argmin())
        hot_slots = slice(hot * slots_per_gpu, (hot + 1) * slots_per_gpu)
        cold_slots = slice(cold * slots_per_gpu, (cold + 1) * slots_per_gpu)
        # Trading slot i of the hot GPU for slot j of the cold one moves this
        # much load. Refuse trades that overshoot, or the pair oscillates.
        gain = per_slot[hot_slots][:, None] - per_slot[cold_slots][None, :]
        allowed = (gain > 0) & (per_gpu[hot] - gain > per_gpu[cold] + gain - 1e-9)
        if not allowed.any():
            break

        i, j = divmod(int(np.where(allowed, gain, -np.inf).argmax()), slots_per_gpu)
        left, right = hot * slots_per_gpu + i, cold * slots_per_gpu + j
        slots[left], slots[right] = slots[right], slots[left]
        per_slot[left], per_slot[right] = per_slot[right], per_slot[left]


def reserve_experts(
    phy2log: np.ndarray,
    dropped: np.ndarray,
    load: np.ndarray,
    dead_ranks: list,
    local_physical_experts: int,
    max_replicas: int,
) -> np.ndarray:
    """Reserve the experts the down ranks load once they come back.

    Args:
        phy2log: [L, num_physical_experts] int32 with the survivors already placed,
            modified in place.
        dropped: [L, K] int, `evict_order`'s first K columns (most expendable first);
            K is the number of experts the survivors' plan dropped.
        load: [L, E] recent token count per logical expert.
        dead_ranks: global FFN rank ids that are down.
        local_physical_experts: slots per FFN rank.
        max_replicas: replica cap per logical expert.

    Returns:
        phy2log.
    """
    capacity = len(dead_ranks) * local_physical_experts
    if capacity == 0:
        return phy2log

    num_layers, num_physical = phy2log.shape
    num_ranks = num_physical // local_physical_experts
    num_experts = load.shape[1]

    survivors = np.flatnonzero((phy2log >= 0).all(axis=0))

    # Every layer runs in lock-step, so all the state carries a leading layer
    # axis and one placement costs O(num_ranks) per layer. A copy of e adds
    # load[e]/(c+1) to its rank and shaves load[e]/(c(c+1)) off every rank
    # already holding it.
    ranks_of_slot = np.repeat(np.arange(num_ranks), local_physical_experts)
    rows = np.arange(num_layers)
    surv_expert = phy2log[:, survivors]  # [L, S] expert ids
    layers_full = np.repeat(rows, survivors.size)
    expert_full = surv_expert.ravel()
    rank_full = np.tile(ranks_of_slot[survivors], num_layers)
    count = np.bincount(layers_full * num_experts + expert_full, minlength=num_layers * num_experts)
    count = count.reshape(num_layers, num_experts)
    # [L, R, E] so `copies[rows, rank]` is a cheap per-rank gather.
    copies = np.bincount(
        layers_full * (num_ranks * num_experts) + rank_full * num_experts + expert_full,
        minlength=num_layers * num_ranks * num_experts,
    ).reshape(num_layers, num_ranks, num_experts)
    per_slot = np.take_along_axis(load, surv_expert, axis=1) / np.take_along_axis(count, surv_expert, axis=1)
    loads = np.bincount(
        layers_full * num_ranks + rank_full, weights=per_slot.ravel(), minlength=num_layers * num_ranks
    ).reshape(num_layers, num_ranks)
    gain = load / (count + 1)
    relief = np.where(count > 0, load / np.maximum(count * (count + 1), 1), 0.0)
    room = count < max_replicas

    free = np.zeros((num_layers, num_ranks), dtype=np.int64)  # next free slot per rank
    left = np.zeros((num_layers, num_ranks), dtype=np.int64)  # free slots left per rank
    dead_mask = np.zeros(num_ranks, dtype=bool)
    for rank in dead_ranks:
        dead_mask[rank] = True
        free[:, rank] = rank * local_physical_experts
        left[:, rank] = local_physical_experts

    def emptiest():
        return np.where(dead_mask & (left > 0), loads, np.inf).argmin(axis=1)

    def place(rank, expert):
        nonlocal loads
        c = count[rows, expert]
        load_e = load[rows, expert]
        # `gain` and `relief` still hold the pre-placement values, and relief is
        # already 0 where c == 0, so both can be used as-is.
        loads -= relief[rows, expert][:, None] * copies[rows, :, expert]
        loads[rows, rank] += gain[rows, expert]
        copies[rows, rank, expert] += 1
        count[rows, expert] = c + 1
        gain[rows, expert] = load_e / (c + 2)
        relief[rows, expert] = load_e / ((c + 1) * (c + 2))
        room[rows, expert] = (c + 1) < max_replicas
        phy2log[rows, free[rows, rank]] = expert
        free[rows, rank] += 1
        left[rows, rank] -= 1

    # Mandatory: every dropped expert gets a slot back, heaviest first, so the
    # emptiest rank takes the biggest load (longest-processing-time first).
    mandatory = np.take_along_axis(dropped, np.argsort(-load[rows[:, None], dropped], axis=1), axis=1)
    for step in range(mandatory.shape[1]):
        place(emptiest(), mandatory[:, step])

    # Filler: hand the emptiest down rank the expert whose copy leaves the
    # smallest peak, preferring the heaviest expert on a tie.
    for _ in range(capacity - mandatory.shape[1]):
        rank = emptiest()
        cand = room & (copies[rows, rank] == 0)
        has_cand = cand.any(axis=1)
        if not has_cand.all():
            cand = np.where(has_cand[:, None], cand, room)  # repeat on the rank if needed
            if not cand.any(axis=1).all():
                raise ValueError(
                    f"cannot fill the down ranks: every expert already has {max_replicas} copies"
                )
        # Peak after the copy: the rank we top up, or the heaviest other rank
        # once every holder of e sheds its relief. Only the heaviest rank can
        # drop, so the top two loads decide it.
        top = loads.argmax(axis=1)
        max1 = loads[rows, top]
        second = np.partition(loads, -2, axis=1)[:, -2] if num_ranks > 1 else np.full(num_layers, -np.inf)
        rest = np.where(copies[rows, top] > 0, np.maximum(max1[:, None] - relief, second[:, None]), max1[:, None])
        peak = np.where(cand, np.maximum(loads[rows, rank][:, None] + gain, rest), np.inf)
        expert = np.where(peak <= peak.min(axis=1)[:, None] + 1e-9, gain, -np.inf).argmax(axis=1)
        place(rank, expert)

    return phy2log
