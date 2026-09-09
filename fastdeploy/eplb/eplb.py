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

from typing import Tuple

import numpy as np


def balanced_packing(
    weight: np.ndarray, num_packs: int, item_group: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.
    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs
        item_group: [X, n], group id per item
    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = np.arange(weight.shape[-1], dtype=np.int32).reshape(1, -1).repeat(num_layers, axis=0)
        rank_in_pack = np.zeros_like(weight, dtype=np.int32)
        return pack_index, rank_in_pack

    if item_group is None:
        item_group = np.arange(num_groups, dtype=np.int32).reshape(1, -1).repeat(num_layers, axis=0)

    indices = np.argsort(-weight.astype(np.float32), axis=-1)
    pack_index = np.empty((num_layers, num_groups), dtype=np.int32)
    rank_in_pack = np.empty((num_layers, num_groups), dtype=np.int32)
    for i in range(num_layers):
        order = indices[i].tolist()
        groups = item_group[i].tolist()
        weights = weight[i].tolist()
        row_pack = [0] * num_groups
        row_rank = [0] * num_groups
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        pack_group_ids = [set() for _ in range(num_packs)]
        # A pack never reopens once full, so drop it instead of re-testing every item.
        open_packs = list(range(num_packs))

        for group in order:
            group_id = groups[group]
            lightest = lightest_spread = -1
            for pack in open_packs:
                pack_weight = pack_weights[pack]
                if lightest < 0 or pack_weight < pack_weights[lightest]:
                    lightest = pack
                if group_id not in pack_group_ids[pack] and (
                    lightest_spread < 0 or pack_weight < pack_weights[lightest_spread]
                ):
                    lightest_spread = pack
            # Co-locate only when no pack with room is free of this group.
            pack = lightest_spread if lightest_spread >= 0 else lightest

            row_pack[group] = pack
            row_rank[group] = pack_items[pack]
            pack_weights[pack] += weights[group]
            pack_items[pack] += 1
            pack_group_ids[pack].add(group_id)
            if pack_items[pack] == groups_per_pack:
                open_packs.remove(pack)

        pack_index[i] = row_pack
        rank_in_pack[i] = row_rank
    return pack_index, rank_in_pack


def replicate_experts(
    weight: np.ndarray, num_phy: int, max_replicas: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.
    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication
        max_replicas: cap on replicas per logical expert
    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    cap = max_replicas if max_replicas is not None and num_phy <= num_log * max_replicas else None
    phy2log = np.arange(num_phy, dtype=np.int32).reshape(1, -1).repeat(n, axis=0)
    rank = np.zeros((n, num_phy), dtype=np.int32)
    logcnt = np.ones((n, num_log), dtype=np.int32)
    arangen = np.arange(n, dtype=np.int32)
    score = np.empty((n, num_log), dtype=np.float64)
    for i in range(num_log, num_phy):
        np.divide(weight, logcnt, out=score)
        if cap is not None:
            score[logcnt >= cap] = -np.inf
        redundant_indices = np.argmax(score, axis=-1)
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_intra_node(
    weight: np.ndarray,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`
    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """

    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    num_redundant_experts = num_physical_experts - num_logical_experts
    assert num_redundant_experts >= 0

    assert num_gpus % num_nodes == 0
    num_gpus_per_node = num_gpus // num_nodes

    assert num_physical_experts % num_gpus == 0
    num_physical_experts_per_gpu = num_physical_experts // num_gpus
    assert num_physical_experts % num_nodes == 0
    num_physical_experts_per_node = num_physical_experts // num_nodes

    assert num_logical_experts % num_physical_experts_per_node == 0
    # num_logical_nodes = num_logical_experts // num_physical_experts_per_node
    assert num_redundant_experts % num_physical_experts_per_node == 0
    # num_redundant_nodes = num_redundant_experts // num_physical_experts_per_node

    def inverse(perm: np.ndarray) -> np.ndarray:
        inv = np.empty_like(perm)
        inv[np.arange(perm.shape[0])[:, None], perm] = np.arange(perm.shape[1], dtype=np.int32).reshape(1, -1)
        return inv

    # Step 1: generate redundant experts by weight.
    # shape of tmp2log, tmprank is [num_layers, num_physical_experts]
    # shape of logcnt is [num_layers, num_logical_experts]
    tmp2log, tmprank, logcnt = replicate_experts(weight, num_physical_experts, num_gpus_per_node)

    # Step 2: compute num_tokens of physical experts
    # shape of tokens_per_tmp is [num_layers * num_nodes, num_physical_experts_per_node]
    tokens_per_tmp = np.take_along_axis(weight / logcnt, tmp2log, axis=-1).reshape(-1, num_physical_experts_per_node)

    # STEP 3: take load balance of gpu cards in node
    # shape of gpu_index, rank_in_gpu, tmp2phy, phy2tmp is [num_layers * num_nodes, num_physical_experts_per_node]
    tmp2log_per_node = tmp2log.reshape(-1, num_physical_experts_per_node)
    gpu_index, rank_in_gpu = balanced_packing(tokens_per_tmp, num_gpus_per_node, tmp2log_per_node)
    tmp2phy = gpu_index * num_physical_experts_per_gpu + rank_in_gpu
    phy2tmp = inverse(tmp2phy)

    # STEP 4: generate final phy2log mapping
    tmp2log = tmp2log.reshape(-1, num_physical_experts_per_node)
    tmprank = tmprank.reshape(-1, num_physical_experts_per_node)
    phy2log = np.take_along_axis(tmp2log, phy2tmp, axis=-1).reshape(-1, num_physical_experts)
    phyrank = np.take_along_axis(tmprank, phy2tmp, axis=-1).reshape(-1, num_physical_experts)
    return phy2log, phyrank, logcnt


def rebalance_experts_hierarchical(
    weight: np.ndarray,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`
    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: np.ndarray) -> np.ndarray:
        inv = np.empty_like(perm)
        inv[np.arange(perm.shape[0])[:, None], perm] = np.arange(perm.shape[1], dtype=np.int32).reshape(1, -1)
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.reshape(num_layers, num_groups, group_size).sum(axis=-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (
        ((group_pack_index * groups_per_node + group_rank_in_pack) * group_size)[:, :, None]
        + np.arange(group_size, dtype=np.int32)
    ).reshape(num_layers, -1)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    tokens_per_mlog = np.take_along_axis(weight, mlog2log, axis=-1).reshape(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes, num_gpus // num_nodes
    )

    # Step 3: pack physical_experts to GPUs
    tokens_per_phy = np.take_along_axis(tokens_per_mlog / mlogcnt, phy2mlog, axis=-1)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes, phy2mlog)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = np.take_along_axis(phy2mlog, pphy2phy, axis=-1)  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (
        pphy2mlog.reshape(num_layers, num_nodes, -1)
        + np.arange(0, num_logical_experts, num_logical_experts // num_nodes, dtype=np.int32).reshape(1, -1, 1)
    ).reshape(num_layers, -1)
    pphy2log = np.take_along_axis(mlog2log, pphy2mlog, axis=-1)
    pphyrank = np.take_along_axis(phyrank, pphy2phy, axis=-1).reshape(num_layers, -1)
    logcnt = np.take_along_axis(mlogcnt.reshape(num_layers, -1), log2mlog, axis=-1)
    return pphy2log, pphyrank, logcnt


def _rebalance_experts(
    weight: np.ndarray,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    eplb_strategy: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Entry point for expert-parallelism load balancer.
    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`
    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.astype(np.float32)
    if eplb_strategy == "balance_intra_node":
        phy2log, phyrank, logcnt = rebalance_experts_intra_node(weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        if num_groups % num_nodes == 0:
            # use hierarchical load-balance policy
            phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
                weight, num_replicas, num_groups, num_nodes, num_gpus
            )
        else:
            # use global load-balance policy
            phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
                weight, num_replicas, 1, 1, num_gpus
            )
    maxlogcnt = logcnt.max()
    log2phy = np.full((num_layers, num_logical_experts, maxlogcnt), -1, dtype=np.int32)
    np.put_along_axis(
        log2phy.reshape(num_layers, -1)[:, :, None],
        (phy2log * maxlogcnt + phyrank)[:, :, None],
        np.arange(num_replicas, dtype=np.int32).reshape(1, -1).repeat(num_layers, axis=0)[:, :, None],
        axis=1,
    )
    return phy2log, log2phy, logcnt


def rebalance_experts(
    weight: np.ndarray,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    eplb_strategy: str = "",
    fd_config=None,
    active_ffn_ranks: list = None,
    keep: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    afd_config = fd_config.afd_config if fd_config is not None else None
    if afd_config is None or not afd_config.enable_afd:
        return _rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus, eplb_strategy)
    return _rebalance_experts_afd(weight, num_groups, num_nodes, eplb_strategy, afd_config, active_ffn_ranks, keep)


def _rebalance_experts_afd(
    weight: np.ndarray,
    num_groups: int,
    num_nodes: int,
    eplb_strategy: str,
    afd_config,
    active_ffn_ranks: list = None,
    keep: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve on the surviving FFN geometry, then lift the result into the global slot space.

    `active_ffn_ranks` and `keep` default to the full layout, so the healthy path is the
    same code with nothing masked out.
    """
    num_layers, num_logical_experts = weight.shape
    ffn_ranks = list(afd_config.ffn_ranks if active_ffn_ranks is None else active_ffn_ranks)
    local_physical_experts = afd_config.num_local_physical_experts
    ffn_slots = local_physical_experts * len(ffn_ranks)

    # eplb only replicates, never drops, so the logical set has to fit before we solve.
    num_kept = keep.sum(axis=1) if keep is not None else np.full(num_layers, num_logical_experts)
    if num_kept.min() != num_kept.max() or num_kept[0] > ffn_slots:
        raise ValueError(
            f"keep must hold one count per layer and fit {ffn_slots} slots, "
            f"got {num_kept.min()}..{num_kept.max()}"
        )

    kept_ids = None
    if keep is not None and not keep.all():
        kept_ids = np.stack([np.flatnonzero(row) for row in keep]).astype(np.int32)
        weight = np.take_along_axis(weight, kept_ids, axis=1)

    phy2log, log2phy, expert_count = _rebalance_experts(
        weight, ffn_slots, num_groups, num_nodes, len(ffn_ranks), eplb_strategy
    )

    if kept_ids is not None:
        rows = np.arange(num_layers)[:, None]
        phy2log = np.take_along_axis(kept_ids, phy2log, axis=1)
        wide_log2phy = np.full((num_layers, num_logical_experts, log2phy.shape[-1]), -1, dtype=np.int32)
        wide_log2phy[rows, kept_ids] = log2phy
        wide_count = np.zeros((num_layers, num_logical_experts), dtype=expert_count.dtype)
        wide_count[rows, kept_ids] = expert_count
        log2phy, expert_count = wide_log2phy, wide_count

    # FFN slot i lives on ffn_ranks[i // local]; attn ranks and lost ranks keep their -1.
    rank_base = np.asarray(ffn_ranks, np.int32)[:, None] * local_physical_experts
    to_global = (rank_base + np.arange(local_physical_experts)).ravel()
    global_phy2log = np.full((num_layers, afd_config.num_physical_experts), -1, dtype=np.int32)
    global_phy2log[:, to_global] = phy2log
    global_log2phy = np.where(log2phy >= 0, to_global[np.maximum(log2phy, 0)], -1).astype(np.int32)
    padding = afd_config.num_redundant_experts + 1 - global_log2phy.shape[-1]
    if padding:
        global_log2phy = np.pad(global_log2phy, ((0, 0), (0, 0), (0, padding)), constant_values=-1)
    return global_phy2log, global_log2phy, expert_count


__all__ = ["rebalance_experts"]


def main():
    """
    main
    """
    num_hidden_layers = 3
    num_expert = 64
    num_groups = 8

    num_replicas = 64
    num_nodes = 4
    num_gpus = 4 * 8

    model_tokens_per_expert_stats_list = np.random.randint(low=1, high=10, size=(num_hidden_layers, num_expert))

    phy2log, phyrank, logcnt = rebalance_experts(
        model_tokens_per_expert_stats_list,
        num_replicas,
        num_groups,
        num_nodes,
        num_gpus,
    )
    print(phy2log)
    print(phyrank)
    print(logcnt)


if __name__ == "__main__":
    main()
