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

import types
import unittest

import numpy as np

from fastdeploy.eplb.eplb import (
    _rebalance_experts,
    balanced_packing,
    rebalance_experts,
    rebalance_experts_hierarchical,
    rebalance_experts_intra_node,
    replicate_experts,
)


def afd_fd_config(num_logical=8, num_redundant=4, ffn_ranks=(2, 3, 4), afd_world_size=5):
    """Minimal stand-in for FDConfig.afd_config in the AFD placement branch."""
    local_physical = (num_logical + num_redundant) // len(ffn_ranks)
    return types.SimpleNamespace(
        afd_config=types.SimpleNamespace(
            enable_afd=True,
            num_logical_experts=num_logical,
            num_redundant_experts=num_redundant,
            ffn_ranks=list(ffn_ranks),
            num_ffn_physical_experts=num_logical + num_redundant,
            num_local_physical_experts=local_physical,
            num_physical_experts=local_physical * afd_world_size,
        )
    )


def replica_ranks(phy2log, layer, expert, local_physical):
    """Which GPU ranks hold a replica of `expert`, from a global phy2log row."""
    slots = np.flatnonzero(phy2log[layer] == expert)
    return sorted({int(slot) // local_physical for slot in slots})


class TestEplb(unittest.TestCase):
    """Test cases for eplb.py"""

    def test_balanced_packing_simple(self):
        """Test balanced_packing with simple case"""
        # Test case with 4 items and 2 packs
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_packs = 2

        pack_index, rank_in_pack = balanced_packing(weight, num_packs)

        expected_pack_index = np.array([[0, 1, 1, 0]], dtype=np.int32)
        expected_rank_in_pack = np.array([[1, 1, 0, 0]], dtype=np.int32)

        np.testing.assert_array_equal(pack_index, expected_pack_index)
        np.testing.assert_array_equal(rank_in_pack, expected_rank_in_pack)

    def test_balanced_packing_single_pack(self):
        """Test balanced_packing with single pack"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_packs = 4  # Each pack gets exactly one item

        pack_index, rank_in_pack = balanced_packing(weight, num_packs)

        expected_pack_index = np.array([[0, 1, 2, 3]], dtype=np.int32)
        expected_rank_in_pack = np.array([[0, 0, 0, 0]], dtype=np.int32)

        np.testing.assert_array_equal(pack_index, expected_pack_index)
        np.testing.assert_array_equal(rank_in_pack, expected_rank_in_pack)

    def test_balanced_packing_multiple_layers(self):
        """Test balanced_packing with multiple layers"""
        weight = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.float32)
        num_packs = 2

        pack_index, rank_in_pack = balanced_packing(weight, num_packs)

        # Verify shape
        self.assertEqual(pack_index.shape, (2, 4))
        self.assertEqual(rank_in_pack.shape, (2, 4))

        # Verify that each pack gets exactly 2 items per layer
        for layer_idx in range(2):
            unique_packs, counts = np.unique(pack_index[layer_idx], return_counts=True)
            np.testing.assert_array_equal(counts, [2, 2])

    def test_replicate_experts_no_redundancy(self):
        """Test replicate_experts with no redundant experts"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_phy = 4  # Same as number of logical experts

        phy2log, rank, logcnt = replicate_experts(weight, num_phy)

        expected_phy2log = np.array([[0, 1, 2, 3]], dtype=np.int32)
        expected_rank = np.array([[0, 0, 0, 0]], dtype=np.int32)
        expected_logcnt = np.array([[1, 1, 1, 1]], dtype=np.int32)

        np.testing.assert_array_equal(phy2log, expected_phy2log)
        np.testing.assert_array_equal(rank, expected_rank)
        np.testing.assert_array_equal(logcnt, expected_logcnt)

    def test_replicate_experts_with_redundancy(self):
        """Test replicate_experts with redundant experts"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_phy = 6  # 2 redundant experts

        phy2log, rank, logcnt = replicate_experts(weight, num_phy)

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 6))
        self.assertEqual(rank.shape, (1, 6))
        self.assertEqual(logcnt.shape, (1, 4))

        # Verify that each logical expert has correct count
        expected_logcnt = np.array([[1, 1, 2, 2]], dtype=np.int32)  # Heaviest and lightest get replicated
        np.testing.assert_array_equal(logcnt, expected_logcnt)

    def test_rebalance_experts_intra_node(self):
        """Test rebalance_experts_intra_node function"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_physical_experts = 4
        num_groups = 1
        num_nodes = 1
        num_gpus = 1

        phy2log, phyrank, logcnt = rebalance_experts_intra_node(
            weight, num_physical_experts, num_groups, num_nodes, num_gpus
        )

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 4))
        self.assertEqual(phyrank.shape, (1, 4))
        self.assertEqual(logcnt.shape, (1, 4))

    def test_rebalance_experts_hierarchical(self):
        """Test rebalance_experts_hierarchical function"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_physical_experts = 4
        num_groups = 2
        num_nodes = 1
        num_gpus = 1

        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_physical_experts, num_groups, num_nodes, num_gpus
        )

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 4))
        self.assertEqual(phyrank.shape, (1, 4))
        self.assertEqual(logcnt.shape, (1, 4))

    def test_rebalance_experts_balance_intra_node(self):
        """Test rebalance_experts with balance_intra_node strategy"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_replicas = 4
        num_groups = 1
        num_nodes = 1
        num_gpus = 1

        phy2log, log2phy, logcnt = rebalance_experts(
            weight, num_replicas, num_groups, num_nodes, num_gpus, "balance_intra_node"
        )

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 4))
        self.assertEqual(log2phy.shape, (1, 4, 1))  # maxlogcnt = 1 when no redundancy
        self.assertEqual(logcnt.shape, (1, 4))

    def test_rebalance_experts_hierarchical_strategy(self):
        """Test rebalance_experts with hierarchical strategy"""
        weight = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float32)
        num_replicas = 8
        num_groups = 4  # Divisible by num_nodes
        num_nodes = 2
        num_gpus = 4

        phy2log, log2phy, logcnt = rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 8))
        self.assertEqual(log2phy.shape, (1, 8, 1))  # maxlogcnt = 1 when no redundancy
        self.assertEqual(logcnt.shape, (1, 8))

    def test_rebalance_experts_global_strategy(self):
        """Test rebalance_experts with global strategy (groups not divisible by nodes)"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_replicas = 4
        num_groups = 3  # Not divisible by num_nodes
        num_nodes = 2
        num_gpus = 2

        phy2log, log2phy, logcnt = rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 4))
        self.assertEqual(log2phy.shape, (1, 4, 1))
        self.assertEqual(logcnt.shape, (1, 4))

    def test_rebalance_experts_with_redundancy(self):
        """Test rebalance_experts with redundant experts"""
        weight = np.array([[1, 2, 3, 4]], dtype=np.float32)
        num_replicas = 6  # 2 redundant experts
        num_groups = 1
        num_nodes = 1
        num_gpus = 1

        phy2log, log2phy, logcnt = rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

        # Verify shape
        self.assertEqual(phy2log.shape, (1, 6))
        self.assertEqual(log2phy.shape, (1, 4, 2))  # maxlogcnt = 2 with redundancy
        self.assertEqual(logcnt.shape, (1, 4))

        # Verify that logical expert counts sum to num_replicas
        self.assertEqual(logcnt.sum(), num_replicas)

    def test_edge_cases(self):
        """Test edge cases for rebalance_experts"""
        # Test with all zero weights
        weight = np.zeros((2, 4), dtype=np.float32)
        num_replicas = 4
        num_groups = 1
        num_nodes = 1
        num_gpus = 1

        phy2log, log2phy, logcnt = rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

        # Should still produce valid results
        self.assertEqual(phy2log.shape, (2, 4))
        self.assertEqual(log2phy.shape, (2, 4, 1))
        self.assertEqual(logcnt.shape, (2, 4))

    def test_large_scale(self):
        """Test with larger scale parameters"""
        num_layers = 10
        num_experts = 64
        weight = np.random.randint(1, 100, size=(num_layers, num_experts)).astype(np.float32)
        num_replicas = 64
        num_groups = 8
        num_nodes = 4
        num_gpus = 32

        phy2log, log2phy, logcnt = rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)

        # Verify shape
        self.assertEqual(phy2log.shape, (num_layers, num_replicas))
        self.assertEqual(log2phy.shape[0], num_layers)
        self.assertEqual(log2phy.shape[1], num_experts)
        self.assertEqual(logcnt.shape, (num_layers, num_experts))

        # Verify that logical expert counts sum to num_replicas for each layer
        for layer_idx in range(num_layers):
            self.assertEqual(logcnt[layer_idx].sum(), num_replicas)


class TestAntiAffinity(unittest.TestCase):
    """Test cases for keeping replicas of one expert on different GPUs"""

    def test_balanced_packing_keeps_groups_apart(self):
        """
        On the shape that ships, every replicated expert gets distinct packs.

        Modelled on what `rebalance_experts_hierarchical` actually hands over: 8 GPUs of 18
        slots, 128 logical experts of which the 16 heaviest are replicated, and replicas of
        one expert carrying an equal share of its load.
        """
        num_packs, items_per_pack, num_experts = 8, 18, 128
        num_replicated = num_packs * items_per_pack - num_experts
        rng = np.random.default_rng(0)
        expert_weight = np.exp(rng.normal(0.0, 1.0, num_experts))
        replicated = np.argsort(-expert_weight)[:num_replicated]

        item_group, weight = [], []
        for expert in range(num_experts):
            replicas = 2 if expert in set(replicated.tolist()) else 1
            for _ in range(replicas):
                item_group.append(expert)
                weight.append(expert_weight[expert] / replicas)
        item_group = np.array(item_group)
        weight = np.array(weight)[None, :]
        self.assertEqual(weight.shape[1], num_packs * items_per_pack)

        pack_index, _ = balanced_packing(weight, num_packs, item_group[None, :])
        for group_id in np.unique(item_group):
            packs = pack_index[0][item_group == group_id]
            self.assertEqual(
                len(set(packs.tolist())),
                len(packs),
                msg=f"expert {group_id} co-located: packs={packs.tolist()}",
            )

    def test_balanced_packing_stays_valid_under_tight_packing(self):
        """
        Spreading is best effort. The greedy fills packs in weight order, so a tight packing
        can leave a group's last item with only its own pack open; the fallback then
        co-locates rather than failing. Pack sizes and ranks must stay correct regardless.
        """
        rng = np.random.default_rng(0)
        for _ in range(30):
            num_packs = int(rng.integers(2, 5))
            items_per_pack = int(rng.integers(2, 5))
            num_items = num_packs * items_per_pack
            group_size = int(rng.integers(1, num_packs + 1))
            item_group = np.repeat(np.arange((num_items + group_size - 1) // group_size), group_size)[:num_items]
            rng.shuffle(item_group)
            weight = rng.random((1, num_items)) * 10

            pack_index, rank_in_pack = balanced_packing(weight, num_packs, item_group[None, :])
            _, counts = np.unique(pack_index[0], return_counts=True)
            np.testing.assert_array_equal(counts, [items_per_pack] * num_packs)
            for pack in range(num_packs):
                ranks = sorted(rank_in_pack[0][pack_index[0] == pack].tolist())
                self.assertEqual(ranks, list(range(items_per_pack)))

    def test_balanced_packing_allows_colocation_when_unavoidable(self):
        """A group with more items than packs still produces a valid packing"""
        weight = np.array([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
        item_group = np.array([[0, 0, 0, 0, 1, 1]])
        pack_index, rank_in_pack = balanced_packing(weight, 2, item_group)

        _, counts = np.unique(pack_index[0], return_counts=True)
        np.testing.assert_array_equal(counts, [3, 3])
        for pack in range(2):
            ranks = sorted(rank_in_pack[0][pack_index[0] == pack].tolist())
            self.assertEqual(ranks, [0, 1, 2])

    def test_balanced_packing_ignores_group_when_one_item_per_pack(self):
        weight = np.array([[1.0, 2.0, 3.0, 4.0]])
        item_group = np.array([[0, 0, 0, 0]])
        pack_index, _ = balanced_packing(weight, 4, item_group)
        np.testing.assert_array_equal(pack_index, [[0, 1, 2, 3]])

    def test_afd_replicas_spread_over_ranks(self):
        """Replicas of one logical expert do not share a GPU, so one loss cannot take both"""
        fd_config = afd_fd_config()
        local_physical = fd_config.afd_config.num_local_physical_experts
        num_ranks = len(fd_config.afd_config.ffn_ranks)
        rng = np.random.default_rng(1)
        weight = rng.random((6, 8)) * 100

        phy2log, _, logcnt = rebalance_experts(
            weight,
            fd_config.afd_config.num_physical_experts,
            1,
            1,
            num_ranks,
            fd_config=fd_config,
        )
        for layer in range(weight.shape[0]):
            for expert in range(weight.shape[1]):
                if logcnt[layer, expert] <= num_ranks:
                    ranks = replica_ranks(phy2log, layer, expert, local_physical)
                    self.assertEqual(
                        len(ranks),
                        int(logcnt[layer, expert]),
                        msg=f"layer {layer} expert {expert} shares a rank: {ranks}",
                    )


    def test_replicate_experts_caps_replicas(self):
        """
        Replicas beyond the GPU count have to share a GPU with a sibling, which splits no
        load while costing HBM, weight reads and a GEMM group. The budget goes elsewhere.
        """
        weight = np.exp(np.random.default_rng(0).normal(0.0, 2.0, (20, 16)))
        _, _, uncapped = replicate_experts(weight, 16 + 48)
        _, _, capped = replicate_experts(weight, 16 + 48, 4)

        self.assertGreater(int(uncapped.max()), 4)
        self.assertLessEqual(int(capped.max()), 4)
        # The budget is still fully spent, just spread wider.
        self.assertEqual(int(capped.sum(axis=1)[0]), 16 + 48)

    def test_replicate_experts_uncapped_by_default(self):
        weight = np.exp(np.random.default_rng(0).normal(0.0, 2.0, (20, 16)))
        _, _, logcnt = replicate_experts(weight, 16 + 48)
        self.assertGreater(int(logcnt.max()), 4)

    def test_replicate_experts_cap_yields_when_slots_exceed_capacity(self):
        """More slots than num_log * cap: the cap has to give, but placement still completes"""
        weight = np.array([[4.0, 3.0, 2.0, 1.0]])
        phy2log, _, logcnt = replicate_experts(weight, 12, 2)
        self.assertEqual(int(logcnt.sum()), 12)
        self.assertEqual(sorted(np.unique(phy2log[0]).tolist()), [0, 1, 2, 3])

    def test_afd_no_duplicate_experts_on_one_rank(self):
        """Skewed load must not park two replicas of one expert on the same GPU"""
        fd_config = afd_fd_config(num_logical=8, num_redundant=16, ffn_ranks=(2, 3, 4, 5), afd_world_size=6)
        afd = fd_config.afd_config
        local_physical = afd.num_local_physical_experts
        weight = np.exp(np.random.default_rng(2).normal(0.0, 3.0, (10, 8)))

        phy2log, _, _ = rebalance_experts(
            weight, afd.num_physical_experts, 1, 1, len(afd.ffn_ranks), fd_config=fd_config
        )
        for layer in range(weight.shape[0]):
            for rank in afd.ffn_ranks:
                block = phy2log[layer, rank * local_physical : (rank + 1) * local_physical]
                self.assertEqual(
                    len(set(block.tolist())),
                    len(block),
                    msg=f"layer {layer} rank {rank} holds duplicates: {block.tolist()}",
                )


    def test_global_policy_balances_and_spreads(self):
        """
        num_groups % num_nodes != 0 takes the global policy. It used to call
        replicate_experts directly, which left GPU assignment to slot index order: no load
        balancing and no anti-affinity. It now goes through the hierarchical solver with
        grouping switched off.
        """
        num_logical, num_replicas, num_gpus = 12, 16, 4
        per_gpu = num_replicas // num_gpus
        weight = np.exp(np.random.default_rng(0).normal(0.0, 2.0, (8, num_logical)))

        phy2log, _, logcnt = _rebalance_experts(weight, num_replicas, 3, 2, num_gpus)

        for layer in range(weight.shape[0]):
            for gpu in range(num_gpus):
                block = phy2log[layer, gpu * per_gpu : (gpu + 1) * per_gpu]
                self.assertEqual(
                    len(set(block.tolist())), len(block), msg=f"gpu {gpu} holds duplicates: {block.tolist()}"
                )
            # Load per GPU, with an expert's load split over its replicas.
            per_slot = weight[layer, phy2log[layer]] / logcnt[layer, phy2log[layer]]
            gpu_load = per_slot.reshape(num_gpus, per_gpu).sum(axis=1)
            self.assertLess(gpu_load.max() / gpu_load.mean(), 1.5)


class TestSurvivingGeometry(unittest.TestCase):
    """Test cases for placing onto a reduced set of FFN ranks"""

    def test_dead_rank_slots_stay_empty(self):
        fd_config = afd_fd_config()
        afd = fd_config.afd_config
        local_physical = afd.num_local_physical_experts
        weight = np.random.default_rng(2).random((3, 8)) * 10

        phy2log, log2phy, logcnt = rebalance_experts(
            weight, afd.num_physical_experts, 1, 1, 2, fd_config=fd_config, active_ffn_ranks=[2, 3]
        )
        dead = slice(4 * local_physical, 5 * local_physical)
        self.assertTrue((phy2log[:, dead] == -1).all())
        # The two survivors hold every slot, and no replica points at the dead rank.
        self.assertEqual(int((phy2log[0] >= 0).sum()), 2 * local_physical)
        self.assertEqual(int(logcnt[0].sum()), 2 * local_physical)
        self.assertTrue(((log2phy < 4 * local_physical) | (log2phy == -1)).all())

    def test_attn_rank_slots_stay_empty(self):
        """Only FFN ranks own expert slots; the attn range is never written"""
        fd_config = afd_fd_config()
        afd = fd_config.afd_config
        local_physical = afd.num_local_physical_experts
        weight = np.random.default_rng(3).random((2, 8)) * 10

        phy2log, _, _ = rebalance_experts(weight, afd.num_physical_experts, 1, 1, 3, fd_config=fd_config)
        self.assertTrue((phy2log[:, : 2 * local_physical] == -1).all())

    def test_keep_drops_experts_with_empty_rows(self):
        """A dropped expert gets count 0 and a -1 row, which the kernel reads as a zero gate"""
        fd_config = afd_fd_config()
        afd = fd_config.afd_config
        weight = np.random.default_rng(4).random((3, 8)) * 10
        keep = np.ones((3, 8), dtype=bool)
        keep[:, [1, 6]] = False

        phy2log, log2phy, logcnt = rebalance_experts(
            weight, afd.num_physical_experts, 1, 1, 3, fd_config=fd_config, keep=keep
        )
        self.assertTrue((logcnt[:, [1, 6]] == 0).all())
        self.assertTrue((log2phy[:, [1, 6]] == -1).all())
        self.assertFalse(np.isin(phy2log, [1, 6]).any())
        # Everything kept is placed at least once.
        for expert in [0, 2, 3, 4, 5, 7]:
            self.assertTrue((logcnt[:, expert] >= 1).all())

    def test_fewer_slots_than_experts(self):
        """One surviving rank cannot hold all 8 experts, so keep must shrink the set first"""
        fd_config = afd_fd_config()
        afd = fd_config.afd_config
        local_physical = afd.num_local_physical_experts
        weight = np.random.default_rng(5).random((2, 8)) * 10
        keep = np.zeros((2, 8), dtype=bool)
        keep[:, :local_physical] = True

        phy2log, log2phy, logcnt = rebalance_experts(
            weight,
            afd.num_physical_experts,
            1,
            1,
            1,
            fd_config=fd_config,
            active_ffn_ranks=[3],
            keep=keep,
        )
        placed = phy2log[0][phy2log[0] >= 0]
        self.assertEqual(sorted(placed.tolist()), list(range(local_physical)))
        self.assertEqual(int(logcnt[0].sum()), local_physical)
        owning_ranks = {int(slot) // local_physical for slot in placed_slots(phy2log, 0)}
        self.assertEqual(owning_ranks, {3})

    def test_rejects_keep_that_cannot_fit(self):
        fd_config = afd_fd_config()
        afd = fd_config.afd_config
        weight = np.random.default_rng(6).random((2, 8)) * 10
        with self.assertRaises(ValueError):
            rebalance_experts(
                weight,
                afd.num_physical_experts,
                1,
                1,
                1,
                fd_config=fd_config,
                active_ffn_ranks=[3],
                keep=np.ones((2, 8), dtype=bool),
            )

    def test_rejects_ragged_keep(self):
        fd_config = afd_fd_config()
        afd = fd_config.afd_config
        weight = np.random.default_rng(7).random((2, 8)) * 10
        keep = np.ones((2, 8), dtype=bool)
        keep[0, 0] = False
        with self.assertRaises(ValueError):
            rebalance_experts(weight, afd.num_physical_experts, 1, 1, 3, fd_config=fd_config, keep=keep)

    def test_defaults_match_full_geometry(self):
        """Passing neither new argument reproduces the untouched placement"""
        fd_config = afd_fd_config()
        afd = fd_config.afd_config
        weight = np.random.default_rng(9).random((4, 8)) * 10

        baseline = rebalance_experts(weight, afd.num_physical_experts, 1, 1, 3, fd_config=fd_config)
        explicit = rebalance_experts(
            weight,
            afd.num_physical_experts,
            1,
            1,
            3,
            fd_config=fd_config,
            active_ffn_ranks=afd.ffn_ranks,
            keep=np.ones((4, 8), dtype=bool),
        )
        for got, expected in zip(explicit, baseline):
            np.testing.assert_array_equal(got, expected)


def placed_slots(phy2log, layer):
    return np.flatnonzero(phy2log[layer] >= 0)


if __name__ == "__main__":
    unittest.main()
