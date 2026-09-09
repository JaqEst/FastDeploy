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

import unittest

import numpy as np

from fastdeploy.eplb.fault_tolerance import (
    gpu_imbalance,
    evict_order,
    keep_from_order,
    coverage_repair,
)


def reference_loss(importance, similarity, keep):
    """Loss(K) straight from the definition, one layer."""
    masked = np.where(keep[None, None, :], similarity, -np.inf)
    return float((importance * (1.0 - masked.max(axis=-1))).sum()) / importance.shape[0]


def brute_force_order(importance, similarity, num_evict):
    """Same greedy, but recomputing Loss for every candidate each round."""
    num_experts = importance.shape[1]
    keep = np.ones(num_experts, dtype=bool)
    order = []
    for _ in range(num_evict):
        base = reference_loss(importance, similarity, keep)
        best_expert, best_delta = None, None
        for expert in np.flatnonzero(keep):
            trial = keep.copy()
            trial[expert] = False
            delta = reference_loss(importance, similarity, trial) - base
            if best_delta is None or delta < best_delta - 1e-12:
                best_expert, best_delta = int(expert), delta
        keep[best_expert] = False
        order.append(best_expert)
    return order, keep


def worked_example():
    """
    8 experts, 2 domains. Importance 1 except expert 5 at 2. Similarity 0.05
    plus a unit diagonal and three strong edges: 0-1 at 0.90 and 6-7 at 0.70 in
    both domains, and 2-3 at 0.90 in domain 0 while 2-4 at 0.90 in domain 1, so
    expert 2 has a good stand-in in each domain but a different one.
    """
    num_experts, num_domains = 8, 2
    importance = np.ones((num_domains, 1, num_experts))
    importance[:, 0, 5] = 2.0
    similarity = np.full((num_domains, 1, num_experts, num_experts), 0.05)
    for domain in range(num_domains):
        np.fill_diagonal(similarity[domain, 0], 1.0)
        similarity[domain, 0, 0, 1] = similarity[domain, 0, 1, 0] = 0.90
        similarity[domain, 0, 6, 7] = similarity[domain, 0, 7, 6] = 0.70
    similarity[0, 0, 2, 3] = similarity[0, 0, 3, 2] = 0.90
    similarity[1, 0, 2, 4] = similarity[1, 0, 4, 2] = 0.90
    return importance, similarity


def random_case(rng, num_domains, num_experts):
    importance = np.exp(rng.normal(0.0, 0.8, (num_domains, num_experts)))
    similarity = rng.random((num_domains, num_experts, num_experts)) * 0.7
    similarity = (similarity + similarity.transpose(0, 2, 1)) / 2
    for domain in range(num_domains):
        np.fill_diagonal(similarity[domain], 1.0)
    return importance, similarity


class TestEvictOrder(unittest.TestCase):
    """Test cases for the expendability order in fault_tolerance.py"""

    def test_worked_example_order(self):
        importance, similarity = worked_example()
        order, cost = evict_order(importance, similarity)
        np.testing.assert_array_equal(order[0], [0, 2, 6, 3, 4, 7, 1, 5])
        np.testing.assert_allclose(cost[0][:3], [0.1, 0.1, 0.3], atol=1e-6)
        self.assertEqual(cost[0][-1], np.inf)

    def test_cost_is_non_decreasing(self):
        """Submodularity: marginal cost never falls"""
        importance, similarity = worked_example()
        _, cost = evict_order(importance, similarity)
        finite = cost[0][:-1]
        self.assertTrue(np.all(np.diff(finite) >= -1e-6), f"cost not monotone: {finite}")

    def test_pair_is_not_broken_twice(self):
        """The trap the per-round recompute exists to avoid"""
        importance, similarity = worked_example()
        order, _ = evict_order(importance, similarity)
        position = {int(expert): i for i, expert in enumerate(order[0])}
        self.assertLess(position[0], position[1])
        self.assertLess(position[6], position[7])
        self.assertGreater(position[1], position[3])

    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            num_domains = int(rng.integers(1, 4))
            num_experts = int(rng.integers(4, 12))
            importance, similarity = random_case(rng, num_domains, num_experts)
            order, _ = evict_order(importance[:, None], similarity[:, None])

            num_evict = int(rng.integers(1, num_experts))
            expected_order, expected_keep = brute_force_order(importance, similarity, num_evict)
            got_keep = keep_from_order(order, num_experts - num_evict)[0]
            self.assertAlmostEqual(
                reference_loss(importance, similarity, got_keep),
                reference_loss(importance, similarity, expected_keep),
                places=9,
                msg=f"order={order[0].tolist()} brute={expected_order}",
            )

    def test_prefix_is_the_answer_for_every_capacity(self):
        """One order serves all slot counts"""
        rng = np.random.default_rng(1)
        num_experts = 10
        importance, similarity = random_case(rng, 3, num_experts)
        order, _ = evict_order(importance[:, None], similarity[:, None])
        for num_evict in range(1, num_experts):
            expected, _ = brute_force_order(importance, similarity, num_evict)
            np.testing.assert_array_equal(
                order[0][:num_evict], expected, err_msg=f"num_evict={num_evict}"
            )

    def test_inputs_are_not_modified(self):
        importance, similarity = worked_example()
        importance_before = importance.copy()
        similarity_before = similarity.copy()
        evict_order(importance, similarity)
        np.testing.assert_array_equal(importance, importance_before)
        np.testing.assert_array_equal(similarity, similarity_before)

    def test_float32_input(self):
        importance, similarity = worked_example()
        order64, _ = evict_order(importance, similarity)
        order32, _ = evict_order(importance.astype(np.float32), similarity.astype(np.float32))
        np.testing.assert_array_equal(order64, order32)

    def test_rejects_non_unit_diagonal(self):
        importance, similarity = worked_example()
        similarity[0, 0, 3, 3] = 0.5
        with self.assertRaises(ValueError):
            evict_order(importance, similarity)

    def test_rejects_mismatched_shapes(self):
        importance, similarity = worked_example()
        with self.assertRaises(ValueError):
            evict_order(importance, similarity[:, :, :, :-1])
        with self.assertRaises(ValueError):
            evict_order(importance[0], similarity)

    def test_layers_are_independent(self):
        """
        Compared against per-layer runs rather than by relabelling experts: ties
        break by lowest index, so permuted ids legitimately reorder.
        """
        importance_a, similarity_a = worked_example()
        rng = np.random.default_rng(7)
        importance_b, similarity_b = random_case(rng, importance_a.shape[0], importance_a.shape[2])
        importance_b = importance_b[:, None]
        similarity_b = similarity_b[:, None]

        alone_a, _ = evict_order(importance_a, similarity_a)
        alone_b, _ = evict_order(importance_b, similarity_b)
        stacked, _ = evict_order(
            np.concatenate([importance_a, importance_b], axis=1),
            np.concatenate([similarity_a, similarity_b], axis=1),
        )
        np.testing.assert_array_equal(stacked[0], alone_a[0])
        np.testing.assert_array_equal(stacked[1], alone_b[0])


class TestKeepFromOrder(unittest.TestCase):
    """Test cases for the runtime lookup in fault_tolerance.py"""

    def test_capacity_sweep(self):
        order = np.array([[0, 2, 6, 3, 4, 7, 1, 5]], dtype=np.int16)
        expected = {
            8: [0, 1, 2, 3, 4, 5, 6, 7],
            7: [1, 2, 3, 4, 5, 6, 7],
            6: [1, 3, 4, 5, 6, 7],
            5: [1, 3, 4, 5, 7],
            4: [1, 4, 5, 7],
            2: [1, 5],
            1: [5],
        }
        for capacity, kept in expected.items():
            keep = keep_from_order(order, capacity)
            np.testing.assert_array_equal(np.flatnonzero(keep[0]), kept, err_msg=f"C={capacity}")
            self.assertEqual(int(keep.sum()), capacity)

    def test_capacity_at_or_above_expert_count(self):
        order = np.array([[0, 2, 6, 3, 4, 7, 1, 5]], dtype=np.int16)
        for capacity in (8, 9, 140):
            self.assertTrue(keep_from_order(order, capacity).all())

    def test_rejects_zero_capacity(self):
        order = np.array([[0, 1]], dtype=np.int16)
        with self.assertRaises(ValueError):
            keep_from_order(order, 0)

    def test_layers_use_their_own_order(self):
        order = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int16)
        keep = keep_from_order(order, 2)
        np.testing.assert_array_equal(np.flatnonzero(keep[0]), [1, 2])
        np.testing.assert_array_equal(np.flatnonzero(keep[1]), [0, 1])


class TestCoverageRepair(unittest.TestCase):
    """Test cases for the coverage repair in fault_tolerance.py"""

    def test_spare_replica_is_taken_first(self):
        """
        3 GPUs of 3 slots hold 6 experts plus 3 replicas. Losing GPU 0 leaves
        expert 2 gone and expert 3 holding two slots, so one write restores full
        coverage at no coverage cost.
        """
        keep = np.ones((1, 6), dtype=bool)
        slots = np.array([[3, 4, 0, 5, 1, 3]])
        importance = np.array([[10.0, 9.0, 8.0, 7.0, 2.0, 1.0]])
        load = np.zeros((1, 6))

        moves, new_slots = coverage_repair(keep, slots, importance, load, num_gpus=2)
        self.assertEqual(len(moves), 1)
        layer, slot, expert = moves[0]
        self.assertEqual((layer, expert), (0, 2))
        self.assertEqual(slots[0][slot], 3)
        np.testing.assert_array_equal(np.bincount(new_slots[0], minlength=6), [1, 1, 1, 1, 1, 1])

    def test_last_copy_is_taken_when_no_spare(self):
        """Without replicas a bring-back has to displace an expert keep drops"""
        keep = np.array([[True, True, True, True, False, False]])
        slots = np.array([[2, 3, 4, 5]])
        importance = np.array([[10.0, 9.0, 8.0, 7.0, 2.0, 1.0]])
        load = np.zeros((1, 6))

        moves, new_slots = coverage_repair(keep, slots, importance, load, num_gpus=2)
        self.assertEqual(len(moves), 2)
        # The most important expert is brought back first, the cheapest gives way.
        self.assertEqual([expert for _, _, expert in moves], [0, 1])
        self.assertEqual(sorted(new_slots[0].tolist()), [0, 1, 2, 3])

    def test_coverage_matches_keep(self):
        """After repair the placement holds exactly the kept experts"""
        rng = np.random.default_rng(3)
        num_experts, num_gpus, slots_per_gpu = 16, 4, 5
        capacity = num_gpus * slots_per_gpu
        for _ in range(20):
            slots = rng.integers(0, num_experts, (1, capacity))
            resident = np.bincount(slots[0], minlength=num_experts) > 0
            keep = resident.copy()
            spare = capacity - int(resident.sum())
            missing = np.flatnonzero(~resident)
            keep[missing[: min(spare, missing.size)]] = True
            importance = rng.random((1, num_experts))
            load = rng.random((1, num_experts))

            _, new_slots = coverage_repair(keep[None], slots, importance, load, num_gpus)
            present = np.bincount(new_slots[0], minlength=num_experts) > 0
            np.testing.assert_array_equal(present, keep)

    def test_move_count_is_bounded_by_the_lost_experts(self):
        """A bring-back can only target an expert that is currently absent"""
        rng = np.random.default_rng(4)
        num_experts, num_gpus, slots_per_gpu = 16, 4, 5
        capacity = num_gpus * slots_per_gpu
        for _ in range(20):
            slots = rng.integers(0, num_experts, (1, capacity))
            resident = np.bincount(slots[0], minlength=num_experts) > 0
            keep = resident.copy()
            missing = np.flatnonzero(~resident)
            keep[missing[: capacity - int(resident.sum())]] = True
            importance = rng.random((1, num_experts))

            moves, _ = coverage_repair(
                keep[None], slots, importance, np.zeros((1, num_experts)), num_gpus
            )
            self.assertLessEqual(len(moves), int((~resident).sum()))

    def test_balance_worked_example(self):
        """
        3 GPUs of 3 slots, experts 0 and 1 replicated. GPU loads start at
        15/12/7; one trade brings the peak to the best a 3-slot split allows.
        """
        keep = np.ones((1, 7), dtype=bool)
        slots = np.array([[0, 1, 2, 3, 4, 0, 5, 6, 1]])
        load = np.array([[10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0]])
        importance = np.ones((1, 7))

        before = gpu_imbalance(slots[0], load[0], 3)
        self.assertAlmostEqual(before, 15 / (34 / 3), places=6)

        moves, new_slots = coverage_repair(keep, slots, importance, load, 3, target_imbalance=1.05)
        self.assertEqual(len(moves), 2)
        np.testing.assert_array_equal(new_slots[0], [6, 1, 2, 3, 4, 0, 5, 0, 1])
        # 34 over three GPUs of three slots cannot go below a peak of 12.
        self.assertAlmostEqual(gpu_imbalance(new_slots[0], load[0], 3), 12 / (34 / 3), places=6)

    def test_balance_stops_when_target_is_unreachable(self):
        """No improving trade left is a normal exit, not a hang"""
        keep = np.ones((1, 7), dtype=bool)
        slots = np.array([[0, 1, 2, 3, 4, 0, 5, 6, 1]])
        load = np.array([[10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0]])
        moves, new_slots = coverage_repair(
            keep, slots, np.ones((1, 7)), load, 3, target_imbalance=1.0
        )
        self.assertGreater(gpu_imbalance(new_slots[0], load[0], 3), 1.0)
        self.assertLess(len(moves), 20)

    def test_balance_never_makes_it_worse(self):
        rng = np.random.default_rng(5)
        num_experts, num_gpus, slots_per_gpu = 16, 4, 5
        for _ in range(20):
            slots = rng.integers(0, num_experts, (1, num_gpus * slots_per_gpu))
            keep = np.bincount(slots[0], minlength=num_experts) > 0
            load = np.exp(rng.normal(0.0, 1.0, (1, num_experts)))
            before = gpu_imbalance(slots[0], load[0], num_gpus)
            _, new_slots = coverage_repair(keep[None], slots, np.ones((1, num_experts)), load, num_gpus)
            self.assertLessEqual(gpu_imbalance(new_slots[0], load[0], num_gpus), before + 1e-9)

    def test_zero_load_is_not_a_division(self):
        """Load stats are zeroed after a reset; balancing must just do nothing"""
        keep = np.ones((1, 4), dtype=bool)
        slots = np.array([[0, 1, 2, 3]])
        moves, new_slots = coverage_repair(keep, slots, np.ones((1, 4)), np.zeros((1, 4)), 2)
        self.assertEqual(moves, [])
        np.testing.assert_array_equal(new_slots, slots)
        self.assertEqual(gpu_imbalance(slots[0], np.zeros(4), 2), 1.0)

    def test_inputs_are_not_modified(self):
        keep = np.ones((1, 7), dtype=bool)
        slots = np.array([[0, 1, 2, 3, 4, 0, 5, 6, 1]])
        load = np.array([[10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0]])
        slots_before = slots.copy()
        load_before = load.copy()
        coverage_repair(keep, slots, np.ones((1, 7)), load, 3)
        np.testing.assert_array_equal(slots, slots_before)
        np.testing.assert_array_equal(load, load_before)

    def test_rejects_keep_larger_than_capacity(self):
        keep = np.ones((1, 6), dtype=bool)
        slots = np.array([[0, 1, 2, 3]])
        with self.assertRaises(ValueError):
            coverage_repair(keep, slots, np.ones((1, 6)), np.zeros((1, 6)), 2)

    def test_rejects_capacity_not_divisible_by_gpus(self):
        keep = np.ones((1, 4), dtype=bool)
        slots = np.array([[0, 1, 2, 3]])
        with self.assertRaises(ValueError):
            coverage_repair(keep, slots, np.ones((1, 4)), np.zeros((1, 4)), 3)

    def test_each_slot_is_written_at_most_once(self):
        """
        A trade can move away an expert the refill just put there, so the plan is reduced
        to the slots whose final content differs. The mover can then treat one entry as
        one H2D.
        """
        rng = np.random.default_rng(8)
        num_experts, num_gpus, slots_per_gpu = 16, 4, 5
        slots = rng.integers(0, num_experts, (3, num_gpus * slots_per_gpu))
        keep = np.stack([np.bincount(slots[i], minlength=num_experts) > 0 for i in range(3)])
        load = np.exp(rng.normal(0.0, 1.0, (3, num_experts)))

        moves, new_slots = coverage_repair(keep, slots, np.ones((3, num_experts)), load, num_gpus)
        targets = [(layer, slot) for layer, slot, _ in moves]
        self.assertEqual(len(targets), len(set(targets)))
        self.assertEqual(len(moves), int((slots != new_slots).sum()))
        # Nothing is written back to the value it already held.
        for layer, slot, expert in moves:
            self.assertNotEqual(slots[layer, slot], expert)

    def test_moves_replay_to_the_returned_placement(self):
        """Applying the move list in order reproduces new_slots"""
        rng = np.random.default_rng(6)
        num_experts, num_gpus, slots_per_gpu = 16, 4, 5
        slots = rng.integers(0, num_experts, (2, num_gpus * slots_per_gpu))
        keep = np.stack([np.bincount(slots[i], minlength=num_experts) > 0 for i in range(2)])
        load = np.exp(rng.normal(0.0, 1.0, (2, num_experts)))

        moves, new_slots = coverage_repair(keep, slots, np.ones((2, num_experts)), load, num_gpus)
        replayed = slots.copy()
        for layer, slot, expert in moves:
            replayed[layer, slot] = expert
        np.testing.assert_array_equal(replayed, new_slots)


if __name__ == "__main__":
    unittest.main()

