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

from fastdeploy.eplb.fault_tolerance import evict_order, keep_from_order


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


if __name__ == "__main__":
    unittest.main()
