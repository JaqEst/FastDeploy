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

import unittest

import numpy as np

from fastdeploy.eplb.afd_utils import (
    build_afd_redundant_expert_tables,
    get_afd_expert_layout_sizes,
)


class TestAFDEPLBUtils(unittest.TestCase):
    """Test AFD-aware EPLB table helpers."""

    def test_get_afd_expert_layout_sizes(self):
        self.assertEqual(
            get_afd_expert_layout_sizes(
                num_logical_experts=128,
                redundant_experts_num=16,
                world_size=4,
                ffn_ranks=[2, 3],
            ),
            (144, 72, 288),
        )

    def test_build_tables_keep_attn_slots_phantom(self):
        weight = np.ones((3, 128), dtype=np.int32)

        phy2log, log2phy, expert_count = build_afd_redundant_expert_tables(
            weight=weight,
            num_replicas=288,
            world_size=4,
            ffn_ranks=[2, 3],
            redundant_experts_num=16,
            num_groups=1,
            num_nodes=1,
            num_gpus=2,
        )

        self.assertEqual(phy2log.shape, (3, 288))
        self.assertEqual(log2phy.shape, (3, 128, 17))
        self.assertEqual(expert_count.shape, (3, 128))
        self.assertTrue((phy2log[:, :144] == -1).all())
        self.assertEqual(set(phy2log[:, 144:].reshape(-1).tolist()), set(range(128)))
        np.testing.assert_array_equal(expert_count.sum(axis=1), np.full([3], 144, dtype=np.int32))
        self.assertTrue((log2phy[log2phy >= 0] >= 144).all())

    def test_build_tables_reject_mismatched_global_width(self):
        weight = np.ones((3, 128), dtype=np.int32)

        with self.assertRaises(ValueError):
            build_afd_redundant_expert_tables(
                weight=weight,
                num_replicas=144,
                world_size=4,
                ffn_ranks=[2, 3],
                redundant_experts_num=16,
                num_groups=1,
                num_nodes=1,
                num_gpus=2,
            )


if __name__ == "__main__":
    unittest.main()
