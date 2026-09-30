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

from fastdeploy.fault_tolerant.strategy import evict_order


class FaultTolStatsManager:
    """The externally produced per-expert importance and similarity tables."""

    def __init__(self, path, num_layers, num_experts):
        self.path = path
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.importance, self.similarity = self._load(path, num_layers, num_experts)
        self.order, _ = evict_order(self.importance, self.similarity)

    @staticmethod
    def _load(path, num_layers, num_experts):
        """-> (importance [D, L, E], similarity [D, L, E, E])."""
        with np.load(path) as f:
            importance, similarity = f["importance"], f["similarity"]

        if importance.ndim != 3 or importance.shape[1:] != (num_layers, num_experts):
            raise ValueError(
                f"{path}: importance must be [D, {num_layers}, {num_experts}], got {importance.shape}"
            )
        expected = (importance.shape[0], num_layers, num_experts, num_experts)
        if similarity.shape != expected:
            raise ValueError(f"{path}: similarity must be {expected}, got {similarity.shape}")

        diagonal = np.einsum("dluu->dlu", similarity)
        if not np.allclose(diagonal, 1.0):
            raise ValueError(
                f"{path}: similarity needs a unit diagonal, observed range "
                f"[{diagonal.min()}, {diagonal.max()}]"
            )
        return importance, similarity
