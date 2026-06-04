"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

"""redundant expert manager."""
from typing import Optional, Tuple

import numpy as np
import paddle
from paddleformers.utils.log import logger

from fastdeploy.eplb.afd_utils import (
    build_afd_redundant_expert_tables,
    get_afd_expert_layout_sizes,
)
from .eplb import rebalance_experts


class RedundantExpertManger:
    """
    RedundantExpertManger
    """

    def __init__(
        self,
        n_routed_experts: int,
        num_hidden_layers: int,
        redundant_experts_num: int,
        ep_size: int,
        enable_afd: bool = False,
    ) -> None:
        """Initialize a redundant expert manager"""
        self.num_expert = n_routed_experts if isinstance(n_routed_experts, int) else n_routed_experts[0]
        self.redundant_experts_num = redundant_experts_num
        self.num_hidden_layers = num_hidden_layers

        self._afd_world_topology = None
        if enable_afd:
            from fastdeploy.model_executor.afd import AFDWorldTopology

            self._afd_world_topology = AFDWorldTopology()

        if self._afd_world_topology is None:
            self.num_replicas = self.num_expert + self.redundant_experts_num
            self.num_nodes = max(ep_size // 8, 8)
            self.num_gpus = ep_size
        else:
            _, _, global_physical_experts = get_afd_expert_layout_sizes(
                num_logical_experts=self.num_expert,
                redundant_experts_num=self.redundant_experts_num,
                world_size=self._afd_world_topology.world_size,
                ffn_ranks=self._afd_world_topology.ffn_ranks,
            )
            self.num_replicas = global_physical_experts
            self.num_nodes = max(len(self._afd_world_topology.ffn_ranks) // 8, 1)
            self.num_gpus = len(self._afd_world_topology.ffn_ranks)
        self.num_groups = 1

        self.export_per_rank = self.num_replicas // ep_size
        assert (
            self.num_replicas % ep_size == 0
        ), f"num_replicas must be divisible by ep_size, \
                but got num_replicas = {self.num_replicas}, ep_size = {ep_size}"

        self.model_ep_rank_to_expert_id_list = paddle.full(
            shape=[
                self.num_hidden_layers,
                self.num_replicas,
            ],
            fill_value=-1,
            dtype="int32",
        )
        self.model_expert_id_to_ep_rank_array = paddle.full(
            shape=[
                self.num_hidden_layers,
                self.num_expert,
                self.redundant_experts_num + 1,
            ],
            fill_value=-1,
            dtype="int32",
        )
        self.model_expert_in_rank_num_list = paddle.full(
            shape=[self.num_hidden_layers, self.num_expert],
            fill_value=0,
            dtype="int32",
        )
        # self.model_ep_rank_to_expert_id_list = paddle.arange(
        #     self.num_expert + self.redundant_experts_num,
        #     dtype="int32").tile([self.num_hidden_layers, 1])
        # self.model_expert_id_to_ep_rank_array = paddle.arange(
        #     self.num_expert,
        #     dtype="int32").reshape([self.num_expert, 1]).tile([self.num_hidden_layers, 1, 1])
        # self.model_expert_in_rank_num_list = paddle.full(
        #     shape=[self.num_hidden_layers, self.num_expert],
        #     fill_value=1,
        #     dtype="int32")

        self.model_tokens_per_expert_stats_list = paddle.ones(
            shape=[self.num_hidden_layers, self.num_expert], dtype="int32"
        )

        rank_expert_list, logical_to_physical_map, expert_count = self._rebalance_experts(
            self.model_tokens_per_expert_stats_list.cpu().numpy()
        )

        self.update_expert_rank_table(rank_expert_list, logical_to_physical_map, expert_count, False)

        logger.info(
            f"moe experts table manager init successfully, ep_size {ep_size} \
            num_replicas {self.num_replicas} export_per_rank {self.export_per_rank}"
        )

    def _rebalance_experts(self, weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._afd_world_topology is None:
            return rebalance_experts(
                weight,
                self.num_replicas,
                self.num_groups,
                self.num_nodes,
                self.num_gpus,
            )

        return build_afd_redundant_expert_tables(
            weight=weight,
            num_replicas=self.num_replicas,
            world_size=self._afd_world_topology.world_size,
            ffn_ranks=self._afd_world_topology.ffn_ranks,
            redundant_experts_num=self.redundant_experts_num,
            num_groups=self.num_groups,
            num_nodes=self.num_nodes,
            num_gpus=self.num_gpus,
        )

    def get_ep_rank_to_expert_id_list_by_layer(
        self, layer_id: int
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        get_ep_rank_to_expert_id_list_by_layer
        """
        return (
            self.model_ep_rank_to_expert_id_list[layer_id],
            self.model_expert_id_to_ep_rank_array[layer_id],
            self.model_expert_in_rank_num_list[layer_id],
            self.model_tokens_per_expert_stats_list[layer_id],
        )

    def get_ep_rank_to_expert_id_list(self) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        get_ep_rank_to_expert_id_list
        """
        return (
            self.model_ep_rank_to_expert_id_list,
            self.model_expert_id_to_ep_rank_array,
            self.model_expert_in_rank_num_list,
        )

    def get_expert_tokens_stats(
        self, verbose: bool = False, clear_stat: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        get_per_expert_tokens_stats
        """
        try:
            if verbose:
                return (
                    self.model_tokens_per_expert_stats_list.cpu().numpy(),
                    self.model_expert_id_to_ep_rank_array.cpu().numpy(),
                    self.model_ep_rank_to_expert_id_list.cpu().numpy(),
                    self.model_expert_in_rank_num_list.cpu().numpy(),
                )
            return (
                self.model_tokens_per_expert_stats_list.cpu().numpy(),
                None,
                None,
                None,
            )
        finally:
            if clear_stat:
                self.model_tokens_per_expert_stats_list.zero_()

    def get_expert_id_to_ep_rank_array(self) -> np.ndarray:
        """
        get_expert_id_to_ep_rank_array
        """
        return self.model_expert_id_to_ep_rank_array.cpu().numpy()

    def update_expert_rank_table(
        self,
        rank_expert_list: np.ndarray,
        logical_to_physical_map: np.ndarray,
        expert_count: np.ndarray,
        clear_stat: bool = True,
    ) -> None:
        """
        update_expert_rank_table
        """
        # update model info
        self.model_ep_rank_to_expert_id_list.copy_(paddle.to_tensor(rank_expert_list), True)
        self.model_expert_id_to_ep_rank_array.fill_(-1)
        self.model_expert_id_to_ep_rank_array[:, :, : logical_to_physical_map.shape[-1]] = paddle.to_tensor(
            logical_to_physical_map
        )
        self.model_expert_in_rank_num_list.copy_(paddle.to_tensor(expert_count), True)

        # reset
        if clear_stat:
            self.model_tokens_per_expert_stats_list.zero_()


if __name__ == "__main__":
    print(RedundantExpertManger(64, 2, 8, 8).model_expert_id_to_ep_rank_array)
