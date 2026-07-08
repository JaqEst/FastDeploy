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

from fastdeploy.eplb.eplb import rebalance_experts
from fastdeploy.eplb.utils import dump_redundant_expert_table_snapshot


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
        fd_config=None,
    ) -> None:
        """Initialize a redundant expert manager"""
        self.num_expert = n_routed_experts if isinstance(n_routed_experts, int) else n_routed_experts[0]
        self.redundant_experts_num = redundant_experts_num
        self.num_hidden_layers = num_hidden_layers
        self.ep_size = ep_size

        self.fd_config = fd_config

        if fd_config is None or not fd_config.afd_config.enable_afd:
            self.num_replicas = self.num_expert + self.redundant_experts_num
            self.num_nodes = max(ep_size // 8, 8)
            self.num_gpus = ep_size
        else:
            self.num_replicas = fd_config.afd_config.afd_num_physical_experts
            self.num_nodes = max(len(fd_config.afd_config.afd_ffn_ranks) // 8, 1)
            self.num_gpus = len(fd_config.afd_config.afd_ffn_ranks)
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
        self.model_active_expert_id_to_ep_rank_array = self.model_expert_id_to_ep_rank_array.clone()
        self.model_active_expert_in_rank_num_list = self.model_expert_in_rank_num_list.clone()
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
        return rebalance_experts(
            weight=weight,
            num_replicas=self.num_replicas,
            num_groups=self.num_groups,
            num_nodes=self.num_nodes,
            num_gpus=self.num_gpus,
            fd_config=self.fd_config,
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

    def get_active_ep_rank_to_expert_id_list_by_layer(
        self, layer_id: int
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """
        get_active_ep_rank_to_expert_id_list_by_layer
        """
        return (
            self.model_ep_rank_to_expert_id_list[layer_id],
            self.model_active_expert_id_to_ep_rank_array[layer_id],
            self.model_active_expert_in_rank_num_list[layer_id],
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
        if self.fd_config is not None and clear_stat:
            try:
                dump_path, snapshot = dump_redundant_expert_table_snapshot(
                    fd_config=self.fd_config,
                    rank_expert_list=self.model_ep_rank_to_expert_id_list,
                    logical_to_physical_map=self.model_expert_id_to_ep_rank_array,
                    expert_count=self.model_expert_in_rank_num_list,
                    source="model_table_before",
                    local_rank=self.fd_config.parallel_config.expert_parallel_rank,
                    clear_stat=clear_stat,
                )
                logger.info(
                    "redundant_expert: dump model routing table before update, "
                    f"path={dump_path}, hash={snapshot['table_hash']}, role={snapshot['role']}, "
                    f"shape={snapshot['shape']}"
                )
            except Exception as e:
                logger.warning(f"redundant_expert: dump model routing table before update failed, {e}")

        # update model info
        self.model_ep_rank_to_expert_id_list.copy_(paddle.to_tensor(rank_expert_list), True)
        self.model_expert_id_to_ep_rank_array.fill_(-1)
        self.model_expert_id_to_ep_rank_array[:, :, : logical_to_physical_map.shape[-1]] = paddle.to_tensor(
            logical_to_physical_map
        )
        self.model_expert_in_rank_num_list.copy_(paddle.to_tensor(expert_count), True)
        self.model_active_expert_id_to_ep_rank_array.copy_(self.model_expert_id_to_ep_rank_array, True)
        self.model_active_expert_in_rank_num_list.copy_(self.model_expert_in_rank_num_list, True)
        if self.fd_config is not None and clear_stat:
            try:
                dump_path, snapshot = dump_redundant_expert_table_snapshot(
                    fd_config=self.fd_config,
                    rank_expert_list=rank_expert_list,
                    logical_to_physical_map=logical_to_physical_map,
                    expert_count=expert_count,
                    source="model_table_after",
                    local_rank=self.fd_config.parallel_config.expert_parallel_rank,
                    clear_stat=clear_stat,
                )
                logger.info(
                    "redundant_expert: dump model routing table after update, "
                    f"path={dump_path}, hash={snapshot['table_hash']}, role={snapshot['role']}, "
                    f"shape={snapshot['shape']}"
                )
            except Exception as e:
                logger.warning(f"redundant_expert: dump model routing table after update failed, {e}")

        # reset
        if clear_stat:
            self.model_tokens_per_expert_stats_list.zero_()

    def refresh_active_expert_rank_table_by_ranks(
        self,
        last_active_ranks: paddle.Tensor,
        active_ranks: paddle.Tensor,
    ):
        """Rebuild the active expert routing table after FFN rank liveness changes.

        The base EPLB table is kept unchanged.  The active table is rewritten
        in place so CUDA graph replay still reads the same tensor address.
        """
        if (
            not self.fd_config.afd_config.enable_afd
            # or self.fd_config.afd_config.afd_role != "attn"
        ):
            return False

        last_active_ranks = last_active_ranks.reshape([-1]).astype("int32")
        active_ranks = active_ranks.reshape([-1]).astype("int32")

        if bool(paddle.all(last_active_ranks == active_ranks)):
            return
        
        local_physical_experts = self.fd_config.afd_config.afd_num_local_physical_experts
        fallback_physical_expert_id = (
            self.fd_config.afd_config.afd_ffn_ranks[0] * local_physical_experts
        )
        
        # Candidate physical-expert ids per (layer, logical_expert), -1 padded.
        base = self.model_expert_id_to_ep_rank_array  # [L, E, C] int32
        num_candidates = base.shape[-1]
        
        # Identify valid candidate physical experts and their owner ranks.
        # In AFD, valid physical experts are only placed on FFN ranks. ATTN rank slots stay -1.
        valid = base >= 0
        owner_rank = paddle.where(
            valid,
            base // local_physical_experts,
            paddle.zeros_like(base),
        ).astype("int64")
        owner_active = (
            paddle.index_select(
                active_ranks,
                owner_rank.reshape([-1]),
                axis=0,
            )
            .reshape(base.shape)
            > 0
        )
        keep = paddle.logical_and(valid, owner_active)

        # Stable compaction: kept candidates keep their original order at the front.
        candidate_index = paddle.arange(
            num_candidates,
            dtype="int64",
        ).reshape([1, 1, num_candidates])
        sort_key = paddle.where(
            keep,
            candidate_index,
            candidate_index + num_candidates,
        )
        order = paddle.argsort(sort_key, axis=-1)
        compacted = paddle.take_along_axis(base, order, axis=-1)

        # Build the active table: front active_count entries are kept, rest are -1.
        active_count = keep.astype("int64").sum(axis=-1)  # [L, E]
        front_mask = candidate_index < active_count.unsqueeze(-1)
        active_table = paddle.where(
            front_mask,
            compacted,
            paddle.full_like(base, -1),
        )

        # Fallback for experts whose replicas are all on inactive ranks.
        no_active = active_count == 0
        first_col = paddle.where(
            no_active,
            paddle.full(
                active_table[:, :, 0].shape,
                fallback_physical_expert_id,
                dtype=active_table.dtype,
            ),
            active_table[:, :, 0],
        )
        active_table[:, :, 0] = first_col
        active_count = paddle.where(
            no_active,
            paddle.ones_like(active_count),
            active_count,
        )

        # Publish refreshed active expert table.
        self.model_active_expert_id_to_ep_rank_array.copy_(
            active_table.astype("int32"),
            True,
        )
        self.model_active_expert_in_rank_num_list.copy_(
            active_count.astype("int32"),
            True,
        )
        changed_ranks = paddle.nonzero(
            last_active_ranks != active_ranks,
        ).reshape([-1])
        
        logger.info(
            "AFD EPLB active expert table refreshed by ranks: "
            f"changed_ranks={changed_ranks.tolist()}, "
            f"active_ranks={active_ranks.tolist()}, "
            f"fallback_physical_expert_id={fallback_physical_expert_id}"
        )

if __name__ == "__main__":
    print(RedundantExpertManger(64, 2, 8, 8).model_expert_id_to_ep_rank_array)
