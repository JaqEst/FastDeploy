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

import threading
import time
from multiprocessing import get_context

import numpy as np
import paddle

from fastdeploy.config import FDConfig
from fastdeploy.eplb.async_expert_loader import load_model_weights_process
from fastdeploy.eplb.eplb import rebalance_experts
from fastdeploy.eplb.utils import (
    RedundantExpertWorkload,
    expand_expert_rank_table,
    derive_expert_tables,
    compact_expert_rank_table,
)
from fastdeploy.inter_communicator import IPCSignal, RearrangeExpertStatus
from fastdeploy.utils import get_logger


class RedundantExpertManager:
    """
    RedundantExpertManger
    """

    def __init__(
        self,
        rank: int = 0,
        ep_size: int = 32,
        fd_config: FDConfig = None,
        ipc_signal_suffix: int = 0,
        shm_fd: int = -1,
    ):
        self.logger = get_logger("eplb_expert_manager", "eplb_{0}.log".format(rank))

        self.rank = rank
        self.ep_size = ep_size
        self.fd_config = fd_config
        self.eplb_config = fd_config.eplb_config
        self.num_redundant_experts = self.eplb_config.redundant_experts_num
        self.num_hidden_layers = self.fd_config.model_config.num_hidden_layers
        self.num_logical_experts = self.fd_config.model_config.moe_num_experts
        self.dp_ipc_signal_suffix = f"{ipc_signal_suffix}_dp{fd_config.parallel_config.local_data_parallel_id}"
        self.local_rank = self.fd_config.parallel_config.tensor_parallel_rank

        if not self.fd_config.afd_config.enable_afd:
            self.num_replicas = self.num_logical_experts + self.num_redundant_experts
            self.num_nodes = max(ep_size // 8, 1)
            self.num_gpus = ep_size
        else:
            self.num_replicas = self.fd_config.afd_config.num_physical_experts
            self.num_nodes = max(len(self.fd_config.afd_config.ffn_ranks) // 8, 1)
            self.num_gpus = len(self.fd_config.afd_config.ffn_ranks)
        self.num_groups = self.num_logical_experts
        self.expert_per_rank = self.num_replicas // ep_size
        assert (
            self.num_replicas % ep_size == 0
        ), f"num_replicas must be divisible by ep_size, \
                but got num_replicas = {self.num_replicas}, ep_size = {ep_size}"

        self.model_ep_rank_to_expert_id_list = np.full(
            (
                self.num_hidden_layers,
                self.num_replicas,
            ),
            -1,
            dtype=np.int32,
        )
        self.model_expert_id_to_ep_rank_array = np.full(
            (
                self.num_hidden_layers,
                self.num_logical_experts,
                self.num_redundant_experts + 1,
            ),
            -1,
            dtype=np.int32,
        )
        self.model_expert_in_rank_num_list = np.zeros(
            (self.num_hidden_layers, self.num_logical_experts), dtype=np.int32
        )

        # backup info
        self.last_model_ep_rank_to_expert_id_list = np.full(
            (
                self.num_hidden_layers,
                self.num_replicas,
            ),
            -1,
            dtype=np.int32,
        )
        self.last_model_expert_id_to_ep_rank_array = np.full(
            (
                self.num_hidden_layers,
                self.num_logical_experts,
                self.num_redundant_experts + 1,
            ),
            -1,
            dtype=np.int32,
        )
        self.last_model_expert_in_rank_num_list = np.zeros(
            (self.num_hidden_layers, self.num_logical_experts), dtype=np.int32
        )

        self.model_tokens_per_expert_stats_list = np.ones(
            (self.num_hidden_layers, self.num_logical_experts), dtype=np.int32
        )

        self.rearrange_experts_signal = None
        self.signal_update_weight_from_tensor_array = None
        self.signal_allreduce_expert_tokens_stats_array = None
        self.signal_allreduce_load_weight_result_array = None
        if self.local_rank == 0:
            self.rearrange_experts_signal = IPCSignal(
                name="rearrange_experts_status",
                array=np.zeros([1], dtype=np.int32),
                dtype=np.int32,
                suffix=self.dp_ipc_signal_suffix,
                create=False,
            )
            self.signal_update_weight_from_tensor_array = IPCSignal(
                name="signal_update_weight_from_tensor",
                array=np.zeros([1], dtype=np.int32),
                dtype=np.int32,
                suffix=self.dp_ipc_signal_suffix,
                create=False,
            )
            self.signal_allreduce_expert_tokens_stats_array = IPCSignal(
                name="signal_allreduce_expert_tokens_stats",
                array=np.zeros([1], dtype=np.int32),
                dtype=np.int32,
                suffix=self.dp_ipc_signal_suffix,
                create=False,
            )
            self.signal_allreduce_load_weight_result_array = IPCSignal(
                name="signal_allreduce_load_weight_result",
                array=np.zeros([1], dtype=np.int32),
                dtype=np.int32,
                suffix=self.dp_ipc_signal_suffix,
                create=False,
            )

        shm_expert_rank_table = np.zeros(
            (self.num_hidden_layers, self.num_logical_experts + self.num_redundant_experts),
            dtype=np.int32,
        )
        self.shm_expert_rank_table = IPCSignal(
            name="expert_rank_table",
            array=shm_expert_rank_table,
            dtype=np.int32,
            suffix=self.dp_ipc_signal_suffix,
            create=False,
        )

        tp_ipc_signal_suffix = f"{self.dp_ipc_signal_suffix}_tp{self.local_rank}"
        self.update_weight_from_disk_result = IPCSignal(
            name="result_update_weight_from_disk",
            array=np.zeros([1], dtype=np.int32),
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=False,
        )

        if not self.seed_expert_rank_table():
            self.calculate_expert_rank_table(True)

        # Handed over by the worker event loop, consumed by the listen thread.
        self.pending_tokens_stats = None
        self.disk_load_in_flight = False
        self.disk_load_begin_ts = 0
        self.need_load_weight_result_allreduce = False
        self.load_weight_begin_ts = 0
        self.load_weight_timeout = 300  # 5min
        self.load_weight_result_allreduce_interval = 3
        self.last_load_weight_result_allreduce_ts = 0
        # 重置重排状态: 'done' -> 'free'
        self.rearrange_end_ts = 0
        self.rearrange_reset_interval = 30

        self.tensor_infos = None

        if not self.fd_config.afd_config.is_attn:
            # Fork explicitly: shm_fd is handed over by inheritance, and create_mmap has
            # already unlinked the expert weight file, so this descriptor is the only way
            # into the buffer.
            ctx = get_context("fork")
            self.parent_data_conn, child_data_conn = ctx.Pipe()
            self.parent_mg_conn, child_mg_conn = ctx.Pipe()
            ctx.Process(
                target=load_model_weights_process,
                name=f"eplb::async_load_model_{rank}",
                args=(
                    self.rank,
                    self.fd_config.model_config.model,
                    self.expert_per_rank,
                    self.fd_config.model_config.moe_layer_start_index,
                    self.eplb_config.moe_quant_type,
                    shm_fd,
                    self.eplb_config,
                    child_data_conn,
                    child_mg_conn,
                    # Handed over only so the child can close them: otherwise it
                    # never sees EOF when parent process exits.
                    self.parent_data_conn,
                    self.parent_mg_conn,
                ),
            ).start()
            child_data_conn.close()
            child_mg_conn.close()

        listen_signal_thread = threading.Thread(target=self.listen_rearrange_expert_signal, args=(), daemon=True)
        listen_signal_thread.start()

        self.logger.info(
            f"redundant_expert: RedundantExpertManager init success, rank {rank}, \
            strategy {self.eplb_config.redundant_expert_eplb_strategy}"
        )

    def get_ep_rank_to_expert_id_list(self):
        """
        get_ep_rank_to_expert_id_list
        """
        return (
            self.model_ep_rank_to_expert_id_list,
            self.model_expert_id_to_ep_rank_array,
            self.model_expert_in_rank_num_list,
        )

    def listen_rearrange_expert_signal(self):
        """
        listen_rearrange_expert_signal
        """
        while True:
            self.advance_rearrange()
            time.sleep(0.5)

    def advance_rearrange(self):
        """
        Drive one step of the rearrange state machine.
        """
        tokens_stats = self.pending_tokens_stats
        if tokens_stats is not None:
            self.pending_tokens_stats = None
            self.begin_rearrange(tokens_stats)

        self.poll_update_weight_from_disk()

        if self.local_rank == 0:
            self.drive_load_weight_result_allreduce()
            now = int(time.time())
            if self.rearrange_experts_signal.value[0] > RearrangeExpertStatus.DOING.value:
                if self.rearrange_end_ts == 0:
                    self.rearrange_end_ts = now
                if now - self.rearrange_end_ts > self.rearrange_reset_interval:
                    # reset rearrange status
                    self.rearrange_experts_signal.value[0] = RearrangeExpertStatus.FREE.value
                    self.rearrange_end_ts = 0

    def seed_expert_rank_table(self) -> bool:
        """
        Seed the tables from the placement the engine published, instead of computing a cold start.
        """
        compact = np.array(self.shm_expert_rank_table.value, dtype=np.int32)
        # The engine fills the table with -1 until it holds a placement.
        if np.any(compact < 0):
            return False

        phy2log = expand_expert_rank_table(compact, self.fd_config.afd_config)
        logical_to_physical_map, expert_count = derive_expert_tables(
            phy2log, self.num_logical_experts, self.num_redundant_experts + 1
        )

        self.model_ep_rank_to_expert_id_list[:] = phy2log[:]
        self.model_expert_id_to_ep_rank_array.fill(-1)
        self.model_expert_id_to_ep_rank_array[..., : logical_to_physical_map.shape[-1]] = logical_to_physical_map[:]
        self.model_expert_in_rank_num_list[:] = expert_count[:]

        # update_weight_from_disk diffs the new placement against this backup, so the backup has to
        # be the placement the weights on device were loaded with, not a cold start.
        self.last_model_ep_rank_to_expert_id_list[:] = self.model_ep_rank_to_expert_id_list[:]
        self.last_model_expert_id_to_ep_rank_array[:] = self.model_expert_id_to_ep_rank_array[:]
        self.last_model_expert_in_rank_num_list[:] = self.model_expert_in_rank_num_list[:]

        self.logger.info("redundant_expert: read the expert rank table published by the engine")
        return True

    def calculate_expert_rank_table(self, is_init=False):
        """
        calculate_expert_rank_table
        """
        num_groups = self.num_groups
        num_nodes = self.num_nodes
        num_gpus = self.num_gpus
        eplb_strategy = self.eplb_config.redundant_expert_eplb_strategy
        if is_init:
            num_groups = 1
            eplb_strategy = ""
            if not self.fd_config.afd_config.enable_afd:
                num_nodes = 8
                num_gpus = 8 * 8
        # eplb
        rank_expert_list, logical_to_physical_map, expert_count = rebalance_experts(
            weight=self.model_tokens_per_expert_stats_list,
            num_replicas=self.num_replicas,
            num_groups=num_groups,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
            eplb_strategy=eplb_strategy,
            fd_config=self.fd_config,
        )

        # backup info
        self.last_model_ep_rank_to_expert_id_list[:] = self.model_ep_rank_to_expert_id_list[:]
        self.last_model_expert_id_to_ep_rank_array[:] = self.model_expert_id_to_ep_rank_array[:]
        self.last_model_expert_in_rank_num_list[:] = self.model_expert_in_rank_num_list[:]

        # update model info
        self.model_ep_rank_to_expert_id_list[:] = rank_expert_list[:]
        self.model_expert_id_to_ep_rank_array.fill(-1)
        self.model_expert_id_to_ep_rank_array[..., : logical_to_physical_map.shape[-1]] = logical_to_physical_map[:]
        self.model_expert_in_rank_num_list[:] = expert_count[:]

        if self.local_rank == 0:
            self.shm_expert_rank_table.value[:] = compact_expert_rank_table(
                self.model_ep_rank_to_expert_id_list, self.fd_config.afd_config
            )

            workload = RedundantExpertWorkload(self.eplb_config.redundant_expert_meta_dir)
            workload.tokens_per_expert_stats_list = self.model_tokens_per_expert_stats_list.tolist()
            workload.ep_rank_to_expert_id_list = rank_expert_list.tolist()
            workload.expert_id_to_ep_rank_array = logical_to_physical_map.tolist()
            workload.expert_in_rank_num_list = expert_count.tolist()
            self.logger.info(workload.dump())

    def update_weight_from_disk(self):
        """
        update_weight_from_disk
        """
        if self.disk_load_in_flight:
            self.logger.warning(f"redundant_expert: a disk load is still in flight, rank {self.rank}")
            return

        self.update_weight_from_disk_result.value[0] = 0
        self.disk_load_begin_ts = int(time.time())
        self.disk_load_in_flight = True
        self.parent_mg_conn.send(
            {
                "old_model_ep_rank_to_expert_id_list": self.last_model_ep_rank_to_expert_id_list,
                "new_model_ep_rank_to_expert_id_list": self.model_ep_rank_to_expert_id_list,
            }
        )
        self.logger.info(f"redundant_expert: update_weight_from_disk send to async process, rank {self.rank}")

    def poll_update_weight_from_disk(self):
        """
        Collect the async loader's answer if it is ready, without blocking.
        """
        if not self.disk_load_in_flight or not self.parent_data_conn.poll():
            return

        response = self.parent_data_conn.recv()
        self.disk_load_in_flight = False
        self.tensor_infos = response["weights"]
        # 更新权重加载结果
        self.update_weight_from_disk_result.value[0] = 1 if response["result"] else -1
        self.logger.info(
            "redundant_expert: update_weight_from_disk end, rank"
            + f" {self.rank} {response['result']}, cost {int(time.time() - self.disk_load_begin_ts)}s"
        )

    def allreduce_expert_tokens_stats(self, tokens_stats, ep_group):
        paddle.distributed.all_reduce(tokens_stats, op=paddle.distributed.ReduceOp.SUM, group=ep_group)

        if self.pending_tokens_stats is not None or self.disk_load_in_flight:
            self.logger.warning("redundant_expert: previous rearrange still in flight, drop this trigger")
            return
        # Invalidate the previous round's result here rather than on the listen thread:
        # every rank runs this in the same event loop step, so the result all-reduce
        # cannot observe a stale success from a rank whose listen thread is still asleep.
        self.update_weight_from_disk_result.value[0] = 0
        self.pending_tokens_stats = np.array(tokens_stats.numpy(), dtype=np.int32)

    def begin_rearrange(self, tokens_stats: np.ndarray):
        """
        Rebalance for the reduced expert load and start loading the weights it needs.
        """
        self.model_tokens_per_expert_stats_list[:] = tokens_stats[:]
        if self.local_rank == 0:
            self.rearrange_experts_signal.value[0] = RearrangeExpertStatus.DOING.value
            self.rearrange_end_ts = 0
            self.need_load_weight_result_allreduce = True
            self.load_weight_begin_ts = int(time.time())
            self.last_load_weight_result_allreduce_ts = 0

        self.calculate_expert_rank_table()
        if self.fd_config.afd_config.is_attn:
            # AFD attn ranks hold no expert weights, only the routing table.
            self.update_weight_from_disk_result.value[0] = 1
        else:
            self.update_weight_from_disk()

    def drive_load_weight_result_allreduce(self):
        """
        Ask the worker event loop to all-reduce(MIN) the per rank disk load results.
        """
        if not self.need_load_weight_result_allreduce:
            return

        now = int(time.time())
        if now - self.load_weight_begin_ts > self.load_weight_timeout:
            self.logger.warning(f"redundant_expert: load weight from disk timeout {self.load_weight_timeout}s")
            self.finish_load_weight_wait()
            return
        if now - self.last_load_weight_result_allreduce_ts <= self.load_weight_result_allreduce_interval:
            return

        self.last_load_weight_result_allreduce_ts = now
        self.signal_allreduce_load_weight_result_array.value[0] = 1

    def finish_load_weight_wait(self):
        """
        Stop waiting for the disk load result, whatever the outcome was.
        """
        self.need_load_weight_result_allreduce = False
        # A request raised but not yet consumed by the event loop would otherwise fire
        # a pointless collective the next time the worker runs a step.
        self.signal_allreduce_load_weight_result_array.value[0] = 0
        self.rearrange_experts_signal.value[0] = RearrangeExpertStatus.LOAD_SUCC.value
        self.rearrange_end_ts = int(time.time())

    def on_load_weight_result_allreduced(self, min_result: int):
        """
        min_result: -1 if any rank failed to load, 0 if any rank is still loading, 1 if all
        ranks succeeded.
        """
        if self.local_rank != 0 or not self.need_load_weight_result_allreduce:
            # Only tp0 owns the dp level signals, and another rank may still be polling
            # after this one already gave up.
            return

        if min_result == 0:
            self.logger.info("redundant_expert: allreduce_load_weight_result waiting")
            return
        if min_result < 0:
            # 如果有DP权重加载异常，结束本次重排
            self.logger.warning("redundant_expert: allreduce_load_weight_result exist fail, terminate this rearrange")
            self.finish_load_weight_wait()
            return

        self.finish_load_weight_wait()
        # prefill需要等待调度屏蔽
        if (
            self.fd_config.afd_config.enable_afd
            or self.fd_config.scheduler_config.splitwise_role == "mixed"
            or self.fd_config.scheduler_config.splitwise_role == "decode"
            or not self.eplb_config.redundant_expert_enable_schedule_cordon
        ):
            self.logger.info("redundant_expert: allreduce_load_weight_result success, notify infer.py")
            self.signal_update_weight_from_tensor_array.value[0] = 1
