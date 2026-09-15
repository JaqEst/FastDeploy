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

import time
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import paddle

from fastdeploy.config import (
    AFDConfig,
    CacheConfig,
    EPLBConfig,
    FDConfig,
    ParallelConfig,
    SchedulerConfig,
)
from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.eplb.experts_manager import RedundantExpertManager
from fastdeploy.inter_communicator import RearrangeExpertStatus


class TestRedundantExpertManager(unittest.TestCase):
    """Test cases for experts_manager.py"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock config objects
        max_num_seqs = 2
        engine_args = EngineArgs(
            max_num_seqs=max_num_seqs,
            num_gpu_blocks_override=102,
            max_num_batched_tokens=3200,
        )
        args = asdict(engine_args)

        cache_cfg = CacheConfig(args)
        model_cfg = SimpleNamespace(enable_mm=True)  # Enable multimodal for feature testing
        speculative_cfg = SimpleNamespace(method=None)
        model_cfg.print = print
        model_cfg.max_model_len = 5120
        model_cfg.num_hidden_layers = 3
        model_cfg.moe_num_experts = 64
        model_cfg.moe_layer_start_index = 1
        model_cfg.model = "/test/model"
        model_cfg.architectures = ["test_model"]
        model_cfg.mm_max_tokens_per_item = None
        model_cfg.version = None  # Required for register_info
        cache_cfg.bytes_per_layer_per_block = 1

        parallel_cfg = ParallelConfig(args)
        scheduler_cfg = SchedulerConfig(args)
        graph_opt_cfg = engine_args.create_graph_optimization_config()

        eplb_args = {
            "redundant_experts_num": 0,
            "redundant_expert_api_user": "test_user",
            "redundant_expert_api_password": "test_pass",
            "redundant_expert_eplb_strategy": "",
            "moe_quant_type": "",
            "redundant_expert_enable_schedule_cordon": False,
        }
        eplb_config = EPLBConfig(eplb_args)
        afd_config = AFDConfig(args)

        self.fd_config = FDConfig(
            model_config=model_cfg,
            cache_config=cache_cfg,
            parallel_config=parallel_cfg,
            graph_opt_config=graph_opt_cfg,
            speculative_config=speculative_cfg,
            scheduler_config=scheduler_cfg,
            eplb_config=eplb_config,
            afd_config=afd_config,
        )
        self.fd_config.parallel_config.local_data_parallel_id = 0
        self.fd_config.splitwise_role = "decode"

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    def test_init(self, mock_thread, mock_process, mock_get_logger):
        """Test RedundantExpertManager initialization"""
        # Mock logger
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Mock process and thread
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Test initialization
        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Verify initialization
        self.assertEqual(manager.rank, 0)
        self.assertEqual(manager.ep_size, 32)
        self.assertEqual(manager.fd_config, self.fd_config)
        self.assertEqual(manager.num_logical_experts, 64)
        self.assertEqual(manager.num_replicas, 64)  # 64 + 0 redundant

        # Verify arrays are created
        self.assertEqual(manager.model_ep_rank_to_expert_id_list.shape, (3, 64))
        self.assertEqual(manager.model_expert_id_to_ep_rank_array.shape, (3, 64, 1))
        self.assertEqual(manager.model_expert_in_rank_num_list.shape, (3, 64))

        # Verify process and thread are started
        mock_process.assert_called_once()
        mock_thread.assert_called_once()

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    def test_init_with_redundant_experts(self, mock_thread, mock_process, mock_get_logger):
        """Test initialization with redundant experts"""
        # Set up redundant experts
        self.fd_config.eplb_config.redundant_experts_num = 16

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=8, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Verify with redundant experts
        self.assertEqual(manager.num_replicas, 80)  # 64 + 16 redundant
        self.assertEqual(manager.model_ep_rank_to_expert_id_list.shape, (3, 80))
        self.assertEqual(manager.model_expert_id_to_ep_rank_array.shape, (3, 64, 17))  # 16 redundant + 1

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    def test_get_ep_rank_to_expert_id_list(self, mock_thread, mock_process, mock_get_logger):
        """Test get_ep_rank_to_expert_id_list method"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Set some test data
        manager.model_ep_rank_to_expert_id_list = np.array([[0, 1, 2, 3]])
        manager.model_expert_id_to_ep_rank_array = np.array([[[0], [1], [2], [3]]])
        manager.model_expert_in_rank_num_list = np.array([[1, 1, 1, 1]])

        result = manager.get_ep_rank_to_expert_id_list()

        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result[0], np.array([[0, 1, 2, 3]]))
        np.testing.assert_array_equal(result[1], np.array([[[0], [1], [2], [3]]]))
        np.testing.assert_array_equal(result[2], np.array([[1, 1, 1, 1]]))

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    def test_caculate_expert_rank_table(self, mock_thread, mock_process, mock_get_logger):
        """Test caculate_expert_rank_table method"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Set up test data
        manager.model_tokens_per_expert_stats_list = np.array([[10, 20, 30, 40], [5, 15, 25, 35]])

        # Mock the rebalance_experts function
        with patch("fastdeploy.eplb.experts_manager.rebalance_experts") as mock_rebalance:
            np_array1 = np.random.randint(0, 100, size=(3, 64))
            np_array2 = np.random.randint(0, 100, size=(3, 64, 1))
            np_array3 = np.random.randint(0, 100, size=(3, 64))
            mock_rebalance.return_value = (
                np_array1,  # phy2log
                np_array2,  # log2phy
                np_array3,  # logcnt
            )

            manager.caculate_expert_rank_table(is_init=True)

            # Verify that rebalance_experts was called with correct parameters
            mock_rebalance.assert_called_once()

            # Verify that arrays are updated
            np.testing.assert_array_equal(manager.model_ep_rank_to_expert_id_list, np_array1)

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    @patch("fastdeploy.eplb.experts_manager.IPCSignal")
    def test_update_weight_from_disk(self, mock_ipc_signal, mock_thread, mock_process, mock_get_logger):
        """Test update_weight_from_disk only sends the request, without waiting for it"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)

        # Mock IPCSignal
        mock_ipc_instance = MagicMock()
        mock_ipc_signal.return_value = mock_ipc_instance
        manager.update_weight_from_disk_result = MagicMock(value=np.array([1]))

        # Mock parent connections
        manager.parent_mg_conn = MagicMock()
        manager.parent_data_conn = MagicMock()

        # Set up test data
        manager.last_model_ep_rank_to_expert_id_list = np.array([[0, 1, 2, 3]])
        manager.model_ep_rank_to_expert_id_list = np.array([[1, 2, 3, 4]])

        manager.update_weight_from_disk()

        # The request is sent, but the answer is never awaited on this thread
        manager.parent_mg_conn.send.assert_called_once()
        manager.parent_data_conn.recv.assert_not_called()
        self.assertTrue(manager.disk_load_in_flight)
        self.assertEqual(manager.update_weight_from_disk_result.value[0], 0)

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    @patch("fastdeploy.eplb.experts_manager.IPCSignal")
    def test_poll_update_weight_from_disk(self, mock_ipc_signal, mock_thread, mock_process, mock_get_logger):
        """Test poll_update_weight_from_disk only consumes a ready answer"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)
        manager.update_weight_from_disk_result = MagicMock(value=np.array([0]))
        manager.parent_data_conn = MagicMock()
        manager.parent_data_conn.poll.return_value = False
        manager.disk_load_in_flight = True

        manager.poll_update_weight_from_disk()
        manager.parent_data_conn.recv.assert_not_called()
        self.assertTrue(manager.disk_load_in_flight)

        manager.parent_data_conn.poll.return_value = True
        manager.parent_data_conn.recv.return_value = {"result": True, "weights": ["weight1", "weight2"]}

        manager.poll_update_weight_from_disk()

        self.assertFalse(manager.disk_load_in_flight)
        self.assertEqual(manager.tensor_infos, ["weight1", "weight2"])
        self.assertEqual(manager.update_weight_from_disk_result.value[0], 1)

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    @patch("fastdeploy.eplb.experts_manager.IPCSignal")
    @patch("fastdeploy.eplb.experts_manager.paddle.distributed.all_reduce")
    def test_allreduce_expert_tokens_stats(
        self, mock_all_reduce, mock_ipc_signal, mock_thread, mock_process, mock_get_logger
    ):
        """Test allreduce_expert_tokens_stats reduces then hands the payload over"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)
        manager.update_weight_from_disk_result = MagicMock(value=np.array([1]))
        manager.calculate_expert_rank_table = MagicMock()
        manager.update_weight_from_disk = MagicMock()

        tokens_stats = paddle.full(
            shape=[manager.num_hidden_layers, manager.num_logical_experts], fill_value=7, dtype="int32"
        )
        manager.allreduce_expert_tokens_stats(tokens_stats, ep_group=None)

        mock_all_reduce.assert_called_once()
        # Nothing slow runs on the caller's thread, and the stale result is invalidated
        manager.calculate_expert_rank_table.assert_not_called()
        manager.update_weight_from_disk.assert_not_called()
        self.assertEqual(manager.update_weight_from_disk_result.value[0], 0)
        np.testing.assert_array_equal(manager.pending_tokens_stats, tokens_stats.numpy())

        # A second trigger while the first is still pending is dropped
        manager.allreduce_expert_tokens_stats(paddle.zeros_like(tokens_stats), ep_group=None)
        np.testing.assert_array_equal(manager.pending_tokens_stats, tokens_stats.numpy())

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    @patch("fastdeploy.eplb.experts_manager.IPCSignal")
    def test_begin_rearrange(self, mock_ipc_signal, mock_thread, mock_process, mock_get_logger):
        """Test begin_rearrange rebalances and starts the disk load on the listen thread"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)
        manager.rearrange_experts_signal = MagicMock(value=np.array([RearrangeExpertStatus.FREE.value]))
        manager.update_weight_from_disk_result = MagicMock(value=np.array([0]))
        manager.update_weight_from_disk = MagicMock()
        manager.calculate_expert_rank_table = MagicMock()

        tokens_stats = np.full((manager.num_hidden_layers, manager.num_logical_experts), 3, dtype=np.int32)
        manager.begin_rearrange(tokens_stats)

        np.testing.assert_array_equal(manager.model_tokens_per_expert_stats_list, tokens_stats)
        manager.calculate_expert_rank_table.assert_called_once()
        manager.update_weight_from_disk.assert_called_once()
        self.assertTrue(manager.need_load_weight_result_allreduce)
        self.assertEqual(manager.rearrange_experts_signal.value[0], RearrangeExpertStatus.DOING.value)

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    @patch("fastdeploy.eplb.experts_manager.IPCSignal")
    def test_drive_load_weight_result_allreduce(self, mock_ipc_signal, mock_thread, mock_process, mock_get_logger):
        """Test the listen thread asks for the result all-reduce and owns the timeout"""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)
        manager.rearrange_experts_signal = MagicMock(value=np.array([RearrangeExpertStatus.DOING.value]))
        manager.signal_allreduce_load_weight_result_array = MagicMock(value=np.array([0]))
        manager.need_load_weight_result_allreduce = True
        manager.load_weight_begin_ts = int(time.time())

        manager.drive_load_weight_result_allreduce()
        self.assertEqual(manager.signal_allreduce_load_weight_result_array.value[0], 1)

        # Within the interval no new request is raised
        manager.signal_allreduce_load_weight_result_array.value[0] = 0
        manager.drive_load_weight_result_allreduce()
        self.assertEqual(manager.signal_allreduce_load_weight_result_array.value[0], 0)

        # Past the timeout the wait ends instead of asking for another collective
        manager.load_weight_begin_ts = int(time.time()) - manager.load_weight_timeout - 1
        manager.last_load_weight_result_allreduce_ts = 0
        manager.drive_load_weight_result_allreduce()
        self.assertEqual(manager.signal_allreduce_load_weight_result_array.value[0], 0)
        self.assertFalse(manager.need_load_weight_result_allreduce)
        self.assertEqual(manager.rearrange_experts_signal.value[0], RearrangeExpertStatus.LOAD_SUCC.value)

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    @patch("fastdeploy.eplb.experts_manager.IPCSignal")
    def test_on_load_weight_result_allreduced_success(
        self, mock_ipc_signal, mock_thread, mock_process, mock_get_logger
    ):
        """Test on_load_weight_result_allreduced with all ranks succeeding (min_result=1)."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)
        manager.rearrange_experts_signal = MagicMock(value=np.array([RearrangeExpertStatus.DOING.value]))
        manager.signal_allreduce_load_weight_result_array = MagicMock(value=np.array([1]))
        manager.signal_update_weight_from_tensor_array = MagicMock(value=np.array([0]))
        manager.need_load_weight_result_allreduce = True
        manager.eplb_config.redundant_expert_enable_schedule_cordon = False

        manager.on_load_weight_result_allreduced(1)

        self.assertFalse(manager.need_load_weight_result_allreduce)
        self.assertEqual(manager.rearrange_experts_signal.value[0], RearrangeExpertStatus.LOAD_SUCC.value)
        self.assertEqual(manager.signal_update_weight_from_tensor_array.value[0], 1)

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    @patch("fastdeploy.eplb.experts_manager.IPCSignal")
    def test_on_load_weight_result_allreduced_fail(self, mock_ipc_signal, mock_thread, mock_process, mock_get_logger):
        """Test on_load_weight_result_allreduced with a failure (min_result=-1)."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)
        manager.rearrange_experts_signal = MagicMock(value=np.array([RearrangeExpertStatus.DOING.value]))
        manager.signal_allreduce_load_weight_result_array = MagicMock(value=np.array([1]))
        manager.signal_update_weight_from_tensor_array = MagicMock(value=np.array([0]))
        manager.need_load_weight_result_allreduce = True

        manager.on_load_weight_result_allreduced(-1)

        self.assertFalse(manager.need_load_weight_result_allreduce)
        self.assertEqual(manager.rearrange_experts_signal.value[0], RearrangeExpertStatus.LOAD_SUCC.value)
        self.assertEqual(manager.signal_update_weight_from_tensor_array.value[0], 0)
        self.assertEqual(manager.signal_allreduce_load_weight_result_array.value[0], 0)

    @patch("fastdeploy.eplb.experts_manager.get_logger")
    @patch("fastdeploy.eplb.experts_manager.Process")
    @patch("fastdeploy.eplb.experts_manager.threading.Thread")
    @patch("fastdeploy.eplb.experts_manager.IPCSignal")
    def test_on_load_weight_result_allreduced_waiting(
        self, mock_ipc_signal, mock_thread, mock_process, mock_get_logger
    ):
        """Test on_load_weight_result_allreduced still waiting (min_result=0)."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        manager = RedundantExpertManager(rank=0, ep_size=32, fd_config=self.fd_config, ipc_signal_suffix=0)
        manager.need_load_weight_result_allreduce = True

        manager.on_load_weight_result_allreduced(0)

        # still waiting, flag remains set
        self.assertTrue(manager.need_load_weight_result_allreduce)


if __name__ == "__main__":
    unittest.main()
