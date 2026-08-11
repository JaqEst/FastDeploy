# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from fastdeploy.engine.common_engine import EngineService
from fastdeploy.engine.elastic_manager import ElasticManager
from fastdeploy.engine.request import ControlRequest


class _Signal:
    def __init__(self, values):
        self.value = np.array(values, dtype=np.int32)


def _cfg(splitwise_role="decode", enable_afd=False):
    return SimpleNamespace(
        router_config=SimpleNamespace(router="http://router.example.com/"),
        scheduler_config=SimpleNamespace(splitwise_role=splitwise_role),
        afd_config=SimpleNamespace(enable_afd=enable_afd, afd_role=None),
    )


class TestElasticManager(unittest.TestCase):
    def test_process_once_sets_ready_after_worker_health_updates_and_router_notified(self):
        signal = _Signal([99, 200])
        manager = ElasticManager(
            cfg=_cfg(enable_afd=True),
            worker_healthy_live_signal=signal,
            is_registered=Mock(return_value=True),
            operation_timeout=60,
        )
        task = manager.submit_recover(ranks=[2], worker_ids=[0])
        task.created_at = 100
        task.updated_at = 100

        with patch("fastdeploy.engine.elastic_manager.requests.post") as post:
            with patch("fastdeploy.engine.elastic_manager.time.time", return_value=101):
                manager.process_once()
            post.assert_not_called()
            self.assertEqual(manager.list_task_status(action="recover")[0]["status"], ElasticManager.STATUS_PENDING)

            post.return_value = Mock(ok=True)
            signal.value[0] = 102

            with patch("fastdeploy.engine.elastic_manager.time.time", return_value=102):
                manager.process_once()

        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://router.example.com/v1/recover/commit")
        self.assertEqual(kwargs["json"], {"ranks": [2], "targets": ["decode", "ffn"]})
        self.assertEqual(task.status, ElasticManager.STATUS_READY)
        self.assertTrue(task.router_notified)
        status = manager.list_task_status(action="recover")[0]
        self.assertEqual(status["status"], ElasticManager.STATUS_READY)
        self.assertTrue(status["router_notified"])

    def test_submit_recover_updates_global_alive_workers(self):
        manager = ElasticManager(
            cfg=_cfg(enable_afd=True),
            worker_healthy_live_signal=_Signal([100, 100, 100]),
            is_registered=Mock(return_value=True),
            operation_timeout=60,
        )

        task = manager.submit_recover(ranks=[2], worker_ids=[1])

        self.assertEqual(manager.get_alive_worker_ids(), [0, 2])
        self.assertNotIn("alive_worker_ids", task.to_dict())

    def test_complete_task_marks_success_and_restores_global_alive_workers(self):
        manager = ElasticManager(
            cfg=_cfg(enable_afd=True),
            worker_healthy_live_signal=_Signal([100, 100, 100]),
            is_registered=Mock(return_value=True),
            operation_timeout=60,
        )
        task = manager.submit_recover(ranks=[2], worker_ids=[1])
        task.status = ElasticManager.STATUS_READY

        completed = manager.complete_task(action="recover", ranks=[2])

        self.assertEqual(task.status, ElasticManager.STATUS_SUCCEEDED)
        self.assertEqual(completed["status"], ElasticManager.STATUS_SUCCEEDED)
        self.assertEqual(manager.get_alive_worker_ids(), [0, 1, 2])
        self.assertEqual(manager.list_task_status(action="recover")[0]["status"], ElasticManager.STATUS_SUCCEEDED)

    def test_not_ready_task_is_moved_to_queue_head(self):
        signal = _Signal([99, 99])
        manager = ElasticManager(
            cfg=_cfg(enable_afd=True),
            worker_healthy_live_signal=signal,
            is_registered=Mock(return_value=True),
            operation_timeout=60,
        )
        first = manager.submit_recover(ranks=[0], worker_ids=[0])
        second = manager.submit_recover(ranks=[1], worker_ids=[1])
        first.created_at = second.created_at = 100
        first.updated_at = second.updated_at = 100

        with patch("fastdeploy.engine.elastic_manager.time.time", return_value=101):
            manager.process_once()

        tasks = manager.list_task_status(action="recover")
        self.assertEqual([task["ranks"] for task in tasks], [[1], [0]])

    def test_process_once_marks_timed_out_recover_failed_and_keeps_history(self):
        manager = ElasticManager(
            cfg=_cfg("mixed"),
            worker_healthy_live_signal=_Signal([0]),
            is_registered=Mock(return_value=True),
            operation_timeout=1,
        )
        task = manager.submit_recover(ranks=[0], worker_ids=[0])
        task.created_at = 100

        with patch("fastdeploy.engine.elastic_manager.time.time", return_value=102):
            manager.process_once()

        self.assertEqual(task.status, ElasticManager.STATUS_FAILED)
        # The task leaves the active queue but stays reportable via history.
        self.assertEqual(len(manager._queue), 0)
        tasks = manager.list_task_status(action="recover")
        self.assertEqual([t["status"] for t in tasks], [ElasticManager.STATUS_FAILED])
        self.assertIs(manager.get_task(action="recover", ranks=[0]), task)

    def test_list_task_status_filters_by_action(self):
        manager = ElasticManager(
            cfg=_cfg(enable_afd=True),
            worker_healthy_live_signal=_Signal([0]),
            is_registered=Mock(return_value=True),
            operation_timeout=60,
        )
        manager.submit_recover(ranks=[0], worker_ids=[0])

        self.assertEqual(len(manager.list_task_status(action="recover")), 1)
        self.assertEqual(manager.list_task_status(action="scaleup"), [])


class TestCommonEngineRecoverElastic(unittest.TestCase):
    def test_control_recover_submits_recover_ranks(self):
        engine = object.__new__(EngineService)
        engine.cfg = SimpleNamespace(
            launch_config=SimpleNamespace(enable_fault_tolerant=True),
            parallel_config=SimpleNamespace(
                local_data_parallel_id=1,
                tensor_parallel_size=2,
                engine_worker_queue_port=[6778],
            ),
        )
        engine.worker_healthy_live_signal = _Signal([100, 195])
        engine.llm_logger = Mock()
        elastic_manager = Mock()
        engine._elastic_manager = elastic_manager

        launcher_client = Mock()
        launcher_client.request.return_value = {"recover_ranks": [2]}

        with (
            patch("fastdeploy.engine.common_engine.time.time", return_value=200),
            patch("fastdeploy.engine.common_engine.envs.FD_WORKER_ALIVE_TIMEOUT", 30),
            patch("fastdeploy.engine.common_engine.ControlSocketClient", return_value=launcher_client),
        ):
            result = EngineService._control_recover(engine, ControlRequest("req-1", "recover"))

        launcher_client.request.assert_called_once_with({"action": "recover", "container_ids": [2]})
        elastic_manager.submit_recover.assert_called_once_with(ranks=[2], worker_ids=[0])
        self.assertEqual(result, {"recover_ranks": [2]})

    def test_recover_status_queries_recover_tasks_without_parameters(self):
        engine = object.__new__(EngineService)
        engine.cfg = SimpleNamespace(
            launch_config=SimpleNamespace(enable_fault_tolerant=True),
        )
        elastic_manager = Mock()
        elastic_manager.list_task_status.return_value = [{"action": "recover", "status": "pending"}]
        engine._elastic_manager = elastic_manager

        result = EngineService._control_recover_status(
            engine,
            ControlRequest("req-1", "recover_status"),
        )

        elastic_manager.list_task_status.assert_called_once_with(action="recover")
        self.assertEqual(result, {"tasks": [{"action": "recover", "status": "pending"}]})

    def test_recover_commit_sends_control_to_alive_and_currently_healthy_workers_then_completes_task(self):
        engine = object.__new__(EngineService)
        engine.cfg = SimpleNamespace(
            launch_config=SimpleNamespace(enable_fault_tolerant=True),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=2,
                local_data_parallel_id=0,
                engine_worker_queue_port=[6778],
            ),
        )
        engine.worker_healthy_live_signal = _Signal([90, 95])
        engine.llm_logger = Mock()
        elastic_manager = Mock()
        task = SimpleNamespace(status=ElasticManager.STATUS_READY, worker_ids=[1], to_dict=lambda: {"status": ElasticManager.STATUS_READY})
        elastic_manager.get_task.return_value = task
        elastic_manager.get_alive_worker_ids.return_value = [0]
        elastic_manager.complete_task.return_value = {"action": "recover", "status": "succeeded", "ranks": [2]}
        engine._elastic_manager = elastic_manager
        engine.engine_worker_queue = Mock()
        engine._wait_for_control_responses = AsyncMock(return_value=[])
        request = ControlRequest("req-1", "recover_commit", {"ranks": [2]})

        with (
            patch("fastdeploy.engine.common_engine.time.time", return_value=100),
            patch("fastdeploy.engine.common_engine.envs.FD_WORKER_ALIVE_TIMEOUT", 30),
        ):
            result = EngineService._control_recover_commit(engine, request)

        elastic_manager.get_task.assert_called_once_with(action="recover", ranks=[2])
        elastic_manager.get_alive_worker_ids.assert_called_once_with()
        put_call = engine.engine_worker_queue.put_tasks.call_args
        commit_request = put_call.args[0][0][0]
        self.assertEqual(commit_request.request_id, "req-1")
        self.assertEqual(commit_request.method, "recover")
        self.assertEqual(commit_request.args, {"ranks": [2], "worker_ids": [0, 1]})
        self.assertEqual(put_call.kwargs["client_ids"], [1])
        engine._wait_for_control_responses.assert_called_once_with(
            "req-1", 30, executors=["worker"], worker_ids=[0, 1]
        )
        elastic_manager.complete_task.assert_called_once_with(action="recover", ranks=[2])
        self.assertEqual(result, None)


if __name__ == "__main__":
    unittest.main()
