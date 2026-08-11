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

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests

from fastdeploy.config import FDConfig
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.utils import elastic_manager_logger as logger


@dataclass
class ElasticTask:
    action: str
    targets: List[str]
    worker_ids: List[int]
    ranks: List[int]
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    router_notified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "action": self.action,
            "worker_ids": list(self.worker_ids),
            "ranks": list(self.ranks),
            "targets": list(self.targets),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "router_notified": self.router_notified,
        }


class ElasticManager:
    """Manage elastic tasks that require async readiness checks."""

    STATUS_PENDING = "pending"
    STATUS_READY = "ready"
    STATUS_SUCCEEDED = "succeeded"
    STATUS_FAILED = "failed"

    ACTION_RECOVER = "recover"

    RECOVER_COMMIT_PATH = "/v1/recover/commit"

    def __init__(
        self,
        cfg: FDConfig,
        worker_healthy_live_signal: IPCSignal,
        is_registered: Callable[[], bool],
        poll_interval: float = 1.0,
        operation_timeout: float = 300.0,
        router_timeout: float = 5.0,
    ):
        self.cfg = cfg
        self.worker_healthy_live_signal = worker_healthy_live_signal
        self.is_registered = is_registered
        self.poll_interval = poll_interval
        self.operation_timeout = operation_timeout
        self.router_timeout = router_timeout

        self._cond = threading.Condition()
        self._queue = deque()
        self._history = deque(maxlen=16)
        self._running = False
        self._thread = None

        self._lock = threading.Lock()
        self._alive_worker_ids = set(range(len(self.worker_healthy_live_signal.value)))

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="fd-elastic-manager", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        with self._cond:
            self._running = False
            self._queue.clear()
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit_recover(self, ranks: List[int], worker_ids: List[int]) -> ElasticTask:
        now = time.time()
        with self._lock:
            for worker_id in worker_ids:
                self._alive_worker_ids.discard(worker_id)
        task = ElasticTask(
            action=self.ACTION_RECOVER,
            targets=self._get_targets(),
            worker_ids=worker_ids,
            ranks=ranks,
            created_at=now,
            updated_at=now,
        )
        with self._cond:
            self._queue.appendleft(task)
            self._cond.notify_all()
        logger.info(f"Submitted recover task: {task.to_dict()}")
        return task

    def list_task_status(self, action: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._cond:
            tasks = list(self._history) + list(reversed(self._queue))
        result = []
        for task in tasks:
            if action is not None and task.action != action:
                continue
            result.append(task.to_dict())
        return result

    def get_task(self, action: str, ranks: List[int]) -> Optional[ElasticTask]:
        normalized_ranks = sorted(ranks)
        with self._cond:
            for task in list(self._queue) + list(reversed(self._history)):
                if task.action == action and sorted(task.ranks) == normalized_ranks:
                    return task
        return None

    def get_alive_worker_ids(self) -> List[int]:
        with self._lock:
            return sorted(self._alive_worker_ids)

    def complete_task(self, action: str, ranks: List[int]) -> Optional[Dict[str, Any]]:
        """Mark elastic task as succeeded."""
        task = self.get_task(action, ranks)
        if not task or task.status != self.STATUS_READY:
            return None
        task.updated_at = time.time()
        task.status = self.STATUS_SUCCEEDED
        with self._lock:
            for worker_id in task.worker_ids:
                self._alive_worker_ids.add(worker_id)
        logger.info(f"Elastic task completed: {task.to_dict()}")
        return task.to_dict()

    def process_once(self) -> bool:
        """Run one manager tick."""
        with self._cond:
            if not self._queue:
                return False
            task = self._queue[-1]

        completed = self._process_task(task)
        with self._cond:
            if not completed:
                self._queue.rotate()
            else:
                self._history.append(self._queue.pop())
        return True

    def _run(self) -> None:
        did_work = False
        while self._running:
            if did_work:
                time.sleep(self.poll_interval)
            did_work = self.process_once()
            with self._cond:
                if not self._queue:
                    did_work = False
                self._cond.wait_for(lambda: not self._running or self._queue)

    def _process_task(self, task: ElasticTask) -> bool:
        if task.status == self.STATUS_SUCCEEDED:
            return True
        if task.status == self.STATUS_FAILED:
            return True

        task.updated_at = time.time()
        if task.updated_at - task.created_at > self.operation_timeout:
            task.status = self.STATUS_FAILED
            logger.error(f"Elastic task timed out: {task.to_dict()}")
            return True

        if task.action == self.ACTION_RECOVER:
            if task.status == self.STATUS_PENDING:
                if not self._recover_ready(task):
                    return task.status == self.STATUS_FAILED
                task.status = self.STATUS_READY
            if not task.router_notified:
                task.router_notified = self._notify_router_recover_commit(task)
            return False

        task.status = self.STATUS_FAILED
        logger.error(f"Unsupported elastic task: {task.to_dict()}")
        return True

    def _recover_ready(self, task: ElasticTask) -> bool:
        if not task.ranks:
            return True

        values = self.worker_healthy_live_signal.value
        recover_start_time = int(task.created_at)
        for worker_id in task.worker_ids:
            if worker_id < 0 or worker_id >= len(values):
                task.status = self.STATUS_FAILED
                logger.error(f"Worker id {worker_id} is out of range")
                return False
            if int(values[worker_id]) < recover_start_time:
                return False

        # instance has to be visible to the router first.
        if self.cfg.router_config.router and not self.is_registered():
            return False
        return True

    def _notify_router_recover_commit(self, task: ElasticTask) -> bool:
        router_url = self.cfg.router_config.router
        if not router_url:
            logger.info("Router is not enabled, skip notify router")
            return True

        try:
            payload = {
                "ranks": task.ranks,
                "targets": task.targets,
            }

            resp = requests.post(
                f"{router_url.rstrip('/')}{self.RECOVER_COMMIT_PATH}",
                json=payload,
                timeout=self.router_timeout,
            )

            if not resp.ok:
                logger.error(f"Router recover commit failed: status={resp.status_code}, body={resp.text}")
                return False
            return True
        except Exception as e:
            logger.error(f"Unexpected error occurred when notifying router: error={e}")
            return False

    def _get_targets(self):
        if self.cfg.afd_config.enable_afd:
            return ["decode", "ffn"]    # ['attn', 'ffn']
        return [self.cfg.scheduler_config.splitwise_role]
