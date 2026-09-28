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

import ctypes
import multiprocessing
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from fastdeploy.eplb.async_expert_loader import load_ep_checkpoint
from fastdeploy.utils import get_logger
from fastdeploy.weight_cache.expert_weight_shm import ExpertWeightShmWriter

PR_SET_PDEATHSIG = 1

READY_TIMEOUT = 600  # seconds
FILL_THREADS = 8


@dataclass
class ExpertBlockSpec:
    """Everything the daemon needs to size and locate the expert weights."""

    model_path: str
    n_routed_experts: int
    hidden_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    moe_layer_start_index: int
    dtype_bytes: int

    @classmethod
    def from_fd_config(cls, fd_config):
        cfg = fd_config.model_config
        return cls(
            model_path=cfg.model,
            n_routed_experts=cfg.n_routed_experts,
            hidden_size=cfg.hidden_size,
            moe_intermediate_size=cfg.moe_intermediate_size,
            num_hidden_layers=cfg.num_hidden_layers,
            moe_layer_start_index=cfg.moe_layer_start_index,
            dtype_bytes=2,  # bf16 only for now
        )

    @property
    def moe_layer_ids(self):
        return range(self.moe_layer_start_index, self.num_hidden_layers)

    @property
    def size(self):
        """Bytes for one up_gate and one down slice per expert."""
        per_expert = (
            self.hidden_size * 2 * self.moe_intermediate_size + self.moe_intermediate_size * self.hidden_size
        ) * self.dtype_bytes
        return len(self.moe_layer_ids) * self.n_routed_experts * per_expert

    def fingerprint(self):
        return {
            "model_path": self.model_path,
            "n_routed_experts": self.n_routed_experts,
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "moe_layer_start_index": self.moe_layer_start_index,
            "dtype_bytes": self.dtype_bytes,
        }


def _is_format_supported(index):
    return any(name.startswith("model.layers.") and ".mlp.experts." in name for name in index)


def _expert_tensors_by_shard(spec, index):
    """{shard_path: [checkpoint_name, ...]} for every expert tensor, in canonical order."""
    by_file = {}
    for layer_id in spec.moe_layer_ids:
        for expert_id in range(spec.n_routed_experts):
            prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                name = f"{prefix}.{proj}.weight"
                if name not in index:
                    raise KeyError(f"expert tensor missing from the checkpoint index: {name}")
                by_file.setdefault(index[name], []).append(name)
    return by_file


def _fill_shard(shard, names, writer):
    """Copy one shard's expert tensors into the block. Runs on a fill thread."""
    from safetensors import safe_open

    with safe_open(shard, framework="paddle", device="cpu") as f:
        for name in names:
            writer.add(name, f.get_tensor(name))


def run_expert_weight_daemon(spec, inst_id):
    """Fill the block, publish it, then idle until terminated."""
    logger = get_logger("expert_weight_daemon", f"expert_weight_daemon_{inst_id}.log")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    # SIGTERM rather than SIGKILL: this process owns the block file and has to unlink it,
    # otherwise a dead engine leaves the whole block behind in /dev/shm.
    ctypes.CDLL(None).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    if os.getppid() == 1:
        os._exit(0)

    index = load_ep_checkpoint(spec.model_path)
    if not _is_format_supported(index):
        raise NotImplementedError(
            "expert weight daemon only supports the checkpoint "
            "naming (model.layers.*.mlp.experts.*)"
        )

    tic = time.perf_counter()
    writer = ExpertWeightShmWriter(inst_id, spec.size)
    try:
        by_file = _expert_tensors_by_shard(spec, index)
        with ThreadPoolExecutor(max_workers=min(len(by_file), FILL_THREADS)) as pool:
            futures = [pool.submit(_fill_shard, shard, names, writer) for shard, names in by_file.items()]
            for future in as_completed(futures):
                future.result()  # re-raise the first failure
                if stop.is_set():
                    return
        writer.publish(spec.fingerprint())
        logger.info(
            f"expert weight block ready: {len(writer.entries)} tensors, {spec.size / 1024**3:.1f} GiB, "
            f"{time.perf_counter() - tic:.1f}s"
        )
        stop.wait()
    finally:
        writer.close()


def spawn_expert_weight_daemon(spec, inst_id):
    """Start the weight daemon."""
    ctx = multiprocessing.get_context("spawn")
    proc = ctx.Process(target=run_expert_weight_daemon, args=(spec, inst_id), name="afd_expert_weight_daemon")
    # daemon=True so the parent's multiprocessing atexit terminates it: it would otherwise be
    # join()ed while it is still idling, hanging engine shutdown, and never unlink the block.
    proc.daemon = True
    proc.start()
    return proc
