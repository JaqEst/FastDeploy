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
from dataclasses import dataclass

from fastdeploy.afd.expert_weight_shm import ExpertWeightShmWriter
from fastdeploy.eplb.async_expert_loader import load_ep_checkpoint
from fastdeploy.utils import get_logger

PR_SET_PDEATHSIG = 1


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


def _is_glm_style(index):
    return any(name.startswith("model.layers.") and ".mlp.experts." in name for name in index)


def _iter_experts(spec, index):
    """Yield (layer_id, expert_id, up_gate, down), reading each shard at most once.

    up_gate and down are stored in the GPU parameter's per-expert layout, i.e. the exact
    bytes FusedMoE's weight loader would place in up_gate_proj_weight[slot] /
    down_proj_weight[slot].
    """
    import paddle
    from safetensors import safe_open

    needed = {}
    for layer_id in spec.moe_layer_ids:
        for expert_id in range(spec.n_routed_experts):
            prefix = f"model.layers.{layer_id}.mlp.experts.{expert_id}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                needed[f"{prefix}.{proj}.weight"] = (layer_id, expert_id, proj)

    by_file = {}
    for name, key in needed.items():
        if name not in index:
            raise KeyError(f"expert tensor missing from the checkpoint index: {name}")
        by_file.setdefault(index[name], []).append(name)

    pending = {}
    for shard, names in by_file.items():
        with safe_open(shard, framework="paddle", device="cpu") as f:
            for name in names:
                layer_id, expert_id, proj = needed[name]
                slot = pending.setdefault((layer_id, expert_id), {})
                slot[proj] = f.get_tensor(name)
                if len(slot) == 3:
                    pending.pop((layer_id, expert_id))
                    # Checkpoints store [out, in]; the GPU parameter is [in, out], with gate in
                    # the first half of the last axis and up in the second. This mirrors
                    # FusedMoE._load_gate_up_weight / _load_down_weight, so the reader copies
                    # bytes straight into the parameter slice.
                    up_gate = paddle.concat(
                        [slot.pop("gate_proj").transpose([1, 0]), slot.pop("up_proj").transpose([1, 0])],
                        axis=-1,
                    )
                    yield layer_id, expert_id, up_gate, slot.pop("down_proj").transpose([1, 0]).contiguous()

    if pending:
        raise RuntimeError(f"incomplete experts after scanning every shard: {sorted(pending)[:4]}")


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
    if not _is_glm_style(index):
        raise NotImplementedError(
            "expert weight daemon only supports the GLM checkpoint naming "
            "(model.layers.*.mlp.experts.*); the quantized ERNIE layout needs its "
            "scale/transpose handling added first"
        )

    tic = time.perf_counter()
    writer = ExpertWeightShmWriter(inst_id, spec.size)
    try:
        count = 0
        for layer_id, expert_id, up_gate, down in _iter_experts(spec, index):
            for key, tensor in ((f"{layer_id}.{expert_id}.up_gate", up_gate), (f"{layer_id}.{expert_id}.down", down)):
                writer.add(key, tensor.data_ptr(), int(tensor.numel().item() * tensor.element_size()))
            count += 1
            if stop.is_set():
                return
        writer.publish(spec.fingerprint())
        logger.info(
            f"expert weight block ready: {count} experts, {spec.size / 1024**3:.1f} GiB, "
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
