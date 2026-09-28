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
import json
import os
import threading

from fastdeploy.eplb.async_expert_loader import cudart, libc

PROT_READ, PROT_WRITE, MAP_SHARED = 0x1, 0x2, 0x01
MAP_FAILED = ctypes.c_void_p(-1).value


def block_paths(inst_id):
    """Paths of the three files that make up one expert weight block."""
    base = f"/dev/shm/fd_afd_expert_{inst_id}"
    return base + ".bin", base + ".json", base + ".ready"


def _cuda_check(ret, what):
    if ret[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{what} failed: {cudart.cudaGetErrorString(ret[0])}")
    return ret[1] if len(ret) > 1 else None


def _mmap(path, size, create):
    fd = os.open(path, os.O_RDWR | (os.O_CREAT if create else 0), 0o600)
    if create:
        os.ftruncate(fd, size)
    ptr = libc.mmap(0, ctypes.c_size_t(size), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
    if ptr == MAP_FAILED:
        os.close(fd)
        raise OSError(f"mmap {path} failed")
    return fd, ptr


class ExpertWeightShmWriter:
    """Daemon side: fill the block, then publish it.

    Entries are keyed by the checkpoint tensor name, e.g.
    "model.layers.3.mlp.experts.7.gate_proj.weight", and stored verbatim in the
    checkpoint's layout, so a reader can hand them straight to the model's weight
    loader. The file is never unlinked while in use: attaching workers reopen it by
    name, and it only goes away when this writer closes.
    """

    def __init__(self, inst_id, size):
        self.bin_path, self.manifest_path, self.ready_path = block_paths(inst_id)
        for p in (self.bin_path, self.manifest_path, self.ready_path):
            if os.path.exists(p):
                os.unlink(p)
        self.size = size
        self.fd, self.ptr = _mmap(self.bin_path, size, create=True)
        self.offset = 0
        self.entries = {}
        self.registered = False
        self._lock = threading.Lock()

    def add(self, key, tensor):
        """Append one entry from a CPU tensor. Safe to call from several threads."""
        nbytes = int(tensor.numel().item() * tensor.element_size())
        with self._lock:
            offset = self.offset
            end = offset + nbytes
            if end > self.size:
                raise IOError(f"expert weight block overflow: {end} > {self.size}")
            self.offset = end
            self.entries[key] = {
                "offset": offset,
                "nbytes": nbytes,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        # Outside the lock: this copy is the bulk of the work and has to overlap across threads.
        # Callers join their threads before publish(), so the bytes are in place by then.
        ctypes.memmove(ctypes.c_void_p(self.ptr + offset), ctypes.c_void_p(tensor.data_ptr()), nbytes)

    def publish(self, fingerprint):
        """Pin the filled block, then write the manifest and ready."""
        # Registering after every page has been touched is far cheaper than letting the driver
        # fault them in, and the faulting is what the threaded fill spreads across cores.
        _cuda_check(cudart.cudaHostRegister(self.ptr, self.size, 0), "cudaHostRegister")
        self.registered = True
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"fingerprint": fingerprint, "size": self.size, "entries": self.entries}, f)
        os.replace(tmp, self.manifest_path)
        with open(self.ready_path, "w") as f:
            f.write(str(os.getpid()))

    def close(self):
        if self.registered:
            _cuda_check(cudart.cudaHostUnregister(self.ptr), "cudaHostUnregister")
        libc.munmap(ctypes.c_void_p(self.ptr), ctypes.c_size_t(self.size))
        os.close(self.fd)
        for p in (self.bin_path, self.manifest_path, self.ready_path):
            if os.path.exists(p):
                os.unlink(p)


class ExpertWeightShm:
    """Worker side: map the daemon's block and hand its entries to the weight loader."""

    def __init__(self, inst_id):
        self.bin_path, self.manifest_path, self.ready_path = block_paths(inst_id)
        with open(self.ready_path) as f:
            self.daemon_pid = int(f.read().strip())
        with open(self.manifest_path) as f:
            manifest = json.load(f)
        self.fingerprint = manifest["fingerprint"]
        self.size = manifest["size"]
        self.entries = manifest["entries"]
        self.fd, self.ptr = _mmap(self.bin_path, self.size, create=False)
        # Per-process registration; the daemon's pin does not make this memory usable here.
        _cuda_check(cudart.cudaHostRegister(self.ptr, self.size, 0), "cudaHostRegister")

    def tensor_infos(self):
        """The entries in the form load_tensor_from_shm_mem expects."""
        import paddle

        return [
            (name, e["offset"], e["nbytes"], e["shape"], getattr(paddle, e["dtype"].split(".")[-1]))
            for name, e in self.entries.items()
        ]

    def close(self):
        _cuda_check(cudart.cudaHostUnregister(self.ptr), "cudaHostUnregister")
        libc.munmap(ctypes.c_void_p(self.ptr), ctypes.c_size_t(self.size))
        os.close(self.fd)
