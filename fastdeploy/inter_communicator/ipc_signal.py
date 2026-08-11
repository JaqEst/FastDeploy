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

import os
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

import _posixshmem
import numpy as np

from fastdeploy.utils import llm_logger

# Blocks created by this process. Their lifecycle belongs to this process, so attaching
# to one of them again must not drop its resource_tracker registration.
_created_shm_names = set()


def untrack_shared_memory(shm: SharedMemory) -> None:
    """Stop the current process from owning the lifecycle of a shared memory block.

    CPython (< 3.13) registers a block with multiprocessing.resource_tracker even when
    the process only attaches to an existing one. As a result the tracker unlinks the
    block once the attaching process exits, which silently destroys a block that other
    processes are still using: they keep their mapping but the name disappears, so any
    process started later (e.g. a worker restarted by recover) fails to attach with
    FileNotFoundError. Only the creator should unlink, via IPCSignal.clear().

    Args:
        shm: The shared memory block to stop tracking in this process.
    """
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception as e:
        llm_logger.warning(f"Failed to untrack shared memory {shm.name}: {e}")


def attach_shared_memory(name: str) -> SharedMemory:
    """Attach to an existing shared memory block without taking over its lifecycle.

    Args:
        name: The unique identifier of the shared memory block.

    Returns:
        The attached shared memory block.
    """
    shm = SharedMemory(name=name)
    if shm._name not in _created_shm_names:
        untrack_shared_memory(shm)
    return shm


def create_shared_memory(name: str, size: int) -> SharedMemory:
    """Create a shared memory block owned by this process.

    Args:
        name: The unique identifier of the shared memory block.
        size: Size of the block in bytes.

    Returns:
        The created shared memory block.
    """
    shm = SharedMemory(create=True, size=size, name=name)
    _created_shm_names.add(shm._name)
    return shm


def shared_memory_exists(name: str) -> bool:
    """Check if a shared memory block with the given name exists.

    The block is probed with shm_open instead of SharedMemory, because attaching via
    SharedMemory would register the block in this process's resource_tracker and a
    probe must not change who owns the block's lifecycle.

    Args:
        name: The unique identifier of the shared memory block.

    Returns:
        True if the shared memory exists, False otherwise.
    """
    try:
        fd = _posixshmem.shm_open("/" + name.lstrip("/"), os.O_RDONLY, mode=0o600)
    except FileNotFoundError:
        return False
    except Exception as e:
        llm_logger.error(f"Unexpected error: {e}")
        return False
    os.close(fd)
    return True


class IPCSignal:
    """A shared memory wrapper for inter-process communication using numpy arrays.

    Allows creating or connecting to existing shared memory blocks and synchronizing
    numpy array data between processes.

    Attributes:
        shm: The underlying SharedMemory object.
        value: Numpy array interface to the shared memory buffer.
    """

    def __init__(
        self,
        name: str,
        array: np.ndarray = None,
        dtype: np.dtype = None,
        suffix: int = None,
        create: bool = True,
        shm_size: int = None,
    ) -> None:
        """Initialize or connect to a shared memory block.

        Args:
            name: Unique identifier for the shared memory block.
            array: Numpy array template defining shape and data type.
            dtype: Data type of the array (must match array.dtype).
            suffix: Suffix number that will be appended to the name.
            create: If True, creates new memory block; otherwise connects to existing.
            shm_size: Size of the shared memory block in bytes.

        Raises:
            AssertionError: If create=True but memory already exists, or dtype mismatch.
        """
        # Set a suffix for name to avoid name conflict while there are multiple engine launched
        if suffix is not None:
            name = name + f".{suffix}"

        if dtype is None or array is None:
            assert shm_size is not None, "shm_size must be specified if array and dtype are None"

            if create:
                llm_logger.debug(f"creating ipc signal: {name}")
                if shared_memory_exists(name):
                    llm_logger.warning(f"ShareMemory: {name} already exists, delete it")
                    SharedMemory(name=name, create=False).unlink()
                self.shm = create_shared_memory(name, shm_size)
                self.value = None
            else:
                llm_logger.debug(f"attaching ipc signal: {name}")
                self.shm = attach_shared_memory(name)
                self.value = None
        else:
            assert isinstance(array, np.ndarray), "Input must be a numpy array"
            assert dtype == array.dtype, "Specified dtype must match array dtype"

            if create:
                llm_logger.debug(f"creating ipc signal: {name}")
                if shared_memory_exists(name):
                    llm_logger.warning(f"ShareMemory: {name} already exists, delete it")
                    SharedMemory(name=name, create=False).unlink()
                self.shm = create_shared_memory(name, array.nbytes)
                self.value: np.ndarray = np.ndarray(array.shape, dtype=array.dtype, buffer=self.shm.buf)
                self.value[:] = array  # Initialize with input array data
            else:
                llm_logger.debug(f"attaching ipc signal: {name}")
                self.shm = attach_shared_memory(name)
                self.value: np.ndarray = np.ndarray(array.shape, dtype=array.dtype, buffer=self.shm.buf)

    def clear(self) -> None:
        """Release system resources and unlink the shared memory block."""
        _created_shm_names.discard(self.shm._name)
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            llm_logger.warning(f"ShareMemory: {self.shm.name} has already been unlinked by another process")
