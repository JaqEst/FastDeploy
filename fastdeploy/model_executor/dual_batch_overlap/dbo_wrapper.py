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

"""Thread-based dual-batch overlap (DBO).

Two micro-batches run the same model body in two threads.  The threads exist only
to interleave kernel *issue* order on the current stream. The yield points live at
the communication boundaries inside the MoE layer and are no-ops outside a DBO
thread, so the same model code serves both the overlapped and the plain path.
"""

from __future__ import annotations

import threading

import paddle

from fastdeploy.model_executor.dual_batch_overlap.dbo_split import (
    split_decode_forward_meta,
)

_EVENTS = (threading.Event(), threading.Event())
# Set by a micro-batch thread that is bailing out, so its peer stops at the next
# handshake instead of running on with a corrupt transfer.
_ABORT = threading.Event()
# thread{i} waits on its own event and wakes its peer's.
_THREAD_EVENTS = {
    "thread0": (_EVENTS[0], _EVENTS[1]),
    "thread1": (_EVENTS[1], _EVENTS[0]),
}
_PEER_NAME = {"thread0": "thread1", "thread1": "thread0"}
_PENDING_HOOKS = {"thread0": None, "thread1": None}


class PeerThreadFailed(RuntimeError):
    """Raised in the surviving micro-batch thread after its peer failed."""


def dbo_enabled():
    """True inside a DBO micro-batch thread, i.e. when a yield has somewhere to go."""
    return threading.current_thread().name in _THREAD_EVENTS


def dbo_register_recv_hook(recv_hook):
    """Hand our transfer's wait to the peer."""
    thread_name = threading.current_thread().name
    if thread_name not in _THREAD_EVENTS or recv_hook is None:
        return
    peer = _PEER_NAME[thread_name]
    assert _PENDING_HOOKS[peer] is None, "previous transfer was never consumed"
    _PENDING_HOOKS[peer] = recv_hook


def dbo_maybe_run_recv_hook():
    """Wait on the peer's transfer, if it left one."""
    thread_name = threading.current_thread().name
    recv_hook = _PENDING_HOOKS.get(thread_name)
    if recv_hook is not None:
        _PENDING_HOOKS[thread_name] = None
        recv_hook()


def dbo_yield():
    """Hand the GPU to the other micro-batch."""
    thread_name = threading.current_thread().name
    events = _THREAD_EVENTS.get(thread_name)
    if events is None:
        return
    own_event, peer_event = events
    peer_event.set()
    own_event.wait()
    own_event.clear()
    if _ABORT.is_set():
        raise PeerThreadFailed("the other micro-batch thread failed")


class DBOWrapper:
    """Wrap a model runnable so one decode step runs as two overlapped micro-batches."""

    def __init__(self, runnable, fd_config):
        self.runnable = runnable
        self.enable_dbo = fd_config.afd_config.enable_dbo
        self.split_inputs = not fd_config.afd_config.is_ffn
        if self.enable_dbo and fd_config.speculative_config.enabled_speculative_decoding():
            # The split assumes one token per live slot, so every token boundary is
            # also a slot boundary; speculative decoding breaks that.
            raise NotImplementedError("AFD DBO does not support speculative decoding yet.")

    def __call__(self, **kwargs):
        if not self.enable_dbo:
            return self.runnable(**kwargs)

        if self.split_inputs:
            forward_meta = kwargs.get("forward_meta")
            # Falling back to a single micro-batch is not an option: every other rank
            # would still run two and the EP collectives would deadlock.
            assert (
                forward_meta is not None and forward_meta.dbo_micro_inputs is not None
            ), "AFD DBO needs dbo_micro_inputs on every step"
            metas = split_decode_forward_meta(forward_meta)
            micro_kwargs = [
                {**kwargs, "forward_meta": meta, "ids_remove_padding": meta.ids_remove_padding} for meta in metas
            ]
        else:
            micro_kwargs = [kwargs, kwargs]

        # The handshake state is module-level; a step that raised could have left an
        # event set, which would let B past its gate too early.
        for event in _EVENTS:
            event.clear()
        for name in _PENDING_HOOKS:
            _PENDING_HOOKS[name] = None
        _ABORT.clear()

        outs = [None, None]
        errors = [None, None]
        device = paddle.device.get_device()

        def body(mb_id, thread_name):
            paddle.device.set_device(device)
            own_event, peer_event = _THREAD_EVENTS[thread_name]
            if mb_id == 1:
                # B must not start until A reaches its first yield.
                own_event.wait()
                own_event.clear()
            try:
                outs[mb_id] = self.runnable(**micro_kwargs[mb_id])
                # Drain the peer's last transfer before releasing it.
                dbo_maybe_run_recv_hook()
            except BaseException as err:  # re-raised on the calling thread
                errors[mb_id] = err
                # Stop the peer at its next handshake instead of letting it run on.
                _ABORT.set()
            finally:
                peer_event.set()

        threads = [
            threading.Thread(target=body, args=(0, "thread0"), name="thread0"),
            threading.Thread(target=body, args=(1, "thread1"), name="thread1"),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Report the original failure, not the peer's induced abort.
        for err in errors:
            if err is not None and not isinstance(err, PeerThreadFailed):
                raise err
        for err in errors:
            if err is not None:
                raise err

        if not self.split_inputs:
            # Both threads ran the same inputs, so there is nothing to stitch back.
            return outs[0]
        return paddle.concat(outs, axis=0)
