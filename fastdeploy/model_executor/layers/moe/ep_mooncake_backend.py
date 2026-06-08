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

import traceback
from types import ModuleType

import paddle
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.utils import singleton

from .ep_backend_base import EPRunnerBase


def load_mooncake_ep() -> ModuleType:
    """
    Load and return the Mooncake EP Module.
    """
    paddle.enable_compat(scope={"mooncake"})
    try:
        import mooncake.mooncake_ep_buffer as ep_module  # type: ignore

        logger.info("FD use Mooncake/EP now.")
        return ep_module
    except ImportError as e:
        logger.error(f"import mooncake.mooncake_ep_buffer failed! type={type(e).__name__}, err={e}")
        logger.error(f"Traceback:{traceback.format_exc()}")
        raise


mooncake = load_mooncake_ep()


@singleton
class MooncakeEPEngine:
    """
    Singleton engine that owns the mooncake EP Buffer and exposes
    low-latency dispatch / combine interfaces.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        ep_size: int,
        num_max_dispatch_tokens_per_rank: int,
        group,
        **kwargs,
    ):
        assert num_max_dispatch_tokens_per_rank <= 1024, (
            "Mooncake EP requires num_max_dispatch_tokens_per_rank <= 1024 "
            "(FINISHED_SUM_TAG constraint)."
        )

        self.active_ranks = paddle.ones((num_experts,), dtype=paddle.int32).cuda()

        num_bytes = mooncake.Buffer.get_ep_buffer_size_hint(
            num_max_dispatch_tokens_per_rank, hidden_size, ep_size, num_experts,
        )
        self.buffer = mooncake.Buffer(group.process_group._mc, num_bytes)

        self.num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        self.num_experts = num_experts
        # First call uses timeout=-1 (no timeout); subsequent calls use 10 s.
        self._first_execution = True
        self._timeout_us = 100_000_000

    def _get_timeout(self) -> int:
        return -1 if self._first_execution else self._timeout_us

    def low_latency_dispatch(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        use_fp8: bool = False,
    ):
        timeout = self._get_timeout()
        packed_recv_hidden, packed_recv_count, handle, _, hook = self.buffer.dispatch(
            hidden_states,
            topk_idx,
            self.active_ranks,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            timeout,
            use_fp8=use_fp8,
            async_finish=False,
            return_recv_hook=True,
        )
        self._first_execution = False
        return packed_recv_hidden, packed_recv_count, handle, hook

    def low_latency_combine(
        self,
        hidden_states: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        handle,
    ):
        timeout = self._get_timeout()
        combined_hidden, _, hook = self.buffer.combine(
            hidden_states,
            topk_idx,
            topk_weights,
            self.active_ranks,
            timeout,
            handle,
            async_finish=False,
            return_recv_hook=True,
        )
        self._first_execution = False
        return combined_hidden, hook


class MooncakeEPRunner(EPRunnerBase):
    """Mooncake-based EP runner base."""

    def __init__(
        self,
        top_k: int,
        hidden_size: int,
        num_experts: int,
        ep_size: int = 1,
        ep_rank: int = 0,
        ep_group=None,
        redundant_experts_num: int = 0,
        **kwargs,  # absorb deepep-only args (splitwise_role, moe_phase, …)
    ):
        super().__init__(top_k, num_experts, redundant_experts_num)
        self.ep_engine = MooncakeEPEngine(
            hidden_size=hidden_size,
            num_experts=num_experts + redundant_experts_num,
            ep_size=ep_size,
            num_max_dispatch_tokens_per_rank=envs.FD_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK,
            group=ep_group,
        )


class MooncakeEPDecoderRunner(MooncakeEPRunner):
    """
    Mooncake EP runner for the decode phase (low-latency path).
    Interface matches DeepEPDecoderRunner so the MoE forward pass needs no changes.
    """

    def dispatch(
        self,
        x: paddle.Tensor,
        topk_idx: paddle.Tensor,
        topk_weights: paddle.Tensor,
        *args,
        **kwargs,
    ):
        use_fp8 = kwargs.get("use_fp8", False)
        recv_hidden_states, recv_expert_count, handle, dispatch_hook = (
            self.ep_engine.low_latency_dispatch(x, topk_idx, use_fp8=use_fp8)
        )
        if dispatch_hook is not None:
            dispatch_hook()

        return recv_hidden_states, recv_expert_count, handle

    def combine(self, ffn_out, topk_idx, topk_weights, handle, **kwargs):
        combined_hidden_states, combine_hook = (
            self.ep_engine.low_latency_combine(ffn_out, topk_idx, topk_weights, handle)
        )
        if combine_hook is not None:
            combine_hook()

        return combined_hidden_states


class MooncakeEPPrefillRunner(MooncakeEPRunner):
    """
    Placeholder for the Mooncake EP prefill runner.
    Normal (high-throughput) mode is not yet implemented by mooncake-ep;
    instantiating this class will raise NotImplementedError.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Mooncake EP does not support the prefill (normal) mode yet. "
            "Only decode (low-latency) mode is available."
        )

    def dispatch(self, *args, **kwargs):
        raise NotImplementedError

    def combine(self, *args, **kwargs):
        raise NotImplementedError


# Canonical public names re-exported via ep.py dispatcher.
EPPrefillRunner = MooncakeEPPrefillRunner
EPDecoderRunner = MooncakeEPDecoderRunner