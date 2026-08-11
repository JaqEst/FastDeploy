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

import paddle

from .afd import AFDA2ABackendBase


class AFDMooncakeA2ABackend(AFDA2ABackendBase):
    """Mooncake implementation for AFD all-to-all communication."""

    name = "mooncake"

    def __init__(self, fd_config):
        if fd_config.parallel_config.ep_group.backend != "mooncake":
            raise TypeError(
                "Mooncake AFD requires fd_config.parallel_config.ep_group to be a Mooncake process group. "
                "Set FD_MOE_A2A_BACKEND=mooncake so FastDeploy initializes Mooncake PG."
            )

        from fastdeploy.model_executor.layers.moe.ep import EPDecoderRunner

        self.ep_runner = EPDecoderRunner(
            fd_config.model_config.num_experts_per_tok,
            fd_config.model_config.hidden_size,
            fd_config.afd_config.num_physical_experts,
            fd_config.scheduler_config.splitwise_role,
            fd_config.model_config.num_max_dispatch_tokens_per_rank,
            ep_size=fd_config.parallel_config.expert_parallel_size,
            ep_rank=fd_config.parallel_config.expert_parallel_rank,
            ep_group=fd_config.parallel_config.ep_group,
            is_extension=fd_config.launch_config.is_extension,
        )

        # active ranks reference
        self.active_ranks = self.ep_runner.ep_engine.active_ranks
        self.last_active_ranks = self.ep_runner.ep_engine.last_active_ranks

    def dispatch_physical(self, x, physical_topk_idx, topk_weights, **kwargs):
        return self.ep_runner.dispatch(
            x,
            physical_topk_idx,
            topk_weights,
            timeout=kwargs.get("timeout_us", -1),
            use_fp8=kwargs.get("use_fp8", False),
        )

    def combine(self, ffn_out, physical_topk_idx, topk_weights, handle, **kwargs):
        return self.ep_runner.combine(
            ffn_out,
            physical_topk_idx,
            topk_weights,
            handle,
            timeout=kwargs.get("timeout_us", -1),
        )
