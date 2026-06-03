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

from abc import abstractmethod

import paddle
from paddle import nn
from paddleformers.utils.log import logger

import fastdeploy


class EPRunnerBase:
    """
    Abstract base class for all EP runner backends (DeepEP, Mooncake, …).

    Subclasses must implement dispatch() and combine().

    Buffer-management hooks (clean_low_latency_buffer, clear/create_ep_buffer)
    default to no-ops.  Backends that manage an explicit buffer should override
    them as needed; backends that do not need buffer management can simply
    inherit the defaults.
    """

    def __init__(self, top_k: int, num_experts: int, redundant_experts_num: int = 0):
        self.top_k = top_k
        self.num_experts = num_experts
        self.redundant_experts_num = redundant_experts_num

    def moe_select(self, layer: nn.Layer, gate_out: paddle.Tensor):
        if layer.redundant_table_manger is not None:
            (
                ep_rank_to_expert_id_list,
                expert_id_to_ep_rank_array,
                expert_in_rank_num_list,
                tokens_per_expert_stats_list,
            ) = layer.redundant_table_manger.get_ep_rank_to_expert_id_list_by_layer(layer.layer_idx)

            if layer.topk_method == "noaux_tc":
                from .moe import get_moe_scores

                score, topk_weights, topk_idx = get_moe_scores(
                    gate_out,
                    layer.n_group,
                    layer.topk_group,
                    layer.top_k,
                    layer.routed_scaling_factor,
                    layer.gate_correction_bias,
                    getattr(layer, "renormalize", True),
                    expert_id_to_ep_rank_array=expert_id_to_ep_rank_array,
                    expert_in_rank_num_list=expert_in_rank_num_list,
                    tokens_per_expert_stats_list=tokens_per_expert_stats_list,
                    redundant_ep_rank_num_plus_one=layer.fd_config.eplb_config.redundant_experts_num + 1,
                    topk_reduce_func=getattr(layer, "topk_reduce_func", None),
                )
            else:
                topk_idx, topk_weights = fastdeploy.model_executor.ops.gpu.moe_redundant_topk_select(
                    gating_logits=gate_out,
                    expert_id_to_ep_rank_array=expert_id_to_ep_rank_array,
                    expert_in_rank_num_list=expert_in_rank_num_list,
                    tokens_per_expert_stats_list=tokens_per_expert_stats_list,
                    bias=layer.gate_correction_bias,
                    moe_topk=self.top_k,
                    apply_norm_weight=True,
                    enable_softmax_top_k_fused=False,
                    redundant_ep_rank_num_plus_one=layer.fd_config.eplb_config.redundant_experts_num + 1,
                )
        else:
            if layer.topk_method == "noaux_tc":
                from fastdeploy.model_executor.layers.moe.moe import get_moe_scores

                score, topk_weights, topk_idx = get_moe_scores(
                    gate_out,
                    layer.n_group,
                    layer.topk_group,
                    layer.top_k,
                    layer.routed_scaling_factor,
                    layer.gate_correction_bias,
                    getattr(layer, "renormalize", True),
                    topk_reduce_func=getattr(layer, "topk_reduce_func", None),
                )
            else:
                topk_idx, topk_weights = fastdeploy.model_executor.ops.gpu.moe_topk_select(
                    gate_out,
                    layer.gate_correction_bias,
                    self.top_k,
                    True,
                    False,
                )
        return topk_idx, topk_weights

    @abstractmethod
    def dispatch(self, *args, **kwargs):
        """
        Scatter input tokens to the target expert ranks.
        """
        raise NotImplementedError

    @abstractmethod
    def combine(self, *args, **kwargs):
        """
        Gather expert outputs back to the originating ranks.
        """
        raise NotImplementedError

    def clean_low_latency_buffer(self):
        """
        Reset low-latency buffer state before each dispatch round.
        Backends that require per-step cleanup should override this.
        """
        pass

    def clear_ep_buffer(self):
        """
        Release buffer resources. Override if the backend manages a buffer.
        """
        pass

    def create_ep_buffer(self):
        """
        Allocate buffer resources. Override if the backend manages a buffer.
        """
        pass