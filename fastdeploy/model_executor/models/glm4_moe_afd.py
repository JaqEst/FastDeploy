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

import re
from typing import Dict

import paddle
from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.glm4_moe import Glm4MoeDecoderLayer
from fastdeploy.model_executor.models.model_base import ModelCategory, ModelForCasualLM, ModelRegistry
from fastdeploy.worker.experts_manager import RedundantExpertManger


@support_graph_optimization
class Glm4AFDAttnModel(nn.Layer):
    def __init__(self, fd_config: FDConfig) -> None:
        super().__init__()

        self.fd_config = fd_config
        fd_config.model_config.pretrained_config.prefix_name = "model"
        self.num_layers = fd_config.model_config.num_hidden_layers

        self.redundant_table_manger = RedundantExpertManger(
            n_routed_experts=fd_config.model_config.n_routed_experts,
            num_hidden_layers=fd_config.model_config.num_hidden_layers,
            redundant_experts_num=fd_config.afd_config.num_redundant_experts,
            ep_size=fd_config.parallel_config.expert_parallel_size,
            fd_config=fd_config,
        )

        self.embed_tokens = VocabParallelEmbedding(
            fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens",
        )
        self.layers = nn.LayerList(
            [
                Glm4MoeDecoderLayer(
                    fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
                    redundant_table_manger=self.redundant_table_manger,
                )
                for i in range(self.num_layers)
            ]
        )
        self.norm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.norm",
        )

    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        residual = None

        for layer_id in range(self.num_layers):
            hidden_states, residual = self.layers[layer_id](forward_meta, hidden_states, residual)

        out = self.norm(hidden_states, residual, forward_meta=forward_meta)[0]

        if self.norm.is_last_norm and self.norm.fd_config.parallel_config.use_sequence_parallel_moe:
            out = self.norm.allgather(out, forward_meta.ids_remove_padding.shape[0])

        return out


class Glm4AFDFFNMoe(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str,
        redundant_table_manger: RedundantExpertManger,
    ) -> None:
        super().__init__()

        self.experts = FusedMoE(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            reduce_results=True,
            renormalize=fd_config.model_config.norm_topk_prob,
            moe_intermediate_size=fd_config.model_config.moe_intermediate_size,
            num_experts=fd_config.model_config.n_routed_experts,
            top_k=fd_config.model_config.num_experts_per_tok,
            topk_method="noaux_tc",
            topk_group=fd_config.model_config.topk_group,
            n_group=fd_config.model_config.n_group,
            routed_scaling_factor=fd_config.model_config.routed_scaling_factor,
            layer_idx=layer_id,
            gate_correction_bias=None,
            redundant_table_manger=redundant_table_manger,
            weight_key_map={
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
            },
            topk_reduce_func=lambda x: x.sum(axis=-1, keepdim=True) + 1e-20,
        )

    def forward(self, forward_meta: ForwardMeta = None) -> None:
        # This rank originates no tokens: FusedMoE.afd_skip_gate makes dispatch pull
        # the routed tokens from the ATTN ranks and combine send the results back.
        self.experts(self.experts.afd_dummy_x, None, forward_meta)


class Glm4AFDFFNDecoderLayer(nn.Layer):
    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str,
        redundant_table_manger: RedundantExpertManger,
    ) -> None:
        super().__init__()

        self.mlp = Glm4AFDFFNMoe(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.mlp",
            redundant_table_manger=redundant_table_manger,
        )

    def forward(self, forward_meta: ForwardMeta) -> None:
        self.mlp(forward_meta)


@support_graph_optimization
class Glm4AFDFFNModel(nn.Layer):
    def __init__(self, fd_config: FDConfig) -> None:
        super().__init__()

        self.fd_config = fd_config
        fd_config.model_config.pretrained_config.prefix_name = "model"
        self.layer_ids = list(range(
            fd_config.model_config.first_k_dense_replace, fd_config.model_config.num_hidden_layers
        ))
        self.num_layers = len(self.layer_ids)

        self.redundant_table_manger = RedundantExpertManger(
            n_routed_experts=fd_config.model_config.n_routed_experts,
            num_hidden_layers=fd_config.model_config.num_hidden_layers,
            redundant_experts_num=fd_config.afd_config.num_redundant_experts,
            ep_size=fd_config.parallel_config.expert_parallel_size,
            fd_config=fd_config,
        )

        self.layers = nn.LayerDict(
            {
                str(layer_id): Glm4AFDFFNDecoderLayer(
                    fd_config=fd_config,
                    layer_id=layer_id,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{layer_id}",
                    redundant_table_manger=self.redundant_table_manger,
                )
                for layer_id in self.layer_ids
            }
        )

    def forward(
        self,
        ids_remove_padding: paddle.Tensor = None,
        forward_meta: ForwardMeta = None,
    ) -> paddle.Tensor:
        # ids_remove_padding/forward_meta are graph-shape selectors for the
        # generic GraphOptBackend.  AFD FFN itself originates no tokens.
        for layer_id in self.layer_ids:
            self.layers[str(layer_id)](forward_meta)

        return paddle.empty(0, dtype=paddle.int32)


@ModelRegistry.register_model_class(
    architecture="Glm4MoeForCausalLM_AFDAttn",
    module_name="glm4_moe_afd",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class Glm4MoeForCausalLM_AFDAttn(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)

        self.model = Glm4AFDAttnModel(fd_config)
        self.redundant_table_manger = self.model.redundant_table_manger

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size

        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

    @classmethod
    def name(cls):
        return "Glm4MoeForCausalLM_AFDAttn"

    @paddle.no_grad()
    def load_weights(self, weights_iterator) -> None:
        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
        )

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("up_gate_proj", "gate_proj", "gate"),
            ("up_gate_proj", "up_proj", "up"),
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
            ("gate.e_score_correction_bias", "gate.e_score_correction_bias", None),
        ]
        if self.fd_config.model_config.use_qk_norm:
            stacked_params_mapping.append(("qk_norm.q_norm", "q_norm", None))
            stacked_params_mapping.append(("qk_norm.k_norm", "k_norm", None))

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)

        for loaded_weight_name, loaded_weight in weights_iterator:
            if ".mlp.experts." in loaded_weight_name:
                continue

            model_param_name = None
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                candidate = loaded_weight_name.replace(weight_name, param_name)
                if candidate not in params_dict:
                    continue
                param = params_dict[candidate]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                model_param_name = candidate
                break

            if model_param_name is None:
                if loaded_weight_name not in params_dict:
                    continue
                param = params_dict[loaded_weight_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight)
                model_param_name = loaded_weight_name

            model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        raise NotImplementedError("AFD models only support loading weights with the default_v1 loader.")

    def forward(self, inputs: Dict, forward_meta: ForwardMeta):
        ids_remove_padding = inputs["ids_remove_padding"]
        return self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

    def compute_logits(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta = None):
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")
        return logits

    def empty_input_forward(self, forward_meta):
        return None

    def clear_grpah_opt_backend(self):
        self.model.clear_grpah_opt_backend(fd_config=self.fd_config)

    @paddle.no_grad()
    def update_state_dict(self, state_dict):
        return None


@ModelRegistry.register_model_class(
    architecture="Glm4MoeForCausalLM_AFDFFN",
    module_name="glm4_moe_afd",
    category=ModelCategory.TEXT_GENERATION,
    primary_use=ModelCategory.TEXT_GENERATION,
)
class Glm4MoeForCausalLM_AFDFFN(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)

        self.model = Glm4AFDFFNModel(fd_config)
        self.redundant_table_manger = self.model.redundant_table_manger

    @classmethod
    def name(cls):
        return "Glm4MoeForCausalLM_AFDFFN"

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        from fastdeploy.model_executor.utils import process_weights_after_loading

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.n_routed_experts,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
        )

        for loaded_weight_name, loaded_weight in weights_iterator:
            if ".mlp.experts." not in loaded_weight_name:
                continue

            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]

                param.weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)

                model_sublayer_name = re.sub(
                    r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name
                )
                process_weights_after_loading_fn(model_sublayer_name, param)
                break

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        raise NotImplementedError("AFD models only support loading weights with the default_v1 loader.")

    def forward(self, inputs: Dict, forward_meta: ForwardMeta):
        ids_remove_padding = inputs["ids_remove_padding"]
        return self.model(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

    def compute_logits(self, hidden_states, forward_meta=None):
        return None

    def empty_input_forward(self, forward_meta):
        return self.forward(None, forward_meta)

    def clear_grpah_opt_backend(self):
        self.model.clear_grpah_opt_backend(fd_config=self.fd_config)

    @paddle.no_grad()
    def update_state_dict(self, state_dict):
        from fastdeploy.model_executor.utils import process_weights_after_loading

        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.n_routed_experts,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
        )

        for loaded_weight_name, loaded_weight in state_dict.items():
            if ".mlp.experts." not in loaded_weight_name:
                continue
            layer_id = int(loaded_weight_name.split(".mlp.experts.", 1)[0].rsplit(".", 1)[-1])
            if str(layer_id) not in self.model.layers:
                continue
            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]

                param.weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)

                model_sublayer_name = re.sub(
                    r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name
                )
                process_weights_after_loading_fn(model_sublayer_name, param)
                break
