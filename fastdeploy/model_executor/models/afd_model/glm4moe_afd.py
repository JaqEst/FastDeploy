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
from collections import defaultdict
from typing import Dict, List, Set

import paddle
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.glm4_moe import (
    Glm4MoeAttention,
    Glm4MoeMLP,
    rms_norm_func,
)
from fastdeploy.model_executor.models.model_base import ModelForCasualLM

import fastdeploy


def _component_name(weight_name: str) -> str:
    if ".self_attn.q_norm." in weight_name or ".self_attn.k_norm." in weight_name:
        return "qk_norm"
    if ".self_attn." in weight_name:
        return "self_attn"
    if ".input_layernorm." in weight_name:
        return "input_layernorm"
    if ".post_attention_layernorm." in weight_name:
        return "post_attention_layernorm"
    if ".mlp.shared_experts." in weight_name:
        return "shared_experts"
    if ".mlp.gate.e_score_correction_bias" in weight_name:
        return "gate_bias"
    if ".mlp.gate." in weight_name:
        return "gate"
    if ".mlp.experts." in weight_name:
        return "routed_experts"
    if ".mlp." in weight_name:
        return "mlp"
    if weight_name.startswith("model.embed_tokens"):
        return "embed_tokens"
    if weight_name.startswith("model.norm"):
        return "norm"
    if weight_name.startswith("lm_head"):
        return "lm_head"
    return "other"


def _layer_index(weight_name: str) -> int | None:
    match = re.search(r"model\.layers\.(\d+)\.", weight_name)
    if match is None:
        return None
    return int(match.group(1))


def _log_actual_loaded_layers(
    role: str,
    loaded_names: List[str],
    moe_layer_ids: Set[int],
    num_layers: int,
) -> None:
    layer_components: Dict[int, Set[str]] = defaultdict(set)
    global_components: Set[str] = set()
    for weight_name in loaded_names:
        comp = _component_name(weight_name)
        layer_id = _layer_index(weight_name)
        if layer_id is None:
            global_components.add(comp)
        else:
            layer_components[layer_id].add(comp)

    dense_count = num_layers - len(moe_layer_ids)
    lines = [
        "",
        f"GLM4 AFD loaded layers summary ({role}):",
        f"  role: {role}",
        f"  total layers: {num_layers}",
        f"  dense layers: {dense_count}",
        f"  MoE layers: {len(moe_layer_ids)}",
    ]

    if global_components:
        lines.append(f"  global components: {', '.join(sorted(global_components))}")

    for layer_id in range(num_layers):
        layer_type = "MoE" if layer_id in moe_layer_ids else "dense"
        comps = sorted(layer_components.get(layer_id, set()))
        comp_text = ", ".join(comps) if comps else "(no weights loaded)"
        lines.append(f"  layer {layer_id:>3} [{layer_type:<5}] {comp_text}")

    logger.info("\n".join(lines))


class Glm4AFDAttnMoeBlock(nn.Layer):
    """The MoE-part that remains on the AFD ATTN worker."""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str) -> None:
        super().__init__()
        self.hidden_size = fd_config.model_config.hidden_size
        self.n_shared_experts = fd_config.model_config.n_shared_experts

        from fastdeploy.model_executor.layers.linear import ReplicatedLinear

        self.gate = ReplicatedLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.gate",
            input_size=self.hidden_size,
            output_size=fd_config.model_config.n_routed_experts,
            with_bias=False,
            skip_quant=True,
            weight_dtype=("float32" if fd_config.model_config.moe_gate_fp32 else ""),
        )
        self.gate.e_score_correction_bias = self.create_parameter(
            shape=[1, fd_config.model_config.n_routed_experts],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        self.shared_experts = None
        if self.n_shared_experts > 0:
            shared_experts_intermediate_size = self.n_shared_experts * fd_config.model_config.moe_intermediate_size
            self.shared_experts = Glm4MoeMLP(
                fd_config=fd_config,
                intermediate_size=shared_experts_intermediate_size,
                layer_id=layer_id,
                prefix=f"{prefix}.shared_experts",
                reduce_results=True,
            )

    def forward(self, x: paddle.Tensor, forward_meta: ForwardMeta = None) -> paddle.Tensor:
        _ = self.gate(x)
        if self.shared_experts is None:
            return paddle.zeros_like(x)
        return self.shared_experts(x, forward_meta)


class Glm4AFDAttnDecoderLayer(nn.Layer):
    def __init__(self, fd_config: FDConfig, prefix: str) -> None:
        super().__init__()

        layer_id = int(prefix.split(sep=".")[-1])
        self.self_attn = Glm4MoeAttention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        if fd_config.model_config.n_routed_experts is not None and layer_id >= fd_config.model_config.first_k_dense_replace:
            self.mlp = Glm4AFDAttnMoeBlock(fd_config, layer_id=layer_id, prefix=f"{prefix}.mlp")
        else:
            self.mlp = Glm4MoeMLP(
                fd_config,
                intermediate_size=fd_config.model_config.intermediate_size,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.input_layernorm",
            layer_id=layer_id,
        )
        self.post_attention_layernorm = RMSNorm(
            fd_config,
            hidden_size=fd_config.model_config.hidden_size,
            eps=fd_config.model_config.rms_norm_eps,
            prefix=f"{prefix}.post_attention_layernorm",
            layer_id=layer_id,
        )

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
        residual: paddle.Tensor = None,
    ):
        proxy_rmsnorm = rms_norm_func if fastdeploy.envs.FD_USE_PHI_RMSNORM else None
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta, proxy_rmsnorm=proxy_rmsnorm
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states, 
            forward_meta=forward_meta
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual, proxy_rmsnorm=proxy_rmsnorm)
        
        hidden_states = self.mlp(hidden_states, forward_meta)
        
        return hidden_states, residual


class Glm4AFDAttnModel(nn.Layer):
    def __init__(self, fd_config: FDConfig) -> None:
        super().__init__()
        self.num_layers = fd_config.model_config.num_hidden_layers
        fd_config.model_config.pretrained_config.prefix_name = "model"

        self.embed_tokens = VocabParallelEmbedding(
            fd_config,
            num_embeddings=fd_config.model_config.vocab_size,
            embedding_dim=fd_config.model_config.hidden_size,
            params_dtype=paddle.get_default_dtype,
            prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.embed_tokens",
        )
        self.layers = nn.LayerList(
            [
                Glm4AFDAttnDecoderLayer(
                    fd_config,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{i}",
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


class Glm4AFDFFNMoeBlock(nn.Layer):
    """The routed expert weights owned by the AFD FFN worker."""

    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str) -> None:
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
            weight_key_map={
                "up_gate_proj_expert_weight_key": f"{prefix}.experts.{{}}.up_gate_proj.weight",
                "down_proj_expert_weight_key": f"{prefix}.experts.{{}}.down_proj.weight",
            },
            topk_reduce_func=lambda x: x.sum(axis=-1, keepdim=True) + 1e-20,
        )


class Glm4MoeForCausalLM_AFDAttn(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)

        self.model = Glm4AFDAttnModel(fd_config)

        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        
        self.lm_head = ParallelLMHead(
            fd_config,
            embedding_dim=fd_config.model_config.hidden_size,
            num_embeddings=fd_config.model_config.vocab_size,
            prefix="lm_head",
        )

        self._moe_layer_ids = list(
            range(fd_config.model_config.first_k_dense_replace, fd_config.model_config.num_hidden_layers)
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
        loaded_names: List[str] = []

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

            loaded_names.append(loaded_weight_name)
            model_sublayer_name = re.sub(r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name)
            process_weights_after_loading_fn(model_sublayer_name, param)

        _log_actual_loaded_layers(
            role="attn",
            loaded_names=loaded_names,
            moe_layer_ids=self._moe_layer_ids,
            num_layers=self.fd_config.model_config.num_hidden_layers,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        pass

    def forward(self, inputs: Dict, forward_meta: ForwardMeta):
        ids_remove_padding = inputs["ids_remove_padding"]
        hidden_states = self.model.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)
        residual = None

        for layer_id in range(self.model.num_layers):
            hidden_states, residual = self.model.layers[layer_id](forward_meta, hidden_states, residual)

        out = self.model.norm(hidden_states, residual, forward_meta=forward_meta)[0]
        if self.model.norm.is_last_norm and self.model.norm.fd_config.parallel_config.use_sequence_parallel_moe:
            out = self.model.norm.allgather(out, forward_meta.ids_remove_padding.shape[0])
        return out

    def compute_logits(self, hidden_states: paddle.Tensor, forward_meta: ForwardMeta = None):
        logits = self.lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        logits[:, self.ori_vocab_size :] = -float("inf")
        return logits

    def empty_input_forward(self, forward_meta):
        return None


class Glm4MoeForCausalLM_AFDFFN(ModelForCasualLM):
    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)
        self._moe_layer_ids = list(
            range(fd_config.model_config.first_k_dense_replace, fd_config.model_config.num_hidden_layers)
        )
        self.moe_blocks = nn.LayerDict(
            {
                str(layer_id): Glm4AFDFFNMoeBlock(
                    fd_config=fd_config,
                    layer_id=layer_id,
                    prefix=f"model.layers.{layer_id}.mlp",
                )
                for layer_id in self._moe_layer_ids
            }
        )

    @classmethod
    def name(cls):
        return "Glm4MoeForCausalLM_AFDFFN"

    @paddle.no_grad()
    def load_weights(self, weights_iterator):
        from fastdeploy.model_executor.utils import process_weights_after_loading

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            num_experts=self.fd_config.model_config.n_routed_experts,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            param_gate_up_proj_name="experts.up_gate_proj_",
            param_down_proj_name="experts.down_proj_",
        )
        params_dict = dict(self.named_parameters())
        process_weights_after_loading_fn = process_weights_after_loading(dict(self.named_sublayers()), self.fd_config)
        loaded_names: List[str] = []

        for loaded_weight_name, loaded_weight in weights_iterator:
            if ".mlp.experts." not in loaded_weight_name:
                continue

            for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                model_param_name = model_param_name.replace("model.layers.", "moe_blocks.")
                model_param_name = model_param_name.replace(".mlp.experts.", ".experts.")
                if model_param_name not in params_dict:
                    continue
                param = params_dict[model_param_name]
                param.weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                loaded_names.append(loaded_weight_name)
                model_sublayer_name = re.sub(
                    r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name
                )
                process_weights_after_loading_fn(model_sublayer_name, param)
                break

        _log_actual_loaded_layers(
            role="ffn",
            loaded_names=loaded_names,
            moe_layer_ids=set(self._moe_layer_ids),
            num_layers=self.fd_config.model_config.num_hidden_layers,
        )

    @paddle.no_grad()
    def set_state_dict(self, state_dict):
        pass

    def forward(self, inputs: Dict, forward_meta: ForwardMeta):
        return None

    def compute_logits(self, hidden_states, forward_meta=None):
        return None

    def empty_input_forward(self, forward_meta):
        return None
