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

from fastdeploy import envs
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.dual_batch_overlap.dbo_runner import (
    DBOMicroState,
    assert_supports_dbo,
    run_dbo_pipeline,
)
from fastdeploy.model_executor.dual_batch_overlap.dbo_split import (
    split_decode_forward_meta,
)
from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead
from fastdeploy.model_executor.layers.moe.moe import FusedMoE, get_moe_scores
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.models.glm4_moe import (
    Glm4MoeAttention,
    Glm4MoeMLP,
    rms_norm_func,
)
from fastdeploy.model_executor.models.model_base import ModelCategory, ModelForCasualLM, ModelRegistry
from fastdeploy.worker.experts_manager import RedundantExpertManger


def afd_comm_kwargs(forward_meta: ForwardMeta):
    """Per-forward EP communication overrides shared by ATTN and FFN workers."""
    if forward_meta is not None and forward_meta.timeout_us:
        return {"timeout": forward_meta.timeout_us}
    return {}


class Glm4AFDAttnMoeBlock(nn.Layer):
    """MoE block on the ATTN worker.

    Holds the *gate* (router) and *shared experts*.  Routed expert
    computation is offloaded to FFN workers via DeepEP dispatch / combine.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str,
        redundant_table_manger: RedundantExpertManger,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = fd_config.model_config.hidden_size
        self.n_shared_experts = fd_config.model_config.n_shared_experts

        # Routing parameters (needed by get_moe_scores)
        self.top_k = fd_config.model_config.num_experts_per_tok
        self.n_group = fd_config.model_config.n_group
        self.topk_group = fd_config.model_config.topk_group
        self.routed_scaling_factor = fd_config.model_config.routed_scaling_factor
        self.renormalize = fd_config.model_config.norm_topk_prob

        from fastdeploy.model_executor.layers.moe.ep import EPDecoderRunner

        self.afd_runner = EPDecoderRunner(
            fd_config.model_config.num_experts_per_tok,
            fd_config.model_config.hidden_size,
            fd_config.afd_config.num_physical_experts,
            fd_config.scheduler_config.splitwise_role,
            fd_config.model_config.num_max_dispatch_tokens_per_rank,
            ep_size=fd_config.parallel_config.expert_parallel_size,
            ep_rank=fd_config.parallel_config.expert_parallel_rank,
            ep_group=fd_config.parallel_config.ep_group,
            is_extension=fd_config.launch_config.is_extension,
            use_internode_ll_two_stage=False,
        )

        self.redundant_table_manger = redundant_table_manger

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
            shared_experts_intermediate_size = (
                self.n_shared_experts * fd_config.model_config.moe_intermediate_size
            )
            self.shared_experts = Glm4MoeMLP(
                fd_config=fd_config,
                intermediate_size=shared_experts_intermediate_size,
                layer_id=layer_id,
                prefix=f"{prefix}.shared_experts",
                reduce_results=True,
            )

    def route(self, x: paddle.Tensor):
        """Gate + top-k selection, mapped into the physical expert id space.

        dispatch/combine must use the same physical id space; the topk kernel reads the
        table manager's placement and already returns AFD physical ids.
        """
        gate_out = self.gate(x)
        gate_out = gate_out.cast("float32")

        (
            _ep_rank_to_expert_id_list,
            expert_id_to_ep_rank_array,
            expert_in_rank_num_list,
            tokens_per_expert_stats_list,
        ) = self.redundant_table_manger.get_ep_rank_to_expert_id_list_by_layer(self.layer_id)

        _score, topk_weights, physical_topk_idx = get_moe_scores(
            gate_out,
            self.n_group,
            self.topk_group,
            self.top_k,
            self.routed_scaling_factor,
            self.gate.e_score_correction_bias,
            self.renormalize,
            expert_id_to_ep_rank_array=expert_id_to_ep_rank_array,
            expert_in_rank_num_list=expert_in_rank_num_list,
            tokens_per_expert_stats_list=tokens_per_expert_stats_list,
            redundant_ep_rank_num_plus_one=self.redundant_table_manger.redundant_experts_num + 1,
        )

        return physical_topk_idx, topk_weights

    def forward(self, x: paddle.Tensor, forward_meta: ForwardMeta = None) -> paddle.Tensor:
        # --- 1. routing ---
        physical_topk_idx, topk_weights = self.route(x)

        # --- 2. dispatch tokens to FFN workers ---
        comm_kwargs = afd_comm_kwargs(forward_meta)
        dummy_recv_x, _, dummy_handle = self.afd_runner.dispatch(
            x,
            physical_topk_idx,
            topk_weights,
            **comm_kwargs,
        )

        # --- 3. shared experts ---
        if self.shared_experts is not None:
            shared_out = self.shared_experts(x, forward_meta)

        # --- 4. combine results from FFN workers ---
        routed_out = self.afd_runner.combine(
            dummy_recv_x,
            physical_topk_idx,
            topk_weights,
            dummy_handle,
            **comm_kwargs
        )

        if self.shared_experts is not None:
            routed_out = routed_out + shared_out

        return routed_out


class Glm4AFDAttnDecoderLayer(nn.Layer):
    supports_dbo = True

    def __init__(
        self,
        fd_config: FDConfig,
        prefix: str,
        redundant_table_manger: RedundantExpertManger,
    ) -> None:
        super().__init__()

        layer_id = int(prefix.split(sep=".")[-1])
        self.self_attn = Glm4MoeAttention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.self_attn",
        )

        if (
            fd_config.model_config.n_routed_experts is not None
            and layer_id >= fd_config.model_config.first_k_dense_replace
        ):
            self.mlp = Glm4AFDAttnMoeBlock(
                fd_config,
                layer_id=layer_id,
                prefix=f"{prefix}.mlp",
                redundant_table_manger=redundant_table_manger,
            )
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
        proxy_rmsnorm = rms_norm_func if envs.FD_USE_PHI_RMSNORM else None
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual_input=residual, forward_meta=forward_meta, proxy_rmsnorm=proxy_rmsnorm
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            forward_meta=forward_meta,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual, proxy_rmsnorm=proxy_rmsnorm)

        hidden_states = self.mlp(hidden_states, forward_meta)

        return hidden_states, residual

    @property
    def is_moe_layer(self) -> bool:
        return isinstance(self.mlp, Glm4AFDAttnMoeBlock)

    def dbo_attn(self, st):
        proxy_rmsnorm = rms_norm_func if envs.FD_USE_PHI_RMSNORM else None
        h, residual = self.input_layernorm(
            st.hidden_states,
            residual_input=st.residual,
            forward_meta=st.forward_meta,
            proxy_rmsnorm=proxy_rmsnorm,
        )
        h = self.self_attn(hidden_states=h, forward_meta=st.forward_meta)
        h, residual = self.post_attention_layernorm(h, residual, proxy_rmsnorm=proxy_rmsnorm)
        st.residual = residual
        st.stash["x"] = h
        st.stash["topk_idx"], st.stash["topk_weights"] = self.mlp.route(h)
        st.stash["comm_kwargs"] = afd_comm_kwargs(st.forward_meta)

    def dbo_dispatch_send(self, st):
        recv, _, handle, hook = self.mlp.afd_runner.dispatch(
            st.stash["x"],
            st.stash["topk_idx"],
            st.stash["topk_weights"],
            return_hook=True,
            **st.stash["comm_kwargs"],
        )
        st.stash["recv"] = recv
        st.stash["handle"] = handle
        st.stash["hook_dispatch"] = hook

    def dbo_dispatch_recv(self, st):
        hook = st.stash.pop("hook_dispatch")
        if hook is not None:
            hook()

    def dbo_local(self, st):
        # Attn worker's local compute is the shared experts; it only depends on
        # the layer input, so it is safe between dispatch and combine.
        st.stash["shared_out"] = (
            self.mlp.shared_experts(st.stash["x"], st.forward_meta)
            if self.mlp.shared_experts is not None
            else None
        )

    def dbo_combine_send(self, st):
        routed, hook = self.mlp.afd_runner.combine(
            st.stash["recv"],
            st.stash["topk_idx"],
            st.stash["topk_weights"],
            st.stash["handle"],
            return_hook=True,
            **st.stash["comm_kwargs"],
        )
        st.stash["routed"] = routed
        st.stash["hook_combine"] = hook

    def dbo_combine_recv(self, st):
        hook = st.stash.pop("hook_combine")
        if hook is not None:
            hook()
        out = st.stash.pop("routed")
        shared_out = st.stash.pop("shared_out")
        if shared_out is not None:
            out = out + shared_out
        st.hidden_states = out
        st.stash.clear()


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
                Glm4AFDAttnDecoderLayer(
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

        self._init_dbo(fd_config)

    def _init_dbo(self, fd_config: FDConfig) -> None:
        """Prepare the dual-batch overlap schedule (no-op unless afd_config.enable_dbo)."""
        self.enable_dbo = fd_config.afd_config.enable_dbo
        if not self.enable_dbo:
            return
        if fd_config.speculative_config.enabled_speculative_decoding():
            # The split assumes one token per live slot, so every token boundary is
            # also a slot boundary; speculative decoding breaks that.
            raise NotImplementedError("AFD DBO does not support speculative decoding yet.")
        # Plain lists reference the already-registered layers in self.layers.
        self.dbo_dense_layers = []
        self.dbo_moe_layers = []
        for layer in self.layers:
            (self.dbo_moe_layers if layer.is_moe_layer else self.dbo_dense_layers).append(layer)
        # _forward_dbo runs every dense layer before entering the pipeline, which
        # only preserves layer order because the dense layers are a prefix
        # (layer_id < first_k_dense_replace).
        num_dense = len(self.dbo_dense_layers)
        if any(self.layers[i].is_moe_layer for i in range(num_dense)):
            raise NotImplementedError("AFD DBO requires the dense layers to be a contiguous prefix.")
        assert_supports_dbo(self.dbo_moe_layers)
        # Swapped in here rather than branched on per step. The graph-opt decorator
        # captures self.forward after __init__ returns, so it picks this up.
        self.forward = self._forward_dbo

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

    def _forward_dbo(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
    ):
        hidden_states = self.embed_tokens(ids_remove_padding=ids_remove_padding, forward_meta=forward_meta)

        residual = None

        for layer in self.dbo_dense_layers:
            hidden_states, residual = layer(forward_meta, hidden_states, residual)

        # Each micro-batch needs its own split-kv plan.
        meta_a, meta_b = split_decode_forward_meta(forward_meta)
        for meta in (meta_a, meta_b):
            forward_meta.attn_backend.plan_split_kv_block(meta)

        num_tokens_a = meta_a.ids_remove_padding.shape[0]
        h_a, h_b = hidden_states[:num_tokens_a], hidden_states[num_tokens_a:]
        r_a, r_b = (None, None) if residual is None else (residual[:num_tokens_a], residual[num_tokens_a:])
        state_a = DBOMicroState(h_a, r_a, meta_a, 0)
        state_b = DBOMicroState(h_b, r_b, meta_b, 1)

        run_dbo_pipeline(self.dbo_moe_layers, state_a, state_b)

        sp_allgather = self.norm.is_last_norm and self.norm.fd_config.parallel_config.use_sequence_parallel_moe

        outs = []
        for st in (state_a, state_b):
            out = self.norm(st.hidden_states, st.residual, forward_meta=st.forward_meta)[0]
            if sp_allgather:
                out = self.norm.allgather(out, st.forward_meta.ids_remove_padding.shape[0])
            outs.append(out)

        return paddle.concat(outs, axis=0)


class Glm4AFDFFNMoeBlock(nn.Layer):
    """Routed-expert weights for one MoE layer on the FFN worker.

    ``FusedMoE`` keeps the model's logical expert count for checkpoint name
    matching; the redundant table manager supplies the inflated AFD physical
    expert space used by dispatch/combine.
    """

    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str,
        redundant_table_manger: RedundantExpertManger,
        dummy_inputs: tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor],
    ) -> None:
        super().__init__()

        from fastdeploy.model_executor.layers.moe.ep import EPDecoderRunner

        self.afd_runner = EPDecoderRunner(
            fd_config.model_config.num_experts_per_tok,
            fd_config.model_config.hidden_size,
            fd_config.afd_config.num_physical_experts,
            fd_config.scheduler_config.splitwise_role,
            fd_config.model_config.num_max_dispatch_tokens_per_rank,
            ep_size=fd_config.parallel_config.expert_parallel_size,
            ep_rank=fd_config.parallel_config.expert_parallel_rank,
            ep_group=fd_config.parallel_config.ep_group,
            is_extension=fd_config.launch_config.is_extension,
            use_internode_ll_two_stage=False,
        )
        self.dummy_x, self.dummy_topk_idx, self.dummy_topk_weights = dummy_inputs

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

    def compute_experts(self, recv_hidden: paddle.Tensor, recv_count: paddle.Tensor) -> paddle.Tensor:
        """Run local routed experts on tokens received from ATTN ranks."""
        layer = self.experts

        dequant_scale = None
        permute_input = recv_hidden
        if isinstance(recv_hidden, (list, tuple)):
            permute_input, dequant_scale = recv_hidden

        num_local_experts = permute_input.shape[0]
        max_num = permute_input.shape[1]
        estimate_total = max_num * num_local_experts

        moe_quant_type = getattr(layer.quant_method, "moe_quant_type", None)
        if moe_quant_type in ["w4a8", "w4afp8"] or layer.with_bias:
            expert_idx_per_token = paddle.arange(num_local_experts, device=permute_input.place)[:, None].tile(
                [1, max_num]
            )
        else:
            expert_idx_per_token = None

        return layer.quant_method.compute_ffn(
            layer,
            permute_input,
            recv_count.cast("int64"),
            expert_idx_per_token,
            True,
            estimate_total,
            dequant_scale,
        )

    def forward(self, forward_meta: ForwardMeta = None) -> None:
        comm_kwargs = afd_comm_kwargs(forward_meta)
        recv_x, recv_count, handle = self.afd_runner.dispatch(
            self.dummy_x,
            self.dummy_topk_idx,
            self.dummy_topk_weights,
            **comm_kwargs,
        )
        ffn_out = self.compute_experts(recv_x, recv_count)
        self.afd_runner.combine(
            ffn_out,
            self.dummy_topk_idx,
            self.dummy_topk_weights,
            handle,
            **comm_kwargs,
        )


class Glm4AFDFFNDecoderLayer(nn.Layer):
    """Dispatch -> local experts -> combine for one MoE layer on the FFN worker."""

    supports_dbo = True

    def __init__(
        self,
        fd_config: FDConfig,
        layer_id: int,
        prefix: str,
        redundant_table_manger: RedundantExpertManger,
        dummy_inputs: tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor],
    ) -> None:
        super().__init__()

        self.mlp = Glm4AFDFFNMoeBlock(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=f"{prefix}.mlp",
            redundant_table_manger=redundant_table_manger,
            dummy_inputs=dummy_inputs,
        )

    def forward(self, forward_meta: ForwardMeta) -> None:
        self.mlp(forward_meta)

    def dbo_attn(self, st):
        st.stash["comm_kwargs"] = afd_comm_kwargs(st.forward_meta)

    def dbo_dispatch_send(self, st):
        recv, count, handle, hook = self.mlp.afd_runner.dispatch(
            self.mlp.dummy_x,
            self.mlp.dummy_topk_idx,
            self.mlp.dummy_topk_weights,
            return_hook=True,
            **st.stash["comm_kwargs"],
        )
        st.stash["recv"] = recv
        st.stash["count"] = count
        st.stash["handle"] = handle
        st.stash["hook_dispatch"] = hook

    def dbo_dispatch_recv(self, st):
        hook = st.stash.pop("hook_dispatch")
        if hook is not None:
            hook()

    def dbo_local(self, st):
        st.stash["ffn_out"] = self.mlp.compute_experts(st.stash["recv"], st.stash["count"])

    def dbo_combine_send(self, st):
        _, hook = self.mlp.afd_runner.combine(
            st.stash["ffn_out"],
            self.mlp.dummy_topk_idx,
            self.mlp.dummy_topk_weights,
            st.stash["handle"],
            return_hook=True,
            **st.stash["comm_kwargs"],
        )
        st.stash["hook_combine"] = hook

    def dbo_combine_recv(self, st):
        hook = st.stash.pop("hook_combine")
        if hook is not None:
            hook()
        st.stash.clear()


@support_graph_optimization
class Glm4AFDFFNModel(nn.Layer):
    """Executable FFN participant body for AFD.

    The outer CausalLM owns loading/logits APIs.  This inner layer owns the
    per-layer dispatch -> local expert compute -> combine sequence so it can be
    captured and replayed independently from the ATTN worker graph.
    """

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

        self._dummy_inputs = (
            paddle.empty(0, fd_config.model_config.hidden_size, dtype=paddle.get_default_dtype()),
            paddle.empty(0, fd_config.model_config.num_experts_per_tok, dtype=paddle.int64),
            paddle.empty(0, fd_config.model_config.num_experts_per_tok, dtype=paddle.float32),
        )

        self.layers = nn.LayerDict(
            {
                str(layer_id): Glm4AFDFFNDecoderLayer(
                    fd_config=fd_config,
                    layer_id=layer_id,
                    prefix=f"{fd_config.model_config.pretrained_config.prefix_name}.layers.{layer_id}",
                    redundant_table_manger=self.redundant_table_manger,
                    dummy_inputs=self._dummy_inputs,
                )
                for layer_id in self.layer_ids
            }
        )

        self._init_dbo(fd_config)

    def _init_dbo(self, fd_config: FDConfig) -> None:
        """Prepare the dual-batch overlap schedule (no-op unless afd_config.enable_dbo)."""
        self.enable_dbo = fd_config.afd_config.enable_dbo
        self._dbo_layers = []
        if not self.enable_dbo:
            return
        if fd_config.speculative_config.enabled_speculative_decoding():
            raise NotImplementedError("AFD DBO does not support speculative decoding yet.")
        self._dbo_layers = [self.layers[str(layer_id)] for layer_id in self.layer_ids]
        assert_supports_dbo(self._dbo_layers)
        self.forward = self._forward_dbo

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

    def _forward_dbo(
        self,
        ids_remove_padding: paddle.Tensor = None,
        forward_meta: ForwardMeta = None,
    ) -> paddle.Tensor:
        state_a = DBOMicroState(None, None, forward_meta, 0)
        state_b = DBOMicroState(None, None, forward_meta, 1)

        run_dbo_pipeline(self._dbo_layers, state_a, state_b)
        return paddle.empty(0, dtype=paddle.int32)


@ModelRegistry.register_model_class(
    architecture="Glm4MoeForCausalLM_AFDAttn",
    module_name="afd_model.glm4moe_afd",
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
    module_name="afd_model.glm4moe_afd",
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
                moe_block = self.model.layers[str(layer_id)].mlp
                weight_loader = getattr(param, "weight_loader", moe_block.experts.weight_loader)
                weight_loader(param, loaded_weight, shard_id=shard_id, expert_id=expert_id)
                model_sublayer_name = re.sub(
                    r"\.(up_gate_proj_weight|down_proj_weight|weight)$", "", model_param_name
                )
                process_weights_after_loading_fn(model_sublayer_name, param)
                break
