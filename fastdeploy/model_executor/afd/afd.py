from __future__ import annotations

import os
from typing import Dict, List

import paddle
from paddleformers.utils.log import logger

from fastdeploy import envs
from fastdeploy.utils import singleton


def _worker_device_from_config(fd_config) -> str:
    """Return the logical Paddle GPU device for the current worker process."""
    if fd_config.device_config.device_type != "cuda":
        return paddle.device.get_device()

    selected_gpus = os.getenv("FLAGS_selected_gpus")
    if selected_gpus:
        selected = [gpu.strip() for gpu in selected_gpus.split(",") if gpu.strip()]
        if len(selected) == 1:
            return f"gpu:{selected[0]}"

    device_ids = str(fd_config.parallel_config.device_ids).split(",")
    local_rank = int(os.getenv("PADDLE_LOCAL_RANK", "0"))
    return f"gpu:{local_rank % max(1, len(device_ids))}"


@singleton
class AFDWorldTopology:
    """Runtime world topology for one AFD worker."""

    def __init__(self, world_size, attn_ranks, ffn_ranks) -> None:
        if world_size <= 0:
            raise ValueError(f"AFD world_size must be positive, got {world_size}")
        if not attn_ranks:
            raise ValueError("AFD requires at least one ATTN rank.")
        if not ffn_ranks:
            raise ValueError("AFD requires at least one FFN rank.")
        combined_ranks = list(attn_ranks) + list(ffn_ranks)
        duplicated_ranks = sorted({rank for rank in combined_ranks if combined_ranks.count(rank) > 1})
        if duplicated_ranks:
            raise ValueError(f"AFD ranks must be unique, duplicated={duplicated_ranks}")
        invalid_ranks = sorted(rank for rank in combined_ranks if rank < 0 or rank >= world_size)
        if invalid_ranks:
            raise ValueError(f"AFD ranks must be in [0, {world_size}), invalid={invalid_ranks}")
        if len(set(attn_ranks).intersection(ffn_ranks)) > 0:
            raise ValueError(f"AFD ATTN/FFN ranks overlap: attn={attn_ranks}, ffn={ffn_ranks}")
        if len(attn_ranks) + len(ffn_ranks) != world_size:
            raise ValueError(
                "AFD rank count mismatch: "
                f"world_size={world_size}, attn={attn_ranks}, ffn={ffn_ranks}"
            )
        missing_ranks = sorted(set(range(world_size)).difference(combined_ranks))
        if missing_ranks:
            raise ValueError(f"AFD topology misses world ranks: {missing_ranks}")
        self.world_size = world_size
        self.attn_ranks: List[int] = sorted(attn_ranks)
        self.ffn_ranks: List[int] = sorted(ffn_ranks)


@singleton
class AFDExpertLayout:
    """Runtime logical-to-physical expert layout for AFD.

    Physical expert space is inflated so that every rank (including ATTN ranks)
    has ``num_local_physical_experts`` DeepEP slots.  ATTN rank slots are
    *phantom* (never routed to); only FFN rank slots carry real experts.

    Mapping formula (logical -> physical):
        ffn_rank_index  = logical_id // num_local_physical_experts
        ffn_global_rank = ffn_ranks[ffn_rank_index]
        physical_id     = ffn_global_rank * num_local_physical_experts
                        + (logical_id % num_local_physical_experts)
    """

    def __init__(self, n_routed_experts: int, redundant_experts_num: int = 0) -> None:
        self.afd_world_topology = AFDWorldTopology()
        self.num_attn_ranks = len(self.afd_world_topology.attn_ranks)
        self.num_ffn_ranks = len(self.afd_world_topology.ffn_ranks)
        self.world_size = self.afd_world_topology.world_size

        self.num_logical_experts = n_routed_experts
        self.redundant_experts_num = redundant_experts_num
        if n_routed_experts <= 0:
            raise ValueError(f"n_routed_experts must be positive, got {n_routed_experts}")
        if redundant_experts_num < 0:
            raise ValueError(f"redundant_experts_num must be non-negative, got {redundant_experts_num}")
        self.num_ffn_physical_experts = n_routed_experts + redundant_experts_num
        if self.num_ffn_physical_experts % self.num_ffn_ranks != 0:
            raise ValueError(
                "AFD requires logical + redundant experts to be evenly sharded over FFN ranks: "
                f"n_routed_experts={n_routed_experts}, "
                f"redundant_experts_num={redundant_experts_num}, "
                f"num_ffn_ranks={self.num_ffn_ranks}"
            )
        self.num_local_physical_experts = self.num_ffn_physical_experts // self.num_ffn_ranks
        self.num_physical_experts = self.num_local_physical_experts * self.world_size

        # log2phy: logical expert -> list of physical expert IDs
        # (one logical expert may map to multiple physical replicas in the future)
        self.log2phy: Dict[int, List[int]] = {}
        self.phy2log: List[int] = [-1] * self.num_physical_experts

        for logical_id in range(n_routed_experts):
            ffn_rank_index = logical_id // self.num_local_physical_experts
            ffn_global_rank = self.afd_world_topology.ffn_ranks[ffn_rank_index]
            local_offset = logical_id % self.num_local_physical_experts
            physical_id = ffn_global_rank * self.num_local_physical_experts + local_offset

            if logical_id not in self.log2phy:
                self.log2phy[logical_id] = []
            self.log2phy[logical_id].append(physical_id)
            self.phy2log[physical_id] = logical_id

        self._log2phy_flat = [self.log2phy[i][0] for i in range(self.num_logical_experts)]
        self._log2phy_tensor_cache: Dict[str, paddle.Tensor] = {}

        logger.info(
            f"AFDExpertLayout: logical={n_routed_experts}, "
            f"redundant={redundant_experts_num}, "
            f"physical={self.num_physical_experts}, "
            f"local_per_rank={self.num_local_physical_experts}, "
            f"attn_ranks={self.afd_world_topology.attn_ranks}, "
            f"ffn_ranks={self.afd_world_topology.ffn_ranks}"
        )

    # ------------------------------------------------------------------
    # scalar helpers
    # ------------------------------------------------------------------
    def router_log2phy(self, logical_expert_id: int) -> int:
        """Convert a single logical expert ID to physical expert ID.

        Currently returns the first physical replica; will support
        load-balanced selection among replicas in the future.
        """
        return self.log2phy[logical_expert_id][0]

    def router_phy2log(self, physical_expert_id: int) -> int:
        """Convert a single physical expert ID to logical expert ID."""
        return self.phy2log[physical_expert_id]

    # ------------------------------------------------------------------
    # batched GPU conversion
    # ------------------------------------------------------------------
    @property
    def log2phy_tensor(self) -> paddle.Tensor:
        """Flat tensor: log2phy_tensor[logical_id] = first physical id.

        Uses the first replica for each logical expert (same as
        ``router_log2phy``).  Will be extended for multi-replica
        selection in the future.
        """
        device = paddle.device.get_device()
        if device not in self._log2phy_tensor_cache:
            self._log2phy_tensor_cache[device] = paddle.to_tensor(
                self._log2phy_flat,
                dtype=paddle.int64,
                place=device,
            )
        return self._log2phy_tensor_cache[device]

    def _log2phy_tensor_for(self, tensor: paddle.Tensor) -> paddle.Tensor:
        place = tensor.place
        cache_key = str(place)
        if cache_key not in self._log2phy_tensor_cache:
            self._log2phy_tensor_cache[cache_key] = paddle.to_tensor(
                self._log2phy_flat,
                dtype=paddle.int64,
                place=place,
            )
        return self._log2phy_tensor_cache[cache_key]

    def batch_log2phy(self, topk_idx: paddle.Tensor) -> paddle.Tensor:
        """Vectorised logical -> physical conversion for a routing tensor.

        Args:
            topk_idx: ``[num_tokens, top_k]`` logical expert IDs.
        Returns:
            Same shape, physical expert IDs.
        """
        if topk_idx.shape[0] == 0:
            return topk_idx
        orig_shape = topk_idx.shape
        return paddle.index_select(
            self._log2phy_tensor_for(topk_idx), topk_idx.reshape([-1]), axis=0
        ).reshape(orig_shape)


class AFDA2ABackendBase:
    """Communication backend used by AFD decode workers."""

    name = "base"

    def dispatch_physical(self, x, physical_topk_idx, topk_weights, **kwargs):
        raise NotImplementedError

    def combine(self, ffn_out, physical_topk_idx, topk_weights, handle, **kwargs):
        raise NotImplementedError

    def runtime_device_info(self):
        return "unavailable"


def _create_afd_a2a_backend(fd_config, afd_layout: AFDExpertLayout) -> AFDA2ABackendBase:
    if envs.FD_MOE_A2A_BACKEND == "deepep":
        from fastdeploy.model_executor.afd.deepep_backend import AFDDeepEPA2ABackend

        return AFDDeepEPA2ABackend(fd_config, afd_layout)
    if envs.FD_MOE_A2A_BACKEND == "mooncake":
        from fastdeploy.model_executor.afd.mooncake_backend import AFDMooncakeM2NA2ABackend

        return AFDMooncakeM2NA2ABackend(fd_config, afd_layout)
    raise ValueError(
        f"Unknown FD_MOE_A2A_BACKEND={envs.FD_MOE_A2A_BACKEND!r}. "
        "Valid options for AFD are ['deepep', 'mooncake']."
    )


@singleton
class AFDDecodeRunner:
    """Decode-phase runner that drives dispatch / combine for AFD.

    Created once per worker process (both ATTN and FFN workers create one).
    The underlying communication backend is process-wide so buffers are
    allocated only once.
    """

    def __init__(self, fd_config, afd_layout: AFDExpertLayout):
        self.fd_config = fd_config
        self.device = _worker_device_from_config(fd_config)
        self._device_touch_tensor = None
        paddle.device.set_device(self.device)
        self._ensure_device("init")
        self.afd_layout = afd_layout
        self.hidden_size = fd_config.model_config.hidden_size
        self.top_k = fd_config.model_config.num_experts_per_tok
        self.num_physical_experts = afd_layout.num_physical_experts
        self.num_local_physical_experts = afd_layout.num_local_physical_experts
        self._logged_dispatch_device = False

        self.a2a_backend = _create_afd_a2a_backend(fd_config, afd_layout)

        logger.info(
            f"AFDDecodeRunner created: physical_experts={self.num_physical_experts}, "
            f"a2a_backend={self.a2a_backend.name}, "
            f"ep_rank={fd_config.parallel_config.expert_parallel_rank}, "
            f"device={self.device}, current_device={paddle.device.get_device()}, "
            f"FLAGS_selected_gpus={os.getenv('FLAGS_selected_gpus')}, "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}, "
            f"PADDLE_LOCAL_RANK={os.getenv('PADDLE_LOCAL_RANK')}"
        )

    # ------------------------------------------------------------------
    def _ensure_device(self, callsite: str) -> None:
        current_device = paddle.device.get_device()
        if current_device != self.device:
            logger.warning(
                f"AFDDecodeRunner reset Paddle device before {callsite}: "
                f"current={current_device}, expected={self.device}, "
                f"FLAGS_selected_gpus={os.getenv('FLAGS_selected_gpus')}, "
                f"PADDLE_LOCAL_RANK={os.getenv('PADDLE_LOCAL_RANK')}"
            )
            paddle.device.set_device(self.device)
        # Paddle set_device does not immediately update CUDA runtime current
        # device. A tiny tensor op on the target place makes DeepEP C++ see the
        # same device through cudaGetDevice/current stream.
        self._device_touch_tensor = paddle.empty([0], dtype="int32")

    def _log_dispatch_device_once(self, x, physical_topk_idx, topk_weights) -> None:
        if self._logged_dispatch_device:
            return
        self._logged_dispatch_device = True
        runtime_local_device_id = self.a2a_backend.runtime_device_info()
        logger.info(
            "AFD dispatch device check: "
            f"runner_device={self.device}, current_device={paddle.device.get_device()}, "
            f"x_place={getattr(x, 'place', None)}, "
            f"topk_idx_place={getattr(physical_topk_idx, 'place', None)}, "
            f"topk_weights_place={getattr(topk_weights, 'place', None)}, "
            f"a2a_backend={self.a2a_backend.name}, "
            f"backend_runtime_device_info={runtime_local_device_id}, "
            f"FLAGS_selected_gpus={os.getenv('FLAGS_selected_gpus')}, "
            f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}, "
            f"PADDLE_LOCAL_RANK={os.getenv('PADDLE_LOCAL_RANK')}"
        )

    def logical_to_physical(self, topk_idx: paddle.Tensor) -> paddle.Tensor:
        """Convert router logical expert IDs to AFD physical expert IDs."""
        self._ensure_device("logical_to_physical")
        return self.afd_layout.batch_log2phy(topk_idx)

    def dispatch_physical(self, x, physical_topk_idx, topk_weights, **kwargs):
        """Low-latency dispatch using physical expert IDs."""
        self._ensure_device("dispatch")
        self._log_dispatch_device_once(x, physical_topk_idx, topk_weights)
        return self.a2a_backend.dispatch_physical(x, physical_topk_idx, topk_weights, **kwargs)

    def dispatch(self, x, topk_idx, topk_weights, **kwargs):
        """Low-latency dispatch using logical expert IDs."""
        physical_topk_idx = self.logical_to_physical(topk_idx)
        return self.dispatch_physical(x, physical_topk_idx, topk_weights, **kwargs)

    def combine(self, ffn_out, physical_topk_idx, topk_weights, handle, **kwargs):
        """Low-latency combine using physical expert IDs."""
        self._ensure_device("combine")
        return self.a2a_backend.combine(ffn_out, physical_topk_idx, topk_weights, handle, **kwargs)
