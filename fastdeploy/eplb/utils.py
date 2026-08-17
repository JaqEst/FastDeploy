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

import json
import os
import time
import hashlib

import numpy as np
import requests

from fastdeploy.config import FDConfig
from fastdeploy.inter_communicator import IPCSignal


class RedundantExpertWorkload:
    """Redundant Expert Workload"""

    def __init__(self, redundant_expert_meta_dir="/tmp/redundant_expert_meta"):
        self.update_timestamp = time.time()
        self.tokens_per_expert_stats_list = None
        self.ep_rank_to_expert_id_list = None
        self.expert_id_to_ep_rank_array = None
        self.expert_in_rank_num_list = None
        self.cost_milliseconds = 0
        self.meta_file_name = f"{redundant_expert_meta_dir}/rearrange-experts.json"
        if not os.path.exists(redundant_expert_meta_dir):
            os.makedirs(redundant_expert_meta_dir, exist_ok=True)

    def __json__(self):
        return self.__dict__

    def dump(self):
        """Dump the object to a JSON file."""
        begin = time.time()
        try:
            with open(self.meta_file_name, "w") as fout:
                json.dump(self.__dict__, fout)
        except Exception as e:
            return f"redundant_expert: dump expert workload failed, {e}"
        cost_time = int((time.time() - begin) * 1000 * 1000)
        return f"redundant_expert: dump expert workload result in {cost_time} us"

    def load(self):
        """Load the object from a JSON file."""
        if not os.path.exists(self.meta_file_name):
            return {}, f"redundant_expert: file {self.meta_file_name} is not exists"
        try:
            with open(self.meta_file_name, "r") as fin:
                meta = json.load(fin)
                self.__dict__.update(meta)
                return self.__json__(), "ok"
        except Exception as e:
            return {}, f"redundant_expert: load file {self.meta_file_name} failed, {e}"


def _as_numpy_int_array(value):
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value, dtype=np.int32)


def build_redundant_expert_table_snapshot(
    *,
    fd_config: FDConfig,
    rank_expert_list,
    logical_to_physical_map,
    expert_count,
    source: str,
    local_rank: int = None,
    clear_stat: bool = None,
):
    """Build a readable EPLB table snapshot without mutating the table."""
    rank_expert_array = _as_numpy_int_array(rank_expert_list)
    logical_to_physical_array = _as_numpy_int_array(logical_to_physical_map)
    expert_count_array = _as_numpy_int_array(expert_count)

    table_payload = {
        "ep_rank_to_expert_id_list": rank_expert_array.tolist(),
        "expert_id_to_ep_rank_array": logical_to_physical_array.tolist(),
        "expert_in_rank_num_list": expert_count_array.tolist(),
    }
    table_hash = hashlib.sha256(json.dumps(table_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    slots_per_rank = int(getattr(fd_config.afd_config, "num_local_physical_experts", 0))

    return {
        "source": source,
        "timestamp": time.time(),
        "role": fd_config.afd_config.afd_role if fd_config.afd_config.enable_afd else "non_afd",
        "local_rank": None if local_rank is None else int(local_rank),
        "clear_stat": clear_stat,
        "afd_node_rank": int(fd_config.afd_config.inst_rank),
        "expert_parallel_rank": int(fd_config.parallel_config.expert_parallel_rank),
        "tensor_parallel_rank": int(fd_config.parallel_config.tensor_parallel_rank),
        "engine_worker_queue_port": fd_config.parallel_config.local_engine_worker_queue_port,
        "shape": {
            "layers": int(rank_expert_array.shape[0]),
            "physical_slots": int(rank_expert_array.shape[1]),
            "slots_per_rank": slots_per_rank,
            "rank_count": int(rank_expert_array.shape[1] // slots_per_rank) if slots_per_rank else None,
            "logical_experts": int(logical_to_physical_array.shape[1]),
            "max_replicas_per_logical_expert": int(logical_to_physical_array.shape[2]),
        },
        "table_hash": table_hash,
        **table_payload,
    }


def dump_redundant_expert_table_snapshot(
    *,
    fd_config: FDConfig,
    rank_expert_list,
    logical_to_physical_map,
    expert_count,
    source: str,
    local_rank: int = None,
    clear_stat: bool = None,
):
    snapshot = build_redundant_expert_table_snapshot(
        fd_config=fd_config,
        rank_expert_list=rank_expert_list,
        logical_to_physical_map=logical_to_physical_map,
        expert_count=expert_count,
        source=source,
        local_rank=local_rank,
        clear_stat=clear_stat,
    )

    dump_dir = os.path.join(fd_config.eplb_config.redundant_expert_meta_dir, "table_dumps")
    os.makedirs(dump_dir, exist_ok=True)
    role = snapshot["role"]
    rank = "none" if local_rank is None else str(int(local_rank))
    timestamp_ms = int(snapshot["timestamp"] * 1000)
    path = os.path.join(dump_dir, f"{timestamp_ms}_{role}_{source}_rank{rank}.json")
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as fout:
        json.dump(snapshot, fout)
    os.replace(tmp_path, path)
    return path, snapshot


def _ffn_slot_pairs(afd_config):
    """Yield (ffn_local_slice, global_slice) pairs, matching the layout eplb produces."""
    local = afd_config.num_local_physical_experts
    for ffn_index, global_rank in enumerate(afd_config.ffn_ranks):
        yield (
            slice(ffn_index * local, (ffn_index + 1) * local),
            slice(global_rank * local, (global_rank + 1) * local),
        )


def compact_expert_rank_table(phy2log: np.ndarray, afd_config) -> np.ndarray:
    """Drop the ATTN placeholder slots, leaving a table independent of the AFD rank layout."""
    if not afd_config.enable_afd:
        return phy2log
    compact = np.full((phy2log.shape[0], afd_config.num_ffn_physical_experts), -1, dtype=np.int32)
    for ffn_slice, global_slice in _ffn_slot_pairs(afd_config):
        compact[:, ffn_slice] = phy2log[:, global_slice]
    return compact


def expand_expert_rank_table(compact: np.ndarray, afd_config) -> np.ndarray:
    """Inverse of compact_expert_rank_table: place FFN-local slots back at global slot ids."""
    if not afd_config.enable_afd:
        return compact
    phy2log = np.full((compact.shape[0], afd_config.num_physical_experts), -1, dtype=np.int32)
    for ffn_slice, global_slice in _ffn_slot_pairs(afd_config):
        phy2log[:, global_slice] = compact[:, ffn_slice]
    return phy2log


def derive_expert_tables(phy2log: np.ndarray, num_logical_experts: int, max_replicas: int):
    """Rebuild (logical_to_physical_map, expert_count) from a phy2log table."""
    num_layers = phy2log.shape[0]
    log2phy = np.full((num_layers, num_logical_experts, max_replicas), -1, dtype=np.int32)
    expert_count = np.zeros((num_layers, num_logical_experts), dtype=np.int32)
    for layer in range(num_layers):
        for slot, logical_expert_id in enumerate(phy2log[layer]):
            if logical_expert_id < 0:
                continue
            log2phy[layer, logical_expert_id, expert_count[layer, logical_expert_id]] = slot
            expert_count[layer, logical_expert_id] += 1
    return log2phy, expert_count


def _fetch_expert_rank_table(config: FDConfig, timeout: float = 5.0, wait_seconds: float = 30.0):
    """Pull the in-use phy2log table from a serving peer instance, discovered via the router."""
    router_url = config.router_config.router if config.router_config else None
    if not router_url:
        return None

    inst_types = ('attn', 'ffn') if config.afd_config.enable_afd else (config.scheduler_config.splitwise_role,)
    eplb_args = {
        "user": config.eplb_config.redundant_expert_api_user,
        "passwd": config.eplb_config.redundant_expert_api_password,
    }

    deadline = time.time() + wait_seconds
    while True:
        try:
            instances = requests.get(f"{router_url.rstrip('/')}/registered", timeout=timeout).json()
        except Exception as e:
            instances = {}

        for type in inst_types:
            for peer in instances.get(type) or []:
                if peer.get("not_ready"):
                    continue
                url = peer.get("url", "")
                if not url.startswith("http"):
                    url = f"http://{url}"
                try:
                    res = requests.post(f"{url}/get_expert_rank_table", json=eplb_args, timeout=timeout)
                    body = res.json()
                    if res.ok and body.get("code") == 0:
                        return np.array(body["data"], dtype=np.int32)
                except Exception as e:
                    pass

        if time.time() >= deadline:
            return None
        time.sleep(2)


def _expert_rank_table_shape(config: FDConfig) -> tuple:
    """Shape of the shared phy2log table: [num_hidden_layers, num_experts+num_redundant_experts]."""
    num_logical_experts = config.model_config.moe_num_experts
    if isinstance(num_logical_experts, list):
        num_logical_experts = num_logical_experts[0]
    return (config.model_config.num_hidden_layers, num_logical_experts + config.eplb_config.redundant_experts_num)


def read_shared_expert_rank_table(config: FDConfig):
    """Read the in-use compact phy2log table the engine published, None when it holds no table."""
    if not config.eplb_config.enable_eplb:
        return None
    try:
        dp_ipc_signal_suffix = (
            f"{config.parallel_config.local_engine_worker_queue_port}_dp{config.parallel_config.local_data_parallel_id}"
        )
        signal = IPCSignal(
            name="expert_rank_table",
            array=np.full(_expert_rank_table_shape(config), -1, dtype=np.int32),
            dtype=np.int32,
            suffix=dp_ipc_signal_suffix,
            create=False,
        )
        table = np.array(signal.value, dtype=np.int32)
    except Exception as e:
        return None
    # The engine fills the table with -1 until it holds a placement.
    return None if np.any(table < 0) else table


def init_eplb_signals(config: FDConfig, ipc_signal_suffix):
    """
    Initialize shared memory to indicate eplb status
    """
    if config.parallel_config.tensor_parallel_rank != 0:
        # only TP rank 0 need to init eplb signals, rank 0 manage all EPLB signals for all TP ranks
        return

    dp_ipc_signal_suffix = f"{ipc_signal_suffix}_dp{config.parallel_config.local_data_parallel_id}"
    # rearrange_experts_status Record the expert's rearrangement status
    rearrange_experts_array = np.zeros([1], dtype=np.int32)
    _ = IPCSignal(
        name="rearrange_experts_status",
        array=rearrange_experts_array,
        dtype=np.int32,
        suffix=dp_ipc_signal_suffix,
        create=True,
    )

    # Record all DP rank IPs when receiving expert rearrangement requests
    rearrange_experts_ips_size_array = np.zeros([1], dtype=np.int32)
    _ = IPCSignal(
        name="rearrange_experts_ips_size",
        array=rearrange_experts_ips_size_array,
        dtype=np.int32,
        suffix=dp_ipc_signal_suffix,
        create=True,
    )
    _ = IPCSignal(
        name="rearrange_experts_ips_list",
        shm_size=config.eplb_config.redundant_expert_ip_shm_size,
        suffix=dp_ipc_signal_suffix,
        create=True,
    )

    # Receive signals for updating weights
    signal_update_weight_from_tensor = np.zeros([1], dtype=np.int32)
    _ = IPCSignal(
        name="signal_update_weight_from_tensor",
        array=signal_update_weight_from_tensor,
        dtype=np.int32,
        suffix=dp_ipc_signal_suffix,
        create=True,
    )

    # In-use phy2log table, served to instances joining the cluster
    expert_rank_table = np.full(_expert_rank_table_shape(config), -1, dtype=np.int32)
    if config.launch_config.is_extension:
        peer_table = _fetch_expert_rank_table(config)
        if peer_table is not None:
            expert_rank_table[:] = peer_table
    _ = IPCSignal(
        name="expert_rank_table",
        array=expert_rank_table,
        dtype=np.int32,
        suffix=dp_ipc_signal_suffix,
        create=True,
    )

    for rank_id in range(config.parallel_config.tensor_parallel_size):
        tp_ipc_signal_suffix = f"{dp_ipc_signal_suffix}_tp{rank_id}"
        # Record expert workload
        experts_token_stats = np.zeros(
            (config.model_config.num_hidden_layers, config.model_config.moe_num_experts),
            dtype=np.int32,
        )
        _ = IPCSignal(
            name="all_experts_token_stats",
            array=experts_token_stats,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=True,
        )
        _ = IPCSignal(
            name="local_experts_token_stats",
            array=experts_token_stats,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=True,
        )

        # Receive signals for loading weights
        signal_update_weight_from_disk = np.zeros([1], dtype=np.int32)
        _ = IPCSignal(
            name="signal_update_weight_from_disk",
            array=signal_update_weight_from_disk,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=True,
        )

        # Receive signals for clearing expert loads
        clear_experts_token_stats = np.zeros([1], dtype=np.int32)
        _ = IPCSignal(
            name="signal_clear_experts_token_stats",
            array=clear_experts_token_stats,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=True,
        )

        result_update_weight_from_disk = np.zeros([1], dtype=np.int32)
        _ = IPCSignal(
            name="result_update_weight_from_disk",
            array=result_update_weight_from_disk,
            dtype=np.int32,
            suffix=tp_ipc_signal_suffix,
            create=True,
        )


if __name__ == "__main__":
    print(RedundantExpertWorkload("/tmp").load())
