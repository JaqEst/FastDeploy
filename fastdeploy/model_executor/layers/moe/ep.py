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

from fastdeploy import envs

_backend = envs.FD_MOE_A2A_BACKEND
_VALID_BACKENDS = ["deepep", "mooncake"]

if _backend not in _VALID_BACKENDS:
    raise ValueError(
        f"Unknown FD_MOE_A2A_BACKEND={_backend!r}. "
        f"Valid options: {_VALID_BACKENDS}"
    )

if _backend == "deepep":
    from .ep_deepep_backend import (  # noqa: F401
        DeepEPBufferManager,
        EPDecoderRunner,
        EPPrefillRunner,
        deep_ep,
    )
elif _backend == "mooncake":
    from .ep_mooncake_backend import (  # noqa: F401
        EPDecoderRunner,
        EPPrefillRunner,
        mooncake,
    )