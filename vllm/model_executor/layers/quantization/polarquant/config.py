# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PolarQuant configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass

PQ_PRESETS: dict[str, dict[str, int]] = {
    "polarquant_4bit": {
        "quant_bits": 4,
        "group_size": 32,
    },
}


@dataclass
class PolarQuantConfig:
    """Configuration for PolarQuant KV-cache compression.

    This v1 integration uses a paged packed-cache layout tailored for vLLM:
    both K and V are stored in rotated polar form with a per-vector radius and
    per-group affine parameters for 4-bit packed direction values.
    """

    head_dim: int = 128
    quant_bits: int = 4
    group_size: int = 32

    def __post_init__(self) -> None:
        if self.quant_bits != 4:
            raise ValueError(
                f"PolarQuant currently supports only 4-bit packing, got {self.quant_bits}"
            )
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")

    @property
    def num_groups(self) -> int:
        return math.ceil(self.head_dim / self.group_size)

    @property
    def padded_head_dim(self) -> int:
        return self.num_groups * self.group_size

    @property
    def packed_dim(self) -> int:
        return self.padded_head_dim // 2

    @property
    def group_params_bytes(self) -> int:
        return self.num_groups * 2

    @property
    def vector_packed_size(self) -> int:
        # radius(fp16) + packed direction + group mins(fp16) + group scales(fp16)
        return 2 + self.packed_dim + self.group_params_bytes * 2

    @property
    def slot_size(self) -> int:
        # K vector payload + V vector payload
        return self.vector_packed_size * 2

    @property
    def slot_size_aligned(self) -> int:
        s = self.slot_size
        return s + (s % 2)

    @property
    def k_radius_offset(self) -> int:
        return 0

    @property
    def k_data_offset(self) -> int:
        return self.k_radius_offset + 2

    @property
    def k_group_mins_offset(self) -> int:
        return self.k_data_offset + self.packed_dim

    @property
    def k_group_scales_offset(self) -> int:
        return self.k_group_mins_offset + self.group_params_bytes

    @property
    def v_radius_offset(self) -> int:
        return self.vector_packed_size

    @property
    def v_data_offset(self) -> int:
        return self.v_radius_offset + 2

    @property
    def v_group_mins_offset(self) -> int:
        return self.v_data_offset + self.packed_dim

    @property
    def v_group_scales_offset(self) -> int:
        return self.v_group_mins_offset + self.group_params_bytes

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> PolarQuantConfig:
        if cache_dtype not in PQ_PRESETS:
            valid = ", ".join(PQ_PRESETS.keys())
            raise ValueError(
                f"Unknown PolarQuant cache dtype: {cache_dtype!r}. Valid presets: {valid}"
            )
        preset = PQ_PRESETS[cache_dtype]
        return PolarQuantConfig(
            head_dim=head_dim,
            quant_bits=preset["quant_bits"],
            group_size=preset["group_size"],
        )
