# SPDX-License-Identifier: Apache-2.0
"""Host-side CPU staging arena: a mirror of the on-card paged KV block pool.

Why this exists
---------------
QAIC keeps KV in the compiled QPC's retained-state buffers **on the card**; vLLM's
Mooncake connectors RDMA KV out of **host-addressable torch tensors** they register
via ``register_kv_caches``. To bridge the two we keep a CPU torch tensor that mirrors
the on-card paged pool ``[num_blocks, num_kv_heads, page_size, head_dim]`` (per layer,
K and V). Mooncake registers these CPU tensors; this arena copies blocks
card<->staging through a small ``CardKVSessionAdapter`` (the only QAIC-coupled part).

With paged KV the on-card pool is already block-structured, so a Mooncake "tile" maps
1:1 to a page block — save/load happen at block granularity.

Verifiability
-------------
This module depends only on numpy/torch + the adapter Protocol, so the arena logic is
unit-tested on CPU with an in-memory mock card (``InMemoryCardKV``). The real adapter
(reading/writing the QPC retained-state via the QAIC session) is exercised on the card.
"""

from __future__ import annotations

from typing import Dict, List, Protocol, Sequence, Tuple

import numpy as np
import torch


class CardKVSessionAdapter(Protocol):
    """Reads/writes physical KV blocks of the on-card paged pool.

    Implemented on the box over the QAIC session (the paged pool is a retained-state
    binding); mocked in tests. ``block_ids`` are physical pool block indices.
    """

    def read_blocks(
        self, layer_idx: int, block_ids: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (k, v), each [len(block_ids), num_kv_heads, page_size, head_dim]."""
        ...

    def write_blocks(
        self, layer_idx: int, block_ids: Sequence[int], k: np.ndarray, v: np.ndarray
    ) -> None:
        """Write (k, v) blocks [len(block_ids), num_kv_heads, page_size, head_dim]."""
        ...


class QaicKVStagingArena:
    """CPU mirror of the on-card paged KV pool, registered with Mooncake."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        num_kv_heads: int,
        page_size: int,
        head_dim: int,
        dtype: torch.dtype,
        session_adapter: CardKVSessionAdapter,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        shape = (num_blocks, num_kv_heads, page_size, head_dim)
        self.k: List[torch.Tensor] = [torch.zeros(shape, dtype=dtype) for _ in range(num_layers)]
        self.v: List[torch.Tensor] = [torch.zeros(shape, dtype=dtype) for _ in range(num_layers)]
        self._sa = session_adapter

    def as_kv_caches(self) -> Dict[str, torch.Tensor]:
        """Host tensors for Mooncake ``register_kv_caches`` (one K and V per layer)."""
        caches: Dict[str, torch.Tensor] = {}
        for i in range(self.num_layers):
            caches[f"layer.{i}.key"] = self.k[i]
            caches[f"layer.{i}.value"] = self.v[i]
        return caches

    def _idx(self, block_ids: Sequence[int]) -> torch.Tensor:
        idx = torch.as_tensor(list(block_ids), dtype=torch.long)
        if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= self.num_blocks):
            raise ValueError(
                f"block_ids out of range [0, {self.num_blocks}): "
                f"min={int(idx.min())} max={int(idx.max())}")
        return idx

    def card_to_staging(self, block_ids: Sequence[int]) -> None:
        """Pull the given physical blocks off the card into the staging mirror.

        Call BEFORE a Mooncake put so the registered host tensors hold fresh KV.
        """
        if len(block_ids) == 0:
            return
        idx = self._idx(block_ids)
        for i in range(self.num_layers):
            k, v = self._sa.read_blocks(i, block_ids)
            self.k[i][idx] = torch.from_numpy(np.ascontiguousarray(k)).to(self.k[i].dtype)
            self.v[i][idx] = torch.from_numpy(np.ascontiguousarray(v)).to(self.v[i].dtype)

    def staging_to_card(self, block_ids: Sequence[int]) -> None:
        """Push staged blocks (filled by a Mooncake get) back onto the card.

        Call AFTER a Mooncake get so the QPC decode sees the loaded KV.
        """
        if len(block_ids) == 0:
            return
        idx = self._idx(block_ids)
        for i in range(self.num_layers):
            k = self.k[i][idx].contiguous().numpy()
            v = self.v[i][idx].contiguous().numpy()
            self._sa.write_blocks(i, block_ids, k, v)


class InMemoryCardKV:
    """Mock on-card pool for unit tests (and a reference for the real adapter)."""

    def __init__(self, num_layers, num_blocks, num_kv_heads, page_size, head_dim, dtype=np.float32):
        shape = (num_blocks, num_kv_heads, page_size, head_dim)
        self.k = [np.zeros(shape, dtype=dtype) for _ in range(num_layers)]
        self.v = [np.zeros(shape, dtype=dtype) for _ in range(num_layers)]

    def read_blocks(self, layer_idx, block_ids):
        bid = np.asarray(list(block_ids), dtype=np.int64)
        return self.k[layer_idx][bid].copy(), self.v[layer_idx][bid].copy()

    def write_blocks(self, layer_idx, block_ids, k, v):
        bid = np.asarray(list(block_ids), dtype=np.int64)
        self.k[layer_idx][bid] = k
        self.v[layer_idx][bid] = v
