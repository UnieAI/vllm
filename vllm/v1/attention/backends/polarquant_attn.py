# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PolarQuant attention backend for vLLM.

This backend reuses the TurboQuant backend shell (metadata, batching,
workspace management, and prefill/decode control flow) but stores K and V in
their own packed PolarQuant layout inside paged vLLM cache slots.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.polarquant.config import (
    PolarQuantConfig,
)
from vllm.v1.attention.backend import AttentionLayer, AttentionType, MultipleOf
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.turboquant_attn import (
    _HAS_FLASH_ATTN,
    _build_hadamard,
    TurboQuantAttentionBackend as _TurboQuantAttentionBackend,
    TurboQuantAttentionImpl as _TurboQuantAttentionImpl,
    TurboQuantMetadata,
    TurboQuantMetadataBuilder,
)
from vllm.v1.attention.ops.triton_polarquant_decode import (
    triton_polarquant_decode_attention,
)
from vllm.v1.attention.ops.triton_polarquant_store import triton_polarquant_store
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)


class PolarQuantAttentionBackend(_TurboQuantAttentionBackend):
    """Attention backend using PolarQuant KV-cache compression."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "polarquant_4bit",
    ]

    @staticmethod
    def get_name() -> str:
        return "POLARQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_impl_cls() -> type["PolarQuantAttentionImpl"]:
        return PolarQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "polarquant_4bit",
    ) -> tuple[int, ...]:
        pq_config = PolarQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, pq_config.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("polarquant_")


class PolarQuantAttentionImpl(_TurboQuantAttentionImpl):
    """PolarQuant attention implementation."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.pq_config = PolarQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)
        self.fa_version = get_flash_attn_version(head_size=head_size)
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        self._reserve_decode_workspace(vllm_config)

    def _ensure_on_device(self, layer, device):
        if not hasattr(layer, "_tq_cached"):
            H = _build_hadamard(self.head_size, str(device))
            layer._tq_PiT = H
            layer._tq_Pi = H
            layer._tq_Pi_half = H.to(torch.float16)
            layer._tq_centroids = torch.empty(0, dtype=torch.float32, device=device)
            layer._tq_cached = True

    def _store_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        triton_polarquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            group_size=self.pq_config.group_size,
            packed_dim=self.pq_config.packed_dim,
            num_groups=self.pq_config.num_groups,
            k_data_offset=self.pq_config.k_data_offset,
            k_group_mins_offset=self.pq_config.k_group_mins_offset,
            k_group_scales_offset=self.pq_config.k_group_scales_offset,
            v_radius_offset=self.pq_config.v_radius_offset,
            v_data_offset=self.pq_config.v_data_offset,
            v_group_mins_offset=self.pq_config.v_group_mins_offset,
            v_group_scales_offset=self.pq_config.v_group_scales_offset,
        )

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        del centroids, PiT, layer

        N, Hq, D = query.shape
        Hk = key.shape[1]
        use_gqa = Hk < Hq

        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )

        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1
        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        if not hasattr(self, "_cu_2"):
            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)

        max_seq = attn_metadata.max_seq_len
        arange_cache: torch.Tensor | None = getattr(self, "_arange_cache", None)
        if arange_cache is None or arange_cache.shape[0] <= max_seq:
            arange_cache = torch.arange(
                0, max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            self._arange_cache = arange_cache

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]

            if q_len == seq_len:
                if _HAS_FLASH_ATTN:
                    self._cu_2[1:2] = q_len
                    cu = self._cu_2
                    out = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
                continue

            cached_len = seq_len - q_len
            synth_seq_lens = arange_cache[cached_len + 1 : seq_len + 1]
            synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
            out = triton_polarquant_decode_attention(
                query=q_seq,
                kv_cache=kv_cache,
                block_table=synth_bt,
                seq_lens=synth_seq_lens,
                Pi=Pi,
                scale=self.scale,
                group_size=self.pq_config.group_size,
                packed_dim=self.pq_config.packed_dim,
                num_groups=self.pq_config.num_groups,
                k_data_offset=self.pq_config.k_data_offset,
                k_group_mins_offset=self.pq_config.k_group_mins_offset,
                k_group_scales_offset=self.pq_config.k_group_scales_offset,
                v_radius_offset=self.pq_config.v_radius_offset,
                v_data_offset=self.pq_config.v_data_offset,
                v_group_mins_offset=self.pq_config.v_group_mins_offset,
                v_group_scales_offset=self.pq_config.v_group_scales_offset,
                max_num_kv_splits=self.max_num_kv_splits,
            )
            output[q_start:q_end] = out.to(query.dtype)

        return output

    def _decode_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: AttentionLayer | None = None,
    ) -> torch.Tensor:
        del centroids, PiT

        B = query.shape[0]
        D = self.head_size
        Hq = self.num_heads
        S = self.max_num_kv_splits
        mid_o_buf = output_buf = lse_buf = None
        if is_workspace_manager_initialized():
            bufs = current_workspace_manager().try_get_simultaneous(
                ((B, Hq, S, D + 1), torch.float32),
                ((B, Hq, D), query.dtype),
                ((B, Hq), torch.float32),
            )
            if bufs is not None:
                mid_o_buf, output_buf, lse_buf = bufs
        if mid_o_buf is None:
            mid_o_buf = torch.empty(
                (B, Hq, S, D + 1), dtype=torch.float32, device=query.device
            )
            output_buf = torch.empty((B, Hq, D), dtype=query.dtype, device=query.device)
            lse_buf = torch.empty((B, Hq), dtype=torch.float32, device=query.device)

        return triton_polarquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            Pi=Pi,
            scale=self.scale,
            group_size=self.pq_config.group_size,
            packed_dim=self.pq_config.packed_dim,
            num_groups=self.pq_config.num_groups,
            k_data_offset=self.pq_config.k_data_offset,
            k_group_mins_offset=self.pq_config.k_group_mins_offset,
            k_group_scales_offset=self.pq_config.k_group_scales_offset,
            v_radius_offset=self.pq_config.v_radius_offset,
            v_data_offset=self.pq_config.v_data_offset,
            v_group_mins_offset=self.pq_config.v_group_mins_offset,
            v_group_scales_offset=self.pq_config.v_group_scales_offset,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=layer,
            max_num_kv_splits=self.max_num_kv_splits,
        )
