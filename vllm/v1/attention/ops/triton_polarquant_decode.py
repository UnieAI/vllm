# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused PolarQuant decode attention."""

from __future__ import annotations

from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2


@triton.jit
def _load_fp16_bytes(KV_cache_ptr, byte_offsets, mask):
    lo = tl.load(KV_cache_ptr + byte_offsets, mask=mask, other=0).to(tl.uint16)
    hi = tl.load(KV_cache_ptr + byte_offsets + 1, mask=mask, other=0).to(tl.uint16)
    return (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _pq_decode_stage1(
    Q_rot_ptr,
    KV_cache_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Mid_o_ptr,
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    K_DATA_OFFSET: tl.constexpr,
    K_GMIN_OFFSET: tl.constexpr,
    K_GSCALE_OFFSET: tl.constexpr,
    V_RADIUS_OFFSET: tl.constexpr,
    V_DATA_OFFSET: tl.constexpr,
    V_GMIN_OFFSET: tl.constexpr,
    V_GSCALE_OFFSET: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)

    kv_head = hid // KV_GROUP_SIZE
    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)
    byte_idx = d_offs // 2
    bit_shift = (d_offs % 2) * 4
    group_byte_offsets = (d_offs // GROUP_SIZE) * 2

    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    bt_base = bid * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        k_radius = _load_fp16_bytes(KV_cache_ptr, slot_bases, kv_mask)
        k_gmins = _load_fp16_bytes(
            KV_cache_ptr,
            slot_bases[:, None] + K_GMIN_OFFSET + group_byte_offsets[None, :],
            kv_mask[:, None] & d_mask[None, :],
        )
        k_gscales = _load_fp16_bytes(
            KV_cache_ptr,
            slot_bases[:, None] + K_GSCALE_OFFSET + group_byte_offsets[None, :],
            kv_mask[:, None] & d_mask[None, :],
        )
        k_bytes = tl.load(
            KV_cache_ptr + slot_bases[:, None] + K_DATA_OFFSET + byte_idx[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        k_idx = ((k_bytes >> bit_shift[None, :]) & 0xF).to(tl.float32)
        k_dir = k_idx * k_gscales + k_gmins

        scores = (
            tl.sum(tl.where(d_mask[None, :], q_rot[None, :] * k_dir, 0.0), axis=1)
            * k_radius
            * ATTN_SCALE
        )
        scores = tl.where(kv_mask, scores, -float("inf"))

        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        v_radius = _load_fp16_bytes(KV_cache_ptr, slot_bases + V_RADIUS_OFFSET, kv_mask)
        v_gmins = _load_fp16_bytes(
            KV_cache_ptr,
            slot_bases[:, None] + V_GMIN_OFFSET + group_byte_offsets[None, :],
            kv_mask[:, None] & d_mask[None, :],
        )
        v_gscales = _load_fp16_bytes(
            KV_cache_ptr,
            slot_bases[:, None] + V_GSCALE_OFFSET + group_byte_offsets[None, :],
            kv_mask[:, None] & d_mask[None, :],
        )
        v_bytes = tl.load(
            KV_cache_ptr + slot_bases[:, None] + V_DATA_OFFSET + byte_idx[None, :],
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        v_idx = ((v_bytes >> bit_shift[None, :]) & 0xF).to(tl.float32)
        values = (v_idx * v_gscales + v_gmins) * v_radius[:, None]

        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    lse = m_prev + tl.log(safe_l)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, lse)


def triton_polarquant_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    Pi: torch.Tensor,
    scale: float,
    *,
    group_size: int,
    packed_dim: int,
    num_groups: int,
    k_data_offset: int,
    k_group_mins_offset: int,
    k_group_scales_offset: int,
    v_radius_offset: int,
    v_data_offset: int,
    v_group_mins_offset: int,
    v_group_scales_offset: int,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,
) -> torch.Tensor:
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    q_rot = (query.float() @ Pi.T.contiguous()).contiguous()

    num_kv_splits = max_num_kv_splits
    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= num_kv_splits
    ):
        mid_o = mid_o_buf[:B, :Hq, :num_kv_splits, :]
    else:
        mid_o = torch.empty(
            B,
            Hq,
            num_kv_splits,
            D + 1,
            dtype=torch.float32,
            device=device,
        )
        if buf_holder is not None:
            buf_holder._pq_mid_o_buf = mid_o

    block_d = triton.next_power_of_2(D)
    grid = (B, Hq, num_kv_splits)
    _pq_decode_stage1[grid](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        mid_o,
        q_rot.stride(0),
        q_rot.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=num_kv_splits,
        KV_GROUP_SIZE=kv_group_size,
        GROUP_SIZE=group_size,
        PACKED_DIM=packed_dim,
        NUM_GROUPS=num_groups,
        ATTN_SCALE=scale,
        BLOCK_D=block_d,
        BLOCK_KV=4,
        K_DATA_OFFSET=k_data_offset,
        K_GMIN_OFFSET=k_group_mins_offset,
        K_GSCALE_OFFSET=k_group_scales_offset,
        V_RADIUS_OFFSET=v_radius_offset,
        V_DATA_OFFSET=v_data_offset,
        V_GMIN_OFFSET=v_group_mins_offset,
        V_GSCALE_OFFSET=v_group_scales_offset,
        num_warps=1,
        num_stages=1,
    )

    out_dtype = query.dtype
    if output_buf is not None and output_buf.shape[0] >= B and output_buf.dtype == out_dtype:
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
        if buf_holder is not None:
            buf_holder._pq_output_buf = output

    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._pq_lse_buf = lse

    grid2 = (B, Hq)
    _fwd_kernel_stage2[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=block_d,
        Lv=D,
        OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
        num_warps=4,
        num_stages=2,
    )
    return (output.float() @ Pi).to(out_dtype)
