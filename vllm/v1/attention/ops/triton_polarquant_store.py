# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for PolarQuant KV store."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _store_fp16_bytes(
    KV_cache_ptr,
    byte_offsets,
    values,
    mask,
):
    vals_f16 = values.to(tl.float16)
    vals_u16 = vals_f16.to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + byte_offsets, (vals_u16 & 0xFF).to(tl.uint8), mask=mask)
    tl.store(
        KV_cache_ptr + byte_offsets + 1,
        ((vals_u16 >> 8) & 0xFF).to(tl.uint8),
        mask=mask,
    )


@triton.jit
def _store_pq_vector(
    Rot_ptr,
    KV_cache_ptr,
    base,
    slot_base,
    d_offs,
    d_mask,
    D_PAD: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    RADIUS_OFFSET: tl.constexpr,
    DATA_OFFSET: tl.constexpr,
    GMIN_OFFSET: tl.constexpr,
    GSCALE_OFFSET: tl.constexpr,
):
    vec = tl.load(Rot_ptr + base + d_offs, mask=d_mask, other=0.0).to(tl.float32)
    norm_sq = tl.sum(vec * vec, axis=0)
    radius = tl.sqrt(norm_sq + 1e-20)
    inv_radius = tl.where(radius > 1e-10, 1.0 / radius, 0.0)
    direction = vec * inv_radius

    dir_groups = tl.reshape(direction, [NUM_GROUPS, GROUP_SIZE])
    gmins = tl.min(dir_groups, axis=1)
    gmaxs = tl.max(dir_groups, axis=1)
    ranges = gmaxs - gmins
    gscales = tl.where(ranges > 1e-8, ranges / 15.0, 1.0)

    normed = (dir_groups - gmins[:, None]) / gscales[:, None]
    q_idx = tl.minimum(tl.maximum((normed + 0.5).to(tl.int32), 0), 15)
    q_flat = tl.reshape(q_idx, [D_PAD])
    q_pairs = tl.reshape(q_flat, [PACKED_DIM, 2])
    packed = tl.sum((q_pairs & 0xF) << (tl.arange(0, 2)[None, :] * 4), axis=1).to(
        tl.uint8
    )

    radius_offset = slot_base + RADIUS_OFFSET
    radius_u16 = radius.to(tl.float16).to(tl.uint16, bitcast=True)
    tl.store(KV_cache_ptr + radius_offset, (radius_u16 & 0xFF).to(tl.uint8))
    tl.store(
        KV_cache_ptr + radius_offset + 1, ((radius_u16 >> 8) & 0xFF).to(tl.uint8)
    )

    byte_offs = tl.arange(0, PACKED_DIM)
    tl.store(KV_cache_ptr + slot_base + DATA_OFFSET + byte_offs, packed)

    group_offs = tl.arange(0, NUM_GROUPS)
    gmin_byte_offsets = slot_base + GMIN_OFFSET + group_offs * 2
    gscale_byte_offsets = slot_base + GSCALE_OFFSET + group_offs * 2
    group_mask = group_offs < NUM_GROUPS
    _store_fp16_bytes(KV_cache_ptr, gmin_byte_offsets, gmins, group_mask)
    _store_fp16_bytes(KV_cache_ptr, gscale_byte_offsets, gscales, group_mask)


@triton.jit
def _pq_fused_store(
    KeyRot_ptr,
    ValueRot_ptr,
    KV_cache_ptr,
    Slot_mapping_ptr,
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    PACKED_DIM: tl.constexpr,
    K_DATA_OFFSET: tl.constexpr,
    K_GMIN_OFFSET: tl.constexpr,
    K_GSCALE_OFFSET: tl.constexpr,
    V_RADIUS_OFFSET: tl.constexpr,
    V_DATA_OFFSET: tl.constexpr,
    V_GMIN_OFFSET: tl.constexpr,
    V_GSCALE_OFFSET: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return

    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    head_idx_i64 = tl.cast(head_idx, tl.int64)
    slot_base = (
        blk * stride_cache_block
        + off * stride_cache_pos
        + head_idx_i64 * stride_cache_head
    )

    base = pid * D
    d_offs = tl.arange(0, D_PAD)
    d_mask = d_offs < D

    _store_pq_vector(
        KeyRot_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        D_PAD=D_PAD,
        GROUP_SIZE=GROUP_SIZE,
        NUM_GROUPS=NUM_GROUPS,
        PACKED_DIM=PACKED_DIM,
        RADIUS_OFFSET=0,
        DATA_OFFSET=K_DATA_OFFSET,
        GMIN_OFFSET=K_GMIN_OFFSET,
        GSCALE_OFFSET=K_GSCALE_OFFSET,
    )
    _store_pq_vector(
        ValueRot_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        D_PAD=D_PAD,
        GROUP_SIZE=GROUP_SIZE,
        NUM_GROUPS=NUM_GROUPS,
        PACKED_DIM=PACKED_DIM,
        RADIUS_OFFSET=V_RADIUS_OFFSET,
        DATA_OFFSET=V_DATA_OFFSET,
        GMIN_OFFSET=V_GMIN_OFFSET,
        GSCALE_OFFSET=V_GSCALE_OFFSET,
    )


def triton_polarquant_store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    PiT: torch.Tensor,
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
) -> None:
    N, H, D = key.shape
    NH = N * H
    block_size = kv_cache.shape[1]
    d_pad = packed_dim * 2

    k_rot = (key.float().reshape(NH, D) @ PiT).contiguous()
    v_rot = (value.float().reshape(NH, D) @ PiT).contiguous()

    stride_block = kv_cache.stride(0)
    stride_pos = kv_cache.stride(1)
    stride_head = kv_cache.stride(2)

    grid = (NH,)
    _pq_fused_store[grid](
        k_rot,
        v_rot,
        kv_cache.view(-1),
        slot_mapping,
        stride_cache_block=stride_block,
        stride_cache_pos=stride_pos,
        stride_cache_head=stride_head,
        D=D,
        D_PAD=d_pad,
        H=H,
        BLOCK_SIZE=block_size,
        GROUP_SIZE=group_size,
        NUM_GROUPS=num_groups,
        PACKED_DIM=packed_dim,
        K_DATA_OFFSET=k_data_offset,
        K_GMIN_OFFSET=k_group_mins_offset,
        K_GSCALE_OFFSET=k_group_scales_offset,
        V_RADIUS_OFFSET=v_radius_offset,
        V_DATA_OFFSET=v_data_offset,
        V_GMIN_OFFSET=v_group_mins_offset,
        V_GSCALE_OFFSET=v_group_scales_offset,
        num_warps=4,
        num_stages=1,
    )
