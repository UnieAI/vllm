# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the QAIC Mooncake staging arena (card<->host KV mirror).

Runs on CPU with an in-memory mock card (no qaicrt / no card). Validates the
block-level copy logic and per-block isolation that the Mooncake save/load path
relies on. The real card I/O (QAIC session) is exercised on the box.
"""

import numpy as np
import torch

from vllm_qaic.kv_connector.qaic_kv_staging import InMemoryCardKV, QaicKVStagingArena


def _make(num_layers=2, num_blocks=8, heads=2, page=4, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    card = InMemoryCardKV(num_layers, num_blocks, heads, page, dim)
    # Fill the mock card with distinct values per (layer, block) so mistakes show up.
    for i in range(num_layers):
        card.k[i][:] = rng.standard_normal(card.k[i].shape).astype(np.float32)
        card.v[i][:] = rng.standard_normal(card.v[i].shape).astype(np.float32)
    arena = QaicKVStagingArena(num_layers, num_blocks, heads, page, dim, torch.float32, card)
    return card, arena


def test_as_kv_caches_shapes():
    _, arena = _make()
    caches = arena.as_kv_caches()
    assert len(caches) == 2 * arena.num_layers
    for name, t in caches.items():
        assert t.shape == (arena.num_blocks, 2, 4, 3), f"{name}: {t.shape}"


def test_card_to_staging_pulls_correct_blocks():
    card, arena = _make()
    block_ids = [1, 3, 6]
    arena.card_to_staging(block_ids)
    for i in range(arena.num_layers):
        for b in block_ids:
            assert np.allclose(arena.k[i][b].numpy(), card.k[i][b]), f"layer{i} blk{b} K"
            assert np.allclose(arena.v[i][b].numpy(), card.v[i][b]), f"layer{i} blk{b} V"
        # untouched staging blocks stay zero (only requested blocks copied)
        untouched = [b for b in range(arena.num_blocks) if b not in block_ids]
        assert np.count_nonzero(arena.k[i][untouched].numpy()) == 0, "copied extra blocks"


def test_staging_to_card_pushes_correct_blocks():
    card, arena = _make(seed=1)
    block_ids = [0, 5]
    # put known values into staging, snapshot the rest of the card
    for i in range(arena.num_layers):
        for b in block_ids:
            arena.k[i][b] = torch.full((2, 4, 3), float(100 + 10 * i + b))
            arena.v[i][b] = torch.full((2, 4, 3), float(200 + 10 * i + b))
    before = [card.k[i].copy() for i in range(arena.num_layers)]
    arena.staging_to_card(block_ids)
    for i in range(arena.num_layers):
        for b in block_ids:
            assert np.allclose(card.k[i][b], 100 + 10 * i + b), f"layer{i} blk{b} K not written"
            assert np.allclose(card.v[i][b], 200 + 10 * i + b), f"layer{i} blk{b} V not written"
        # blocks NOT in block_ids must be unchanged (no cross-block corruption)
        for b in range(arena.num_blocks):
            if b not in block_ids:
                assert np.allclose(card.k[i][b], before[i][b]), f"layer{i} blk{b} corrupted"


def test_roundtrip_integrity():
    card, arena = _make(seed=2)
    block_ids = [2, 4, 7]
    orig = [card.k[i][block_ids].copy() for i in range(arena.num_layers)]
    arena.card_to_staging(block_ids)          # card -> staging
    arena.staging_to_card(block_ids)          # staging -> card (no change expected)
    for i in range(arena.num_layers):
        assert np.allclose(card.k[i][block_ids], orig[i]), f"layer{i} roundtrip changed K"


def test_empty_block_ids_noop():
    card, arena = _make(seed=3)
    snap = [card.k[i].copy() for i in range(arena.num_layers)]
    arena.card_to_staging([])
    arena.staging_to_card([])
    for i in range(arena.num_layers):
        assert np.allclose(card.k[i], snap[i])


if __name__ == "__main__":
    test_as_kv_caches_shapes()
    test_card_to_staging_pulls_correct_blocks()
    test_staging_to_card_pushes_correct_blocks()
    test_roundtrip_integrity()
    test_empty_block_ids_noop()
    print("QAIC KV STAGING: ALL PASS")
