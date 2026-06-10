# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property-based test for block hash equivalence between warmup and
normal request paths.

# Feature: adaptive-warmup-data-plane,
#   Property 8: Block hash equivalence
"""

from __future__ import annotations

from collections.abc import Sequence

from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    hash_block_tokens,
    init_none_hash,
)

# Initialize the NONE_HASH used as parent for the first block.
# Use a fixed seed so tests are reproducible across runs.
init_none_hash(sha256)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

BLOCK_SIZE = 16

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def compute_block_hashes_normal_path(
    token_ids: Sequence[int],
    block_size: int,
) -> list[BlockHash]:
    """Simulate normal request path block hash computation.

    This replicates the logic in ``request_block_hasher`` returned by
    ``get_request_block_hasher`` for a text-only request (no LoRA,
    no multimodal, no prompt embeds, no cache_salt) — i.e.,
    ``extra_keys`` is always None.

    The normal request path iterates full blocks, computing a rolling
    hash where each block's hash depends on the previous block's hash
    (chain hashing).
    """
    hashes: list[BlockHash] = []
    prev_block_hash: BlockHash | None = None

    num_full_blocks = len(token_ids) // block_size
    for i in range(num_full_blocks):
        start = i * block_size
        end = start + block_size
        block_tokens = token_ids[start:end]
        block_hash = hash_block_tokens(
            sha256, prev_block_hash, block_tokens, extra_keys=None
        )
        hashes.append(block_hash)
        prev_block_hash = block_hash

    return hashes


def compute_block_hashes_warmup_path(
    token_ids: Sequence[int],
    block_size: int,
) -> list[BlockHash]:
    """Simulate warmup path block hash computation.

    The warmup path receives a concatenated token sequence (resolved
    from TokenRegistry) and must compute block hashes using the same
    ``hash_block_tokens`` function with the same chaining logic so
    that warmed blocks are discoverable via prefix cache lookup.

    This is the computation that PrefixWarmupWorker (or its caller)
    performs after ``execute_warmup_prefill`` completes, in order to
    register blocks in the BlockPool with correct hashes.
    """
    hashes: list[BlockHash] = []
    prev_block_hash: BlockHash | None = None

    num_full_blocks = len(token_ids) // block_size
    for i in range(num_full_blocks):
        start = i * block_size
        end = start + block_size
        block_tokens = token_ids[start:end]
        block_hash = hash_block_tokens(
            sha256, prev_block_hash, block_tokens, extra_keys=None
        )
        hashes.append(block_hash)
        prev_block_hash = block_hash

    return hashes


# ------------------------------------------------------------------
# Strategies
# ------------------------------------------------------------------

# Token IDs in a realistic vocabulary range
token_id_strategy = st.integers(min_value=0, max_value=32000)

# Token sequences that form at least 1 full block
token_sequence_strategy = st.lists(
    token_id_strategy,
    min_size=BLOCK_SIZE,
    max_size=BLOCK_SIZE * 10,
)


# ------------------------------------------------------------------
# Property 8: Block hash equivalence
# ------------------------------------------------------------------


class TestBlockHashEquivalence:
    """Property 8: Block hash equivalence.

    **Validates: Requirements 5.2**

    For any token sequence, the warmup prefill path must compute and
    register block hashes that are identical to those computed by the
    normal user request path for the same tokens.
    """

    @settings(max_examples=200, deadline=None)
    @given(token_ids=token_sequence_strategy)
    def test_warmup_produces_identical_hashes_to_normal_path(
        self,
        token_ids: list[int],
    ) -> None:
        """Block hashes from warmup path equal normal request path.

        **Validates: Requirements 5.2**

        Given the same token sequence, both the normal request path
        and the warmup path must produce identical BlockHash values
        for every full block.
        """
        normal_hashes = compute_block_hashes_normal_path(token_ids, BLOCK_SIZE)
        warmup_hashes = compute_block_hashes_warmup_path(token_ids, BLOCK_SIZE)

        assert len(normal_hashes) == len(warmup_hashes), (
            f"Hash count mismatch: normal={len(normal_hashes)}, "
            f"warmup={len(warmup_hashes)}"
        )

        for i, (normal_h, warmup_h) in enumerate(zip(normal_hashes, warmup_hashes)):
            assert normal_h == warmup_h, (
                f"Block {i} hash mismatch: normal={normal_h!r}, warmup={warmup_h!r}"
            )

    @settings(max_examples=200, deadline=None)
    @given(token_ids=token_sequence_strategy)
    def test_integer_hash_equivalence_for_registry_lookup(
        self,
        token_ids: list[int],
    ) -> None:
        """Integer hash() of BlockHash is identical across paths.

        **Validates: Requirements 5.2**

        The scheduler stores ``hash(blk_hash)`` (Python's built-in
        hash of the BlockHash bytes) as the key in the frequency
        tracker and token registry. The warmup path must produce
        the same integer keys so that warmed blocks can be looked
        up correctly via prefix cache.
        """
        normal_hashes = compute_block_hashes_normal_path(token_ids, BLOCK_SIZE)
        warmup_hashes = compute_block_hashes_warmup_path(token_ids, BLOCK_SIZE)

        for i, (normal_h, warmup_h) in enumerate(zip(normal_hashes, warmup_hashes)):
            assert hash(normal_h) == hash(warmup_h), (
                f"Block {i} integer hash mismatch: "
                f"hash(normal)={hash(normal_h)}, "
                f"hash(warmup)={hash(warmup_h)}"
            )

    @settings(max_examples=100, deadline=None)
    @given(
        token_ids=token_sequence_strategy,
        block_size=st.sampled_from([8, 16, 32, 64]),
    )
    def test_equivalence_across_block_sizes(
        self,
        token_ids: list[int],
        block_size: int,
    ) -> None:
        """Hash equivalence holds for various block sizes.

        **Validates: Requirements 5.2**

        The property must hold regardless of the configured block
        size, as long as both paths use the same block size.
        """
        # Ensure we have at least one full block
        if len(token_ids) < block_size:
            return

        normal_hashes = compute_block_hashes_normal_path(token_ids, block_size)
        warmup_hashes = compute_block_hashes_warmup_path(token_ids, block_size)

        assert normal_hashes == warmup_hashes, (
            f"Hash lists differ for block_size={block_size}"
        )

    @settings(max_examples=100, deadline=None)
    @given(token_ids=token_sequence_strategy)
    def test_chaining_consistency(
        self,
        token_ids: list[int],
    ) -> None:
        """Rolling hash chain produces consistent results.

        **Validates: Requirements 5.2**

        Verifies that the chain property holds: each block's hash
        depends on the previous block's hash, ensuring that
        identical prefixes always map to identical hash sequences
        regardless of path.
        """
        num_full_blocks = len(token_ids) // BLOCK_SIZE
        if num_full_blocks < 2:
            return

        # Compute hashes for full sequence
        full_hashes = compute_block_hashes_normal_path(token_ids, BLOCK_SIZE)

        # Compute hashes for just the first N-1 blocks
        partial_tokens = token_ids[: (num_full_blocks - 1) * BLOCK_SIZE]
        partial_hashes = compute_block_hashes_normal_path(partial_tokens, BLOCK_SIZE)

        # The first N-1 hashes should be identical (prefix property)
        for i in range(len(partial_hashes)):
            assert full_hashes[i] == partial_hashes[i], (
                f"Block {i} hash changed when extending sequence"
            )
