# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Property-based tests for TokenRegistry.

# Feature: adaptive-warmup-data-plane, Property 1: TokenRegistry 註冊與讀取 round-trip
# Feature: adaptive-warmup-data-plane, Property 3: TokenRegistry 淘汰正確性
# Feature: adaptive-warmup-data-plane, Property 4: TokenRegistry 序列化 round-trip
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.v1.core.adaptive.token_registry import TokenRegistry

# --- Constants ---
BLOCK_SIZE = 16

# --- Strategies ---

block_hashes = st.integers(min_value=0, max_value=2**63 - 1)
token_ids_st = st.lists(st.integers(0, 32000), min_size=BLOCK_SIZE, max_size=BLOCK_SIZE)


def token_ids_strategy(block_size: int):
    """Generate a valid token ID list of exactly block_size length."""
    return st.lists(
        st.integers(min_value=0, max_value=32000),
        min_size=block_size,
        max_size=block_size,
    )


# --- Property 1: TokenRegistry 註冊與讀取 round-trip ---


class TestTokenRegistryRoundTrip:
    """Property 1: TokenRegistry 註冊與讀取 round-trip.

    **Validates: Requirements 2.1, 2.4**

    For any valid block hash (integer) and any token ID sequence of
    length equal to block_size, calling `register(hash, tokens)`
    followed by `get_tokens(hash)` should return a list equal to
    the original token sequence. For any unregistered block hash,
    `get_tokens` should return None.
    """

    @settings(max_examples=100)
    @given(block_hash=block_hashes, tokens=token_ids_st)
    def test_register_then_get_returns_equal_tokens(
        self,
        block_hash: int,
        tokens: list[int],
    ):
        """register(hash, tokens) followed by get_tokens(hash)
        returns equal tokens.

        **Validates: Requirements 2.1, 2.4**
        """
        registry = TokenRegistry(max_entries=1024, block_size=BLOCK_SIZE)
        registry.register(block_hash, tokens)
        result = registry.get_tokens(block_hash)
        assert result == tokens

    @settings(max_examples=100)
    @given(block_hash=block_hashes)
    def test_get_tokens_returns_none_for_unregistered(
        self,
        block_hash: int,
    ):
        """get_tokens returns None for unregistered hashes.

        **Validates: Requirements 2.4**
        """
        registry = TokenRegistry(max_entries=1024, block_size=BLOCK_SIZE)
        result = registry.get_tokens(block_hash)
        assert result is None


# --- Property 3: TokenRegistry 淘汰正確性 ---


def registry_entries_strategy(max_entries: int, block_size: int):
    """Generate a list of (block_hash, token_ids) entries to fill a
    registry."""
    return st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=2**63 - 1),
            token_ids_strategy(block_size),
        ),
        min_size=max_entries,
        max_size=max_entries * 2,
    )


# Feature: adaptive-warmup-data-plane, Property 3: TokenRegistry 淘汰正確性


@settings(max_examples=100)
@given(
    max_entries=st.integers(min_value=2, max_value=20),
    block_size=st.integers(min_value=1, max_value=16),
    data=st.data(),
)
def test_evict_stale_removes_only_inactive_entries(
    max_entries: int,
    block_size: int,
    data: st.DataObject,
) -> None:
    """Property 3a: evict_stale correctness.

    After calling evict_stale(active_hashes), all remaining entries
    in the registry must have their block hash in active_hashes.

    **Validates: Requirements 2.3**
    """
    registry = TokenRegistry(max_entries=max_entries, block_size=block_size)

    # Generate entries to fill the registry
    entries = data.draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=2**63 - 1),
                token_ids_strategy(block_size),
            ),
            min_size=1,
            max_size=max_entries * 2,
        )
    )

    for block_hash, token_ids in entries:
        registry.register(block_hash, token_ids)

    # Draw a subset of registered hashes as active
    registered_hashes = list(registry._entries.keys())
    if registered_hashes:
        active_subset_size = data.draw(
            st.integers(min_value=0, max_value=len(registered_hashes))
        )
        active_hashes = set(registered_hashes[:active_subset_size])
    else:
        active_hashes = set()

    registry.evict_stale(active_hashes)

    # Property: all remaining entries must be in active_hashes
    for remaining_hash in registry._entries:
        assert remaining_hash in active_hashes, (
            f"Entry with hash {remaining_hash} survived eviction "
            f"but is not in active_hashes"
        )


@settings(max_examples=100)
@given(
    max_entries=st.integers(min_value=2, max_value=20),
    block_size=st.integers(min_value=1, max_value=16),
    data=st.data(),
)
def test_register_at_capacity_evicts_oldest_entry(
    max_entries: int,
    block_size: int,
    data: st.DataObject,
) -> None:
    """Property 3b: LRU fallback eviction on register().

    When the registry is at capacity and register() is called with
    a new entry, the oldest entry (first in insertion order) is
    evicted.

    **Validates: Requirements 2.3**
    """
    registry = TokenRegistry(max_entries=max_entries, block_size=block_size)

    # Fill registry to capacity with unique hashes
    fill_hashes: list[int] = []
    fill_entries = data.draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=2**63 - 1),
                token_ids_strategy(block_size),
            ),
            min_size=max_entries * 3,
            max_size=max_entries * 3,
        )
    )

    # Use unique hashes to fill exactly max_entries
    seen: set[int] = set()
    for block_hash, token_ids in fill_entries:
        if block_hash not in seen:
            registry.register(block_hash, token_ids)
            seen.add(block_hash)
            fill_hashes.append(block_hash)
        if len(fill_hashes) == max_entries:
            break

    # If we couldn't generate enough unique hashes, skip
    if len(registry) < max_entries:
        return

    assert len(registry) == max_entries

    # Record which hash is oldest (first inserted)
    oldest_hash = fill_hashes[0]
    assert oldest_hash in registry._entries

    # Register a brand new hash not already in the registry
    new_hash = data.draw(
        st.integers(min_value=0, max_value=2**63 - 1).filter(
            lambda h: h not in registry._entries
        )
    )
    new_tokens = data.draw(token_ids_strategy(block_size))

    registry.register(new_hash, new_tokens)

    # Property: size is still bounded
    assert len(registry) <= max_entries

    # Property: the oldest entry was evicted
    assert oldest_hash not in registry._entries, (
        f"Oldest hash {oldest_hash} was not evicted when "
        f"registering new hash {new_hash} at capacity"
    )

    # Property: the new entry is present
    assert new_hash in registry._entries


# --- Property 4: TokenRegistry 序列化 round-trip ---


@st.composite
def registry_entries_for_roundtrip(draw, block_size: int, max_count: int):
    """Generate a list of unique (block_hash, token_ids) entries."""
    count = draw(st.integers(min_value=0, max_value=max_count))
    entries: list[tuple[int, list[int]]] = []
    seen: set[int] = set()
    for _ in range(count):
        h = draw(st.integers(min_value=0, max_value=2**63 - 1))
        if h in seen:
            continue
        seen.add(h)
        tokens = draw(token_ids_strategy(block_size))
        entries.append((h, tokens))
    return entries


# Feature: adaptive-warmup-data-plane, Property 4: TokenRegistry 序列化 round-trip


class TestTokenRegistrySerializationRoundTrip:
    """Property 4: TokenRegistry 序列化 round-trip.

    For any valid TokenRegistry state (arbitrary number of entries,
    each containing a block_hash and a block_size-length token
    sequence), calling to_dict() followed by from_dict() should
    produce a TokenRegistry containing exactly the same mappings.

    **Validates: Requirements 2.6**
    """

    @settings(max_examples=100)
    @given(
        max_entries=st.integers(min_value=1, max_value=100),
        block_size=st.integers(min_value=1, max_value=64),
        data=st.data(),
    )
    def test_serialization_round_trip(
        self,
        max_entries: int,
        block_size: int,
        data: st.DataObject,
    ) -> None:
        """Serialize then deserialize preserves all mappings.

        # Feature: adaptive-warmup-data-plane, Property 4: TokenRegistry 序列化 round-trip
        """  # noqa: E501
        entries = data.draw(
            registry_entries_for_roundtrip(block_size=block_size, max_count=max_entries)
        )

        # Build registry with random entries
        registry = TokenRegistry(max_entries=max_entries, block_size=block_size)
        for block_hash, token_ids in entries:
            registry.register(block_hash, token_ids)

        # Serialize
        serialized = registry.to_dict()

        # Deserialize
        restored = TokenRegistry.from_dict(
            serialized,
            max_entries=max_entries,
            block_size=block_size,
        )

        # Verify same number of entries
        assert len(restored) == len(registry)

        # Verify all mappings are preserved
        for block_hash, _ in entries:
            original_tokens = registry.get_tokens(block_hash)
            restored_tokens = restored.get_tokens(block_hash)
            assert restored_tokens == original_tokens, (
                f"Mismatch for hash {block_hash}: "
                f"original={original_tokens}, "
                f"restored={restored_tokens}"
            )

    @settings(max_examples=100)
    @given(
        max_entries=st.integers(min_value=1, max_value=50),
        block_size=st.integers(min_value=1, max_value=32),
        data=st.data(),
    )
    def test_serialization_round_trip_with_evictions(
        self,
        max_entries: int,
        block_size: int,
        data: st.DataObject,
    ) -> None:
        """Round-trip works after entries have been evicted.

        # Feature: adaptive-warmup-data-plane, Property 4: TokenRegistry 序列化 round-trip
        """  # noqa: E501
        # Generate more entries than max to trigger evictions
        num_entries = data.draw(st.integers(min_value=0, max_value=max_entries * 2))
        registry = TokenRegistry(max_entries=max_entries, block_size=block_size)

        for _ in range(num_entries):
            block_hash = data.draw(st.integers(min_value=0, max_value=2**63 - 1))
            token_ids = data.draw(token_ids_strategy(block_size))
            registry.register(block_hash, token_ids)

        # Serialize and restore
        serialized = registry.to_dict()
        restored = TokenRegistry.from_dict(
            serialized,
            max_entries=max_entries,
            block_size=block_size,
        )

        # Verify same size
        assert len(restored) == len(registry)

        # Verify all current entries match
        for key in list(registry._entries.keys()):
            assert restored.get_tokens(key) == registry.get_tokens(key)
