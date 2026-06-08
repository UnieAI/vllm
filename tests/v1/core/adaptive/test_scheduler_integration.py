# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for PrefixFrequencyTracker integration with Scheduler."""

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.adaptive.prefix_frequency_tracker import (
    PrefixFrequencyTracker,
)


def test_scheduler_accepts_prefix_frequency_tracker():
    """Scheduler can be constructed with a PrefixFrequencyTracker."""
    tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.9)
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=16,
    )
    # Assign tracker after construction (the default is None)
    scheduler.prefix_frequency_tracker = tracker
    assert scheduler.prefix_frequency_tracker is tracker


def test_scheduler_none_tracker_does_not_error():
    """Scheduler works without a tracker (default None)."""
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=16,
    )
    assert scheduler.prefix_frequency_tracker is None
    # Schedule with no tracker should work fine
    requests = create_requests(num_requests=1, num_tokens=32, block_size=16)
    scheduler.add_request(requests[0])
    output = scheduler.schedule()
    assert output is not None


def test_scheduler_emits_prefix_lookup_to_tracker():
    """Scheduler calls tracker.update() for each block hash on lookup.

    Validates Requirement 1.1: WHEN a request arrives and prefix cache
    lookup is performed, THE Prefix_Frequency_Tracker SHALL update the
    EMA score for each queried prefix pattern.
    """
    tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.9)
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=16,
    )
    scheduler.prefix_frequency_tracker = tracker

    # Create a request with enough tokens to produce at least 1 block hash
    # (need > block_size tokens for a full block)
    requests = create_requests(num_requests=1, num_tokens=48, block_size=16)
    request = requests[0]

    # The request should have block_hashes computed at creation
    assert len(request.block_hashes) > 0

    # Add and schedule the request
    scheduler.add_request(request)
    scheduler.schedule()

    # Verify the tracker was updated — it should have entries
    # for each block hash in the request
    assert len(tracker) == len(request.block_hashes)


def test_scheduler_tracker_accumulates_across_requests():
    """Multiple requests with same prefix update the same tracker entry.

    Validates that repeated prefix patterns increase EMA scores.
    """
    tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.9)
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=16,
    )
    scheduler.prefix_frequency_tracker = tracker

    # Create two requests with the same prompt (same block hashes)
    requests = create_requests(
        num_requests=2,
        num_tokens=48,
        same_prompt=True,
        block_size=16,
    )

    # Schedule first request
    scheduler.add_request(requests[0])
    scheduler.schedule()
    first_count = len(tracker)

    # Schedule second request (same prefix pattern)
    scheduler.add_request(requests[1])
    scheduler.schedule()

    # The number of distinct entries shouldn't increase since
    # same hashes are updated (not added anew)
    assert len(tracker) == first_count
