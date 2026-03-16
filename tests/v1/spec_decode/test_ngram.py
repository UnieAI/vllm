# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np

from vllm.config import SpeculativeConfig
from vllm.v1.spec_decode.ngram_proposer import (
    NgramProposer,
    _find_longest_matched_ngram_and_propose_tokens,
)


def _make_ngram_proposer(
    *,
    min_n: int,
    max_n: int,
    k: int,
    max_model_len: int = 1024,
    max_num_seqs: int = 16,
    search_window: int | None = None,
) -> NgramProposer:
    return NgramProposer(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=max_model_len),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=min_n,
                prompt_lookup_max=max_n,
                prompt_lookup_window=search_window,
                num_speculative_tokens=k,
                method="ngram",
            ),
        )
    )


def _materialize_match(
    tokens: np.ndarray,
    *,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    start_position, draft_len = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens,
        min_ngram=min_ngram,
        max_ngram=max_ngram,
        max_model_len=max_model_len,
        k=k,
    )
    return tokens[start_position : start_position + draft_len]


def test_find_longest_matched_ngram_and_propose_tokens():
    tokens = np.array([1, 2, 3, 4, 1, 2, 3, 5, 6], dtype=np.int32)
    np.testing.assert_array_equal(
        _materialize_match(
            tokens,
            min_ngram=2,
            max_ngram=2,
            max_model_len=1024,
            k=2,
        ),
        np.array([], dtype=np.int32),
    )

    tokens = np.array([1, 2, 3, 4, 1, 2, 3], dtype=np.int32)
    np.testing.assert_array_equal(
        _materialize_match(
            tokens,
            min_ngram=2,
            max_ngram=2,
            max_model_len=1024,
            k=3,
        ),
        np.array([4, 1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        _materialize_match(
            tokens,
            min_ngram=2,
            max_ngram=2,
            max_model_len=1024,
            k=2,
        ),
        np.array([4, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        _materialize_match(
            tokens,
            min_ngram=1,
            max_ngram=1,
            max_model_len=1024,
            k=3,
        ),
        np.array([4, 1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        _materialize_match(
            tokens,
            min_ngram=1,
            max_ngram=1,
            max_model_len=1024,
            k=2,
        ),
        np.array([4, 1], dtype=np.int32),
    )

    tokens = np.array([1, 3, 6, 2, 3, 4, 1, 2, 3], dtype=np.int32)
    np.testing.assert_array_equal(
        _materialize_match(
            tokens,
            min_ngram=2,
            max_ngram=2,
            max_model_len=1024,
            k=3,
        ),
        np.array([4, 1, 2], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        _materialize_match(
            tokens,
            min_ngram=1,
            max_ngram=1,
            max_model_len=1024,
            k=2,
        ),
        np.array([6, 2], dtype=np.int32),
    )


def test_ngram_proposer():
    token_ids_cpu = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    result = _make_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 0

    token_ids_cpu = np.array([[1, 2, 3, 4, 1, 2, 3]], dtype=np.int32)
    result = _make_ngram_proposer(min_n=4, max_n=4, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 0

    result = _make_ngram_proposer(min_n=3, max_n=4, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert result == [[4, 1]]

    token_ids_cpu = np.array([[2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4]], dtype=np.int32)
    result = _make_ngram_proposer(min_n=3, max_n=4, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert result == [[1, 2]]

    token_ids_cpu = np.array([[3, 4, 5, 2, 3, 4, 1, 2, 3, 4]], dtype=np.int32)
    result = _make_ngram_proposer(min_n=2, max_n=4, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert result == [[1, 2]]

    token_ids_cpu = np.array(
        [[1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3]],
        dtype=np.int32,
    )
    result = _make_ngram_proposer(min_n=3, max_n=3, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert result == [[100, 1]]

    token_ids_cpu = np.array([[]], dtype=np.int32)
    result = _make_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 0

    token_ids_cpu = np.array([[1, 2, 3, 1, 2], [4, 5, 6, -1, -1]], dtype=np.int32)
    result = _make_ngram_proposer(min_n=2, max_n=2, k=2, max_num_seqs=2).propose(
        sampled_token_ids=[[0], [1]],
        num_tokens_no_spec=np.array([5, 3], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert result[0] == [3, 1]
    assert result[1] == []

    proposer = _make_ngram_proposer(min_n=2, max_n=2, k=2, max_model_len=20)
    token_ids_cpu = np.zeros((3, 20), dtype=np.int32)
    token_ids_cpu[0, :5] = [1, 2, 3, 1, 2]
    token_ids_cpu[1, :3] = [4, 5, 6]
    token_ids_cpu[2, :5] = [7, 8, 9, 7, 8]
    result = proposer.propose(
        sampled_token_ids=[[2], [], [8]],
        num_tokens_no_spec=np.array([5, 3, 5], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert result == [[3, 1], [], [9, 7]]
    assert proposer.valid_ngram_num_drafts[0] == 2
    assert proposer.valid_ngram_num_drafts[1] == 0
    assert proposer.valid_ngram_num_drafts[2] == 2
    np.testing.assert_array_equal(proposer.valid_ngram_draft[0, :2], [3, 1])
    np.testing.assert_array_equal(proposer.valid_ngram_draft[2, :2], [9, 7])


def test_ngram_proposer_search_window():
    token_ids_cpu = np.array([[1, 2, 3, 4, 1, 2, 3]], dtype=np.int32)

    full_history_result = _make_ngram_proposer(
        min_n=3,
        max_n=3,
        k=2,
        search_window=None,
    ).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([7], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert full_history_result == [[4, 1]]

    recent_window_result = _make_ngram_proposer(
        min_n=3,
        max_n=3,
        k=2,
        search_window=4,
    ).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([7], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert recent_window_result == [[]]


def test_ngram_proposer_request_filter_hook():
    class FilteredNgramProposer(NgramProposer):
        def __init__(self, *args, skipped_requests: set[int], **kwargs):
            self.skipped_requests = skipped_requests
            super().__init__(*args, **kwargs)

        def should_propose_for_request(
            self,
            request_index: int,
            sampled_ids: list[int],
            num_tokens: int,
        ) -> bool:
            return (
                request_index not in self.skipped_requests
                and super().should_propose_for_request(
                    request_index, sampled_ids, num_tokens
                )
            )

    proposer = FilteredNgramProposer(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=1024),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=2),
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=2,
                prompt_lookup_max=2,
                num_speculative_tokens=2,
                method="ngram",
            ),
        ),
        skipped_requests={0},
    )

    token_ids_cpu = np.array([[1, 2, 3, 1, 2], [7, 8, 9, 7, 8]], dtype=np.int32)
    result = proposer.propose(
        sampled_token_ids=[[0], [1]],
        num_tokens_no_spec=np.array([5, 5], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    assert result[0] == []
    assert result[1] == [9, 7]
