# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import numpy as np
import pytest

from vllm.config import (
    ModelConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.spec_decode.ngram_proposer import (
    NgramProposer,
    _find_longest_matched_ngram_and_propose_tokens,
)
from vllm.v1.spec_decode.ngram_dsc_controller import NgramDSCController


def test_find_longest_matched_ngram_and_propose_tokens():
    tokens = np.array([1, 2, 3, 4, 1, 2, 3, 5, 6])
    result = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=2
    )
    assert len(result) == 0

    tokens = np.array([1, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=3
        ),
        np.array([4, 1, 2]),
    )
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=2
        ),
        np.array([4, 1]),
    )
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=1, max_ngram=1, max_model_len=1024, k=3
        ),
        np.array([4, 1, 2]),
    )
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=1, max_ngram=1, max_model_len=1024, k=2
        ),
        np.array([4, 1]),
    )

    tokens = np.array([1, 3, 6, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=3
        ),
        np.array([4, 1, 2]),
    )
    # Return on the first match
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=1, max_ngram=1, max_model_len=1024, k=2
        ),
        np.array([6, 2]),
    )


def test_ngram_proposer():
    def get_ngram_proposer(min_n: int, max_n: int, k: int) -> NgramProposer:
        # Dummy model config. Just to set max_model_len.
        model_config = ModelConfig(model="facebook/opt-125m")
        return NgramProposer(
            vllm_config=VllmConfig(
                model_config=model_config,
                speculative_config=SpeculativeConfig(
                    prompt_lookup_min=min_n,
                    prompt_lookup_max=max_n,
                    num_speculative_tokens=k,
                    method="ngram",
                ),
            )
        )

    # No match.
    token_ids_cpu = np.array([[1, 2, 3, 4, 5]])
    result = get_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 0

    # No match for 4-gram.
    token_ids_cpu = np.array([[1, 2, 3, 4, 1, 2, 3]])
    result = get_ngram_proposer(min_n=4, max_n=4, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 0

    # No match for 4-gram but match for 3-gram.
    token_ids_cpu = np.array([[1, 2, 3, 4, 1, 2, 3]])
    result = get_ngram_proposer(min_n=3, max_n=4, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(result, np.array([[4, 1]]))

    # Match for both 4-gram and 3-gram.
    # In this case, the proposer should return the 4-gram match.
    token_ids_cpu = np.array([[2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4]])
    result = get_ngram_proposer(min_n=3, max_n=4, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(result, np.array([[1, 2]]))  # Not [5, 1]]

    # Match for 2-gram and 3-gram, but not 4-gram.
    token_ids_cpu = np.array([[3, 4, 5, 2, 3, 4, 1, 2, 3, 4]])
    result = get_ngram_proposer(min_n=2, max_n=4, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(result, np.array([[1, 2]]))  # Not [5, 2]]

    # Multiple 3-gram matched, but always pick the first one.
    token_ids_cpu = np.array([[1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3]])
    result = get_ngram_proposer(min_n=3, max_n=3, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(result, np.array([[100, 1]]))

    # check empty input
    token_ids_cpu = np.array([[]])
    result = get_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 0

    # check multibatch input
    # first request has 5 tokens and a match
    # second request has 3 tokens and no match. Padded with -1 for max len 5
    token_ids_cpu = np.array([[1, 2, 3, 1, 2], [4, 5, 6, -1, -1]])
    result = get_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        sampled_token_ids=[[0], [1]],
        num_tokens_no_spec=np.array([5, 3]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 2
    assert np.array_equal(result[0], np.array([3, 1]))
    assert np.array_equal(result[1], np.array([]))

    # Test non-contiguous indices: requests 0 and 2 need proposals,
    # request 1 is in prefill
    proposer = get_ngram_proposer(min_n=2, max_n=2, k=2)
    max_model_len = 20
    token_ids_cpu = np.zeros((3, max_model_len), dtype=np.int32)
    token_ids_cpu[0, :5] = [1, 2, 3, 1, 2]
    token_ids_cpu[1, :3] = [4, 5, 6]
    token_ids_cpu[2, :5] = [7, 8, 9, 7, 8]
    num_tokens_no_spec = np.array([5, 3, 5], dtype=np.int32)
    sampled_token_ids = [[2], [], [8]]  # Empty list for request 1 simulates prefill
    result = proposer.propose(
        sampled_token_ids=sampled_token_ids,
        num_tokens_no_spec=num_tokens_no_spec,
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result) == 3
    assert np.array_equal(result[0], [3, 1])
    assert len(result[1]) == 0
    assert np.array_equal(result[2], [9, 7])
    # Verify internal arrays written to correct indices
    assert proposer.valid_ngram_num_drafts[0] == 2
    assert proposer.valid_ngram_num_drafts[1] == 0
    assert proposer.valid_ngram_num_drafts[2] == 2
    assert np.array_equal(proposer.valid_ngram_draft[0, :2], [3, 1])
    assert np.array_equal(proposer.valid_ngram_draft[2, :2], [9, 7])

    # test if 0 threads available: can happen if TP size > CPU count
    ngram_proposer = get_ngram_proposer(min_n=2, max_n=2, k=2)
    ngram_proposer.num_numba_thread_available = 0
    # set max_model_len to 2 * threshold to ensure multithread is used
    num_tokens_threshold = ngram_proposer.num_tokens_threshold
    ngram_proposer.max_model_len = 2 * num_tokens_threshold
    # using multibatch test
    middle_integer = num_tokens_threshold // 2
    input_1 = [_ for _ in range(num_tokens_threshold)]
    input_1 += [middle_integer, middle_integer + 1]
    input_2 = [-1] * len(input_1)
    input_2[:3] = [4, 5, 6]
    token_ids_cpu = np.array([input_1, input_2])
    result = ngram_proposer.propose(
        sampled_token_ids=[[0], [1]],
        num_tokens_no_spec=np.array([len(input_1), 3]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 2
    assert np.array_equal(result[0], np.array([middle_integer + 2, middle_integer + 3]))
    assert np.array_equal(result[1], np.array([]))


def test_ngram_dsc_method_alias():
    speculative_config = SpeculativeConfig(
        method="ngram_dsc",
        num_speculative_tokens=2,
    )
    assert speculative_config.method == "ngram"
    assert speculative_config.ngram_dsc is True


def test_ngram_proposer_runtime_effective_spec_tokens():
    model_config = ModelConfig(model="facebook/opt-125m")
    proposer = NgramProposer(
        vllm_config=VllmConfig(
            model_config=model_config,
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=2,
                prompt_lookup_max=2,
                num_speculative_tokens=3,
                method="ngram",
            ),
        )
    )

    token_ids_cpu = np.array([[1, 2, 3, 4, 1, 2, 3]])
    num_tokens_no_spec = np.array([len(token_ids_cpu[0])])

    result = proposer.propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=num_tokens_no_spec,
        token_ids_cpu=token_ids_cpu,
        effective_num_spec_tokens=1,
    )
    assert np.array_equal(result, np.array([[4]]))

    result = proposer.propose(
        sampled_token_ids=[[0]],
        num_tokens_no_spec=num_tokens_no_spec,
        token_ids_cpu=token_ids_cpu,
        effective_num_spec_tokens=0,
    )
    assert result == [[]]


def test_ngram_dsc_controller_argmax_goodput_can_choose_k0():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.0,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_initial_max_k=4,
            ngram_dsc_base_latency_tokens=32.0,
            ngram_dsc_disable_decode_tokens=1,
            ngram_dsc_enable_decode_tokens=1,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=5, now=1.0
    )
    assert state.load_regime == "adaptive"
    assert state.baseline_k == 0
    assert state.effective_num_spec_tokens == 0


def test_ngram_dsc_controller_default_cold_start_is_less_optimistic():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_base_latency_tokens=32.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    state = controller.decide(
        decode_token_load=0, total_num_scheduled_tokens=268, now=1.0
    )
    assert state.load_regime == "adaptive"
    assert state.best_candidate_k >= 1
    assert state.effective_num_spec_tokens >= 1


def test_ngram_dsc_controller_argmax_goodput_can_choose_max_k():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.80,
            ngram_dsc_position_acceptance_prior_rate=0.95,
            ngram_dsc_position_acceptance_prior_decay=1.0,
            ngram_dsc_position_acceptance_prior_strength=0.0,
            ngram_dsc_position_acceptance_confidence_z=0.0,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_initial_max_k=4,
            ngram_dsc_base_latency_tokens=32.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=5, now=1.0
    )
    assert state.load_regime == "adaptive"
    assert state.baseline_k == 4
    assert state.effective_num_spec_tokens == 4


def test_ngram_dsc_controller_idle_step_can_choose_best_k():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.8,
            ngram_dsc_position_acceptance_prior_rate=0.95,
            ngram_dsc_position_acceptance_prior_decay=1.0,
            ngram_dsc_position_acceptance_prior_strength=0.0,
            ngram_dsc_position_acceptance_confidence_z=0.0,
            ngram_dsc_initial_max_k=1,
            ngram_dsc_base_latency_tokens=32.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    state = controller.decide(
        decode_token_load=0, total_num_scheduled_tokens=16, now=1.0
    )
    assert state.load_regime == "adaptive"
    assert state.baseline_k == 4
    assert state.best_candidate_k == 4
    assert state.effective_num_spec_tokens == 4


def test_ngram_dsc_controller_rejections_reduce_selected_k():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.8,
            ngram_dsc_position_acceptance_prior_rate=0.95,
            ngram_dsc_position_acceptance_prior_decay=1.0,
            ngram_dsc_position_acceptance_prior_strength=0.0,
            ngram_dsc_position_acceptance_confidence_z=0.0,
            ngram_dsc_base_latency_tokens=32.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    state = controller.decide(
        decode_token_load=0, total_num_scheduled_tokens=16, now=1.0
    )
    assert state.load_regime == "adaptive"
    assert state.effective_num_spec_tokens == 4

    controller.observe_draft(
        num_draft_tokens=3,
        num_accepted_tokens=0,
        position_draft_counts=(1, 1, 1, 0),
        position_accept_counts=(0, 0, 0, 0),
    )
    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=4, now=2.0
    )
    assert state.load_regime == "adaptive"
    assert state.effective_num_spec_tokens == 2

    controller.observe_draft(
        num_draft_tokens=3,
        num_accepted_tokens=0,
        position_draft_counts=(1, 1, 1, 0),
        position_accept_counts=(0, 0, 0, 0),
    )
    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=2, now=3.0
    )
    assert state.load_regime == "adaptive"
    assert state.baseline_k == state.best_candidate_k
    assert state.effective_num_spec_tokens == 1


def test_ngram_dsc_controller_sparse_zero_acceptance_keeps_nonzero_posterior():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.8,
            ngram_dsc_position_acceptance_prior_rate=0.4,
            ngram_dsc_position_acceptance_prior_decay=0.5,
            ngram_dsc_position_acceptance_prior_strength=8.0,
            ngram_dsc_position_acceptance_confidence_z=0.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    controller.observe_draft(
        num_draft_tokens=1,
        num_accepted_tokens=0,
        speculation_attempted=True,
        num_eligible_decode_reqs=8,
        num_draft_reqs=1,
        position_draft_counts=(1, 0, 0, 0),
        position_accept_counts=(0, 0, 0, 0),
    )

    conservative_rates = controller._get_conservative_position_acceptance_rates()
    assert controller.proposal_coverage_ema == pytest.approx(0.7375)
    assert conservative_rates[0] > 0.0
    assert conservative_rates[1] <= conservative_rates[0]


def test_ngram_dsc_controller_reports_adaptive_mode():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.8,
            ngram_dsc_base_latency_tokens=32.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    state = controller.decide(
        decode_token_load=8, total_num_scheduled_tokens=268, now=1.0
    )
    assert state.load_regime == "adaptive"
    assert state.baseline_k == state.best_candidate_k


def test_ngram_dsc_controller_goodput_holds_k_during_dwell_window():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_initial_max_k=4,
            ngram_dsc_base_latency_tokens=32.0,
            ngram_dsc_goodput_margin=0.0,
            ngram_dsc_goodput_increase_margin=0.0,
            ngram_dsc_goodput_min_dwell_sec=1.0,
            ngram_dsc_max_step_delta=1,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=5, now=1.0
    )
    assert state.effective_num_spec_tokens == 4

    controller.observe_draft(
        num_draft_tokens=16,
        num_accepted_tokens=0,
        position_draft_counts=(16, 0, 0, 0),
        position_accept_counts=(0, 0, 0, 0),
    )
    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=2, now=2.0
    )
    assert state.effective_num_spec_tokens == 0

    controller.observe_draft(
        num_draft_tokens=8,
        num_accepted_tokens=8,
        position_draft_counts=(8, 0, 0, 0),
        position_accept_counts=(8, 0, 0, 0),
    )
    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=3, now=2.5
    )
    assert state.effective_num_spec_tokens == 0

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=3, now=3.1
    )
    assert state.effective_num_spec_tokens == 1


def test_ngram_dsc_controller_goodput_smooths_scheduled_token_load():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_initial_max_k=4,
            ngram_dsc_base_latency_tokens=32.0,
            ngram_dsc_goodput_margin=0.0,
            ngram_dsc_goodput_increase_margin=0.0,
            ngram_dsc_scheduled_tokens_ema_alpha=0.5,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=6, now=1.0
    )
    assert state.smoothed_total_num_scheduled_tokens == 5.0

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=2, now=2.0
    )
    assert state.smoothed_total_num_scheduled_tokens == 3.5


def test_ngram_dsc_controller_cold_start_caps_initial_k():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_initial_max_k=1,
            ngram_dsc_upward_min_position_samples=8,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=5, now=1.0
    )
    assert state.effective_num_spec_tokens == 1


def test_ngram_dsc_controller_fast_fails_after_early_bad_acceptance():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_initial_max_k=1,
            ngram_dsc_fast_fail_min_steps=2,
            ngram_dsc_fast_fail_max_steps=3,
            ngram_dsc_fast_fail_max_acceptance_rate=0.05,
            ngram_dsc_goodput_min_dwell_sec=10.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=5, now=1.0
    )
    assert state.effective_num_spec_tokens == 1

    controller.observe_draft(num_draft_tokens=1, num_accepted_tokens=0)
    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=4, now=2.0
    )
    assert state.effective_num_spec_tokens == 1

    controller.observe_draft(num_draft_tokens=1, num_accepted_tokens=0)
    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=3, now=3.0
    )
    assert state.effective_num_spec_tokens == 0


def test_ngram_dsc_controller_does_not_reenter_same_request_after_fast_fail():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_initial_max_k=1,
            ngram_dsc_fast_fail_min_steps=2,
            ngram_dsc_fast_fail_max_steps=3,
            ngram_dsc_fast_fail_max_acceptance_rate=0.05,
            ngram_dsc_goodput_min_dwell_sec=0.0,
            ngram_dsc_upward_min_position_samples=1,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=5, now=1.0
    )
    assert state.effective_num_spec_tokens == 1

    controller.observe_draft(num_draft_tokens=1, num_accepted_tokens=0)
    controller.observe_draft(num_draft_tokens=1, num_accepted_tokens=0)
    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=3, now=2.0
    )
    assert state.effective_num_spec_tokens == 0

    controller.acceptance_rate_ema = 1.0
    controller.position_acceptance_rate_ema[:] = 1.0
    controller.position_acceptance_opportunities[:] = 16

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=2, now=3.0
    )
    assert state.effective_num_spec_tokens == 0

    state = controller.decide(
        decode_token_load=8, total_num_scheduled_tokens=8, now=3.5
    )
    assert state.effective_num_spec_tokens > 0

    state = controller.decide(
        decode_token_load=0, total_num_scheduled_tokens=0, now=4.0
    )
    assert state.effective_num_spec_tokens == 0
    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=2, now=5.0
    )
    assert state.effective_num_spec_tokens == 1


def test_ngram_dsc_controller_fast_fail_lock_holds_only_for_cold_start_load():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_initial_max_k=1,
            ngram_dsc_fast_fail_min_steps=2,
            ngram_dsc_fast_fail_max_steps=3,
            ngram_dsc_fast_fail_max_acceptance_rate=0.05,
            ngram_dsc_goodput_min_dwell_sec=0.0,
            ngram_dsc_upward_min_position_samples=1,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    controller.decide(decode_token_load=1, total_num_scheduled_tokens=5, now=1.0)
    controller.observe_draft(num_draft_tokens=1, num_accepted_tokens=0)
    controller.observe_draft(num_draft_tokens=1, num_accepted_tokens=0)

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=3, now=2.0
    )
    assert state.effective_num_spec_tokens == 0

    controller.acceptance_rate_ema = 1.0
    controller.position_acceptance_rate_ema[:] = 1.0
    controller.position_acceptance_opportunities[:] = 16

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=2, now=3.0
    )
    assert state.effective_num_spec_tokens == 0

    state = controller.decide(
        decode_token_load=8, total_num_scheduled_tokens=8, now=4.0
    )
    assert state.effective_num_spec_tokens > 0


def test_ngram_dsc_controller_fast_fail_penalizes_unobserved_positions():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_initial_max_k=1,
            ngram_dsc_fast_fail_min_steps=2,
            ngram_dsc_fast_fail_max_steps=3,
            ngram_dsc_fast_fail_max_acceptance_rate=0.05,
            ngram_dsc_goodput_min_dwell_sec=0.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    controller.decide(decode_token_load=1, total_num_scheduled_tokens=5, now=1.0)
    controller.observe_draft(num_draft_tokens=1, num_accepted_tokens=0)
    controller.observe_draft(num_draft_tokens=1, num_accepted_tokens=0)
    controller.decide(decode_token_load=1, total_num_scheduled_tokens=3, now=2.0)

    assert controller.position_acceptance_rate_ema[1] == pytest.approx(0.0)
    assert controller.position_acceptance_rate_ema[2] == pytest.approx(0.0)
    assert controller.position_acceptance_rate_ema[3] == pytest.approx(0.0)


def test_ngram_dsc_controller_profiled_latency_model_changes_k():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_initial_max_k=4,
            ngram_dsc_latency_model="profiled",
            ngram_dsc_profiled_latency_intercept_s=0.10,
            ngram_dsc_profiled_latency_spec_tokens_coeff_s=0.09,
            ngram_dsc_goodput_margin=0.0,
            ngram_dsc_goodput_increase_margin=0.0,
            ngram_dsc_max_step_delta=4,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    state = controller.decide(
        decode_token_load=1, total_num_scheduled_tokens=2, now=1.0
    )
    assert state.latency_model == "profiled"
    assert state.load_regime == "adaptive"
    assert state.baseline_k == 0
    assert state.effective_num_spec_tokens == 0
    assert state.selected_latency_s == pytest.approx(0.10)
    assert state.selected_expected_tokens == pytest.approx(1.0)
    assert state.candidate_latencies_s[0] == pytest.approx(0.10)


def test_ngram_dsc_controller_prefers_higher_goodput_by_default():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=2,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    assert controller._select_best_candidate(  # type: ignore[attr-defined]
        (
            (10.00, 0.010, 1.0),
            (10.10, 0.020, 1.1),
            (9.00, 0.005, 1.0),
        )
    ) == 1


def test_ngram_dsc_controller_keeps_probing_when_k0_edge_is_small_and_evidence_is_sparse():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=2,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_min_spec_realized_samples_before_k0=8,
            ngram_dsc_k0_sparse_evidence_margin=0.10,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    controller.realized_k_step_counts[1] = 3

    assert controller._select_sparse_evidence_probe_candidate(  # type: ignore[attr-defined]
        (
            (30.3090, 0.032993, 1.000),
            (29.5189, 0.038431, 1.134),
            (26.2719, 0.043869, 1.153),
        )
    ) == 1


def test_ngram_dsc_controller_allows_k0_after_enough_speculative_evidence():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=2,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_min_spec_realized_samples_before_k0=8,
            ngram_dsc_k0_sparse_evidence_margin=0.10,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    controller.realized_k_step_counts[1] = 8

    assert controller._select_sparse_evidence_probe_candidate(  # type: ignore[attr-defined]
        (
            (30.3090, 0.032993, 1.000),
            (29.5189, 0.038431, 1.134),
            (26.2719, 0.043869, 1.153),
        )
    ) == 0


def test_ngram_dsc_controller_prefers_lower_latency_within_near_best_goodput_band():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=2,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_near_best_goodput_ratio=0.02,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    assert controller._select_best_candidate(  # type: ignore[attr-defined]
        (
            (10.00, 0.010, 1.0),
            (10.10, 0.020, 1.1),
            (9.00, 0.005, 1.0),
        )
    ) == 0


def test_ngram_dsc_controller_does_not_decay_proposal_coverage_when_speculation_is_off():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_acceptance_ema_alpha=0.5,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    controller.observe_draft(
        num_draft_tokens=1,
        num_accepted_tokens=0,
        speculation_attempted=True,
        num_eligible_decode_reqs=4,
        num_draft_reqs=1,
        position_draft_counts=(1, 0, 0, 0),
        position_accept_counts=(0, 0, 0, 0),
    )
    observed_coverage = controller.proposal_coverage_ema

    controller.observe_draft(
        num_draft_tokens=0,
        num_accepted_tokens=0,
        speculation_attempted=False,
        num_eligible_decode_reqs=4,
        num_draft_reqs=0,
    )

    assert controller.proposal_coverage_ema == pytest.approx(observed_coverage)


def test_ngram_dsc_controller_online_latency_scale_reduces_profiled_overestimate():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_latency_model="profiled",
            ngram_dsc_profiled_latency_intercept_s=0.20,
            ngram_dsc_online_latency_fitting=True,
            ngram_dsc_online_latency_fit_min_samples=64,
            ngram_dsc_online_latency_fit_warmup_samples=0,
            ngram_dsc_online_latency_fit_ema_alpha=1.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    predicted_latency_before = controller._estimate_latency(  # type: ignore[attr-defined]
        0,
        decode_token_load=1,
        total_num_scheduled_tokens=2.0,
    )
    controller.observe_realized_latency(
        decode_token_load=1,
        smoothed_total_num_scheduled_tokens=2.0,
        effective_num_spec_tokens=0,
        realized_latency_s=0.02,
        realized_generated_tokens=1.0,
    )
    predicted_latency_after = controller._estimate_latency(  # type: ignore[attr-defined]
        0,
        decode_token_load=1,
        total_num_scheduled_tokens=2.0,
    )
    assert predicted_latency_after < predicted_latency_before
    assert predicted_latency_after == pytest.approx(0.02)


def test_ngram_dsc_controller_position_rates_are_monotone_for_unobserved_late_positions():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.8,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_ema_alpha=1.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    controller.observe_draft(
        num_draft_tokens=3,
        num_accepted_tokens=0,
        position_draft_counts=(1, 1, 1, 0),
        position_accept_counts=(0, 0, 0, 0),
    )
    state = controller.decide(
        decode_token_load=1,
        total_num_scheduled_tokens=2,
        now=1.0,
    )
    assert state.position_acceptance_rates == pytest.approx((0.0, 0.0, 0.0, 0.0))


def test_ngram_dsc_controller_skips_unstable_k0_realized_samples():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_latency_model="profiled",
            ngram_dsc_online_latency_fitting=True,
            ngram_dsc_online_latency_fit_min_samples=1,
            ngram_dsc_online_latency_fit_warmup_samples=0,
            ngram_dsc_online_latency_fit_refit_interval_samples=1,
            ngram_dsc_realized_sample_min_decode_token_load=1,
            ngram_dsc_realized_sample_min_smoothed_scheduled_tokens=1.0,
            ngram_dsc_realized_sample_min_latency_s=0.001,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    assert (
        controller.should_observe_realized_sample(
            decode_token_load=0,
            smoothed_total_num_scheduled_tokens=0.64,
            effective_num_spec_tokens=0,
            realized_latency_s=0.000351,
        )
        is False
    )

    update = controller.observe_realized_latency(
        decode_token_load=0,
        smoothed_total_num_scheduled_tokens=0.64,
        effective_num_spec_tokens=0,
        realized_latency_s=0.000351,
        realized_generated_tokens=1.0,
    )
    assert update is None
    assert controller.realized_k_step_counts[0] == 0


def test_ngram_dsc_controller_throttles_normal_decode_realized_logging():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_normal_decode_realized_log_interval=3,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    assert (
        controller.should_log_realized_sample(
            effective_num_spec_tokens=0,
            sample_observed=False,
        )
        is False
    )
    assert (
        controller.should_log_realized_sample(
            effective_num_spec_tokens=0,
            sample_observed=True,
        )
        is True
    )
    assert (
        controller.should_log_realized_sample(
            effective_num_spec_tokens=0,
            sample_observed=True,
        )
        is False
    )
    assert (
        controller.should_log_realized_sample(
            effective_num_spec_tokens=0,
            sample_observed=True,
        )
        is True
    )
    assert (
        controller.should_log_realized_sample(
            effective_num_spec_tokens=2,
            sample_observed=True,
        )
        is True
    )


def test_speculative_config_loads_profiled_latency_coefficients_from_file(tmp_path):
    coeff_path = tmp_path / "ngram_dsc_latency_coeffs.json"
    coeff_path.write_text(
        json.dumps(
            {
                "coefficients": {
                    "intercept_s": 0.011,
                    "decode_token_load_coeff_s": 0.022,
                    "scheduled_tokens_coeff_s": 0.033,
                    "spec_tokens_coeff_s": 0.044,
                    "spec_scheduled_tokens_interaction_coeff_s": 0.055,
                }
            }
        ),
        encoding="utf-8",
    )

    config = SpeculativeConfig(
        method="ngram_dsc",
        num_speculative_tokens=4,
        ngram_dsc=True,
        ngram_dsc_strategy="goodput",
        ngram_dsc_profiled_latency_coefficients_path=str(coeff_path),
    )

    assert config.ngram_dsc_latency_model == "profiled"
    assert config.ngram_dsc_profiled_latency_intercept_s == pytest.approx(0.011)
    assert config.ngram_dsc_profiled_latency_decode_token_load_coeff_s == pytest.approx(
        0.022
    )
    assert config.ngram_dsc_profiled_latency_scheduled_tokens_coeff_s == pytest.approx(
        0.033
    )
    assert config.ngram_dsc_profiled_latency_spec_tokens_coeff_s == pytest.approx(
        0.044
    )
    assert (
        config.ngram_dsc_profiled_latency_spec_scheduled_tokens_interaction_coeff_s
        == pytest.approx(0.055)
    )


def test_ngram_dsc_controller_online_latency_fit_waits_for_min_samples():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_latency_model="profiled",
            ngram_dsc_profiled_latency_intercept_s=0.05,
            ngram_dsc_online_latency_fitting=True,
            ngram_dsc_online_latency_fit_warmup_samples=1,
            ngram_dsc_online_latency_fit_min_samples=4,
            ngram_dsc_online_latency_fit_refit_interval_samples=2,
            ngram_dsc_online_latency_fit_max_latency_ratio_to_median=10.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    samples = [
        (1, 2.0, 0, 0.018),
        (1, 2.0, 1, 0.024),
        (2, 3.0, 1, 0.030),
        (1, 4.0, 2, 0.036),
        (2, 5.0, 2, 0.045),
    ]
    for index, sample in enumerate(samples[:-1], start=1):
        update = controller.observe_realized_latency(
            decode_token_load=sample[0],
            smoothed_total_num_scheduled_tokens=sample[1],
            effective_num_spec_tokens=sample[2],
            realized_latency_s=sample[3],
            realized_generated_tokens=1.0,
        )
        assert update is None, f"unexpected fit update before sample {index}"

    update = controller.observe_realized_latency(
        decode_token_load=samples[-1][0],
        smoothed_total_num_scheduled_tokens=samples[-1][1],
        effective_num_spec_tokens=samples[-1][2],
        realized_latency_s=samples[-1][3],
        realized_generated_tokens=1.0,
    )
    assert update is not None
    assert update.num_valid_samples == 4
    assert controller.profiled_latency_intercept_s != pytest.approx(0.05)


def test_ngram_dsc_controller_online_latency_fit_filters_outliers():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_latency_model="profiled",
            ngram_dsc_profiled_latency_intercept_s=0.02,
            ngram_dsc_online_latency_fitting=True,
            ngram_dsc_online_latency_fit_min_samples=5,
            ngram_dsc_online_latency_fit_refit_interval_samples=1,
            ngram_dsc_online_latency_fit_max_latency_ratio_to_median=2.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    stable_samples = [
        (1, 2.0, 0, 0.018),
        (1, 2.0, 1, 0.024),
        (2, 3.0, 1, 0.030),
        (1, 4.0, 2, 0.036),
        (2, 5.0, 2, 0.045),
    ]
    update = None
    for sample in stable_samples:
        update = controller.observe_realized_latency(
            decode_token_load=sample[0],
            smoothed_total_num_scheduled_tokens=sample[1],
            effective_num_spec_tokens=sample[2],
            realized_latency_s=sample[3],
            realized_generated_tokens=1.0,
        )
    assert update is not None
    assert update.num_dropped_samples == 0

    outlier_update = controller.observe_realized_latency(
        decode_token_load=1,
        smoothed_total_num_scheduled_tokens=2.0,
        effective_num_spec_tokens=1,
        realized_latency_s=2.0,
        realized_generated_tokens=1.0,
    )
    assert outlier_update is None

    outlier_update = controller.observe_realized_latency(
        decode_token_load=3,
        smoothed_total_num_scheduled_tokens=6.0,
        effective_num_spec_tokens=3,
        realized_latency_s=0.057,
        realized_generated_tokens=1.0,
    )
    assert outlier_update is not None
    assert outlier_update.num_dropped_samples >= 1
    assert (
        controller._estimate_latency(
            1,
            decode_token_load=1,
            total_num_scheduled_tokens=2.0,
        )
        < 0.1
    )


def test_ngram_dsc_controller_uses_global_acceptance_rate_for_expected_tokens():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.9,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_prior_strength=0.0,
            ngram_dsc_position_acceptance_confidence_z=0.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    controller.observe_draft(
        num_draft_tokens=12,
        num_accepted_tokens=6,
        position_draft_counts=(4, 4, 4, 0),
        position_accept_counts=(4, 2, 0, 0),
    )

    assert controller._expected_generated_tokens(3) == pytest.approx(1.875)


def test_ngram_dsc_controller_sparse_global_acceptance_uses_prior_smoothed_estimate():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.8,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_prior_strength=8.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    for _ in range(4):
        controller.observe_draft(
            num_draft_tokens=1,
            num_accepted_tokens=0,
            speculation_attempted=True,
            num_eligible_decode_reqs=1,
            num_draft_reqs=1,
            position_draft_counts=(1, 0, 0, 0),
            position_accept_counts=(0, 0, 0, 0),
        )

    # Raw EMA collapses to 0, but the estimator used for TurboSpec goodput
    # should remain prior-smoothed under sparse evidence.
    assert controller.acceptance_rate_ema == pytest.approx(0.0)
    assert controller._expected_generated_tokens(1) > 1.0


def test_ngram_dsc_controller_shrinks_sparse_position_acceptance_with_confidence():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.8,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_prior_strength=8.0,
            ngram_dsc_position_acceptance_confidence_z=1.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    controller.observe_draft(
        num_draft_tokens=4,
        num_accepted_tokens=4,
        position_draft_counts=(1, 1, 1, 1),
        position_accept_counts=(1, 1, 1, 1),
    )

    conservative_rates = controller._get_conservative_position_acceptance_rates()
    assert conservative_rates[0] < controller.position_acceptance_rate_ema[0]
    assert controller._expected_generated_tokens(4) < 5.0


def test_ngram_dsc_controller_conservative_position_acceptance_is_monotone():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.8,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_prior_rate=0.0,
            ngram_dsc_position_acceptance_prior_strength=0.0,
            ngram_dsc_position_acceptance_confidence_z=0.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    controller.observe_draft(
        num_draft_tokens=12,
        num_accepted_tokens=8,
        position_draft_counts=(3, 3, 3, 3),
        position_accept_counts=(3, 1, 2, 2),
    )

    conservative_rates = controller._get_conservative_position_acceptance_rates()
    assert conservative_rates[0] == pytest.approx(1.0)
    assert conservative_rates[1] == pytest.approx(1.0 / 3.0)
    assert conservative_rates[2] == pytest.approx(1.0 / 3.0)
    assert conservative_rates[3] == pytest.approx(1.0 / 3.0)


def test_ngram_dsc_controller_blocks_upward_move_without_position_samples():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_ema_alpha=1.0,
            ngram_dsc_goodput_margin=0.0,
            ngram_dsc_goodput_increase_margin=0.0,
            ngram_dsc_upward_min_position_samples=4,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )
    controller.current_effective_num_spec_tokens = 1
    controller.position_acceptance_opportunities[:] = np.array([8, 0, 0, 0])

    state = controller.decide(
        decode_token_load=1,
        total_num_scheduled_tokens=2,
        now=1.0,
    )
    assert state.effective_num_spec_tokens == 1

    controller.position_acceptance_opportunities[:] = np.array([8, 8, 8, 8])
    state = controller.decide(
        decode_token_load=1,
        total_num_scheduled_tokens=2,
        now=2.0,
    )
    assert state.effective_num_spec_tokens == 2


def test_ngram_dsc_controller_tracks_recent_realized_goodput_by_k():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_realized_goodput_ema_alpha=0.5,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    controller.observe_realized_latency(
        decode_token_load=1,
        smoothed_total_num_scheduled_tokens=2.0,
        effective_num_spec_tokens=2,
        realized_latency_s=0.1,
        realized_generated_tokens=2.0,
    )
    controller.observe_realized_latency(
        decode_token_load=1,
        smoothed_total_num_scheduled_tokens=2.0,
        effective_num_spec_tokens=2,
        realized_latency_s=0.2,
        realized_generated_tokens=1.0,
    )

    state = controller.decide(
        decode_token_load=1,
        total_num_scheduled_tokens=2,
        now=1.0,
    )
    assert state.realized_goodputs_by_k[2] == pytest.approx(12.5)
    assert state.realized_expected_tokens_by_k[2] == pytest.approx(1.5)
    assert state.realized_speedups_vs_k0_by_k[2] == pytest.approx(0.0)
    assert state.realized_speedups_vs_predicted_k0_by_k[2] == pytest.approx(0.0)
    assert state.realized_speedups_vs_realized_k0_by_k[2] == pytest.approx(0.0)
    assert state.realized_samples_by_k[2] == 2


def test_ngram_dsc_controller_blocks_upward_move_after_recent_bad_realized_goodput():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_ema_alpha=1.0,
            ngram_dsc_goodput_margin=0.0,
            ngram_dsc_goodput_increase_margin=0.0,
            ngram_dsc_upward_min_position_samples=0,
            ngram_dsc_realized_goodput_guard_min_samples=2,
            ngram_dsc_realized_goodput_guard_margin=0.0,
            ngram_dsc_max_step_delta=1,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )
    controller.current_effective_num_spec_tokens = 1

    for _ in range(2):
        controller.observe_realized_latency(
            decode_token_load=1,
            smoothed_total_num_scheduled_tokens=2.0,
            effective_num_spec_tokens=1,
            realized_latency_s=0.02,
            realized_generated_tokens=1.0,
        )
        controller.observe_realized_latency(
            decode_token_load=1,
            smoothed_total_num_scheduled_tokens=2.0,
            effective_num_spec_tokens=2,
            realized_latency_s=0.05,
            realized_generated_tokens=1.0,
        )

    state = controller.decide(
        decode_token_load=1,
        total_num_scheduled_tokens=2,
        now=1.0,
    )
    assert state.best_candidate_k >= 2
    assert state.effective_num_spec_tokens == 1


def test_ngram_dsc_controller_argmax_goodput_selects_k0_when_speculation_is_expensive():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.2,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_prior_rate=0.0,
            ngram_dsc_position_acceptance_prior_strength=0.0,
            ngram_dsc_position_acceptance_confidence_z=0.0,
            ngram_dsc_latency_model="profiled",
            ngram_dsc_profiled_latency_intercept_s=0.02,
            ngram_dsc_profiled_latency_spec_tokens_coeff_s=0.05,
            ngram_dsc_goodput_margin=0.0,
            ngram_dsc_goodput_increase_margin=0.0,
            ngram_dsc_max_step_delta=4,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )
    controller.current_effective_num_spec_tokens = 2

    state = controller.decide(
        decode_token_load=1,
        total_num_scheduled_tokens=2,
        now=1.0,
    )
    assert state.best_candidate_k == 0
    assert state.effective_num_spec_tokens == 0


def test_ngram_dsc_controller_recent_realized_samples_do_not_override_argmax():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_ema_alpha=1.0,
            ngram_dsc_goodput_margin=0.0,
            ngram_dsc_goodput_increase_margin=0.0,
            ngram_dsc_goodput_min_dwell_sec=10.0,
            ngram_dsc_realized_goodput_guard_min_samples=2,
            ngram_dsc_realized_goodput_guard_margin=0.05,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )
    controller.current_effective_num_spec_tokens = 1

    for _ in range(2):
        controller.observe_realized_latency(
            decode_token_load=1,
            smoothed_total_num_scheduled_tokens=2.0,
            effective_num_spec_tokens=1,
            realized_latency_s=0.04,
            realized_generated_tokens=1.0,
            predicted_k0_goodput=50.0,
            baseline_goodput=50.0,
        )

    state = controller.decide(
        decode_token_load=1,
        total_num_scheduled_tokens=2,
        now=1.0,
    )
    assert state.best_candidate_k >= 0
    assert state.realized_speedups_vs_k0_by_k[1] < 1.0


def test_ngram_dsc_controller_ignores_dwell_when_using_pure_argmax():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.2,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_prior_rate=0.0,
            ngram_dsc_position_acceptance_prior_strength=0.0,
            ngram_dsc_position_acceptance_confidence_z=0.0,
            ngram_dsc_latency_model="profiled",
            ngram_dsc_profiled_latency_intercept_s=0.02,
            ngram_dsc_profiled_latency_spec_tokens_coeff_s=0.05,
            ngram_dsc_goodput_margin=0.0,
            ngram_dsc_goodput_increase_margin=0.0,
            ngram_dsc_goodput_min_dwell_sec=10.0,
            ngram_dsc_max_step_delta=4,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )
    controller.current_effective_num_spec_tokens = 3
    controller.last_switch_time = 1.0

    state = controller.decide(
        decode_token_load=1,
        total_num_scheduled_tokens=2,
        now=2.0,
    )
    assert state.best_candidate_k == 0
    assert state.effective_num_spec_tokens == 0


def test_ngram_dsc_controller_uses_realized_k0_baseline_when_available():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_disable_decode_tokens=1,
            ngram_dsc_enable_decode_tokens=1,
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_k0_baseline_min_samples=2,
            ngram_dsc_realized_goodput_guard_min_samples=2,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    for _ in range(2):
        controller.observe_realized_latency(
            decode_token_load=1,
            smoothed_total_num_scheduled_tokens=2.0,
            effective_num_spec_tokens=0,
            realized_latency_s=0.02,
            realized_generated_tokens=1.0,
            predicted_k0_goodput=10.0,
            baseline_goodput=10.0,
        )

    state = controller.decide(
        decode_token_load=1,
        total_num_scheduled_tokens=2,
        now=1.0,
    )
    assert state.baseline_source == "realized_k0"
    assert state.baseline_goodput == pytest.approx(50.0)
    assert state.baseline_latency_s == pytest.approx(0.02)
    assert state.realized_latencies_by_k[0] == pytest.approx(0.02)


def test_ngram_dsc_controller_splits_predicted_and_realized_k0_speedups():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_disable_decode_tokens=1,
            ngram_dsc_enable_decode_tokens=1,
            ngram_dsc_k0_baseline_min_samples=1,
            ngram_dsc_realized_goodput_guard_min_samples=1,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    controller.observe_realized_latency(
        decode_token_load=1,
        smoothed_total_num_scheduled_tokens=2.0,
        effective_num_spec_tokens=2,
        realized_latency_s=0.05,
        realized_generated_tokens=1.0,
        predicted_k0_goodput=20.0,
    )
    controller.observe_realized_latency(
        decode_token_load=1,
        smoothed_total_num_scheduled_tokens=2.0,
        effective_num_spec_tokens=0,
        realized_latency_s=0.02,
        realized_generated_tokens=1.0,
        predicted_k0_goodput=20.0,
    )
    controller.observe_realized_latency(
        decode_token_load=1,
        smoothed_total_num_scheduled_tokens=2.0,
        effective_num_spec_tokens=2,
        realized_latency_s=0.05,
        realized_generated_tokens=1.0,
        predicted_k0_goodput=20.0,
    )

    state = controller.decide(
        decode_token_load=1,
        total_num_scheduled_tokens=2,
        now=1.0,
    )
    assert state.baseline_source == "realized_k0"
    assert state.realized_speedups_vs_predicted_k0_by_k[2] == pytest.approx(1.0)
    assert state.realized_speedups_vs_realized_k0_by_k[2] == pytest.approx(0.4)


def test_ngram_dsc_controller_low_load_can_choose_narrower_width_than_ngram():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_initial_acceptance_rate=0.95,
            ngram_dsc_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_ema_alpha=1.0,
            ngram_dsc_position_acceptance_prior_strength=0.0,
            ngram_dsc_position_acceptance_confidence_z=0.0,
            ngram_dsc_latency_model="profiled",
            ngram_dsc_profiled_latency_intercept_s=0.01,
            ngram_dsc_profiled_latency_spec_tokens_coeff_s=0.008,
            ngram_dsc_goodput_margin=0.0,
            ngram_dsc_goodput_increase_margin=0.0,
            ngram_dsc_goodput_min_dwell_sec=0.0,
            ngram_dsc_max_step_delta=4,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )
    controller.observe_draft(
        num_draft_tokens=30,
        num_accepted_tokens=16,
        position_draft_counts=(10, 10, 10, 0),
        position_accept_counts=(10, 5, 1, 0),
    )

    state = controller.decide(
        decode_token_load=1,
        total_num_scheduled_tokens=2,
        now=1.0,
    )
    assert state.load_regime == "adaptive"
    assert state.baseline_k == 0
    assert state.best_candidate_k == 0
    assert state.effective_num_spec_tokens == 0


def test_ngram_dsc_controller_online_latency_fit_freezes_underdetermined_features():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_latency_model="profiled",
            ngram_dsc_profiled_latency_intercept_s=0.02,
            ngram_dsc_profiled_latency_decode_token_load_coeff_s=0.01,
            ngram_dsc_profiled_latency_scheduled_tokens_coeff_s=0.03,
            ngram_dsc_profiled_latency_spec_tokens_coeff_s=0.04,
            ngram_dsc_profiled_latency_spec_scheduled_tokens_interaction_coeff_s=0.05,
            ngram_dsc_online_latency_fitting=True,
            ngram_dsc_online_latency_fit_min_samples=4,
            ngram_dsc_online_latency_fit_refit_interval_samples=1,
            ngram_dsc_online_latency_fit_min_feature_range=0.5,
            ngram_dsc_online_latency_fit_ema_alpha=1.0,
            ngram_dsc_online_latency_fit_max_relative_update=1.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    update = None
    for latency_s in (0.030, 0.031, 0.029, 0.030):
        update = controller.observe_realized_latency(
            decode_token_load=1,
            smoothed_total_num_scheduled_tokens=4.0,
            effective_num_spec_tokens=2,
            realized_latency_s=latency_s,
            realized_generated_tokens=1.0,
        )

    assert update is not None
    assert "scheduled_tokens" in update.frozen_coefficients
    assert "spec_tokens" in update.frozen_coefficients
    assert (
        controller.profiled_latency_scheduled_tokens_coeff_s == pytest.approx(0.03)
    )
    assert controller.profiled_latency_spec_tokens_coeff_s == pytest.approx(0.04)


def test_ngram_dsc_controller_online_latency_fit_freezes_structure_without_enough_spec_samples():
    controller = NgramDSCController(
        speculative_config=SpeculativeConfig(
            method="ngram_dsc",
            num_speculative_tokens=4,
            ngram_dsc=True,
            ngram_dsc_strategy="goodput",
            ngram_dsc_latency_model="profiled",
            ngram_dsc_profiled_latency_intercept_s=0.02,
            ngram_dsc_profiled_latency_decode_token_load_coeff_s=0.01,
            ngram_dsc_profiled_latency_scheduled_tokens_coeff_s=0.03,
            ngram_dsc_profiled_latency_spec_tokens_coeff_s=0.04,
            ngram_dsc_profiled_latency_spec_scheduled_tokens_interaction_coeff_s=0.05,
            ngram_dsc_online_latency_fitting=True,
            ngram_dsc_online_latency_fit_min_samples=4,
            ngram_dsc_online_latency_fit_refit_interval_samples=1,
            ngram_dsc_online_latency_fit_min_feature_range=0.5,
            ngram_dsc_online_latency_fit_min_nonzero_k_samples=3,
            ngram_dsc_online_latency_fit_ema_alpha=1.0,
            ngram_dsc_online_latency_fit_max_relative_update=1.0,
        ),
        max_num_running_reqs=16,
        max_num_scheduled_tokens=8192,
    )

    update = None
    samples = (
        (1, 8.0, 2, 0.050),
        (1, 4.0, 0, 0.020),
        (8, 8.0, 0, 0.026),
        (8, 12.0, 0, 0.028),
    )
    for sample in samples:
        update = controller.observe_realized_latency(
            decode_token_load=sample[0],
            smoothed_total_num_scheduled_tokens=sample[1],
            effective_num_spec_tokens=sample[2],
            realized_latency_s=sample[3],
            realized_generated_tokens=1.0,
        )

    assert update is not None
    assert "decode_load" in update.frozen_coefficients
    assert "scheduled_tokens" in update.frozen_coefficients
    assert "spec_tokens" in update.frozen_coefficients
    assert "spec_x_scheduled_tokens" in update.frozen_coefficients
    assert controller.profiled_latency_decode_token_load_coeff_s == pytest.approx(0.01)
    assert controller.profiled_latency_scheduled_tokens_coeff_s == pytest.approx(0.03)
    assert controller.profiled_latency_spec_tokens_coeff_s == pytest.approx(0.04)
