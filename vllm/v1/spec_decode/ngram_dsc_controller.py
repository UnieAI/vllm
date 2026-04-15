# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from vllm.config import SpeculativeConfig


@dataclass
class NgramDSCState:
    effective_num_spec_tokens: int
    spec_enabled: bool
    acceptance_rate_ema: float
    load_regime: str = "adaptive"
    baseline_k: int = 0
    latency_model: str = "heuristic"
    smoothed_total_num_scheduled_tokens: float = 0.0
    proposal_coverage_ema: float = 1.0
    current_goodput: float = 0.0
    selected_goodput: float = 0.0
    current_latency_s: float = 0.0
    selected_latency_s: float = 0.0
    selected_expected_tokens: float = 0.0
    baseline_goodput: float = 0.0
    baseline_latency_s: float = 0.0
    baseline_source: str = "predicted"
    best_candidate_k: int = 0
    candidate_goodputs: tuple[float, ...] = ()
    candidate_latencies_s: tuple[float, ...] = ()
    position_acceptance_rates: tuple[float, ...] = ()
    conservative_position_acceptance_rates: tuple[float, ...] = ()
    realized_goodputs_by_k: tuple[float, ...] = ()
    realized_expected_tokens_by_k: tuple[float, ...] = ()
    realized_speedups_vs_k0_by_k: tuple[float, ...] = ()
    realized_speedups_vs_predicted_k0_by_k: tuple[float, ...] = ()
    realized_speedups_vs_realized_k0_by_k: tuple[float, ...] = ()
    realized_latencies_by_k: tuple[float, ...] = ()
    realized_samples_by_k: tuple[int, ...] = ()


class _LatencySample(NamedTuple):
    decode_token_load: int
    smoothed_total_num_scheduled_tokens: float
    effective_num_spec_tokens: int
    realized_latency_s: float


@dataclass
class NgramDSCLatencyFitUpdate:
    num_total_samples: int
    num_valid_samples: int
    num_dropped_samples: int
    intercept_s: float
    decode_token_load_coeff_s: float
    scheduled_tokens_coeff_s: float
    spec_tokens_coeff_s: float
    spec_scheduled_tokens_interaction_coeff_s: float
    active_coefficients: tuple[str, ...]
    frozen_coefficients: tuple[str, ...]


class NgramDSCController:
    """Controls n-gram speculation for scheduler-side DSC policies."""

    def __init__(
        self,
        speculative_config: SpeculativeConfig,
        max_num_running_reqs: int,
        max_num_scheduled_tokens: int,
    ) -> None:
        self.strategy = speculative_config.ngram_dsc_strategy
        self.max_num_spec_tokens = speculative_config.num_speculative_tokens
        self.initial_acceptance_rate = (
            speculative_config.ngram_dsc_initial_acceptance_rate
        )
        self.acceptance_rate_ema = self.initial_acceptance_rate
        self.acceptance_ema_alpha = speculative_config.ngram_dsc_acceptance_ema_alpha
        self.position_acceptance_ema_alpha = (
            speculative_config.ngram_dsc_position_acceptance_ema_alpha
            or self.acceptance_ema_alpha
        )
        self.position_acceptance_prior_rate = (
            speculative_config.ngram_dsc_position_acceptance_prior_rate
            if speculative_config.ngram_dsc_position_acceptance_prior_rate is not None
            else self.acceptance_rate_ema
        )
        self.position_acceptance_prior_decay = (
            speculative_config.ngram_dsc_position_acceptance_prior_decay
        )
        self.position_acceptance_prior_strength = (
            speculative_config.ngram_dsc_position_acceptance_prior_strength
        )
        self.position_acceptance_confidence_z = (
            speculative_config.ngram_dsc_position_acceptance_confidence_z
        )
        self.base_latency_tokens = speculative_config.ngram_dsc_base_latency_tokens
        self.latency_model = speculative_config.ngram_dsc_latency_model
        self.profiled_latency_intercept_s = (
            speculative_config.ngram_dsc_profiled_latency_intercept_s
        )
        self.profiled_latency_decode_token_load_coeff_s = (
            speculative_config.ngram_dsc_profiled_latency_decode_token_load_coeff_s
        )
        self.profiled_latency_scheduled_tokens_coeff_s = (
            speculative_config.ngram_dsc_profiled_latency_scheduled_tokens_coeff_s
        )
        self.profiled_latency_spec_tokens_coeff_s = (
            speculative_config.ngram_dsc_profiled_latency_spec_tokens_coeff_s
        )
        self.profiled_latency_spec_scheduled_tokens_interaction_coeff_s = (
            speculative_config.ngram_dsc_profiled_latency_spec_scheduled_tokens_interaction_coeff_s
        )
        self.online_latency_fitting = (
            speculative_config.ngram_dsc_online_latency_fitting
        )
        self.online_latency_fit_min_samples = (
            speculative_config.ngram_dsc_online_latency_fit_min_samples
        )
        self.online_latency_fit_warmup_samples = (
            speculative_config.ngram_dsc_online_latency_fit_warmup_samples
        )
        self.online_latency_fit_refit_interval_samples = (
            speculative_config.ngram_dsc_online_latency_fit_refit_interval_samples
        )
        self.online_latency_fit_max_samples = (
            speculative_config.ngram_dsc_online_latency_fit_max_samples
        )
        self.online_latency_fit_max_latency_ratio_to_median = (
            speculative_config.ngram_dsc_online_latency_fit_max_latency_ratio_to_median
        )
        self.online_latency_fit_ema_alpha = (
            speculative_config.ngram_dsc_online_latency_fit_ema_alpha
        )
        self.online_latency_fit_max_relative_update = (
            speculative_config.ngram_dsc_online_latency_fit_max_relative_update
        )
        self.online_latency_fit_min_feature_range = (
            speculative_config.ngram_dsc_online_latency_fit_min_feature_range
        )
        self.online_latency_fit_min_nonzero_k_samples = (
            speculative_config.ngram_dsc_online_latency_fit_min_nonzero_k_samples
        )
        self.online_latency_fit_max_realized_latency_s = (
            speculative_config.ngram_dsc_online_latency_fit_max_realized_latency_s
        )
        self.realized_sample_min_decode_token_load = (
            speculative_config.ngram_dsc_realized_sample_min_decode_token_load
        )
        self.realized_sample_min_smoothed_scheduled_tokens = (
            speculative_config.ngram_dsc_realized_sample_min_smoothed_scheduled_tokens
        )
        self.realized_sample_min_latency_s = (
            speculative_config.ngram_dsc_realized_sample_min_latency_s
        )
        self.normal_decode_realized_log_interval = (
            speculative_config.ngram_dsc_normal_decode_realized_log_interval
        )
        self.near_best_goodput_ratio = (
            speculative_config.ngram_dsc_near_best_goodput_ratio
        )
        self.switch_hysteresis_ratio = (
            speculative_config.ngram_dsc_switch_hysteresis_ratio
        )
        self.goodput_margin = speculative_config.ngram_dsc_goodput_margin
        self.goodput_increase_margin = (
            speculative_config.ngram_dsc_goodput_increase_margin
        )
        self.realized_goodput_ema_alpha = (
            speculative_config.ngram_dsc_realized_goodput_ema_alpha
        )
        self.k0_baseline_min_samples = (
            speculative_config.ngram_dsc_k0_baseline_min_samples
        )
        self.initial_max_k = min(
            self.max_num_spec_tokens,
            speculative_config.ngram_dsc_initial_max_k,
        )
        self.min_spec_realized_samples_before_k0 = (
            speculative_config.ngram_dsc_min_spec_realized_samples_before_k0
        )
        self.k0_sparse_evidence_margin = (
            speculative_config.ngram_dsc_k0_sparse_evidence_margin
        )
        self.fast_fail_min_steps = speculative_config.ngram_dsc_fast_fail_min_steps
        self.fast_fail_max_steps = speculative_config.ngram_dsc_fast_fail_max_steps
        self.fast_fail_max_acceptance_rate = (
            speculative_config.ngram_dsc_fast_fail_max_acceptance_rate
        )
        self.realized_goodput_guard_min_samples = (
            speculative_config.ngram_dsc_realized_goodput_guard_min_samples
        )
        self.realized_goodput_guard_margin = (
            speculative_config.ngram_dsc_realized_goodput_guard_margin
        )
        self.upward_min_position_samples = (
            speculative_config.ngram_dsc_upward_min_position_samples
        )
        self.goodput_min_dwell_sec = (
            speculative_config.ngram_dsc_goodput_min_dwell_sec
        )
        self.max_step_delta = speculative_config.ngram_dsc_max_step_delta
        self.scheduled_tokens_ema_alpha = (
            speculative_config.ngram_dsc_scheduled_tokens_ema_alpha
        )

        decode_capacity = max(
            1, min(max_num_running_reqs, max_num_scheduled_tokens)
        )
        auto_disable = max(1, min(128, decode_capacity // 2))
        disable_decode_tokens = (
            speculative_config.ngram_dsc_disable_decode_tokens or auto_disable
        )
        enable_decode_tokens = speculative_config.ngram_dsc_enable_decode_tokens
        if enable_decode_tokens is None:
            enable_decode_tokens = max(1, int(disable_decode_tokens * 0.8))

        self.disable_decode_tokens = disable_decode_tokens
        self.enable_decode_tokens = min(enable_decode_tokens, disable_decode_tokens)
        self.switch_cooldown_sec = speculative_config.ngram_dsc_switch_cooldown_sec
        self.load_regime = "adaptive"
        self.current_effective_num_spec_tokens = (
            self.initial_max_k
            if self.strategy == "goodput"
            else self.max_num_spec_tokens
        )
        self.spec_enabled = self.current_effective_num_spec_tokens > 0
        self.last_switch_time = 0.0
        self.smoothed_total_num_scheduled_tokens = float(
            max(1, self.current_effective_num_spec_tokens)
        )
        self.position_acceptance_rate_ema = np.full(
            self.max_num_spec_tokens,
            self.acceptance_rate_ema,
            dtype=np.float64,
        )
        self.position_acceptance_successes = np.zeros(
            self.max_num_spec_tokens,
            dtype=np.float64,
        )
        self.position_acceptance_opportunities = np.zeros(
            self.max_num_spec_tokens,
            dtype=np.float64,
        )
        self.acceptance_successes = 0.0
        self.acceptance_opportunities = 0.0
        self.proposal_coverage_ema = 1.0
        self.proposal_coverage_observations = 0
        self.realized_k_step_counts = np.zeros(
            self.max_num_spec_tokens + 1,
            dtype=np.int64,
        )
        self.realized_k_total_generated_tokens = np.zeros(
            self.max_num_spec_tokens + 1,
            dtype=np.float64,
        )
        self.realized_k_total_latency_s = np.zeros(
            self.max_num_spec_tokens + 1,
            dtype=np.float64,
        )
        self.realized_k_goodput_ema = np.zeros(
            self.max_num_spec_tokens + 1,
            dtype=np.float64,
        )
        self.realized_k_expected_tokens_ema = np.ones(
            self.max_num_spec_tokens + 1,
            dtype=np.float64,
        )
        self.realized_k_speedup_vs_predicted_k0_ema = np.zeros(
            self.max_num_spec_tokens + 1,
            dtype=np.float64,
        )
        self.realized_k_speedup_vs_realized_k0_ema = np.zeros(
            self.max_num_spec_tokens + 1,
            dtype=np.float64,
        )
        self.realized_k_speedup_vs_predicted_k0_counts = np.zeros(
            self.max_num_spec_tokens + 1,
            dtype=np.int64,
        )
        self.realized_k_speedup_vs_realized_k0_counts = np.zeros(
            self.max_num_spec_tokens + 1,
            dtype=np.int64,
        )
        self.realized_k_latency_s_ema = np.zeros(
            self.max_num_spec_tokens + 1,
            dtype=np.float64,
        )
        self.fast_fail_lock_active = False
        self.cold_start_spec_steps = 0
        self.cold_start_total_draft_tokens = 0
        self.cold_start_total_accepted_tokens = 0
        self._latency_samples: list[_LatencySample] = []
        self._latency_observation_count = 0
        self._last_latency_refit_valid_sample_count = 0
        self._normal_decode_realized_sample_count = 0
        self._spec_decode_realized_sample_count = 0
        self.profiled_latency_scale = 1.0

    def decide(
        self, decode_token_load: int, total_num_scheduled_tokens: int, now: float
    ) -> NgramDSCState:
        if decode_token_load <= 0:
            self.fast_fail_lock_active = False
            self._reset_cold_start_state()
        elif self.fast_fail_lock_active and decode_token_load > 1:
            self.fast_fail_lock_active = False
            self._reset_cold_start_state()
        smoothed_total_num_scheduled_tokens = 0.0
        current_goodput = 0.0
        selected_goodput = 0.0
        current_latency_s = 0.0
        selected_latency_s = 0.0
        selected_expected_tokens = 0.0
        proposal_coverage_ema = self.proposal_coverage_ema
        baseline_goodput = 0.0
        baseline_latency_s = 0.0
        baseline_source = "predicted"
        load_regime = self.load_regime
        baseline_k = 0
        best_candidate_k = 0
        candidate_goodputs: tuple[float, ...] = ()
        candidate_latencies_s: tuple[float, ...] = ()
        conservative_position_acceptance_rates: tuple[float, ...] = ()
        if self.strategy == "threshold":
            effective_num_spec_tokens = self._decide_threshold(
                decode_token_load=decode_token_load, now=now
            )
        else:
            (
                effective_num_spec_tokens,
                smoothed_total_num_scheduled_tokens,
                proposal_coverage_ema,
                current_goodput,
                selected_goodput,
                current_latency_s,
                selected_latency_s,
                selected_expected_tokens,
                baseline_goodput,
                baseline_latency_s,
                baseline_source,
                load_regime,
                baseline_k,
                best_candidate_k,
                candidate_goodputs,
                candidate_latencies_s,
                conservative_position_acceptance_rates,
            ) = self._decide_goodput(
                decode_token_load=decode_token_load,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                now=now,
            )
            self.spec_enabled = effective_num_spec_tokens > 0
        previous_effective_num_spec_tokens = self.current_effective_num_spec_tokens
        if effective_num_spec_tokens != previous_effective_num_spec_tokens:
            self.last_switch_time = now
            if (
                effective_num_spec_tokens <= 0
                or previous_effective_num_spec_tokens <= 0
            ):
                self._reset_cold_start_state()
        self.current_effective_num_spec_tokens = effective_num_spec_tokens

        return NgramDSCState(
            effective_num_spec_tokens=effective_num_spec_tokens,
            spec_enabled=self.spec_enabled,
            acceptance_rate_ema=self.acceptance_rate_ema,
            load_regime=load_regime,
            baseline_k=baseline_k,
            latency_model=self.latency_model,
            smoothed_total_num_scheduled_tokens=smoothed_total_num_scheduled_tokens,
            proposal_coverage_ema=proposal_coverage_ema,
            current_goodput=current_goodput,
            selected_goodput=selected_goodput,
            current_latency_s=current_latency_s,
            selected_latency_s=selected_latency_s,
            selected_expected_tokens=selected_expected_tokens,
            baseline_goodput=baseline_goodput,
            baseline_latency_s=baseline_latency_s,
            baseline_source=baseline_source,
            best_candidate_k=best_candidate_k,
            candidate_goodputs=candidate_goodputs,
            candidate_latencies_s=candidate_latencies_s,
            position_acceptance_rates=tuple(
                float(value) for value in self._get_position_acceptance_rates()
            ),
            conservative_position_acceptance_rates=(
                conservative_position_acceptance_rates
            ),
            realized_goodputs_by_k=self._get_realized_goodputs_by_k(),
            realized_expected_tokens_by_k=tuple(
                float(value) for value in self.realized_k_expected_tokens_ema
            ),
            realized_speedups_vs_k0_by_k=self._get_selection_speedups_by_k(),
            realized_speedups_vs_predicted_k0_by_k=tuple(
                float(value)
                for value in self.realized_k_speedup_vs_predicted_k0_ema
            ),
            realized_speedups_vs_realized_k0_by_k=tuple(
                float(value)
                for value in self.realized_k_speedup_vs_realized_k0_ema
            ),
            realized_latencies_by_k=tuple(
                float(value) for value in self.realized_k_latency_s_ema
            ),
            realized_samples_by_k=tuple(
                int(value) for value in self.realized_k_step_counts
            ),
        )

    def observe_draft(
        self,
        *,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        speculation_attempted: bool = False,
        num_eligible_decode_reqs: int | None = None,
        num_draft_reqs: int | None = None,
        position_draft_counts: tuple[int, ...] | None = None,
        position_accept_counts: tuple[int, ...] | None = None,
    ) -> None:
        if self.strategy != "goodput":
            return

        if (speculation_attempted and num_eligible_decode_reqs is not None
                and num_eligible_decode_reqs > 0):
            drafted_reqs = max(
                0,
                min(
                    num_eligible_decode_reqs,
                    num_draft_reqs if num_draft_reqs is not None else 0,
                ),
            )
            proposal_coverage = drafted_reqs / num_eligible_decode_reqs
            self.proposal_coverage_ema = (
                (1.0 - self.acceptance_ema_alpha) * self.proposal_coverage_ema
                + self.acceptance_ema_alpha * proposal_coverage
            )
            self.proposal_coverage_observations += 1

        if num_draft_tokens <= 0:
            return

        acceptance_rate = min(1.0, max(0.0, num_accepted_tokens / num_draft_tokens))
        acceptance_decay = 1.0 - self.acceptance_ema_alpha
        self.acceptance_successes *= acceptance_decay
        self.acceptance_opportunities *= acceptance_decay
        self.acceptance_successes += max(
            0.0, min(float(num_draft_tokens), float(num_accepted_tokens))
        )
        self.acceptance_opportunities += float(num_draft_tokens)
        self.acceptance_rate_ema = (
            (1.0 - self.acceptance_ema_alpha) * self.acceptance_rate_ema
            + self.acceptance_ema_alpha * acceptance_rate
        )
        self.cold_start_spec_steps += 1
        self.cold_start_total_draft_tokens += num_draft_tokens
        self.cold_start_total_accepted_tokens += num_accepted_tokens
        if (
            position_draft_counts is None
            or position_accept_counts is None
            or len(position_draft_counts) != self.max_num_spec_tokens
            or len(position_accept_counts) != self.max_num_spec_tokens
        ):
            return

        decay = 1.0 - self.position_acceptance_ema_alpha
        self.position_acceptance_successes *= decay
        self.position_acceptance_opportunities *= decay

        for index, (draft_count, accept_count) in enumerate(
            zip(position_draft_counts, position_accept_counts)
        ):
            if draft_count <= 0:
                continue
            position_acceptance_rate = min(
                1.0, max(0.0, accept_count / draft_count)
            )
            self.position_acceptance_rate_ema[index] = (
                (1.0 - self.position_acceptance_ema_alpha)
                * self.position_acceptance_rate_ema[index]
                + self.position_acceptance_ema_alpha * position_acceptance_rate
            )
            self.position_acceptance_successes[index] += max(
                0, min(draft_count, accept_count)
            )
            self.position_acceptance_opportunities[index] += draft_count

    def observe_realized_latency(
        self,
        *,
        decode_token_load: int,
        smoothed_total_num_scheduled_tokens: float,
        effective_num_spec_tokens: int,
        realized_latency_s: float,
        realized_generated_tokens: float,
        predicted_k0_goodput: float | None = None,
        baseline_goodput: float | None = None,
    ) -> NgramDSCLatencyFitUpdate | None:
        if self.strategy != "goodput":
            return None

        effective_num_spec_tokens = max(
            0, min(self.max_num_spec_tokens, effective_num_spec_tokens)
        )
        if not self.should_observe_realized_sample(
            decode_token_load=decode_token_load,
            smoothed_total_num_scheduled_tokens=smoothed_total_num_scheduled_tokens,
            effective_num_spec_tokens=effective_num_spec_tokens,
            realized_latency_s=realized_latency_s,
        ):
            return None
        if self.latency_model == "profiled" and self.online_latency_fitting:
            self._update_profiled_latency_scale(
                decode_token_load=decode_token_load,
                smoothed_total_num_scheduled_tokens=smoothed_total_num_scheduled_tokens,
                effective_num_spec_tokens=effective_num_spec_tokens,
                realized_latency_s=realized_latency_s,
            )
        realized_goodput = max(0.0, realized_generated_tokens) / max(
            1e-6, realized_latency_s
        )
        self.realized_k_step_counts[effective_num_spec_tokens] += 1
        self._update_realized_k_ema(
            self.realized_k_expected_tokens_ema,
            effective_num_spec_tokens,
            max(0.0, realized_generated_tokens),
        )
        self._update_realized_k_ema(
            self.realized_k_goodput_ema,
            effective_num_spec_tokens,
            realized_goodput,
        )
        self._update_realized_k_ema(
            self.realized_k_latency_s_ema,
            effective_num_spec_tokens,
            max(1e-6, realized_latency_s),
        )
        if predicted_k0_goodput is not None and predicted_k0_goodput > 0.0:
            self._update_realized_k_ema(
                self.realized_k_speedup_vs_predicted_k0_ema,
                effective_num_spec_tokens,
                realized_goodput / predicted_k0_goodput,
            )
            self.realized_k_speedup_vs_predicted_k0_counts[
                effective_num_spec_tokens
            ] += 1
        if self._has_ready_realized_k0_baseline():
            realized_k0_goodput = float(self.realized_k_goodput_ema[0])
            if realized_k0_goodput > 0.0:
                self._update_realized_k_ema(
                    self.realized_k_speedup_vs_realized_k0_ema,
                    effective_num_spec_tokens,
                    realized_goodput / realized_k0_goodput,
                )
                self.realized_k_speedup_vs_realized_k0_counts[
                    effective_num_spec_tokens
                ] += 1
        self.realized_k_total_generated_tokens[effective_num_spec_tokens] += max(
            0.0, realized_generated_tokens
        )
        self.realized_k_total_latency_s[effective_num_spec_tokens] += max(
            1e-6, realized_latency_s
        )
        if self.latency_model != "profiled" or not self.online_latency_fitting:
            return None

        self._latency_observation_count += 1
        if self._latency_observation_count <= self.online_latency_fit_warmup_samples:
            return None

        self._latency_samples.append(
            _LatencySample(
                decode_token_load=max(1, decode_token_load),
                smoothed_total_num_scheduled_tokens=max(
                    1.0, smoothed_total_num_scheduled_tokens
                ),
                effective_num_spec_tokens=max(0, effective_num_spec_tokens),
                realized_latency_s=max(1e-6, realized_latency_s),
            )
        )
        if len(self._latency_samples) > self.online_latency_fit_max_samples:
            self._latency_samples = self._latency_samples[
                -self.online_latency_fit_max_samples :
            ]

        valid_samples = self._filtered_latency_samples(self._latency_samples)
        num_valid_samples = len(valid_samples)
        if num_valid_samples < self.online_latency_fit_min_samples:
            return None
        if (
            num_valid_samples - self._last_latency_refit_valid_sample_count
            < self.online_latency_fit_refit_interval_samples
        ):
            return None

        x, y = self._design_matrix(valid_samples)
        coefficients, active_mask = self._fit_latency_coefficients(
            x, y, valid_samples
        )
        self.profiled_latency_intercept_s = float(coefficients[0])
        self.profiled_latency_decode_token_load_coeff_s = float(coefficients[1])
        self.profiled_latency_scheduled_tokens_coeff_s = float(coefficients[2])
        self.profiled_latency_spec_tokens_coeff_s = float(coefficients[3])
        self.profiled_latency_spec_scheduled_tokens_interaction_coeff_s = float(
            coefficients[4]
        )
        self._last_latency_refit_valid_sample_count = num_valid_samples
        coefficient_names = (
            "intercept",
            "decode_load",
            "scheduled_tokens",
            "spec_tokens",
            "spec_x_scheduled_tokens",
        )
        return NgramDSCLatencyFitUpdate(
            num_total_samples=len(self._latency_samples),
            num_valid_samples=num_valid_samples,
            num_dropped_samples=len(self._latency_samples) - num_valid_samples,
            intercept_s=self.profiled_latency_intercept_s,
            decode_token_load_coeff_s=(
                self.profiled_latency_decode_token_load_coeff_s
            ),
            scheduled_tokens_coeff_s=(
                self.profiled_latency_scheduled_tokens_coeff_s
            ),
            spec_tokens_coeff_s=self.profiled_latency_spec_tokens_coeff_s,
            spec_scheduled_tokens_interaction_coeff_s=(
                self.profiled_latency_spec_scheduled_tokens_interaction_coeff_s
            ),
            active_coefficients=tuple(
                name
                for name, is_active in zip(coefficient_names, active_mask)
                if is_active
            ),
            frozen_coefficients=tuple(
                name
                for name, is_active in zip(coefficient_names, active_mask)
                if not is_active
            ),
        )

    def should_observe_realized_sample(
        self,
        *,
        decode_token_load: int,
        smoothed_total_num_scheduled_tokens: float,
        effective_num_spec_tokens: int,
        realized_latency_s: float,
    ) -> bool:
        if effective_num_spec_tokens > 0:
            return True
        if decode_token_load < self.realized_sample_min_decode_token_load:
            return False
        if (
            smoothed_total_num_scheduled_tokens
            < self.realized_sample_min_smoothed_scheduled_tokens
        ):
            return False
        if realized_latency_s < self.realized_sample_min_latency_s:
            return False
        return True

    def should_log_realized_sample(
        self,
        *,
        effective_num_spec_tokens: int,
        sample_observed: bool,
    ) -> bool:
        if effective_num_spec_tokens > 0:
            if not sample_observed:
                return False
            self._spec_decode_realized_sample_count += 1
            return self._spec_decode_realized_sample_count == 1
        if not sample_observed:
            return False
        self._normal_decode_realized_sample_count += 1
        return self._normal_decode_realized_sample_count == 1

    def _decide_threshold(self, *, decode_token_load: int, now: float) -> int:
        if self.spec_enabled:
            if decode_token_load >= self.disable_decode_tokens:
                self.spec_enabled = False
                self.last_switch_time = now
            return self.max_num_spec_tokens if self.spec_enabled else 0

        if now - self.last_switch_time < self.switch_cooldown_sec:
            return 0

        if decode_token_load <= self.enable_decode_tokens:
            self.spec_enabled = True
            self.last_switch_time = now

        return self.max_num_spec_tokens if self.spec_enabled else 0

    def _decide_goodput(
        self, *, decode_token_load: int, total_num_scheduled_tokens: int, now: float
    ) -> tuple[
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        str,
        str,
        int,
        int,
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
    ]:
        smoothed_total_num_scheduled_tokens = (
            self._update_smoothed_total_num_scheduled_tokens(
                total_num_scheduled_tokens
            )
        )
        current_k = self.current_effective_num_spec_tokens
        conservative_position_acceptance_rates = tuple(
            float(value)
            for value in self._get_conservative_position_acceptance_rates()
        )
        if total_num_scheduled_tokens <= 0:
            bootstrap_k = self._get_bootstrap_verification_width()
            return (
                bootstrap_k,
                smoothed_total_num_scheduled_tokens,
                self.proposal_coverage_ema,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                "bootstrap_empty_decode_batch",
                "adaptive",
                bootstrap_k,
                bootstrap_k,
                (),
                (),
                conservative_position_acceptance_rates,
            )
        candidate_metrics = self._estimate_candidate_metrics(
            decode_token_load=decode_token_load,
            total_num_scheduled_tokens=smoothed_total_num_scheduled_tokens,
        )
        candidate_goodputs = tuple(metric[0] for metric in candidate_metrics)
        candidate_latencies_s = tuple(metric[1] for metric in candidate_metrics)
        load_regime = "adaptive"
        current_goodput = candidate_goodputs[current_k]
        current_latency_s = candidate_metrics[current_k][1]
        best_candidate_k = self._select_best_candidate(candidate_metrics)
        if best_candidate_k == 0:
            sparse_probe_k = self._select_sparse_evidence_probe_candidate(
                candidate_metrics
            )
            if sparse_probe_k > 0:
                best_candidate_k = sparse_probe_k
        best_candidate_k = self._apply_warmup_downshift_guard(
            current_k=current_k,
            best_candidate_k=best_candidate_k,
            candidate_metrics=candidate_metrics,
        )
        selected_k = self._apply_switch_hysteresis(
            current_k=current_k,
            best_candidate_k=best_candidate_k,
            candidate_metrics=candidate_metrics,
        )
        baseline_k = best_candidate_k
        baseline_goodput = candidate_goodputs[best_candidate_k]
        baseline_latency_s = candidate_metrics[best_candidate_k][1]
        baseline_source = "argmax_goodput"
        return (
            selected_k,
            smoothed_total_num_scheduled_tokens,
            self.proposal_coverage_ema,
            current_goodput,
            candidate_goodputs[selected_k],
            current_latency_s,
            candidate_metrics[selected_k][1],
            candidate_metrics[selected_k][2],
            baseline_goodput,
            baseline_latency_s,
            baseline_source,
            load_regime,
            baseline_k,
            best_candidate_k,
            candidate_goodputs,
            candidate_latencies_s,
            conservative_position_acceptance_rates,
        )

    def _select_best_candidate(
        self,
        candidate_metrics: tuple[tuple[float, float, float], ...],
    ) -> int:
        if not candidate_metrics:
            return 0
        best_goodput = max(metric[0] for metric in candidate_metrics)
        goodput_tolerance = max(
            1e-9,
            abs(best_goodput) * max(0.0, self.near_best_goodput_ratio),
        )
        near_best_candidates = [
            k
            for k, (goodput, _, _) in enumerate(candidate_metrics)
            if best_goodput - goodput <= goodput_tolerance
        ]
        return min(
            near_best_candidates,
            key=lambda k: (
                -candidate_metrics[k][0],
                candidate_metrics[k][1],
                -candidate_metrics[k][2],
                k,
            ),
        )

    def _apply_switch_hysteresis(
        self,
        *,
        current_k: int,
        best_candidate_k: int,
        candidate_metrics: tuple[tuple[float, float, float], ...],
    ) -> int:
        if not candidate_metrics or best_candidate_k == current_k:
            return best_candidate_k
        if current_k < 0 or current_k >= len(candidate_metrics):
            return best_candidate_k

        current_goodput = candidate_metrics[current_k][0]
        best_goodput = candidate_metrics[best_candidate_k][0]
        required_goodput = current_goodput * (
            1.0 + max(0.0, self.switch_hysteresis_ratio)
        )
        if best_goodput <= required_goodput:
            return current_k
        return best_candidate_k

    def _apply_warmup_downshift_guard(
        self,
        *,
        current_k: int,
        best_candidate_k: int,
        candidate_metrics: tuple[tuple[float, float, float], ...],
    ) -> int:
        if (
            not candidate_metrics
            or current_k <= 0
            or best_candidate_k >= current_k
            or current_k >= len(candidate_metrics)
            or best_candidate_k < 0
            or best_candidate_k >= len(candidate_metrics)
        ):
            return best_candidate_k
        if self.cold_start_spec_steps >= self.fast_fail_max_steps:
            return best_candidate_k
        warmup_draft_budget = self.max_num_spec_tokens * max(
            1, self.fast_fail_max_steps
        )
        if self.cold_start_total_draft_tokens >= warmup_draft_budget:
            return best_candidate_k

        current_goodput = candidate_metrics[current_k][0]
        challenger_goodput = candidate_metrics[best_candidate_k][0]
        required_goodput = current_goodput * (
            1.0 + max(self.goodput_margin, self.switch_hysteresis_ratio)
        )
        if challenger_goodput < required_goodput:
            return current_k
        return best_candidate_k

    def _get_bootstrap_verification_width(self) -> int:
        if self.max_num_spec_tokens <= 0:
            return 0
        return self.max_num_spec_tokens

    def _select_sparse_evidence_probe_candidate(
        self,
        candidate_metrics: tuple[tuple[float, float, float], ...],
    ) -> int:
        if len(candidate_metrics) <= 1:
            return 0
        realized_spec_samples = int(np.sum(self.realized_k_step_counts[1:]))
        if realized_spec_samples >= self.min_spec_realized_samples_before_k0:
            return 0

        positive_candidates = range(1, len(candidate_metrics))
        best_positive_k = max(
            positive_candidates,
            key=lambda k: (
                candidate_metrics[k][0],
                -candidate_metrics[k][1],
                candidate_metrics[k][2],
                -k,
            ),
        )
        best_positive_goodput = candidate_metrics[best_positive_k][0]
        k0_goodput = candidate_metrics[0][0]
        required_goodput = best_positive_goodput * (
            1.0 + self.k0_sparse_evidence_margin
        )
        if k0_goodput < required_goodput:
            return best_positive_k
        return 0

    def _estimate_candidate_metrics(
        self,
        *,
        decode_token_load: int,
        total_num_scheduled_tokens: float,
    ) -> tuple[tuple[float, float, float], ...]:
        return tuple(
            self._estimate_goodput(
                k,
                decode_token_load=decode_token_load,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
            )
            for k in range(self.max_num_spec_tokens + 1)
        )

    def _update_smoothed_total_num_scheduled_tokens(
        self, total_num_scheduled_tokens: int
    ) -> float:
        observed_total_num_scheduled_tokens = float(
            max(0, total_num_scheduled_tokens)
        )
        self.smoothed_total_num_scheduled_tokens = (
            (1.0 - self.scheduled_tokens_ema_alpha)
            * self.smoothed_total_num_scheduled_tokens
            + self.scheduled_tokens_ema_alpha
            * observed_total_num_scheduled_tokens
        )
        return max(0.0, self.smoothed_total_num_scheduled_tokens)

    def _estimate_goodput(
        self,
        k: int,
        *,
        decode_token_load: int,
        total_num_scheduled_tokens: float,
    ) -> tuple[float, float, float]:
        expected_tokens = self._expected_generated_tokens(k)
        estimated_latency = self._estimate_latency(
            k,
            decode_token_load=decode_token_load,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
        )
        return expected_tokens / estimated_latency, estimated_latency, expected_tokens

    def _estimate_latency(
        self,
        k: int,
        *,
        decode_token_load: int,
        total_num_scheduled_tokens: float,
    ) -> float:
        if self.latency_model == "profiled":
            estimated_latency = (
                self.profiled_latency_intercept_s
                + self.profiled_latency_decode_token_load_coeff_s
                * max(1, decode_token_load)
                + self.profiled_latency_scheduled_tokens_coeff_s
                * max(1.0, total_num_scheduled_tokens)
                + self.profiled_latency_spec_tokens_coeff_s * k
                + self.profiled_latency_spec_scheduled_tokens_interaction_coeff_s
                * k
                * max(1.0, total_num_scheduled_tokens)
            )
            return max(1e-6, estimated_latency * self.profiled_latency_scale)

        base_latency = self.base_latency_tokens + max(1.0, total_num_scheduled_tokens)
        per_token_latency = max(1, decode_token_load)
        return max(1e-6, base_latency + per_token_latency * k)

    def _expected_generated_tokens(self, k: int) -> float:
        if k <= 0:
            return 1.0
        acceptance_rate = self._get_acceptance_rate_estimate()
        if acceptance_rate >= 1.0:
            expected_tokens = 1.0 + float(k)
        else:
            expected_tokens = (1.0 - acceptance_rate**(k + 1)) / max(
                1e-9, 1.0 - acceptance_rate
            )
        proposal_coverage = self._get_proposal_coverage()
        return 1.0 + proposal_coverage * (expected_tokens - 1.0)

    def _get_conservative_position_acceptance_rates(self) -> np.ndarray:
        if self.max_num_spec_tokens <= 0:
            return np.zeros(0, dtype=np.float64)

        prior_strength = max(0.0, self.position_acceptance_prior_strength)
        prior_rate = min(1.0, max(0.0, self.position_acceptance_prior_rate))
        conservative_rates = np.empty(
            self.max_num_spec_tokens,
            dtype=np.float64,
        )
        for index in range(self.max_num_spec_tokens):
            opportunities = float(max(0, self.position_acceptance_opportunities[index]))
            prior_rate = self._get_position_acceptance_prior_rate(index)
            if opportunities > 0.0:
                raw_rate = float(
                    min(
                        1.0,
                        max(
                            0.0,
                            self.position_acceptance_successes[index]
                            / opportunities,
                        ),
                    ))
            else:
                raw_rate = float(
                    min(
                        self._get_position_acceptance_bootstrap_rate(index),
                        min(1.0, max(0.0, self.position_acceptance_rate_ema[index])),
                    )
                )
            posterior_total = opportunities + prior_strength
            if posterior_total <= 0.0:
                conservative_rates[index] = prior_rate
                continue
            successes = raw_rate * opportunities
            posterior_mean = (
                successes + prior_strength * prior_rate
            ) / posterior_total
            posterior_var = (
                posterior_mean
                * (1.0 - posterior_mean)
                / max(1.0, posterior_total + 1.0)
            )
            conservative_rate = posterior_mean - (
                self.position_acceptance_confidence_z * float(np.sqrt(posterior_var))
            )
            capped_rate = min(1.0, max(0.0, conservative_rate))
            if opportunities >= max(1.0, prior_strength):
                capped_rate = min(raw_rate, capped_rate)
            conservative_rates[index] = capped_rate
        for index in range(1, self.max_num_spec_tokens):
            conservative_rates[index] = min(
                conservative_rates[index],
                conservative_rates[index - 1],
            )
        return conservative_rates

    def _get_position_acceptance_rates(self) -> np.ndarray:
        if self.max_num_spec_tokens <= 0:
            return np.zeros(0, dtype=np.float64)
        rates = np.empty(self.max_num_spec_tokens, dtype=np.float64)
        for index in range(self.max_num_spec_tokens):
            opportunities = float(max(0, self.position_acceptance_opportunities[index]))
            if opportunities > 0.0:
                rates[index] = min(
                    1.0,
                    max(0.0, self.position_acceptance_successes[index] / opportunities),
                )
            else:
                rates[index] = min(
                    self._get_position_acceptance_bootstrap_rate(index),
                    min(
                        1.0,
                        max(0.0, self.position_acceptance_rate_ema[index]),
                    ),
                )
            if index > 0:
                rates[index] = min(rates[index], rates[index - 1])
        return rates

    def _get_position_acceptance_prior_rate(self, index: int) -> float:
        return float(
            min(
                1.0,
                max(
                    0.0,
                    self.position_acceptance_prior_rate
                    * (self.position_acceptance_prior_decay**index),
                ),
            )
        )

    def _get_position_acceptance_bootstrap_rate(self, index: int) -> float:
        return self._get_position_acceptance_prior_rate(index)

    def _get_acceptance_rate_estimate(self) -> float:
        prior_strength = max(0.0, self.position_acceptance_prior_strength)
        prior_rate = float(min(1.0, max(0.0, self.initial_acceptance_rate)))
        opportunities = max(0.0, self.acceptance_opportunities)
        if opportunities <= 0.0:
            return prior_rate
        posterior_total = opportunities + prior_strength
        if posterior_total <= 0.0:
            return prior_rate
        posterior_mean = (
            self.acceptance_successes + prior_strength * prior_rate
        ) / posterior_total
        return float(min(1.0, max(0.0, posterior_mean)))

    def _get_proposal_coverage(self) -> float:
        if self.proposal_coverage_observations <= 0:
            return 1.0
        return float(min(1.0, max(0.0, self.proposal_coverage_ema)))

    def _has_ready_realized_k0_baseline(self) -> bool:
        return (
            self.realized_k_step_counts[0] >= self.k0_baseline_min_samples
            and self.realized_k_goodput_ema[0] > 0.0
            and self.realized_k_latency_s_ema[0] > 0.0
        )

    def _reset_cold_start_state(self) -> None:
        self.cold_start_spec_steps = 0
        self.cold_start_total_draft_tokens = 0
        self.cold_start_total_accepted_tokens = 0

    def _get_selection_speedups_by_k(self) -> tuple[float, ...]:
        if self._has_ready_realized_k0_baseline():
            return tuple(
                float(value) for value in self.realized_k_speedup_vs_realized_k0_ema
            )
        return tuple(
            float(value) for value in self.realized_k_speedup_vs_predicted_k0_ema
        )

    def _filtered_latency_samples(
        self, samples: list[_LatencySample]
    ) -> list[_LatencySample]:
        filtered = samples
        if self.online_latency_fit_max_realized_latency_s is not None:
            filtered = [
                sample
                for sample in filtered
                if sample.realized_latency_s
                <= self.online_latency_fit_max_realized_latency_s
            ]
        if (
            self.online_latency_fit_max_latency_ratio_to_median is not None
            and filtered
        ):
            median_latency = float(
                np.median([sample.realized_latency_s for sample in filtered])
            )
            latency_limit = (
                median_latency * self.online_latency_fit_max_latency_ratio_to_median
            )
            filtered = [
                sample
                for sample in filtered
                if sample.realized_latency_s <= latency_limit
            ]
        return filtered

    def _design_matrix(
        self, samples: list[_LatencySample]
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.array(
            [
                [
                    1.0,
                    float(sample.decode_token_load),
                    sample.smoothed_total_num_scheduled_tokens,
                    float(sample.effective_num_spec_tokens),
                    float(sample.effective_num_spec_tokens)
                    * sample.smoothed_total_num_scheduled_tokens,
                ]
                for sample in samples
            ],
            dtype=np.float64,
        )
        y = np.array(
            [sample.realized_latency_s for sample in samples], dtype=np.float64
        )
        return x, y

    def _update_profiled_latency_scale(
        self,
        *,
        decode_token_load: int,
        smoothed_total_num_scheduled_tokens: float,
        effective_num_spec_tokens: int,
        realized_latency_s: float,
    ) -> None:
        raw_predicted_latency_s = (
            self.profiled_latency_intercept_s
            + self.profiled_latency_decode_token_load_coeff_s
            * max(1, decode_token_load)
            + self.profiled_latency_scheduled_tokens_coeff_s
            * max(1.0, smoothed_total_num_scheduled_tokens)
            + self.profiled_latency_spec_tokens_coeff_s
            * effective_num_spec_tokens
            + self.profiled_latency_spec_scheduled_tokens_interaction_coeff_s
            * effective_num_spec_tokens
            * max(1.0, smoothed_total_num_scheduled_tokens)
        )
        raw_predicted_latency_s = max(1e-6, raw_predicted_latency_s)
        target_scale = np.clip(
            realized_latency_s / raw_predicted_latency_s,
            0.05,
            20.0,
        )
        self.profiled_latency_scale = max(
            1e-3,
            (1.0 - self.online_latency_fit_ema_alpha) * self.profiled_latency_scale
            + self.online_latency_fit_ema_alpha * float(target_scale),
        )

    def _fit_latency_coefficients(
        self,
        x: np.ndarray,
        y: np.ndarray,
        samples: list[_LatencySample],
    ) -> tuple[np.ndarray, np.ndarray]:
        current_coefficients = self._current_latency_coefficients()
        feature_ranges = np.ptp(x, axis=0)
        num_nonzero_k_samples = sum(
            sample.effective_num_spec_tokens > 0 for sample in samples
        )
        allow_structural_refit = (
            num_nonzero_k_samples >= self.online_latency_fit_min_nonzero_k_samples
        )
        active_mask = np.array(
            [
                True,
                allow_structural_refit
                and feature_ranges[1] >= self.online_latency_fit_min_feature_range,
                allow_structural_refit
                and feature_ranges[2] >= self.online_latency_fit_min_feature_range,
                allow_structural_refit
                and feature_ranges[3] >= self.online_latency_fit_min_feature_range,
                allow_structural_refit
                and feature_ranges[4] >= self.online_latency_fit_min_feature_range,
            ],
            dtype=bool,
        )
        frozen_mask = ~active_mask

        fitted_coefficients = current_coefficients.copy()
        residual_y = y.copy()
        if np.any(frozen_mask):
            residual_y = residual_y - (x[:, frozen_mask] @ current_coefficients[frozen_mask])
        if np.any(active_mask):
            fitted_coefficients[active_mask] = self._fit_nonnegative_least_squares(
                x[:, active_mask], residual_y
            )

        mixed_coefficients = (
            (1.0 - self.online_latency_fit_ema_alpha) * current_coefficients
            + self.online_latency_fit_ema_alpha * fitted_coefficients
        )
        max_relative_update = self.online_latency_fit_max_relative_update
        if max_relative_update > 0.0:
            max_delta = max_relative_update * np.maximum(
                np.maximum(np.abs(current_coefficients), np.abs(fitted_coefficients)),
                1e-6,
            )
            mixed_coefficients = np.clip(
                mixed_coefficients,
                current_coefficients - max_delta,
                current_coefficients + max_delta,
            )

        return np.maximum(mixed_coefficients, 0.0), active_mask

    def _fit_nonnegative_least_squares(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        active = list(range(x.shape[1]))
        coefficients = np.zeros(x.shape[1], dtype=np.float64)

        while active:
            active_x = x[:, active]
            active_coefficients, *_ = np.linalg.lstsq(active_x, y, rcond=None)
            negative_indices = [
                index
                for index, value in enumerate(active_coefficients)
                if value < 0.0
            ]
            if not negative_indices:
                coefficients[np.array(active)] = active_coefficients
                break
            most_negative = min(
                negative_indices, key=lambda index: active_coefficients[index]
            )
            del active[most_negative]

        return coefficients

    def _current_latency_coefficients(self) -> np.ndarray:
        return np.array(
            [
                self.profiled_latency_intercept_s,
                self.profiled_latency_decode_token_load_coeff_s,
                self.profiled_latency_scheduled_tokens_coeff_s,
                self.profiled_latency_spec_tokens_coeff_s,
                self.profiled_latency_spec_scheduled_tokens_interaction_coeff_s,
            ],
            dtype=np.float64,
        )

    def _update_realized_k_ema(
        self,
        values: np.ndarray,
        k: int,
        observation: float,
    ) -> None:
        if self.realized_k_step_counts[k] <= 1:
            values[k] = observation
            return
        values[k] = (
            (1.0 - self.realized_goodput_ema_alpha) * values[k]
            + self.realized_goodput_ema_alpha * observation
        )

    def _get_realized_goodputs_by_k(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.realized_k_goodput_ema)
