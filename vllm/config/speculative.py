# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from pydantic import Field, SkipValidation, model_validator
from typing_extensions import Self

from vllm.config import LoadConfig
from vllm.config.kernel import MoEBackend
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_text_config
from vllm.utils.hashing import safe_hash
from vllm.utils.import_utils import LazyLoader, has_arctic_inference

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    import vllm.model_executor.layers.quantization as me_quant
else:
    PretrainedConfig = Any

    me_quant = LazyLoader(
        "model_executor", globals(), "vllm.model_executor.layers.quantization"
    )

logger = init_logger(__name__)

_NGRAM_DSC_TURBOSPEC_DEFAULTS: dict[str, object] = {
    "ngram_dsc_strategy": "goodput",
    "ngram_dsc_initial_acceptance_rate": 0.8,
    "ngram_dsc_acceptance_ema_alpha": 0.3,
    "ngram_dsc_position_acceptance_prior_rate": 0.4,
    "ngram_dsc_position_acceptance_prior_decay": 0.5,
    "ngram_dsc_position_acceptance_prior_strength": 8.0,
    "ngram_dsc_position_acceptance_confidence_z": 1.0,
    "ngram_dsc_base_latency_tokens": 32.0,
    "ngram_dsc_latency_model": "profiled",
    "ngram_dsc_profiled_latency_intercept_s": 0.015,
    "ngram_dsc_profiled_latency_decode_token_load_coeff_s": 0.002,
    "ngram_dsc_profiled_latency_scheduled_tokens_coeff_s": 0.006,
    "ngram_dsc_profiled_latency_spec_tokens_coeff_s": 0.004,
    "ngram_dsc_profiled_latency_spec_scheduled_tokens_interaction_coeff_s": 0.0005,
    "ngram_dsc_online_latency_fitting": True,
    "ngram_dsc_online_latency_fit_min_samples": 32,
    "ngram_dsc_online_latency_fit_warmup_samples": 4,
    "ngram_dsc_online_latency_fit_refit_interval_samples": 8,
    "ngram_dsc_online_latency_fit_max_samples": 256,
    "ngram_dsc_online_latency_fit_max_latency_ratio_to_median": 3.0,
    "ngram_dsc_online_latency_fit_ema_alpha": 0.25,
    "ngram_dsc_online_latency_fit_max_relative_update": 0.5,
    "ngram_dsc_online_latency_fit_min_feature_range": 0.5,
    "ngram_dsc_online_latency_fit_min_nonzero_k_samples": 8,
    "ngram_dsc_realized_sample_min_decode_token_load": 1,
    "ngram_dsc_realized_sample_min_smoothed_scheduled_tokens": 1.0,
    "ngram_dsc_realized_sample_min_latency_s": 0.001,
    "ngram_dsc_normal_decode_realized_log_interval": 64,
    "ngram_dsc_near_best_goodput_ratio": 0.0,
    "ngram_dsc_switch_hysteresis_ratio": 0.02,
    "ngram_dsc_goodput_margin": 0.05,
    "ngram_dsc_goodput_increase_margin": 0.10,
    "ngram_dsc_realized_goodput_ema_alpha": 0.25,
    "ngram_dsc_k0_baseline_min_samples": 1,
    "ngram_dsc_initial_max_k": 1,
    "ngram_dsc_min_spec_realized_samples_before_k0": 8,
    "ngram_dsc_k0_sparse_evidence_margin": 0.10,
    "ngram_dsc_fast_fail_min_steps": 2,
    "ngram_dsc_fast_fail_max_steps": 3,
    "ngram_dsc_fast_fail_max_acceptance_rate": 0.05,
    "ngram_dsc_realized_goodput_guard_min_samples": 8,
    "ngram_dsc_realized_goodput_guard_margin": 0.05,
    "ngram_dsc_goodput_min_dwell_sec": 0.25,
    "ngram_dsc_upward_min_position_samples": 8,
    "ngram_dsc_scheduled_tokens_ema_alpha": 0.2,
    "ngram_dsc_max_step_delta": 1,
}


def _load_ngram_dsc_profiled_latency_coefficients(
    path_str: str,
) -> dict[str, float]:
    path = Path(path_str).expanduser()
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            "ngram_dsc profiled latency coefficient file must contain a JSON object."
        )

    coefficients_obj = payload.get("coefficients", payload)
    if not isinstance(coefficients_obj, dict):
        raise ValueError(
            "ngram_dsc profiled latency coefficient file must contain a "
            "'coefficients' object or direct coefficient keys."
        )

    aliases = {
        "intercept_s": "ngram_dsc_profiled_latency_intercept_s",
        "decode_token_load_coeff_s": (
            "ngram_dsc_profiled_latency_decode_token_load_coeff_s"
        ),
        "scheduled_tokens_coeff_s": (
            "ngram_dsc_profiled_latency_scheduled_tokens_coeff_s"
        ),
        "spec_tokens_coeff_s": "ngram_dsc_profiled_latency_spec_tokens_coeff_s",
        "spec_scheduled_tokens_interaction_coeff_s": (
            "ngram_dsc_profiled_latency_spec_scheduled_tokens_interaction_coeff_s"
        ),
    }

    coefficients: dict[str, float] = {}
    for key, value in coefficients_obj.items():
        normalized_key = aliases.get(key, key)
        if normalized_key not in {
            "ngram_dsc_profiled_latency_intercept_s",
            "ngram_dsc_profiled_latency_decode_token_load_coeff_s",
            "ngram_dsc_profiled_latency_scheduled_tokens_coeff_s",
            "ngram_dsc_profiled_latency_spec_tokens_coeff_s",
            "ngram_dsc_profiled_latency_spec_scheduled_tokens_interaction_coeff_s",
        }:
            continue
        coefficients[normalized_key] = float(value)
    return coefficients

MTPModelTypes = Literal[
    "deepseek_mtp",
    "mimo_mtp",
    "glm4_moe_mtp",
    "glm4_moe_lite_mtp",
    "glm_ocr_mtp",
    "ernie_mtp",
    "nemotron_h_mtp",
    "exaone_moe_mtp",
    "qwen3_next_mtp",
    "qwen3_5_mtp",
    "longcat_flash_mtp",
    "mtp",
    "pangu_ultra_moe_mtp",
    "step3p5_mtp",
]
EagleModelTypes = Literal["eagle", "eagle3", "extract_hidden_states", MTPModelTypes]
NgramGPUTypes = Literal["ngram_gpu"]
SpeculativeMethod = Literal[
    "ngram",
    "ngram_dsc",
    "medusa",
    "mlp_speculator",
    "draft_model",
    "suffix",
    EagleModelTypes,
    NgramGPUTypes,
]
RejectionSampleMethod = Literal["strict", "probabilistic"]
NgramDSCStrategy = Literal["threshold", "goodput"]
NgramDSCLatencyModel = Literal["heuristic", "profiled"]


@config
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    enforce_eager: bool | None = None
    """Override the default enforce_eager from model_config"""
    # General speculative decoding control
    num_speculative_tokens: int = Field(default=None, gt=0)  # type: ignore[assignment]
    """The number of speculative tokens, if provided. It will default to the
    number in the draft model config if present, otherwise, it is required."""
    model: str | None = None
    """The name of the draft model, eagle head, or additional weights, if
    provided."""
    method: SpeculativeMethod | None = None
    """The name of the speculative method to use. If users provide and set the
    `model` param, the speculative method type will be detected automatically
    if possible, if `model` param is not provided, the method name must be
    provided.

    If using `ngram` method, the related configuration `prompt_lookup_max` and
    `prompt_lookup_min` should be considered."""
    draft_tensor_parallel_size: int | None = Field(default=None, ge=1)
    """The degree of the tensor parallelism for the draft model. Can only be 1
    or the same as the target model's tensor parallel size."""
    tensor_parallel_size: int | None = None
    """Users should pass "draft_tensor_parallel_size". This parameter's purpose is to
    warn users when they mistakenly provide the wrong argument."""

    # Draft model configuration
    quantization: me_quant.QuantizationMethods | str | None = None
    """Quantization method that was used to quantize the draft model weights.
    If `None`, we assume the model weights are not quantized. Note that it only
    takes effect when using the draft model-based speculative method."""
    moe_backend: MoEBackend | None = None
    """MoE backend to use for the draft model. When `None`, the draft model
    inherits the target model's `--moe-backend` setting. Useful when the
    drafter and generator require different MoE kernels (e.g. quantized
    generator with unquantized drafter)."""
    max_model_len: int | None = Field(default=None, ge=1)
    """The maximum model length of the draft model. Used when testing the
    ability to skip speculation for some sequences."""
    revision: str | None = None
    """The specific model version to use for the draft model. It can be a
    branch name, a tag name, or a commit id. If unspecified, will use the
    default version."""
    code_revision: str | None = None
    """The specific revision to use for the draft model code on Hugging Face
    Hub. It can be a branch name, a tag name, or a commit id. If unspecified,
    will use the default version."""

    # Advanced control
    disable_padded_drafter_batch: bool = False
    """Disable input padding for speculative decoding. If set to True,
    speculative input batches can contain sequences of different lengths,
    which may only be supported by certain attention backends. This currently
    only affects the EAGLE method of speculation."""
    use_local_argmax_reduction: bool = False
    """Use vocab-parallel local argmax instead of all-gathering full logits
    for draft token generation. Reduces communication from O(vocab_size) to
    O(2 * tp_size) per token. Only applies to greedy draft selection in
    non-tree speculation."""

    # Ngram proposer configuration
    prompt_lookup_max: int | None = Field(default=None, ge=1)
    """Maximum size of ngram token window when using Ngram proposer, required
    when method is set to ngram."""
    prompt_lookup_min: int | None = Field(default=None, ge=1)
    """Minimum size of ngram token window when using Ngram proposer, if
    provided. Defaults to 1."""
    ngram_dsc: bool = False
    """Enable dynamic switching control (DSC) for ngram speculation. When
    enabled, the scheduler dynamically controls the speculative ngram length.
    The default threshold strategy disables ngram speculation under heavy
    decode load and re-enables it after load drops and cooldown elapses."""
    ngram_dsc_strategy: NgramDSCStrategy = "threshold"
    """Scheduler policy used by ngram DSC. `threshold` preserves the existing
    binary on/off behavior. `goodput` uses a TurboSpec-style goodput estimate
    to choose an effective speculative length in [0, num_speculative_tokens]."""
    ngram_dsc_disable_decode_tokens: int | None = Field(default=None, ge=1)
    """Disable ngram speculation when the current number of running decode
    tokens reaches this threshold. If None, the scheduler picks an automatic
    threshold based on runtime capacity."""
    ngram_dsc_enable_decode_tokens: int | None = Field(default=None, ge=1)
    """Re-enable ngram speculation when running decode tokens fall to this
    threshold or below. If None, the scheduler derives it from the disable
    threshold to provide hysteresis."""
    ngram_dsc_switch_cooldown_sec: float = Field(default=30.0, ge=0.0)
    """Minimum time in seconds between DSC mode switches, used to avoid
    oscillation under fluctuating load."""
    ngram_dsc_initial_acceptance_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    """Initial acceptance-rate estimate used by the goodput-based ngram DSC
    controller before any online observations are available."""
    ngram_dsc_acceptance_ema_alpha: float = Field(default=0.3, gt=0.0, le=1.0)
    """Smoothing factor for the goodput-based ngram DSC acceptance-rate EMA.
    Larger values adapt faster to recent acceptance observations."""
    ngram_dsc_position_acceptance_ema_alpha: float | None = Field(
        default=None, gt=0.0, le=1.0
    )
    """Optional smoothing factor for per-position acceptance-rate EMAs. When
    omitted, the controller reuses `ngram_dsc_acceptance_ema_alpha`."""
    ngram_dsc_position_acceptance_prior_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    """Optional prior mean used to shrink per-position acceptance estimates
    under sparse evidence. When omitted, the controller reuses
    `ngram_dsc_initial_acceptance_rate`."""
    ngram_dsc_position_acceptance_prior_decay: float = Field(
        default=1.0, ge=0.0, le=1.0
    )
    """Per-position multiplicative decay applied to the cold-start prior
    acceptance rate. Values below 1.0 make later draft positions more
    pessimistic before any evidence exists."""
    ngram_dsc_position_acceptance_prior_strength: float = Field(
        default=8.0, ge=0.0
    )
    """Pseudo-count strength for the per-position acceptance prior. Larger
    values make the controller rely more on the prior until enough
    observations are collected."""
    ngram_dsc_position_acceptance_confidence_z: float = Field(
        default=1.0, ge=0.0
    )
    """Confidence penalty applied to per-position posterior estimates. The
    controller subtracts `z * stddev` from the posterior mean to avoid
    overestimating late-position acceptance under sparse evidence."""
    ngram_dsc_base_latency_tokens: float = Field(default=32.0, gt=0.0)
    """Fixed per-step latency term for the goodput-based ngram DSC estimator.
    This approximates non-token-dependent target-model overhead."""
    ngram_dsc_latency_model: NgramDSCLatencyModel = "heuristic"
    """Latency estimator used by the goodput-based ngram DSC controller.
    `heuristic` preserves the current token-count proxy. `profiled` uses a
    fitted offline latency model with coefficients supplied below."""
    ngram_dsc_profiled_latency_coefficients_path: str | None = None
    """Optional path to a JSON file containing fitted profiled latency
    coefficients. When provided, the file values override the inline profiled
    latency coefficients and force `ngram_dsc_latency_model='profiled'`."""
    ngram_dsc_profiled_latency_intercept_s: float = Field(default=0.0, ge=0.0)
    """Intercept term for the profiled latency model, in seconds."""
    ngram_dsc_profiled_latency_decode_token_load_coeff_s: float = Field(
        default=0.0, ge=0.0
    )
    """Coefficient for running decode load in the profiled latency model,
    in seconds per decode request."""
    ngram_dsc_profiled_latency_scheduled_tokens_coeff_s: float = Field(
        default=0.0, ge=0.0
    )
    """Coefficient for total scheduled tokens in the profiled latency model,
    in seconds per scheduled token."""
    ngram_dsc_profiled_latency_spec_tokens_coeff_s: float = Field(
        default=0.0, ge=0.0
    )
    """Coefficient for speculative width k in the profiled latency model,
    in seconds per speculative token."""
    ngram_dsc_profiled_latency_spec_scheduled_tokens_interaction_coeff_s: float = (
        Field(default=0.0, ge=0.0)
    )
    """Interaction coefficient for speculative width and scheduled tokens in
    the profiled latency model, in seconds per (k * scheduled_tokens)."""
    ngram_dsc_online_latency_fitting: bool = False
    """Enable automatic online refitting of the profiled latency model from
    realized decode-step latency samples observed during serving."""
    ngram_dsc_online_latency_fit_min_samples: int = Field(default=32, ge=1)
    """Minimum number of filtered realized-latency samples required before
    the controller updates profiled latency coefficients online."""
    ngram_dsc_online_latency_fit_warmup_samples: int = Field(default=4, ge=0)
    """Number of initial realized-latency samples to ignore before online
    fitting starts. This helps avoid warmup and graph-capture skew."""
    ngram_dsc_online_latency_fit_refit_interval_samples: int = Field(
        default=8, ge=1
    )
    """Minimum number of new filtered samples required between online
    latency-model refits."""
    ngram_dsc_online_latency_fit_max_samples: int = Field(default=256, ge=1)
    """Maximum number of recent realized-latency samples retained for online
    fitting. Older samples are evicted FIFO once this cap is reached."""
    ngram_dsc_online_latency_fit_max_latency_ratio_to_median: float = Field(
        default=3.0, gt=0.0
    )
    """Discard realized-latency samples above median * ratio during online
    fitting. This filters warmup spikes and transient outliers."""
    ngram_dsc_online_latency_fit_ema_alpha: float = Field(
        default=0.25, gt=0.0, le=1.0
    )
    """EMA factor used to blend each new online latency-model fit into the
    existing coefficients. Smaller values preserve the bootstrap profile
    longer and reduce oscillation."""
    ngram_dsc_online_latency_fit_max_relative_update: float = Field(
        default=0.5, ge=0.0
    )
    """Maximum per-refit relative coefficient change for online latency-model
    updates. This bounds abrupt shifts when the regression is underdetermined."""
    ngram_dsc_online_latency_fit_min_feature_range: float = Field(
        default=0.5, ge=0.0
    )
    """Minimum feature range required before a non-intercept latency
    coefficient is allowed to refit online. Features with insufficient
    variation remain frozen at their current values."""
    ngram_dsc_online_latency_fit_min_nonzero_k_samples: int = Field(
        default=0, ge=0
    )
    """Minimum number of recent valid realized-latency samples with
    `effective_num_spec_tokens > 0` required before any structural profiled
    latency coefficients other than the intercept are allowed to refit
    online. This keeps narrow `k=0` traces from overwriting the bootstrap
    latency shape."""
    ngram_dsc_online_latency_fit_max_realized_latency_s: float | None = Field(
        default=None, gt=0.0
    )
    """Optional hard upper bound for realized-latency samples included in
    online latency-model fitting."""
    ngram_dsc_realized_sample_min_decode_token_load: int = Field(default=1, ge=0)
    """Minimum decode-token load required before a normal-decode (`k=0`)
    realized sample is considered stable enough for TurboSpec telemetry and
    online latency fitting."""
    ngram_dsc_realized_sample_min_smoothed_scheduled_tokens: float = Field(
        default=1.0, ge=0.0
    )
    """Minimum smoothed scheduled-token count required before a normal-decode
    (`k=0`) realized sample is considered stable enough for TurboSpec
    telemetry and online latency fitting."""
    ngram_dsc_realized_sample_min_latency_s: float = Field(default=0.001, ge=0.0)
    """Minimum realized decode-step latency required before a normal-decode
    (`k=0`) realized sample is considered stable enough for TurboSpec
    telemetry and online latency fitting."""
    ngram_dsc_normal_decode_realized_log_interval: int = Field(default=64, ge=1)
    """Emit at most one detailed `NGRAM_DSC_REALIZED` log per this many valid
    normal-decode (`k=0`) samples. Speculative realized steps are always
    logged."""
    ngram_dsc_near_best_goodput_ratio: float = Field(default=0.0, ge=0.0)
    """Relative tolerance for TTFT-aware tie-breaking. When multiple
    candidates are within this fraction of the best predicted goodput, the
    controller chooses the lower-latency candidate."""
    ngram_dsc_switch_hysteresis_ratio: float = Field(default=0.02, ge=0.0)
    """Minimum relative predicted-goodput improvement required before the
    controller changes from the current speculative width to a different
    width. This suppresses flapping on near-tied candidates."""
    ngram_dsc_goodput_margin: float = Field(default=0.05, ge=0.0)
    """Minimum relative goodput improvement required before the goodput-based
    DSC controller changes the current speculative length."""
    ngram_dsc_goodput_increase_margin: float = Field(default=0.10, ge=0.0)
    """Minimum relative goodput improvement required before the goodput-based
    DSC controller increases the current speculative length. This is
    intentionally stricter than the general margin to reduce oscillation."""
    ngram_dsc_realized_goodput_ema_alpha: float = Field(
        default=0.25, gt=0.0, le=1.0
    )
    """EMA factor used for recent realized-goodput tracking by speculative
    width. This keeps the controller anchored to recent serving behavior
    instead of a full-history average."""
    ngram_dsc_k0_baseline_min_samples: int = Field(default=1, ge=1)
    """Minimum number of realized `k=0` decode steps required before the
    controller is allowed to replace the predicted `k=0` baseline with the
    realized normal-decode baseline."""
    ngram_dsc_initial_max_k: int = Field(default=1, ge=0)
    """Maximum speculative width allowed at cold start before the controller
    has accumulated meaningful acceptance evidence. This reduces one-request
    speculative waste by starting conservatively."""
    ngram_dsc_min_spec_realized_samples_before_k0: int = Field(default=8, ge=0)
    """Minimum number of realized speculative (`k>0`) samples before the
    controller is allowed to disable speculation on a small predicted
    goodput advantage alone."""
    ngram_dsc_k0_sparse_evidence_margin: float = Field(default=0.10, ge=0.0)
    """When speculative evidence is still sparse, `k=0` must beat the best
    positive speculative width by at least this relative goodput margin
    before the controller shuts prompt lookup off."""
    ngram_dsc_fast_fail_min_steps: int = Field(default=2, ge=1)
    """Minimum number of early speculative verify steps required before the
    controller can trigger an immediate fast-fail to `k=0`."""
    ngram_dsc_fast_fail_max_steps: int = Field(default=3, ge=1)
    """Maximum number of early speculative verify steps during which the
    fast-fail rule is active. After this window, normal goodput control
    takes over."""
    ngram_dsc_fast_fail_max_acceptance_rate: float = Field(
        default=0.05, ge=0.0, le=1.0
    )
    """If early accepted / drafted token ratio stays at or below this value
    during the fast-fail window, the controller immediately disables
    speculation by switching to `k=0`."""
    ngram_dsc_realized_goodput_guard_min_samples: int = Field(default=8, ge=1)
    """Minimum number of realized samples required before recent realized
    goodput for a speculative width can be used to block upward moves."""
    ngram_dsc_realized_goodput_guard_margin: float = Field(default=0.05, ge=0.0)
    """Relative underperformance margin used by the realized-goodput guard.
    If a wider speculative width underperforms a narrower recent baseline by
    more than this margin, the controller will not move upward into it."""
    ngram_dsc_upward_min_position_samples: int = Field(default=8, ge=0)
    """Minimum number of observed opportunities required for each new draft
    position before the goodput controller is allowed to increase `k` into
    that position."""
    ngram_dsc_goodput_min_dwell_sec: float = Field(default=0.25, ge=0.0)
    """Minimum time to keep the current speculative length before the
    goodput-based DSC controller can switch again."""
    ngram_dsc_max_step_delta: int = Field(default=1, ge=1)
    """Maximum change in effective speculative length per scheduler step for
    the goodput-based DSC controller. This reduces oscillation."""
    ngram_dsc_scheduled_tokens_ema_alpha: float = Field(default=0.2, gt=0.0, le=1.0)
    """Smoothing factor for the scheduler-load EMA used by the goodput-based
    DSC controller. Smaller values trade responsiveness for stability."""

    # Alternative drafting strategies
    speculative_token_tree: str | None = None
    """Specifies the tree structure for speculative token generation.
    """
    parallel_drafting: bool = False
    """Enable parallel drafting, where all speculative tokens are generated
    in parallel rather than sequentially. This can improve performance but
    requires the speculative model be trained to support parallel drafting.
    Only compatible with EAGLE and draft model methods."""

    # required configuration params passed from engine
    target_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the target model."""
    target_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the target model."""

    # params generated in the post-init stage
    draft_model_config: SkipValidation[ModelConfig] = None  # type: ignore
    """The configuration of the draft model initialized internal."""
    draft_parallel_config: SkipValidation[ParallelConfig] = None  # type: ignore
    """The parallel configuration for the draft model initialized internal."""

    # Suffix decoding configuration
    suffix_decoding_max_tree_depth: int = 24
    """The maximum depth of the suffix decoding global and prompt trees. The
    tree depth limits the sum of the prefix match and speculation lengths."""

    suffix_decoding_max_cached_requests: int = 10000
    """The maximum number of requests to cache in the global suffix tree. If
    exceeded, will trigger eviction in FIFO order. If set to 0, the global
    suffix tree is disabled and past responses are not cached (prompt trees
    are still used)."""

    suffix_decoding_max_spec_factor: float = 1.0
    """The maximum spec factor for suffix decoding. The spec factor controls
    speculation lengths based on the prefix match length: max_spec_tokens =
    max_spec_factor * prefix_match_length."""

    suffix_decoding_min_token_prob: float = 0.1
    """The minimum token probability for suffix decoding. Will only speculate
    tokens with estimated probability (based on frequency counts) greater than
    or equal to this value."""

    draft_load_config: LoadConfig | None = None
    """Load config for the draft model. If not specified, will use the load
    config from the target model."""

    rejection_sample_method: RejectionSampleMethod = "strict"
    """Whether to use strict (target and draft sampled tokens match exactly)
    or probabilistic rejection sampling. Both respect the target model
    distribution, but the latter yields a higher acceptance rate at the cost
    of more memory to cache draft logits."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: list[Any] = []
        # Eagle3 and extract_hidden_states affect the computation graph because
        # they return intermediate hidden states in addition to the final hidden state.
        uses_aux_hidden_states = self.method in ("eagle3", "extract_hidden_states")
        factors.append(uses_aux_hidden_states)

        # The specific layers used also affect the computation graph
        if uses_aux_hidden_states and self.draft_model_config is not None:
            layer_ids = getattr(
                self.draft_model_config.hf_config,
                "eagle_aux_hidden_state_layer_ids",
                None,
            )
            if layer_ids is not None:
                # Convert to tuple to make it hashable
                factors.append(tuple(layer_ids))

        hash_str = safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()
        return hash_str

    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
        if hf_config.model_type in ("deepseek_v3", "deepseek_v32", "glm_moe_dsa"):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
            )
        if hf_config.model_type in ("pangu_ultra_moe"):
            hf_config.model_type = "pangu_ultra_moe_mtp"
        if hf_config.model_type == "pangu_ultra_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]}
            )

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["MiMoMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeLiteForCausalLM":
            hf_config.model_type = "glm4_moe_lite_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeLiteMTPModel"],
                }
            )

        if hf_config.architectures[0] == "GlmOcrForConditionalGeneration":
            hf_config.model_type = "glm_ocr_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["GlmOcrMTPModel"],
                }
            )

        if hf_config.model_type == "ernie4_5_moe":
            hf_config.model_type = "ernie_mtp"
        if hf_config.model_type == "ernie_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ErnieMTPModel"]}
            )

        if (
            hf_config.model_type in {"nemotron_h", "nemotron_h_puzzle"}
            and hasattr(hf_config, "num_nextn_predict_layers")
            and hf_config.num_nextn_predict_layers > 0
        ):
            # Check if this is an MTP variant
            hf_config.model_type = "nemotron_h_mtp"
        if hf_config.model_type == "nemotron_h_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["NemotronHMTPModel"]}
            )

        if hf_config.model_type == "qwen3_next":
            hf_config.model_type = "qwen3_next_mtp"
        if hf_config.model_type == "qwen3_next_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}
            )

        if hf_config.model_type == "exaone_moe":
            hf_config.model_type = "exaone_moe_mtp"
        if hf_config.model_type == "exaone_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ExaoneMoeMTP"]}
            )

        if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
            is_moe = hf_config.model_type == "qwen3_5_moe"
            hf_config.model_type = "qwen3_5_mtp"
            n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
                }
            )
        if hf_config.model_type == "longcat_flash":
            hf_config.model_type = "longcat_flash_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]}
            )

        if hf_config.model_type == "step3p5":
            hf_config.model_type = "step3p5_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

        if initial_architecture == "MistralLarge3ForCausalLM":
            hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

        return hf_config

    def __post_init__(self):
        fields_set = set(getattr(self, "__pydantic_fields_set__", set()))
        uses_ngram_dsc_alias = (
            self.method == "ngram_dsc"
            or (self.method is None and self.model == "ngram_dsc")
        )

        # Note: "method" is a new parameter that helps to extend the
        # configuration of non-model-based proposers, and the "model" parameter
        # will be used to set the draft model, eagle head, or additional weight
        # when needed. If users do not specify "method", the speculative method
        # will be detected automatically if possible. If the speculative method
        # can not be detected, it will be considered as the "draft_model" by
        # default.

        # infer method from user args
        if self.method is None:
            if self.model in ("ngram", "[ngram]", "ngram_dsc"):
                if self.model == "ngram_dsc":
                    self.ngram_dsc = True
                self.method = "ngram"
            else:
                self.method = "draft_model"

        if self.method in get_args(MTPModelTypes) and self.method != "mtp":
            logger.warning(
                "method `%s` is deprecated and replaced with mtp.", self.method
            )
            self.method = "mtp"

        if self.model is None and self.num_speculative_tokens is not None:
            if self.method == "mtp":
                if self.target_model_config is None:
                    raise ValueError("target_model_config must be present for mtp")
                if self.target_model_config.hf_text_config.model_type == "deepseek_v32":
                    # FIXME(luccafong): cudagraph with v32 MTP is not supported,
                    # remove this when the issue is fixed.
                    self.enforce_eager = True
                # use the draft model from the same model:
                self.model = self.target_model_config.model
                # Align the quantization of draft model for cases such as
                # --quantization fp8 with a bf16 checkpoint.
                if not self.quantization:
                    self.quantization = self.target_model_config.quantization
            elif self.method in ("ngram", "[ngram]", "ngram_dsc"):
                if self.method == "ngram_dsc":
                    self.ngram_dsc = True
                self.model = "ngram"
            elif self.method == "ngram_gpu":
                self.model = "ngram_gpu"
            elif self.method == "suffix":
                self.model = "suffix"
            elif self.method == "extract_hidden_states":
                self.model = "extract_hidden_states"
            else:
                raise ValueError(
                    "num_speculative_tokens was provided but without speculative model."
                )

        if self.method in ("ngram", "[ngram]", "ngram_dsc"):
            if self.method == "ngram_dsc":
                self.ngram_dsc = True
            self.method = "ngram"

        if uses_ngram_dsc_alias:
            for field_name, value in _NGRAM_DSC_TURBOSPEC_DEFAULTS.items():
                if field_name not in fields_set:
                    setattr(self, field_name, value)

        if self.method in ("ngram", "ngram_gpu"):
            # Set default values if not provided
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                # TODO(woosuk): Tune these values. They are arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                if self.prompt_lookup_max is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                if self.prompt_lookup_min is None:
                    raise ValueError(
                        "Either prompt_lookup_max or prompt_lookup_min must be "
                        "provided when using the ngram method."
                    )
                self.prompt_lookup_max = self.prompt_lookup_min

            # Validate values
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must "
                    f"be <= prompt_lookup_max={self.prompt_lookup_max}"
                )

            # TODO: current we still need extract vocab_size from target model
            # config, in future, we may try refactor it out, and set
            # draft related config as None here.
            self.draft_model_config = self.target_model_config
            self.draft_parallel_config = self.target_parallel_config
        elif self.method == "suffix":
            self._validate_suffix_decoding()
        elif self.method == "extract_hidden_states":
            from vllm.transformers_utils.configs.extract_hidden_states import (
                ExtractHiddenStatesConfig,
            )

            # ExtractHiddenStatesModel is instantiated manually in load_model()
            # We just need to store the target model config for KV cache shape info
            self.model = "extract_hidden_states"
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if hasattr(self.draft_model_config, "hf_config"):
                hf_config = self.draft_model_config.hf_config.to_dict()
            elif (
                isinstance(self.draft_model_config, dict)
                and "hf_config" in self.draft_model_config
            ):
                hf_config = self.draft_model_config["hf_config"]
            else:
                hf_config = {}

            self.draft_model_config = copy.copy(self.target_model_config)
            self.draft_model_config.hf_config = ExtractHiddenStatesConfig(
                self.draft_model_config.hf_config, **hf_config
            )
            self.update_arch_()
            self.draft_parallel_config = self.target_parallel_config

        else:
            self.prompt_lookup_max = 0
            self.prompt_lookup_min = 0

            if self.model is not None:
                self.draft_model_config = ModelConfig(
                    model=self.model,
                    runner="draft",
                    tokenizer=self.target_model_config.tokenizer,
                    tokenizer_mode=self.target_model_config.tokenizer_mode,
                    trust_remote_code=self.target_model_config.trust_remote_code,
                    allowed_local_media_path=self.target_model_config.allowed_local_media_path,
                    allowed_media_domains=self.target_model_config.allowed_media_domains,
                    dtype=self.target_model_config.dtype,
                    seed=self.target_model_config.seed,
                    revision=self.revision,
                    code_revision=self.code_revision,
                    tokenizer_revision=self.target_model_config.tokenizer_revision,
                    spec_target_max_model_len=self.target_model_config.max_model_len,
                    quantization=self.quantization,
                    enforce_eager=self.target_model_config.enforce_eager,
                    max_logprobs=self.target_model_config.max_logprobs,
                    hf_overrides=SpeculativeConfig.hf_config_override,
                    config_format=self.target_model_config.config_format,
                )

                # Automatically detect the method
                if self.method in ("eagle", "eagle3"):
                    pass
                # examples:
                # yuhuili/EAGLE-LLaMA3-Instruct-8B
                # yuhuili/EAGLE3-LLaMA3.1-Instruct-8B
                # AngelSlim/Qwen3-8B_eagle3
                elif "eagle-" in self.draft_model_config.model.lower():
                    self.method = "eagle"
                elif "eagle3" in self.draft_model_config.model.lower():
                    self.method = "eagle3"
                elif self.draft_model_config.hf_config.model_type == "medusa":
                    self.method = "medusa"
                elif self.draft_model_config.hf_config.model_type == "mlp_speculator":
                    self.method = "mlp_speculator"
                elif self.draft_model_config.hf_config.model_type in get_args(
                    MTPModelTypes
                ):
                    self.method = "mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                            "Enabling num_speculative_tokens > 1 will run "
                            "multiple times of forward on same MTP layer"
                            ",which may result in lower acceptance rate"
                        )
                elif self.draft_model_config.hf_config.model_type in (
                    "longcat_flash_mtp"
                ):
                    self.method = "longcat_flash_mtp"
                    if self.num_speculative_tokens > 1:
                        logger.warning(
                            "LongCat MTP models only have "
                            "one layer. Might need some code changes "
                            "to support multiple layers."
                        )
                elif self.method == "draft_model":
                    pass
                else:
                    raise NotImplementedError(
                        f"Unsupported speculative method: '{self.method}'"
                    )

                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3"):
                    from vllm.transformers_utils.configs.eagle import EAGLEConfig
                    from vllm.transformers_utils.configs.speculators import (
                        SpeculatorsConfig,
                    )

                    if isinstance(
                        self.draft_model_config.hf_config,
                        (EAGLEConfig, SpeculatorsConfig),
                    ):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method,
                            model_type="eagle",
                        )
                        self.draft_model_config.hf_config = eagle_config
                        self.update_arch_()

                if self.num_speculative_tokens is not None and hasattr(
                    self.draft_model_config.hf_config, "num_lookahead_tokens"
                ):
                    self.draft_model_config.hf_config.num_lookahead_tokens = (
                        self.num_speculative_tokens
                    )

                n_predict = getattr(
                    self.draft_model_config.hf_config, "n_predict", None
                )
                if n_predict is not None:
                    if self.num_speculative_tokens is None:
                        # Default to max value defined in draft model config.
                        self.num_speculative_tokens = n_predict
                    elif (
                        self.num_speculative_tokens > n_predict
                        and self.num_speculative_tokens % n_predict != 0
                    ):
                        # Ensure divisibility for MTP module reuse.
                        raise ValueError(
                            f"num_speculative_tokens:{self.num_speculative_tokens}"
                            f" must be divisible by {n_predict=}"
                        )

                if self.speculative_token_tree is None:
                    if self.num_speculative_tokens is None:
                        raise ValueError(
                            "A speculative model was provided, but neither "
                            "`speculative_token_tree` nor `num_speculative_tokens` "
                            "was provided"
                        )

                    # Generate chain of tokens.
                    self.speculative_token_tree = str(
                        [(i + 1) * (0,) for i in range(self.num_speculative_tokens)]
                    )
                else:
                    # Sort the token tree breadth-first.
                    tree_choices = ast.literal_eval(self.speculative_token_tree)
                    self.speculative_token_tree = str(
                        sorted(tree_choices, key=lambda t: (len(t), t))
                    )

                self.draft_tensor_parallel_size = (
                    SpeculativeConfig._verify_and_get_draft_tp(
                        self.target_parallel_config,
                        self.draft_tensor_parallel_size,
                        self.draft_model_config.hf_config,
                    )
                )

                self.draft_model_config.max_model_len = (
                    SpeculativeConfig._maybe_override_draft_max_model_len(
                        self.max_model_len,
                        self.draft_model_config.max_model_len,
                        self.target_model_config.max_model_len,
                    )
                )

                self.draft_parallel_config = (
                    SpeculativeConfig.create_draft_parallel_config(
                        self.target_parallel_config, self.draft_tensor_parallel_size
                    )
                )
        return self

    def _validate_suffix_decoding(self):
        if not has_arctic_inference():
            raise ImportError(
                "Arctic Inference is required for suffix decoding. "
                "Install via `pip install arctic-inference==0.1.1`."
            )
        if self.num_speculative_tokens is None:
            # Suffix decoding decides the actual number of speculative tokens
            # dynamically and treats num_speculative_tokens as a maximum limit.
            self.num_speculative_tokens = self.suffix_decoding_max_tree_depth
            logger.warning(
                "Defaulted num_speculative_tokens to %s for suffix decoding.",
                self.num_speculative_tokens,
            )
        # Validate values
        if self.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{self.suffix_decoding_max_tree_depth} must be >= 1"
            )
        if self.suffix_decoding_max_cached_requests < 0:
            raise ValueError(
                f"suffix_decoding_max_cached_requests="
                f"{self.suffix_decoding_max_cached_requests} must be >= 0"
            )
        if self.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{self.suffix_decoding_max_spec_factor} must be >= 0"
            )
        if not 0 <= self.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{self.suffix_decoding_min_token_prob} must be in [0, 1]"
            )

    @staticmethod
    def _maybe_override_draft_max_model_len(
        speculative_max_model_len: int | None,
        draft_max_model_len: int,
        target_max_model_len: int,
    ) -> int:
        """Determine the max sequence len for the draft model. This is usually
        the draft_max_model_len, but may be the target_max_model_len if it is
        less than the draft_max_model_len, or may be speculative_max_model_len
        if it is specified.

        This is necessary so that sequences do not exceed the capacity of the
        draft model or the target model.

        speculative_max_model_len is mainly used for testing that sequences can
        skip speculation.
        """

        if speculative_max_model_len is not None:
            if speculative_max_model_len > draft_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {draft_max_model_len=}"
                )

            if speculative_max_model_len > target_max_model_len:
                raise ValueError(
                    f"{speculative_max_model_len=} cannot be "
                    f"larger than {target_max_model_len=}"
                )

            return speculative_max_model_len

        return min(
            draft_max_model_len,
            target_max_model_len,
        )

    @staticmethod
    def _verify_and_get_draft_tp(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int | None,
        draft_hf_config: PretrainedConfig,
    ) -> int:
        """
        Verifies and adjusts the tensor parallel size for a draft model
        specified using speculative_draft_tensor_parallel_size.
        """
        # If speculative_draft_tensor_parallel_size is unset then set it
        # appropriately else verify that it is set correctly.
        if speculative_draft_tensor_parallel_size is None:
            if draft_hf_config.model_type == "mlp_speculator":
                speculative_draft_tensor_parallel_size = 1
                if target_parallel_config.tensor_parallel_size > 1:
                    logger.warning(
                        "%s cannot currently be run with tp>1; "
                        "setting speculative_draft_tensor_parallel_size=1",
                        draft_hf_config.model_type,
                    )
            else:
                speculative_draft_tensor_parallel_size = (
                    target_parallel_config.tensor_parallel_size
                )
        elif speculative_draft_tensor_parallel_size not in (
            1,
            target_parallel_config.tensor_parallel_size,
        ):
            raise ValueError(
                f"{speculative_draft_tensor_parallel_size=} cannot be "
                f"other value than 1 or target model tensor_parallel_size"
            )
        return speculative_draft_tensor_parallel_size

    def update_arch_(self):
        """
        EagleConfig and ExtractHiddenStatesConfig update architectures, so update all
        architectures-related fields in self.draft_model_config
        """
        self.draft_model_config.hf_text_config = get_hf_text_config(
            self.draft_model_config.hf_config
        )
        self.draft_model_config.model_arch_config = (
            self.draft_model_config.get_model_arch_config()
        )
        model_info, arch = self.draft_model_config.registry.inspect_model_cls(
            self.draft_model_config.architectures,
            self.draft_model_config,
        )
        self.draft_model_config._model_info = model_info
        self.draft_model_config._architecture = arch

    @staticmethod
    def create_draft_parallel_config(
        target_parallel_config: ParallelConfig,
        speculative_draft_tensor_parallel_size: int,
    ) -> ParallelConfig:
        """Create a parallel config for use by the draft worker.

        This is mostly a copy of the target parallel config, except the tp_size.
        """
        draft_parallel_config = ParallelConfig(
            pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,
            tensor_parallel_size=speculative_draft_tensor_parallel_size,
            distributed_executor_backend=target_parallel_config.distributed_executor_backend,
            max_parallel_loading_workers=target_parallel_config.max_parallel_loading_workers,
            disable_custom_all_reduce=target_parallel_config.disable_custom_all_reduce,
            ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight,
            placement_group=target_parallel_config.placement_group,
        )

        return draft_parallel_config

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        if self.tensor_parallel_size is not None:
            raise ValueError(
                "'tensor_parallel_size' is not a valid argument in the "
                "speculative_config. Please pass 'draft_tensor_parallel_size' instead."
            )

        if self.num_speculative_tokens is None:
            raise ValueError(
                "num_speculative_tokens must be provided with "
                "speculative model unless the draft model config contains an "
                "n_predict parameter."
            )

        if self.num_speculative_tokens <= 0:
            raise ValueError(
                "Expected num_speculative_tokens to be greater "
                f"than zero ({self.num_speculative_tokens})."
            )

        if (
            self.ngram_dsc_enable_decode_tokens is not None
            and self.ngram_dsc_disable_decode_tokens is not None
            and self.ngram_dsc_enable_decode_tokens
            > self.ngram_dsc_disable_decode_tokens
        ):
            raise ValueError(
                "ngram_dsc_enable_decode_tokens must be <= "
                "ngram_dsc_disable_decode_tokens."
            )
        if self.ngram_dsc_strategy == "goodput" and not self.ngram_dsc:
            raise ValueError(
                "ngram_dsc_strategy='goodput' requires ngram_dsc to be enabled."
            )
        if self.ngram_dsc_profiled_latency_coefficients_path is not None:
            try:
                coefficients = _load_ngram_dsc_profiled_latency_coefficients(
                    self.ngram_dsc_profiled_latency_coefficients_path
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    "Failed to load ngram_dsc profiled latency coefficients from "
                    f"{self.ngram_dsc_profiled_latency_coefficients_path!r}: {exc}"
                ) from exc
            for field_name, value in coefficients.items():
                setattr(self, field_name, value)
            self.ngram_dsc_latency_model = "profiled"
        if self.ngram_dsc_online_latency_fitting:
            self.ngram_dsc_latency_model = "profiled"
        if (
            self.ngram_dsc_latency_model == "profiled"
            and (
                self.ngram_dsc_profiled_latency_intercept_s
                + self.ngram_dsc_profiled_latency_decode_token_load_coeff_s
                + self.ngram_dsc_profiled_latency_scheduled_tokens_coeff_s
                + self.ngram_dsc_profiled_latency_spec_tokens_coeff_s
                + self.ngram_dsc_profiled_latency_spec_scheduled_tokens_interaction_coeff_s
            )
            <= 0.0
        ):
            raise ValueError(
                "ngram_dsc_latency_model='profiled' requires at least one "
                "positive profiled latency coefficient."
            )
        if (
            self.ngram_dsc_online_latency_fit_min_samples
            > self.ngram_dsc_online_latency_fit_max_samples
        ):
            raise ValueError(
                "ngram_dsc_online_latency_fit_min_samples must be <= "
                "ngram_dsc_online_latency_fit_max_samples."
            )

        if self.draft_model_config:
            self.draft_model_config.verify_with_parallel_config(
                self.draft_parallel_config
            )

        aux_hidden_states_supported = [
            "llama",
            "qwen",
            "minicpm",
            "gpt_oss",
            "hunyuan_vl",
            "hunyuan_v1_dense",
            "afmoe",
            "nemotron_h",
            "deepseek_v2",
            "deepseek_v3",
            "kimi_k2",
            "kimi_k25",
        ]
        if (
            self.method in ("eagle3", "extract_hidden_states")
            and self.target_model_config
            and not any(
                supported_model in self.target_model_config.hf_text_config.model_type
                for supported_model in aux_hidden_states_supported
            )
        ):
            raise ValueError(
                f"{self.method} is only supported for {aux_hidden_states_supported}"
                f" models. Got {self.target_model_config.hf_text_config.model_type=}"
            )
        self.verify_equal_vocab_size_if_draft_model()
        return self

    def verify_equal_vocab_size_if_draft_model(self):
        if (
            self.method == "draft_model"
            and self.target_model_config is not None
            and self.draft_model_config is not None
        ):
            target_vocab_size = self.target_model_config.get_vocab_size()
            draft_vocab_size = self.draft_model_config.get_vocab_size()
            if target_vocab_size != draft_vocab_size:
                raise ValueError(
                    f"Target and draft model should have the same vocabulary size. "
                    f"Target model vocab_size={target_vocab_size}. "
                    f"Draft model vocab_size={draft_vocab_size}. "
                    f"Using models with different tokenizers can cause out-of-bounds "
                    f"errors during speculative decoding."
                )

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        """
        Calculate the maximum number of new slots that might be added to the batch
        when drafting.
        """
        slots_per_req = 0  # for serial non-draft-model methods, no change needed
        if self.parallel_drafting:
            # For parallel drafting, we need one new slot per 'masked' token
            slots_per_req = self.num_speculative_tokens - 1
        if self.uses_draft_model():
            # For draft model-based speculation, we need one new slot per request
            # Since we do not slice the draft tokens
            slots_per_req += 1
        return slots_per_req

    def use_eagle(self) -> bool:
        return self.method in ("eagle", "eagle3", "mtp")

    def uses_draft_model(self) -> bool:
        return self.method == "draft_model"

    def uses_extract_hidden_states(self) -> bool:
        return self.method == "extract_hidden_states"

    def use_ngram_gpu(self) -> bool:
        return self.method == "ngram_gpu"

    def __repr__(self) -> str:
        method = self.method
        model = (
            None
            if method in ("ngram", "suffix", "extract_hidden_states")
            else self.draft_model_config.model
        )
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"
