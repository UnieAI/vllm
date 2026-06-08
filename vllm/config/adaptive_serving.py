# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from pydantic import Field, model_validator

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

PROFILE_DEFAULTS: dict[str, dict[str, float]] = {
    "dev": {
        "warmup_ema_decay": 0.8,
        "warmup_budget_ms": 200.0,
        "warmup_vram_budget_ratio": 0.5,
        "warmup_min_hit_count": 3.0,
    },
    "production": {
        "warmup_ema_decay": 0.95,
        "warmup_budget_ms": 100.0,
        "warmup_vram_budget_ratio": 0.3,
        "warmup_min_hit_count": 20.0,
    },
}


@config
class AdaptiveServingConfig:
    """Configuration for Adaptive Speculative Serving."""

    # Layer 1: Prefix Warmup
    enable_adaptive_warmup: bool = True
    """Enable the Prefix Frequency Tracker and Prefix Warmup Worker
    to pre-warm high-frequency prefixes during idle GPU windows."""

    warmup_max_prefixes: int = Field(default=1000, gt=0)
    """Maximum number of tracked prefix patterns in the frequency
    map."""

    warmup_ema_decay: float | None = Field(default=None, gt=0.0, lt=1.0)
    """EMA decay rate for prefix frequency tracking. If None, set
    by the selected adaptive_profile."""

    warmup_budget_ms: float | None = Field(default=None, gt=0.0)
    """Maximum GPU time (ms) for prefix warmup per idle window.
    If None, set by the selected adaptive_profile."""

    warmup_vram_budget_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    """Fraction of free KV cache blocks allowed for warmup in a
    single idle window. If None, set by the selected
    adaptive_profile."""

    warmup_min_hit_count: float | None = Field(default=None, ge=0.0)
    """Minimum EMA score a prefix must reach before becoming a
    warmup candidate. If None, set by the selected
    adaptive_profile."""

    # Layer 2: Self-Speculation
    enable_self_speculation: bool = False
    """Enable the Self-Speculation Proposer to speculatively
    pre-compute next-token KV cache and logits during decode."""

    self_spec_confidence_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    """Top-1 softmax probability threshold for triggering
    speculative pre-computation."""

    self_spec_min_hit_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    """Hit rate below which speculation is disabled for a context
    pattern."""

    self_spec_activation_hit_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    """Hit rate at or above which speculation is enabled for a
    context pattern."""

    # General
    adaptive_profile: Literal["dev", "production"] = "production"
    """Profile that selects sensible defaults for warmup
    parameters. 'dev' learns faster; 'production' is more
    conservative."""

    self_spec_stats_persist_path: str | None = None
    """Path for persisting learned prefix frequencies and
    confidence thresholds across restarts. If None, state is
    not persisted."""

    persist_interval_seconds: float = Field(default=300.0, gt=0.0)
    """Interval (seconds) between periodic persistence writes."""

    # Memory safety
    warmup_pause_threshold: float = Field(default=0.9, gt=0.0, le=1.0)
    """KV cache usage fraction at which warmup is paused."""

    warmup_resume_threshold: float = Field(default=0.8, gt=0.0, lt=1.0)
    """KV cache usage fraction below which warmup resumes after
    being paused."""

    high_load_queue_depth: int = Field(default=10, gt=0)
    """Queue depth threshold for detecting sustained high load."""

    high_load_duration_seconds: float = Field(default=5.0, gt=0.0)
    """Duration (seconds) that queue depth must exceed the
    threshold before warmup is disabled."""

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> "AdaptiveServingConfig":
        """Fill None-valued warmup parameters from profile."""
        defaults = PROFILE_DEFAULTS[self.adaptive_profile]
        if self.warmup_ema_decay is None:
            self.warmup_ema_decay = defaults["warmup_ema_decay"]
        if self.warmup_budget_ms is None:
            self.warmup_budget_ms = defaults["warmup_budget_ms"]
        if self.warmup_vram_budget_ratio is None:
            self.warmup_vram_budget_ratio = defaults["warmup_vram_budget_ratio"]
        if self.warmup_min_hit_count is None:
            self.warmup_min_hit_count = defaults["warmup_min_hit_count"]
        return self

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "AdaptiveServingConfig":
        """Validate that pause > resume and activation > min."""
        if self.warmup_pause_threshold <= self.warmup_resume_threshold:
            raise ValueError(
                "warmup_pause_threshold must be greater than warmup_resume_threshold"
            )
        if self.self_spec_activation_hit_rate <= self.self_spec_min_hit_rate:
            raise ValueError(
                "self_spec_activation_hit_rate must be greater "
                "than self_spec_min_hit_rate"
            )
        return self
