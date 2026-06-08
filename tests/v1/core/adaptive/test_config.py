# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AdaptiveServingConfig profile resolution and defaults."""

import pytest
from pydantic import ValidationError

from vllm.config.adaptive_serving import (
    PROFILE_DEFAULTS,
    AdaptiveServingConfig,
)


class TestDevProfileDefaults:
    """Test that dev profile defaults are correctly applied.

    Requirements: 7.7
    """

    def test_dev_profile_fills_warmup_ema_decay(self):
        cfg = AdaptiveServingConfig(adaptive_profile="dev")
        assert cfg.warmup_ema_decay == 0.8

    def test_dev_profile_fills_warmup_budget_ms(self):
        cfg = AdaptiveServingConfig(adaptive_profile="dev")
        assert cfg.warmup_budget_ms == 200.0

    def test_dev_profile_fills_warmup_vram_budget_ratio(self):
        cfg = AdaptiveServingConfig(adaptive_profile="dev")
        assert cfg.warmup_vram_budget_ratio == 0.5

    def test_dev_profile_fills_warmup_min_hit_count(self):
        cfg = AdaptiveServingConfig(adaptive_profile="dev")
        assert cfg.warmup_min_hit_count == 3.0

    def test_dev_profile_matches_profile_defaults_dict(self):
        cfg = AdaptiveServingConfig(adaptive_profile="dev")
        for key, value in PROFILE_DEFAULTS["dev"].items():
            assert getattr(cfg, key) == value


class TestProductionProfileDefaults:
    """Test that production profile defaults are correctly applied.

    Requirements: 7.8
    """

    def test_production_profile_fills_warmup_ema_decay(self):
        cfg = AdaptiveServingConfig(adaptive_profile="production")
        assert cfg.warmup_ema_decay == 0.95

    def test_production_profile_fills_warmup_budget_ms(self):
        cfg = AdaptiveServingConfig(adaptive_profile="production")
        assert cfg.warmup_budget_ms == 100.0

    def test_production_profile_fills_warmup_vram_budget_ratio(self):
        cfg = AdaptiveServingConfig(adaptive_profile="production")
        assert cfg.warmup_vram_budget_ratio == 0.3

    def test_production_profile_fills_warmup_min_hit_count(self):
        cfg = AdaptiveServingConfig(adaptive_profile="production")
        assert cfg.warmup_min_hit_count == 20.0

    def test_production_is_default_profile(self):
        cfg = AdaptiveServingConfig()
        assert cfg.adaptive_profile == "production"
        assert cfg.warmup_ema_decay == 0.95

    def test_production_profile_matches_profile_defaults_dict(self):
        cfg = AdaptiveServingConfig(adaptive_profile="production")
        for key, value in PROFILE_DEFAULTS["production"].items():
            assert getattr(cfg, key) == value


class TestExplicitOverrides:
    """Test that explicit parameter overrides take precedence.

    Requirements: 7.7, 7.8
    """

    def test_explicit_ema_decay_overrides_dev_profile(self):
        cfg = AdaptiveServingConfig(adaptive_profile="dev", warmup_ema_decay=0.5)
        assert cfg.warmup_ema_decay == 0.5

    def test_explicit_budget_ms_overrides_production_profile(self):
        cfg = AdaptiveServingConfig(
            adaptive_profile="production", warmup_budget_ms=50.0
        )
        assert cfg.warmup_budget_ms == 50.0

    def test_explicit_vram_ratio_overrides_profile(self):
        cfg = AdaptiveServingConfig(
            adaptive_profile="dev", warmup_vram_budget_ratio=0.1
        )
        assert cfg.warmup_vram_budget_ratio == 0.1

    def test_explicit_min_hit_count_overrides_profile(self):
        cfg = AdaptiveServingConfig(
            adaptive_profile="production", warmup_min_hit_count=5.0
        )
        assert cfg.warmup_min_hit_count == 5.0

    def test_partial_override_fills_remaining_from_profile(self):
        cfg = AdaptiveServingConfig(adaptive_profile="dev", warmup_ema_decay=0.5)
        # Explicitly set
        assert cfg.warmup_ema_decay == 0.5
        # Filled from dev profile
        assert cfg.warmup_budget_ms == 200.0
        assert cfg.warmup_vram_budget_ratio == 0.5
        assert cfg.warmup_min_hit_count == 3.0


class TestAutoEnablePrefixCaching:
    """Test auto-enable prefix caching behavior at VllmConfig level.

    Requirements: 7.9
    """

    def test_prefix_caching_auto_enabled_when_warmup_active(self):
        """When adaptive warmup is enabled but prefix caching is off,
        VllmConfig should auto-enable prefix caching."""
        from vllm.config import CacheConfig, VllmConfig

        cache_cfg = CacheConfig(enable_prefix_caching=False)
        adaptive_cfg = AdaptiveServingConfig(enable_adaptive_warmup=True)
        vllm_cfg = VllmConfig(
            cache_config=cache_cfg,
            adaptive_serving=adaptive_cfg,
        )
        assert vllm_cfg.cache_config.enable_prefix_caching is True

    def test_prefix_caching_unchanged_when_warmup_disabled(self):
        """When adaptive warmup is disabled, prefix caching is not
        modified."""
        from vllm.config import CacheConfig, VllmConfig

        cache_cfg = CacheConfig(enable_prefix_caching=False)
        adaptive_cfg = AdaptiveServingConfig(enable_adaptive_warmup=False)
        vllm_cfg = VllmConfig(
            cache_config=cache_cfg,
            adaptive_serving=adaptive_cfg,
        )
        assert vllm_cfg.cache_config.enable_prefix_caching is False

    def test_prefix_caching_already_enabled_stays_enabled(self):
        """When prefix caching is already enabled, it stays enabled
        regardless of adaptive warmup."""
        from vllm.config import CacheConfig, VllmConfig

        cache_cfg = CacheConfig(enable_prefix_caching=True)
        adaptive_cfg = AdaptiveServingConfig(enable_adaptive_warmup=True)
        vllm_cfg = VllmConfig(
            cache_config=cache_cfg,
            adaptive_serving=adaptive_cfg,
        )
        assert vllm_cfg.cache_config.enable_prefix_caching is True


class TestValidation:
    """Test validation constraints on AdaptiveServingConfig."""

    def test_invalid_profile_rejected(self):
        with pytest.raises(ValidationError):
            AdaptiveServingConfig(adaptive_profile="invalid")

    def test_pause_must_exceed_resume_threshold(self):
        with pytest.raises(ValueError):
            AdaptiveServingConfig(
                warmup_pause_threshold=0.8,
                warmup_resume_threshold=0.8,
            )

    def test_activation_must_exceed_min_hit_rate(self):
        with pytest.raises(ValueError):
            AdaptiveServingConfig(
                self_spec_activation_hit_rate=0.5,
                self_spec_min_hit_rate=0.5,
            )
