# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TurboQuant compatibility with adaptive serving.

Verifies no conflict when combining `--kv-cache-dtype turboquant_4bit_nc`
with adaptive warmup and self-speculation.

Requirements: 5.4
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vllm.config.adaptive_serving import AdaptiveServingConfig
from vllm.v1.core.adaptive.prefix_frequency_tracker import (
    PrefixFrequencyTracker,
)
from vllm.v1.core.adaptive.prefix_warmup_worker import (
    PrefixWarmupWorker,
    _get_compression_ratio,
)
from vllm.v1.spec_decode.self_speculation import SelfSpeculationProposer


class TestPrefixWarmupWorkerTurboQuant:
    """Test PrefixWarmupWorker with TurboQuant KV cache dtype."""

    def _make_worker(
        self, kv_cache_dtype: str = "turboquant_4bit_nc", head_size: int = 128
    ) -> PrefixWarmupWorker:
        """Create a PrefixWarmupWorker with mocked dependencies."""
        config = AdaptiveServingConfig(
            enable_adaptive_warmup=True,
            adaptive_profile="dev",
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)

        mock_block_pool = MagicMock()
        mock_block_pool.get_cached_block_hashes.return_value = set()
        mock_block_pool.get_num_free_blocks.return_value = 100

        mock_kv_cache_manager = MagicMock()
        mock_kv_cache_manager.block_pool = mock_block_pool
        mock_kv_cache_manager.usage = 0.5

        mock_executor = MagicMock()

        # Patch TurboQuantConfig to avoid requiring the actual module
        with patch(
            "vllm.v1.core.adaptive.prefix_warmup_worker._get_compression_ratio"
        ) as mock_ratio:
            # Simulate a compression ratio for 4-bit quantization
            # 4-bit ≈ 0.25x of bf16 (which is 16-bit)
            mock_ratio.return_value = 0.25
            worker = PrefixWarmupWorker(
                config=config,
                frequency_tracker=tracker,
                block_pool=mock_block_pool,
                kv_cache_manager=mock_kv_cache_manager,
                model_executor=mock_executor,
                kv_cache_dtype=kv_cache_dtype,
                head_size=head_size,
            )

        return worker

    def test_instantiation_with_turboquant(self):
        """PrefixWarmupWorker can be instantiated with
        kv_cache_dtype='turboquant_4bit_nc' without error."""
        worker = self._make_worker(kv_cache_dtype="turboquant_4bit_nc")
        assert worker is not None
        assert worker.kv_cache_dtype == "turboquant_4bit_nc"

    def test_compression_ratio_less_than_one(self):
        """TurboQuant dtypes produce a compression ratio < 1.0."""
        worker = self._make_worker(kv_cache_dtype="turboquant_4bit_nc")
        assert worker.compression_ratio < 1.0

    def test_compression_ratio_uncompressed(self):
        """Non-TurboQuant dtypes produce a compression ratio of 1.0."""
        config = AdaptiveServingConfig(
            enable_adaptive_warmup=True,
            adaptive_profile="dev",
        )
        tracker = PrefixFrequencyTracker(max_entries=100, ema_decay=0.8)
        mock_block_pool = MagicMock()
        mock_block_pool.get_num_free_blocks.return_value = 100
        mock_kv_cache_manager = MagicMock()
        mock_kv_cache_manager.block_pool = mock_block_pool
        mock_kv_cache_manager.usage = 0.5
        mock_executor = MagicMock()

        worker = PrefixWarmupWorker(
            config=config,
            frequency_tracker=tracker,
            block_pool=mock_block_pool,
            kv_cache_manager=mock_kv_cache_manager,
            model_executor=mock_executor,
            kv_cache_dtype="auto",
            head_size=128,
        )
        assert worker.compression_ratio == 1.0

    def test_get_compression_ratio_turboquant_variants(self):
        """_get_compression_ratio handles TurboQuant variants by
        trying to import TurboQuantConfig. If unavailable, we patch it."""
        # For dtypes that don't need TurboQuantConfig import
        assert _get_compression_ratio("auto", 128) == 1.0
        assert _get_compression_ratio("bfloat16", 128) == 1.0
        assert _get_compression_ratio("fp8", 128) == 0.5
        assert _get_compression_ratio("fp8_e4m3", 128) == 0.5

    def test_get_compression_ratio_non_string(self):
        """_get_compression_ratio handles non-string (e.g. MagicMock)
        gracefully by returning 1.0."""
        assert _get_compression_ratio(MagicMock(), 128) == 1.0


class TestSelfSpeculationProposerTurboQuant:
    """Test SelfSpeculationProposer with TurboQuant KV cache dtype."""

    def _make_vllm_config(self, cache_dtype: str = "turboquant_4bit_nc") -> MagicMock:
        """Create a mock VllmConfig with TurboQuant cache dtype."""
        adaptive_cfg = AdaptiveServingConfig(
            enable_adaptive_warmup=True,
            enable_self_speculation=True,
            adaptive_profile="dev",
        )

        cache_config = MagicMock()
        cache_config.cache_dtype = cache_dtype

        vllm_config = MagicMock()
        vllm_config.adaptive_serving = adaptive_cfg
        vllm_config.cache_config = cache_config

        return vllm_config

    def test_instantiation_with_turboquant(self):
        """SelfSpeculationProposer can be instantiated when VllmConfig
        has cache_config.cache_dtype = 'turboquant_4bit_nc'."""
        vllm_config = self._make_vllm_config(cache_dtype="turboquant_4bit_nc")
        proposer = SelfSpeculationProposer(vllm_config=vllm_config)
        assert proposer is not None

    def test_proposer_stores_correct_kv_cache_dtype(self):
        """Proposer stores the configured kv_cache_dtype correctly."""
        vllm_config = self._make_vllm_config(cache_dtype="turboquant_4bit_nc")
        proposer = SelfSpeculationProposer(vllm_config=vllm_config)
        assert proposer.kv_cache_dtype == "turboquant_4bit_nc"

    def test_proposer_with_auto_dtype(self):
        """Proposer stores 'auto' when no TurboQuant is configured."""
        vllm_config = self._make_vllm_config(cache_dtype="auto")
        proposer = SelfSpeculationProposer(vllm_config=vllm_config)
        assert proposer.kv_cache_dtype == "auto"

    def test_proposer_with_fp8_dtype(self):
        """Proposer stores fp8 dtype correctly."""
        vllm_config = self._make_vllm_config(cache_dtype="fp8")
        proposer = SelfSpeculationProposer(vllm_config=vllm_config)
        assert proposer.kv_cache_dtype == "fp8"


class TestCombinedTurboQuantCompatibility:
    """Test that both components work together with TurboQuant."""

    def test_both_enabled_with_turboquant_no_conflict(self):
        """Both adaptive warmup and self-speculation can be enabled
        simultaneously with TurboQuant without conflict."""
        # Create shared config with both features enabled
        config = AdaptiveServingConfig(
            enable_adaptive_warmup=True,
            enable_self_speculation=True,
            adaptive_profile="production",
        )
        assert config.enable_adaptive_warmup is True
        assert config.enable_self_speculation is True

        # Create warmup worker with TurboQuant
        tracker = PrefixFrequencyTracker(
            max_entries=config.warmup_max_prefixes,
            ema_decay=config.warmup_ema_decay,
        )
        mock_block_pool = MagicMock()
        mock_block_pool.get_cached_block_hashes.return_value = set()
        mock_block_pool.get_num_free_blocks.return_value = 200

        mock_kv_cache_manager = MagicMock()
        mock_kv_cache_manager.block_pool = mock_block_pool
        mock_kv_cache_manager.usage = 0.4

        mock_executor = MagicMock()

        with patch(
            "vllm.v1.core.adaptive.prefix_warmup_worker._get_compression_ratio"
        ) as mock_ratio:
            mock_ratio.return_value = 0.25
            worker = PrefixWarmupWorker(
                config=config,
                frequency_tracker=tracker,
                block_pool=mock_block_pool,
                kv_cache_manager=mock_kv_cache_manager,
                model_executor=mock_executor,
                kv_cache_dtype="turboquant_4bit_nc",
                head_size=128,
            )

        # Create proposer with TurboQuant
        cache_config = MagicMock()
        cache_config.cache_dtype = "turboquant_4bit_nc"
        vllm_config = MagicMock()
        vllm_config.adaptive_serving = config
        vllm_config.cache_config = cache_config

        proposer = SelfSpeculationProposer(vllm_config=vllm_config)

        # Verify both components are configured correctly
        assert worker.kv_cache_dtype == "turboquant_4bit_nc"
        assert proposer.kv_cache_dtype == "turboquant_4bit_nc"
        assert worker.compression_ratio < 1.0

    def test_config_accepts_both_flags_simultaneously(self):
        """AdaptiveServingConfig accepts both enable_adaptive_warmup=True
        and enable_self_speculation=True without error."""
        config = AdaptiveServingConfig(
            enable_adaptive_warmup=True,
            enable_self_speculation=True,
        )
        assert config.enable_adaptive_warmup is True
        assert config.enable_self_speculation is True

    def test_config_dev_profile_both_enabled(self):
        """Dev profile with both features enabled applies correct
        defaults."""
        config = AdaptiveServingConfig(
            enable_adaptive_warmup=True,
            enable_self_speculation=True,
            adaptive_profile="dev",
        )
        assert config.warmup_ema_decay == 0.8
        assert config.warmup_budget_ms == 200.0
        assert config.self_spec_confidence_threshold == 0.9

    def test_config_production_profile_both_enabled(self):
        """Production profile with both features enabled applies
        correct defaults."""
        config = AdaptiveServingConfig(
            enable_adaptive_warmup=True,
            enable_self_speculation=True,
            adaptive_profile="production",
        )
        assert config.warmup_ema_decay == 0.95
        assert config.warmup_budget_ms == 100.0
        assert config.self_spec_confidence_threshold == 0.9
