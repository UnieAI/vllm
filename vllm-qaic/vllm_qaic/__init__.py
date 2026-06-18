"""vllm-qaic: out-of-tree vLLM platform plugin for Qualcomm Cloud AI 100.

Two entry points are exposed (see pyproject.toml):

  * ``register_platform``  -> group ``vllm.platform_plugins``
        Called by vLLM during platform resolution. Must return the fully
        qualified class name of our Platform, or ``None`` if QAIC is not
        available on this host (so vLLM falls back to CPU/CUDA/etc.).

  * ``register``           -> group ``vllm.general_plugins``
        Executed by vLLM at startup. Used for out-of-tree registration that
        does NOT require patching vLLM core: quantization config, KV
        connector, custom models, etc.
"""

import sys
import platform


def _qaic_available() -> bool:
    """True iff the Cloud AI SDK runtime (qaicrt) can be imported.

    Mirrors the detection the fork did in platforms/__init__.py, minus the
    ``"qaic" in version("vllm")`` hack (the plugin entry point replaces it).
    """
    def _add_path(p: str) -> None:  # avoid duplicate sys.path entries
        if p not in sys.path:
            sys.path.append(p)

    try:
        try:
            import qaicrt  # noqa: F401
        except ImportError:
            _add_path(f"/opt/qti-aic/dev/lib/{platform.machine()}")
            import qaicrt  # noqa: F401
        try:
            import QAicApi_pb2  # noqa: F401
        except ImportError:
            _add_path("/opt/qti-aic/dev/python")
            import QAicApi_pb2  # noqa: F401
        return True
    except Exception:
        return False


def register_platform():
    """vllm.platform_plugins entry point."""
    if _qaic_available():
        return "vllm_qaic.platform.QaicPlatform"
    return None


def register():
    """vllm.general_plugins entry point. Runs at startup in every process."""
    # Import lazily so a non-QAIC host that happens to have this package
    # installed does not blow up at import time.
    if not _qaic_available():
        return
    from vllm_qaic.quant import register_qaic_quant
    register_qaic_quant()
    # QAIC Mooncake bridge: a host-staging subclass of the stock MooncakeStoreConnector
    # (cross-instance KV reuse). Lazy registration; only imported when selected via
    # --kv-transfer-config '{"kv_connector":"QaicMooncakeStoreConnector",...}'.
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import (
            KVConnectorFactory,
        )

        KVConnectorFactory.register_connector(
            "QaicMooncakeStoreConnector",
            "vllm_qaic.kv_connector.qaic_mooncake_store_connector",
            "QaicMooncakeStoreConnector",
        )
    except Exception:  # pragma: no cover - core may not expose the factory
        pass
