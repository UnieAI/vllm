"""mxfp6 quantization registration (out-of-tree, no core patch).

The fork added "mxfp6" -> QaicQuantConfig by editing
vllm/model_executor/layers/quantization/__init__.py. We do it from outside
instead, via the public register_quantization_config() decorator
(verified at vllm/model_executor/layers/quantization/__init__.py:59).

PORTING: copy QaicQuantConfig from
    vllm/model_executor/layers/quantization/qaic_quant.py  (68 lines)
into this file, then keep the register_qaic_quant() wrapper below.
"""

from vllm.model_executor.layers.quantization import register_quantization_config


def register_qaic_quant() -> None:
    # TODO(port): paste QaicQuantConfig here and register it, e.g.:
    #
    #   from vllm.model_executor.layers.quantization.base_config import (
    #       QuantizationConfig)
    #
    #   @register_quantization_config("mxfp6")
    #   class QaicQuantConfig(QuantizationConfig):
    #       ...   # body from qaic_quant.py
    #
    # Registration must be idempotent (general_plugins can run per-process).
    pass
