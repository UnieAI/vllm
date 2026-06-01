#!/usr/bin/env bash
# Port the large Qualcomm-proprietary QAIC files from the v1_ngram fork into the
# vllm-qaic package, and rewrite intra-fork imports to the package namespace.
#
# WHY a script (not files committed here): these three files are ~2400 lines of
# Qualcomm "Confidential and Proprietary" code that we cannot execute/verify off
# the AIC machine. You already have them in your fork checkout — copying them is
# mechanical, so we make it reproducible instead of pasting unverifiable code.
#
# Usage:
#   ./scripts/port_from_fork.sh /path/to/UnieAI-vllm-checkout
#
# After running, do the two MANUAL adaptations the script flags at the end.

set -euo pipefail

FORK="${1:?usage: port_from_fork.sh <path-to-fork-checkout>}"
PKG="$(cd "$(dirname "$0")/.." && pwd)/vllm_qaic"
ML="$FORK/vllm/model_executor/model_loader"

declare -A MAP=(
  ["$ML/qaic_session_np.py"]="$PKG/session.py"        # BSD-3-Clause, pure numpy+qaicrt
  ["$ML/qaic_v1.py"]="$PKG/model_loader.py"           # QaicCausalLM + load_qaic_model
  ["$ML/qaic.py"]="$PKG/compile_config.py"            # _clean_config + QEfficient compile cfg
)

for src in "${!MAP[@]}"; do
  dst="${MAP[$src]}"
  [ -f "$src" ] || { echo "MISSING: $src"; exit 1; }
  cp "$src" "$dst"
  # Rewrite intra-fork qaic imports -> package namespace.
  sed -i.bak \
    -e 's#from vllm\.model_executor\.model_loader\.qaic_session_np import#from vllm_qaic.session import#g' \
    -e 's#from vllm\.model_executor\.model_loader\.qaic_v1 import#from vllm_qaic.model_loader import#g' \
    -e 's#from vllm\.model_executor\.model_loader\.qaic import#from vllm_qaic.compile_config import#g' \
    "$dst"
  rm -f "$dst.bak"
  echo "ported: $src -> $dst"
done

cat <<'EOF'

DONE (mechanical). Now the TWO manual adaptations (search & fix):

  1) QAIC config source:
     OLD: vllm_config.model_config.override_qaic_config
     NEW: vllm_config.additional_config
     (the plugin moves QAIC knobs onto --additional-config; see platform.py)

  2) Platform check:
     OLD: current_platform.is_qaic()
     NEW: current_platform.device_type == "qaic"
     (the 0.21 OOT platform has no is_qaic())

  Then smoke-import on the AIC machine (torch 2.11 env, qaicrt on path):
     python -c "import vllm_qaic.session, vllm_qaic.compile_config, vllm_qaic.model_loader; print('import OK')"
EOF
