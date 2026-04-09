#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config="$SCRIPT_DIR/../configs/exp_name.yaml"
RAYLOG="$SCRIPT_DIR/../raylog"
test_script="$SCRIPT_DIR/../testings/test.py"

echo "$config"

# ── Conda init (must happen before the loop, once) ──────────────────────────
# Find conda base: prefer $CONDA_EXE (set when conda is active), then search PATH
if command -v conda &>/dev/null; then
  CONDA_BASE="$(conda info --base)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  CONDA_BASE="$HOME/anaconda3"
else
  echo "ERROR: conda not found. Install conda or update CONDA_BASE manually." >&2
  exit 1
fi

export PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ray210           # ✅ Activate once, outside the loop

cd "$SCRIPT_DIR/../testings"    # ✅ Change directory once, outside the loop

# ── Checkpoints ─────────────────────────────────────────────────────────────
#Place checkpoints here : 1 per line
checkpoints=(
  "local-debug_env000-debug/_trainable_flexenv_893b3_00000_0_2026-04-07_16-14-56/checkpoint_000000"
)



# ── Loop ─────────────────────────────────────────────────────────────
for cp in "${checkpoints[@]}"; do
  cp_abs="${RAYLOG}/${cp}"
  echo "Loading checkpoint: $cp_abs"

  # Validate checkpoint path exists before updating config
  if [ ! -e "$cp_abs" ]; then
    echo "WARNING: checkpoint path does not exist: $cp_abs" >&2
    continue
  fi

  yq eval ".checkpoint = \"$cp_abs\"" -i "$config"

  python "$test_script"
done
