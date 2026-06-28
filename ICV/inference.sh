#!/usr/bin/env bash
# Self-contained ICV inference. Creates the venv on first run, sets the HF cache,
# then runs the in-context-vector smoke test. Re-runnable on any machine.
#
#   ./inference.sh                 # both demos (sentiment + safety), falcon-7b 8-bit
#   ./inference.sh --demo sentiment
#   ./inference.sh --gpus 0,1      # spread across GPUs if a single card OOMs
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) HuggingFace cache location (edit ../config.sh to change where models land).
source "$HERE/../config.sh"

# 2) First-run bootstrap of the virtualenv.
if [ ! -d "$HERE/.venv" ]; then
  echo "[inference] creating venv + installing requirements (first run only)..."
  python3 -m venv "$HERE/.venv"
  "$HERE/.venv/bin/python" -m pip install -q -U pip wheel
  "$HERE/.venv/bin/python" -m pip install -q -r "$HERE/requirements.txt"
fi

# 3) Activate and run.
source "$HERE/.venv/bin/activate"
cd "$HERE"
exec python infer.py "$@"
