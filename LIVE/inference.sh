#!/usr/bin/env bash
# LIVE inference launcher.
#
# STATUS: DEFERRED. A real LIVE inference run needs three things this workspace
# does not set up automatically:
#   1. a large multimodal model (IDEFICS-9B / IDEFICS2-8B, ~16-18 GB),
#   2. the COCO + VQAv2/OKVQA datasets, and
#   3. a TRAINED ICV checkpoint (the authors publish none -> you must run train.py first).
#
# This script bootstraps the minimal env, confirms it is healthy, then prints the
# exact steps to run the full pipeline. It intentionally does NOT pretend to infer.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1) HuggingFace cache location (shared with the rest of the workspace).
source "$HERE/../config.sh"

# 2) First-run bootstrap of the minimal virtualenv.
if [ ! -d "$HERE/.venv" ]; then
  echo "[inference] creating venv + installing minimal requirements (first run only)..."
  python3 -m venv "$HERE/.venv"
  "$HERE/.venv/bin/python" -m pip install -q -U pip wheel
  "$HERE/.venv/bin/python" -m pip install -q -r "$HERE/requirements.txt"
fi
source "$HERE/.venv/bin/activate"

# 3) Environment sanity check.
echo "[inference] environment check:"
python - <<'PY'
import torch, transformers, hydra, omegaconf
print(f"  torch        {torch.__version__}  (cuda={torch.cuda.is_available()}, gpus={torch.cuda.device_count()})")
print(f"  transformers {transformers.__version__}")
print(f"  hydra-core   {hydra.__version__}")
print("  env OK")
PY

# 4) If a real run is requested, check prerequisites and forward to inference.py.
if [ "${1:-}" = "--run" ]; then
  shift
  [ -f "$HERE/.env" ] || { echo "ERROR: copy .env.example -> .env and fill in absolute paths first."; exit 1; }
  echo "[inference] launching: python inference.py $*"
  exec python "$HERE/inference.py" "$@"
fi

cat <<'EOF'

LIVE inference is deferred. To run the full pipeline:

  # 1. Full deps (heavy: deepspeed, faiss-gpu, lightning, ...)
  pip install -r requirements-full.txt
  pip install transformers==4.28.1        # only if using OpenFlamingo
  pip install git+https://github.com/ForJadeForest/lmm_icl_interface.git
  pip install git+https://github.com/davidbau/baukit.git

  # 2. Download a model into the HF cache, e.g.
  huggingface-cli download HuggingFaceM4/idefics2-8b-base

  # 3. Prepare datasets (COCO + VQAv2/OKVQA) and set paths:
  cp .env.example .env   # then edit RESULT_DIR / VQAV2_PATH / OKVQA_PATH / COCO_PATH / MODEL_CPK_DIR

  # 4. Train an ICV (produces icv_cpk.pth), then run inference:
  python train.py     run_name="okvqa_idefics2_icv" data_cfg/task/datasets=ok_vqa lmm=idefics2-8B-base ...
  ./inference.sh --run run_name="okvqa_idefics2_icv" data_cfg/task/datasets=ok_vqa lmm=idefics2-8B-base

See README.md (this folder) for the full command options.
EOF
