# Shared configuration for the Query-aware-steering workspace.
# Sourced by ICV/inference.sh and LIVE/inference.sh.
#
# ┌──────────────────────────────────────────────────────────────────────┐
# │ EDIT ME: where HuggingFace models & datasets get cached.              │
# │ Keep this OFF the home disk (home/'/' is often near-full).            │
# │ You can point this at ANY disk on ANY machine.                        │
# └──────────────────────────────────────────────────────────────────────┘
export HF_CACHE_DIR="${HF_CACHE_DIR:-/datastarnas01/hf_cache}"

# Everything below is derived from HF_CACHE_DIR — no need to edit.
export HF_HOME="$HF_CACHE_DIR"                       # hub cache -> $HF_HOME/hub, datasets -> $HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR/hub"
# ICV reads HF_CACHE_DIR directly for its own checkpoint cache (see ICV/anchor.py).
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p "$HUGGINGFACE_HUB_CACHE"
echo "[config] HF cache -> $HF_CACHE_DIR"
