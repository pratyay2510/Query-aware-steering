from pathlib import Path
import os

root = Path(__file__).parent
data_root = root.joinpath("data")
inference_root = root.joinpath("inference")

logger_root = root.joinpath("logger")
dump_root = root.joinpath("dump")

# HuggingFace cache location. Honors the HF_CACHE_DIR env var (set by ../config.sh)
# so the cache can live on any disk; falls back to the standard HF cache otherwise.
checkpoints_root = Path(
    os.environ.get("HF_CACHE_DIR", str(Path.home() / ".cache" / "huggingface"))
)

hf_datasets_root = root.joinpath("datasets")
