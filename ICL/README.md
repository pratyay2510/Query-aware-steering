# ICL Demo

This directory contains a Hugging Face in-context learning (ICL) demo for math problem solving on the MATH dataset. The default model is `Qwen/Qwen2.5-7B-Instruct`, and the default dataset is `EleutherAI/hendrycks_math`.

Use the terminal commands below as the primary workflow. The notebook is kept for interactive debugging.

## Directory Structure

```text
ICL/
├── ICL-demo.py                            # Main terminal script and reusable notebook functions
├── ICL-demo.ipynb                         # Two-cell notebook for debugging no-ICL and ICL runs
├── visualize_icl_steering_embeddings.py   # Final-layer hidden-state extraction and UMAP plots
├── README.md                              # This runbook
└── requirements.txt                       # Python dependencies
```

External locations used by default:

```text
/p/vast1/dutta5/envs/ICL-venv       # Python environment
/p/vast1/dutta5/cache-dirs          # Model/tokenizer cache
/p/vast1/dutta5/datasets            # Dataset cache
/p/vast1/dutta5/embeddings          # Saved hidden states and UMAP plots
```

For Qwen 2.5 7B, model files are cached under:

```text
/p/vast1/dutta5/cache-dirs/models/Qwen__Qwen2.5-7B-Instruct
```

For the MATH algebra split, dataset files are cached under:

```text
/p/vast1/dutta5/datasets/EleutherAI__hendrycks_math/algebra
```

The MATH algebra test split has 1187 examples. The script also creates deterministic 400-example test chunks under:

```text
/p/vast1/dutta5/datasets/EleutherAI__hendrycks_math/algebra/test_chunks
```

Current chunk files:

```text
test_chunks/
├── 0.jsonl        # test examples 0-399
├── 1.jsonl        # test examples 400-799
├── 2.jsonl        # test examples 800-1186
└── manifest.json  # chunk metadata
```

## What The Script Does

`ICL-demo.py` has two modes:

- `--mode no_icl`: normal inference. Each MATH test problem is sent to the model without examples.
- `--mode oracle`: ICL/oracle experiment. Each test problem is evaluated with no ICL and with multiple sampled ICL contexts.

For each evaluated problem, the script:

1. Loads the problem and gold solution.
2. Extracts the gold answer, preferring LaTeX `\boxed{...}`.
3. Builds either a no-ICL prompt or an ICL prompt.
4. Runs deterministic generation with `do_sample=False`.
5. Extracts the model answer.
6. Compares normalized prediction vs. normalized gold answer.
7. Optionally saves detailed generations and summary metrics.

The oracle setting is an analysis baseline. It answers: “If one of the sampled ICL contexts worked, how much could performance improve?” It is not deployable as-is because it uses correctness after generation to identify the best context.

## Setup

Run from the ICL directory:

```bash
cd /usr/WS2/dutta5/Projects/Query-aware-steering/ICL
source /p/vast1/dutta5/envs/ICL-venv/bin/activate
python -m pip install -r requirements.txt
```

If the model has not been downloaded before, the first real inference run downloads model weight shards into `/p/vast1/dutta5/cache-dirs/models/...`. Qwen 2.5 7B, Llama 3B, and Llama 8B runs need a GPU node with enough memory for practical inference.

The embedding visualization script requires `umap-learn`, which is included in `requirements.txt`. On this cluster, UMAP may need a writable Numba cache. The script sets `NUMBA_CACHE_DIR=/tmp/numba-codex-cache` by default.

## Smoke Test

This verifies imports, CLI, notebook shape, tokenizer/config access, and dataset cache location. It does not download full 7B model weights and does not need a GPU.

```bash
cd /usr/WS2/dutta5/Projects/Query-aware-steering/ICL
source /p/vast1/dutta5/envs/ICL-venv/bin/activate

python -m py_compile ICL-demo.py visualize_icl_steering_embeddings.py
python ICL-demo.py --help
python visualize_icl_steering_embeddings.py --help

python - <<'PY'
import nbformat
nb = nbformat.read("ICL-demo.ipynb", as_version=4)
assert len(nb.cells) == 2
assert all(cell.cell_type == "code" for cell in nb.cells)
print("notebook_cells:", len(nb.cells))
PY

python - <<'PY'
from pathlib import Path
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

model = "Qwen/Qwen2.5-7B-Instruct"
model_cache = Path("/p/vast1/dutta5/cache-dirs/models/Qwen__Qwen2.5-7B-Instruct")
dataset_cache = Path("/p/vast1/dutta5/datasets/EleutherAI__hendrycks_math/algebra")

model_cache.mkdir(parents=True, exist_ok=True)
dataset_cache.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(model, cache_dir=str(model_cache))
cfg = AutoConfig.from_pretrained(model, cache_dir=str(model_cache))
ds = load_dataset("EleutherAI/hendrycks_math", "algebra", cache_dir=str(dataset_cache))

print("tokenizer:", tok.__class__.__name__)
print("model_type:", cfg.model_type)
print("train_size:", len(ds["train"]))
print("test_size:", len(ds["test"]))
print("dataset_cache:", dataset_cache)
PY

python - <<'PY'
from pathlib import Path
import json

chunk_dir = Path("/p/vast1/dutta5/datasets/EleutherAI__hendrycks_math/algebra/test_chunks")
manifest = json.loads((chunk_dir / "manifest.json").read_text())
print("chunk_size:", manifest["chunk_size"])
print("num_chunks:", manifest["num_chunks"])
print("chunk_counts:", [c["num_examples"] for c in manifest["chunks"]])
PY
```

Expected output includes:

```text
notebook_cells: 2
tokenizer: Qwen2TokenizerFast
model_type: qwen2
train_size: 1744
test_size: 1187
dataset_cache: /p/vast1/dutta5/datasets/EleutherAI__hendrycks_math/algebra
chunk_size: 400
num_chunks: 3
chunk_counts: [400, 400, 387]
```

Additional smoke tests for the visualization script:

```bash
# This should fail fast on a non-GPU node instead of silently loading Llama on CPU.
python visualize_icl_steering_embeddings.py \
  --models llama3.2-3b \
  --device cuda \
  --load-only \
  --test-scope single \
  --example-id 0 \
  --max-examples 1 \
  --skip-umap

# On a GPU node, this checks tokenizer/model/dataset loading without extracting vectors.
python visualize_icl_steering_embeddings.py \
  --models llama3.2-3b \
  --device auto \
  --load-only \
  --test-scope single \
  --example-id 0 \
  --max-examples 1 \
  --skip-umap
```

## Result Files

Result writing is enabled by default. Disable it with `--no-save-results`.

For `Qwen/Qwen2.5-7B-Instruct`, output filenames use the prefix `qwen2.57B`:

```text
qwen2.57B_no_icl.csv
qwen2.57B_no_icl.parquet
qwen2.57B_no_icl_summary.csv
qwen2.57B_split0_no_icl.csv
qwen2.57B_split0_no_icl.parquet
qwen2.57B_split0_no_icl_summary.csv
qwen2.57B_fixed_contexts.csv
qwen2.57B_oracle_context.csv
qwen2.57B_oracle_context.parquet
qwen2.57B_success_matrix.csv
qwen2.57B_oracle_summary.csv
```

Full-split runs use the plain prefix, such as `qwen2.57B_no_icl.csv`. Chunked subset runs include the split index, such as `qwen2.57B_split1_no_icl.csv`. Single-example runs include the example id, such as `qwen2.57B_example25_no_icl.csv`.

CSV is convenient for quick inspection. Parquet is more compact for raw generation tables. Use `--output-dir <dir>` to write results somewhere other than the current directory.

## Test Scope

Use `--test-scope` to choose how much of the test set to run:

- `single`: run one test example selected by `--example-id`.
- `subset`: run one deterministic 400-example chunk selected by `--split-index`.
- `full`: run the entire selected MATH test split.

For MATH algebra, valid split indices are:

```text
--split-index 0  # examples 0-399
--split-index 1  # examples 400-799
--split-index 2  # examples 800-1186
```

Runtime for ICL/oracle mode is roughly:

```text
num_test * (num_contexts + 1)
```

For example, `--test-scope subset --split-index 0 --num-contexts 20` runs `400 * 21 = 8400` generations.

## No-ICL Inference Commands

Single test point:

```bash
python ICL-demo.py \
  --mode no_icl \
  --test-scope single \
  --example-id 0 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device auto \
  --math-config algebra \
  --max-new-tokens 64 \
  --print-examples 1
```

Subset of test points:

```bash
python ICL-demo.py \
  --mode no_icl \
  --test-scope subset \
  --split-index 0 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device auto \
  --math-config algebra \
  --max-new-tokens 768 \
  --print-examples 3
```

Entire test split:

```bash
python ICL-demo.py \
  --mode no_icl \
  --test-scope full \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device auto \
  --math-config algebra \
  --max-new-tokens 768 \
  --print-examples 5
```

No-ICL single example without writing files:

```bash
python ICL-demo.py \
  --mode no_icl \
  --test-scope single \
  --example-id 0 \
  --max-new-tokens 64 \
  --no-save-results
```

## ICL / Oracle Commands

Single test point:

```bash
python ICL-demo.py \
  --mode oracle \
  --test-scope single \
  --example-id 0 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device auto \
  --math-config algebra \
  --num-contexts 2 \
  --k 2 \
  --max-new-tokens 128 \
  --print-examples 1
```

Subset of test points:

```bash
python ICL-demo.py \
  --mode oracle \
  --test-scope subset \
  --split-index 0 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device auto \
  --math-config algebra \
  --num-contexts 5 \
  --k 4 \
  --max-new-tokens 512 \
  --print-examples 3
```

Entire test split:

```bash
python ICL-demo.py \
  --mode oracle \
  --test-scope full \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device auto \
  --math-config algebra \
  --num-contexts 20 \
  --k 4 \
  --max-new-tokens 768 \
  --print-examples 5
```

ICL/oracle run without writing files:

```bash
python ICL-demo.py \
  --mode oracle \
  --test-scope single \
  --example-id 0 \
  --num-contexts 2 \
  --k 2 \
  --max-new-tokens 128 \
  --no-save-results
```

## ICL Steering Embedding Visualization

`visualize_icl_steering_embeddings.py` extracts final-layer final-token hidden states before and after ICL, saves the vectors under `/p/vast1/dutta5/embeddings`, fits UMAP in two dimensions, and plots paired movement arrows.

The script imports and reuses the existing setup and prompt functions from `ICL-demo.py` instead of reimplementing inference plumbing. It reuses:

- `prepare_cache_dirs`
- `load_experiment_resources`
- `select_test_examples`
- `build_fixed_contexts`
- `make_messages_no_icl`
- `make_messages_icl`
- `extract_gold`

The embedding definition is:

```text
outputs.hidden_states[-1] at the last non-padding prompt token after
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

This is intentionally not the model input embedding and not the generated answer token. It is the final transformer layer representation of the final prompt token immediately before generation would begin. For both no-ICL and ICL prompts, the extraction point is the same semantic position: the model is about to answer the target problem.

The script tokenizes once without truncation to count prompt length. By default, it raises if the prompt exceeds `--max-length`, because truncation can remove context examples or the target problem and make the before/after vector comparison logically invalid. Use `--allow-truncation` only for a deliberate diagnostic run.

Default model aliases:

```text
llama3.2-3b  -> meta-llama/Llama-3.2-3B-Instruct
llama3.2-8b  -> meta-llama/Meta-Llama-3.1-8B-Instruct
```

Meta does not provide a text-only Llama 3.2 8B checkpoint. The `llama3.2-8b` alias therefore maps to `Meta-Llama-3.1-8B-Instruct` and prints a warning. If you have a different local 8B checkpoint, pass the exact Hugging Face ID or local path with `--models`.

Run a small visualization on a GPU node:

```bash
cd /usr/WS2/dutta5/Projects/Query-aware-steering/ICL
source /p/vast1/dutta5/envs/ICL-venv/bin/activate

python visualize_icl_steering_embeddings.py \
  --models llama3.2-3b llama3.2-8b \
  --device auto \
  --math-config algebra \
  --test-scope subset \
  --split-index 0 \
  --max-examples 50 \
  --num-contexts 4 \
  --k 4
```

Run only one model:

```bash
python visualize_icl_steering_embeddings.py \
  --models meta-llama/Llama-3.2-3B-Instruct \
  --device auto \
  --math-config algebra \
  --test-scope subset \
  --split-index 0 \
  --max-examples 50 \
  --num-contexts 4 \
  --k 4
```

Extract vectors without plotting:

```bash
python visualize_icl_steering_embeddings.py \
  --models llama3.2-3b \
  --device auto \
  --test-scope single \
  --example-id 0 \
  --max-examples 1 \
  --num-contexts 2 \
  --k 2 \
  --skip-umap
```

Use `--device cuda` when you want an explicit GPU requirement. With `--device auto`, the script fails fast if CUDA is unavailable, unless `--allow-auto-cpu` is provided. This avoids accidentally loading Llama checkpoints on CPU.

### Visualization Output Files

For MATH algebra, Llama 3.2 3B, split 0, 50 examples, output is written under:

```text
/p/vast1/dutta5/embeddings/
└── EleutherAI__hendrycks_math/
    └── algebra/
        └── meta-llama__Llama-3.2-3B-Instruct/
            └── split0_n50/
```

Files include:

```text
*_final_token_hidden_states.npz  # no_icl_vectors, icl_vectors, distances, question ids
*_pairs.csv                      # per-question/per-context metadata and movement metrics
*_contexts.csv                   # sampled fixed context ids
*_config.json                    # model, dataset, extraction definition, paths
*_umap_pairs.csv                 # paired no-ICL and ICL UMAP coordinates
*_umap_movement.png              # arrow plot in the shared UMAP space
*_movement_hist.png              # histogram of cosine movement magnitudes
```

Important shape convention in the `.npz` file:

```text
no_icl_vectors: [num_questions, hidden_size]
icl_vectors:    [num_questions, num_contexts, hidden_size]
```

### Visualization Recommendations

The main recommended plot is a paired arrow plot in a single UMAP space per model. Fit UMAP on the concatenation of `no_icl_vectors` and flattened `icl_vectors`, then draw arrows from each no-ICL point to its ICL endpoint. This directly satisfies the rule that both vectors must live in the same latent space.

Use color saturation for original-space movement magnitude, not just 2D distance. The script colors arrows and ICL endpoints by cosine movement:

```text
1 - cosine_similarity(no_icl_vector, icl_vector)
```

This avoids overinterpreting UMAP geometry. UMAP is useful for layout, but the color encodes movement measured in the model hidden space.

Recommended variants:

- `All context arrows`: draw one arrow per question-context pair. Best for small runs such as 25-100 questions and 2-5 contexts.
- `Mean context endpoint`: average the ICL vectors for each question before UMAP or average endpoints after UMAP. Best when the all-arrow plot is too dense.
- `Context fan plot`: keep the same no-ICL start point and show all context endpoints as a fan. Best for inspecting whether different demonstrations steer a problem consistently.
- `Magnitude histogram`: always include this beside the UMAP plot. It shows whether movement is broadly distributed or driven by a few extreme examples.
- `Correctness overlay`: after generation results exist, color or marker-shape by no-ICL wrong/right and ICL wrong/right. This is the most useful next plot for asking whether larger movement correlates with better answers.

Do not compare Llama 3B and Llama 8B points in a single UMAP fit. Their hidden dimensions and learned representation spaces are not shared. Make one UMAP per model, then compare movement distributions or aggregate statistics across models.

## Inspect Results

After a no-ICL run:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_parquet("qwen2.57B_no_icl.parquet")
print(df[["question_id", "gold", "prediction", "correct"]].head())
print(df.loc[0, "generation"])
PY
```

After an ICL/oracle run:

```bash
python - <<'PY'
import pandas as pd
summary = pd.read_csv("qwen2.57B_oracle_summary.csv")
matrix = pd.read_csv("qwen2.57B_success_matrix.csv")
print(summary.T)
print(matrix.head())
PY
```

## Interactive Notebook

Open `ICL-demo.ipynb` for debugging. It has two code cells:

1. Cell 1 imports `ICL-demo.py`, loads the model/dataset, and runs no-ICL inference.
2. Cell 2 reuses the loaded resources and runs the ICL/oracle experiment.

For a quick notebook smoke test on a GPU node, set these variables in Cell 1:

```python
TEST_SCOPE = "single"
EXAMPLE_ID = 0
MAX_NEW_TOKENS = 64
SAVE_RESULTS = False
```

For a full notebook run, use:

```python
TEST_SCOPE = "full"
SAVE_RESULTS = True
```

For a chunked notebook run, use:

```python
TEST_SCOPE = "subset"
SPLIT_INDEX = 0
SAVE_RESULTS = True
```

## Useful Arguments

For `ICL-demo.py`:

- `--model`: Hugging Face model name or local model path.
- `--cache-root`: Model/tokenizer cache root. Default: `/p/vast1/dutta5/cache-dirs`.
- `--dataset-root`: Dataset cache root. Default: `/p/vast1/dutta5/datasets`.
- `--output-dir`: Directory for result files. Default: current directory.
- `--mode`: `no_icl` or `oracle`.
- `--device`: `auto`, `cuda`, or `cpu`.
- `--math-config`: MATH split, such as `algebra`, `geometry`, `number_theory`, `prealgebra`, `precalculus`, `intermediate_algebra`, or `counting_and_probability`.
- `--test-scope`: `single`, `subset`, or `full`.
- `--example-id`: Test-set index used when `--test-scope single`.
- `--split-index`: 400-example chunk index used when `--test-scope subset`.
- `--num-test`: Legacy argument retained for compatibility; chunked subset selection uses `--split-index`.
- `--num-contexts`: Number of random ICL contexts to sample.
- `--k`: Number of demonstrations in each ICL context.
- `--max-new-tokens`: Maximum generated answer length.
- `--max-length`: Maximum tokenized prompt length before generation.
- `--seed`: Random seed for context sampling.
- `--print-examples`: Number of no-ICL generations to print.
- `--save-results` / `--no-save-results`: Enable or disable CSV/Parquet output.

For `visualize_icl_steering_embeddings.py`:

- `--models`: One or more model aliases, Hugging Face IDs, or local paths. Default: `llama3.2-3b llama3.2-8b`.
- `--embedding-root`: Root directory for saved vectors and plots. Default: `/p/vast1/dutta5/embeddings`.
- `--max-examples`: Cap the selected test scope before extraction. Default: `50`.
- `--num-contexts`: Number of fixed random ICL contexts to compare against each no-ICL vector. Default: `4`.
- `--k`: Number of demonstrations per ICL context. Default: `4`.
- `--skip-umap`: Save hidden states and metadata without producing UMAP plots.
- `--load-only`: Load tokenizer, model, and dataset, then exit. Useful for GPU smoke tests.
- `--allow-truncation`: Permit over-length prompts to be truncated. Avoid this for real comparisons unless intentionally debugging.
- `--allow-auto-cpu`: Let `--device auto` continue on CPU when CUDA is unavailable. The default is to fail fast.
- `--umap-neighbors`, `--umap-min-dist`: UMAP layout controls.
- `--max-arrows`: Maximum number of arrows drawn in the UMAP plot. Embeddings and coordinate files still contain all pairs.

## Notes

- `--mode oracle` requires `--num-contexts > 0`.
- The answer checker is string-based, so mathematically equivalent forms can be marked wrong if normalization does not make them identical.
- The first full Qwen 2.5 7B inference run downloads several `.safetensors` weight shards. Later runs reuse the cache.
- The visualization script does not generate answers. It only runs forward passes to collect prompt hidden states.
