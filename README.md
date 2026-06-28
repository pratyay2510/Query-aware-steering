# Query-aware Steering (QAS)

Experimentation workspace for **making in-context learning (ICL) query-aware**.

Standard ICL prepends a fixed set of demonstrations to every prompt. A growing line
of work instead distills demonstrations into a single **in-context vector (ICV)**
that is added to the model's hidden states at inference time — cheaper and often
stronger than prompting. The goal of this project is to make that steering
**query-aware**: condition the vector on the actual query rather than using one
static direction for all inputs.

This repo vendors two prior works as starting points and gives each a clean,
one-command inference path:

| Folder | Work | Modality | Status here |
|--------|------|----------|-------------|
| [`ICV/`](ICV/)  | [In-Context Vectors](https://github.com/shengliu66/ICV) (Liu et al.) | Text LLM | ✅ runnable smoke test (script + notebook) |
| [`LIVE/`](LIVE/) | [LIVE: Learnable In-Context Vector](https://github.com/ForJadeForest/LIVE-Learnable-In-Context-Vector) (Peng et al., NeurIPS'24) | Multimodal VQA | ⏸️ env + docs only (see note below) |

---

## Layout

```
QAS/
├── config.sh        # EDIT ME: one place to set the HuggingFace cache location
├── push.sh          # commit (asks for a message) + push to GitHub
├── pull.sh          # pull latest from GitHub
├── ICV/             # In-Context Vectors — working smoke test
│   ├── inference.sh #   self-bootstrapping runner (creates venv, runs infer.py)
│   ├── infer.py     #   terminal demo: generation BEFORE vs AFTER steering
│   └── demo_notebook.ipynb
└── LIVE/            # Learnable ICV (multimodal) — deferred
    ├── inference.sh #   env sanity check + documented full-run instructions
    ├── requirements.txt        # minimal env to import the code
    └── requirements-full.txt   # authors' full frozen environment
```

## Prerequisites

- Linux, Python 3.10, an NVIDIA GPU (developed on 3× RTX 3080 10 GB).
- Internet access to GitHub / PyPI / HuggingFace.
- The virtualenvs are **not** committed — each `inference.sh` creates its own on
  first run, so the repo stays portable across machines.

## First-time setup (any machine)

1. **Pick a cache location.** Models are large; keep them off a near-full home
   disk. Edit one line in [`config.sh`](config.sh):

   ```bash
   export HF_CACHE_DIR="${HF_CACHE_DIR:-/datastarnas01/hf_cache}"   # <-- point anywhere
   ```

   Every script sources this, so it's the single source of truth for where
   HuggingFace models, hub files and datasets are stored.

2. **Run a technique.** No manual `pip install` needed — see below.

---

## ICV — in-context vectors (working)

```bash
./ICV/inference.sh                 # sentiment demo, falcon-7b in 8-bit on one GPU
./ICV/inference.sh --demo safety   # dialogue-safety steering
./ICV/inference.sh --demo both
./ICV/inference.sh --gpus 0,1      # spread across GPUs if a single card OOMs
```

First run creates `ICV/.venv`, installs dependencies, downloads `tiiuae/falcon-7b`
(~14 GB) into your cache, then prints generations before and after steering, e.g.:

```
[BEFORE steering]  ... paraphrase: "Worst restaurant ever!"
[AFTER  steering]  ... paraphrase: "Best restaurant ever!"
```

### Notebook

[`ICV/demo_notebook.ipynb`](ICV/demo_notebook.ipynb) is the interactive version
(safety + sentiment). To use it:

```bash
./ICV/inference.sh                 # once, to create the venv
ICV/.venv/bin/python -m ipykernel install --user --name qas-icv --display-name "QAS ICV (.venv)"
```

Then open the notebook and select the **QAS ICV (.venv)** kernel. The first cell
pins the GPU and cache (mirrors `config.sh`); run cells top to bottom.

### What was fixed vs upstream
- Added the missing `bitsandbytes` dependency (needed for 8-bit loading) and
  trimmed `requirements.txt` to what inference actually uses (the heavy
  `parlai`/`fschat` deps are optional, kept as comments).
- HuggingFace cache is now configurable via `HF_CACHE_DIR` (was a hard-coded path).
- Task imports are loaded defensively, so a missing optional dep no longer breaks
  the demo; the GPU-selection / 8-bit init order bug was fixed.

---

## LIVE — learnable in-context vector (deferred)

```bash
./LIVE/inference.sh                # bootstraps a minimal venv + environment check
```

LIVE is multimodal (VQA) and a real inference run needs three heavy ingredients
this workspace does **not** download automatically:

1. a large multimodal model (IDEFICS-9B / IDEFICS2-8B, ~16–18 GB),
2. the COCO + VQAv2/OKVQA datasets, and
3. a **trained ICV checkpoint** — the authors publish none, so you must run
   `LIVE/train.py` first.

`LIVE/inference.sh` therefore verifies the environment and prints the exact steps
for the full pipeline (full deps via `requirements-full.txt`, the two GitHub
dependencies, model download, dataset prep, `.env` setup, train, then
`./LIVE/inference.sh --run ...`). See [`LIVE/README.md`](LIVE/README.md) for the
upstream command reference.

---

## Syncing with GitHub

Remote: `https://github.com/pratyay2510/Query-aware-steering`

```bash
./push.sh     # stages everything, asks for a commit message, commits, pushes
./pull.sh     # fast-forward pull
```

First push needs a GitHub **Personal Access Token** (this machine has no `gh` CLI
or stored credentials). When git prompts, use your username + a PAT as the
password; `credential.helper store` is enabled so it's cached afterwards.
Alternatively switch the remote to SSH:
`git remote set-url origin git@github.com:pratyay2510/Query-aware-steering.git`.

## Notes on cache & disk

`config.sh` redirects all HuggingFace storage to `HF_CACHE_DIR`. The default
(`/datastarnas01/hf_cache`) is on large network storage; for faster local I/O
point it at a local SSD instead (e.g. `/ak_drive/hf_cache`). The home disk is
intentionally avoided. Virtualenvs, caches, model weights and datasets are all
git-ignored.

## Credits

- ICV — *In-Context Vectors: Making In-Context Learning More Effective and
  Controllable Through Latent Space Steering*, Liu et al.
- LIVE — *Learnable In-Context Vector for Visual Question Answering*, Peng et al.,
  NeurIPS 2024 ([arXiv:2406.13185](https://arxiv.org/abs/2406.13185)).
