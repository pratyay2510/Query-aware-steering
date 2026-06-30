#!/usr/bin/env python3
"""
Visualize ICL steering in final-layer final-token hidden states.

For each selected MATH test problem, this script compares:

1. The final-layer hidden state at the last prompt token without ICL.
2. The same hidden-state position after adding a sampled ICL context.

The UMAP reducer is fit on the combined no-ICL and ICL vectors for one model,
so paired points are plotted in the same 2D latent space. Different model
families are plotted separately because their hidden spaces are not shared.
"""

import argparse
import importlib.util
import json
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-codex-cache")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


DATASET_NAME = "EleutherAI/hendrycks_math"
DATASET_SLUG = "EleutherAI__hendrycks_math"
DEFAULT_EMBEDDING_ROOT = "/p/vast1/dutta5/embeddings"

MODEL_ALIASES = {
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama32-3b": "meta-llama/Llama-3.2-3B-Instruct",
    # Meta did not release a text-only Llama 3.2 8B. This alias keeps the
    # requested 8B scale while making the substitution explicit at runtime.
    "llama3.2-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-3.2-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama32-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama31-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def load_demo_module():
    demo_path = Path(__file__).with_name("ICL-demo.py")
    spec = importlib.util.spec_from_file_location("icl_demo", demo_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import demo helpers from {demo_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    demo = load_demo_module()

    p = argparse.ArgumentParser(
        description=(
            "Extract final-layer final-token hidden states before/after ICL, "
            "save them under vast1 embeddings, and plot paired UMAP movement."
        )
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["llama3.2-3b", "llama3.2-8b"],
        help=(
            "Model aliases or HF/local IDs. Comma-separated values are also "
            "accepted. Default: llama3.2-3b llama3.2-8b. The 8B alias maps to "
            "Meta-Llama-3.1-8B-Instruct because Meta has no text-only Llama 3.2 8B."
        ),
    )
    p.add_argument("--cache-root", type=str, default=demo.DEFAULT_CACHE_ROOT)
    p.add_argument("--dataset-root", type=str, default=demo.DEFAULT_DATASET_ROOT)
    p.add_argument("--embedding-root", type=str, default=DEFAULT_EMBEDDING_ROOT)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument(
        "--allow-auto-cpu",
        action="store_true",
        help=(
            "Allow --device auto to proceed when CUDA is unavailable. Without this, "
            "auto fails fast to avoid accidentally loading Llama checkpoints on CPU."
        ),
    )
    p.add_argument("--math-config", type=str, default="algebra")
    p.add_argument("--test-scope", choices=["subset", "full", "single"], default="subset")
    p.add_argument("--example-id", type=int, default=0)
    p.add_argument("--split-index", type=int, default=0)
    p.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Optional cap after selecting the test scope. Use 0 or negative for no cap.",
    )
    p.add_argument("--num-contexts", type=int, default=4)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-length", type=int, default=7500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--umap-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.08)
    p.add_argument("--max-arrows", type=int, default=1000)
    p.add_argument(
        "--allow-truncation",
        action="store_true",
        help=(
            "Allow tokenized prompts longer than --max-length. By default this "
            "raises, because truncation can remove the target problem and make "
            "the before/after comparison invalid."
        ),
    )
    p.add_argument("--skip-umap", action="store_true", help="Only extract and save vectors.")
    p.add_argument(
        "--load-only",
        action="store_true",
        help="Load tokenizer, model, and dataset for each model, then exit.",
    )
    p.add_argument(
        "--no-save-csv",
        action="store_true",
        help="Skip CSV metadata files. NPZ and JSON config are still saved.",
    )
    return p.parse_args()


def normalize_model_args(model_args):
    raw = []
    for item in model_args:
        raw.extend(part for part in item.split(",") if part.strip())

    models = []
    for model in raw:
        key = model.strip()
        resolved = MODEL_ALIASES.get(key.lower(), key)
        if key.lower() in {"llama3.2-8b", "llama-3.2-8b", "llama32-8b"}:
            warnings.warn(
                "Meta has no text-only Llama 3.2 8B release. "
                "Using meta-llama/Meta-Llama-3.1-8B-Instruct for the 8B run. "
                "Pass an exact --models ID if you have a different local checkpoint.",
                RuntimeWarning,
                stacklevel=2,
            )
        models.append(resolved)

    if not models:
        raise ValueError("At least one model must be provided.")

    return models


def safe_slug(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "__", str(value)).strip("_") or "value"


def scope_slug(args):
    if args.test_scope == "single":
        base = f"single{args.example_id}"
    elif args.test_scope == "subset":
        base = f"split{args.split_index}"
    else:
        base = "full"

    if args.max_examples and args.max_examples > 0:
        base = f"{base}_n{args.max_examples}"
    return base


def model_run_dir(args, model_name):
    return (
        Path(args.embedding_root).expanduser().resolve()
        / DATASET_SLUG
        / args.math_config
        / safe_slug(model_name)
        / scope_slug(args)
    )


def make_demo_args(demo, args, model_name, output_dir):
    return demo.make_experiment_args(
        model=model_name,
        cache_root=args.cache_root,
        dataset_root=args.dataset_root,
        output_dir=str(output_dir),
        mode="oracle",
        device=args.device,
        math_config=args.math_config,
        test_scope=args.test_scope,
        example_id=args.example_id,
        split_index=args.split_index,
        num_contexts=args.num_contexts,
        k=args.k,
        max_new_tokens=1,
        max_length=args.max_length,
        seed=args.seed,
        print_examples=0,
        save_results=False,
    )


def input_device_for_model(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def tokenize_prompt(tokenizer, prompt, max_length, allow_truncation):
    untruncated = tokenizer(prompt, return_tensors="pt", truncation=False)
    prompt_tokens = int(untruncated.input_ids.shape[1])

    if prompt_tokens > max_length and not allow_truncation:
        raise ValueError(
            f"Prompt has {prompt_tokens} tokens, which exceeds --max-length {max_length}. "
            "Increase --max-length, lower --k/--num-contexts, or pass --allow-truncation "
            "only if you intentionally accept a potentially invalid comparison."
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return inputs, prompt_tokens, bool(prompt_tokens > max_length)


@torch.no_grad()
def final_prompt_hidden_state(tokenizer, model, messages, max_length, allow_truncation):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs, prompt_tokens, was_truncated = tokenize_prompt(
        tokenizer,
        prompt,
        max_length,
        allow_truncation,
    )

    device = input_device_for_model(model)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    outputs = model(
        **inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    final_hidden = outputs.hidden_states[-1]
    last_token = inputs["attention_mask"].sum(dim=1) - 1
    batch_index = torch.arange(final_hidden.shape[0], device=final_hidden.device)
    vector = final_hidden[batch_index, last_token.to(final_hidden.device)]

    return {
        "vector": vector[0].detach().float().cpu().numpy(),
        "prompt_tokens": prompt_tokens,
        "was_truncated": was_truncated,
    }


def vector_distances(no_vector, icl_vector):
    no = no_vector.astype(np.float64)
    icl = icl_vector.astype(np.float64)
    denom = np.linalg.norm(no) * np.linalg.norm(icl)
    cosine_similarity = float(np.dot(no, icl) / denom) if denom > 0 else float("nan")
    return {
        "cosine_similarity": cosine_similarity,
        "cosine_distance": float(1.0 - cosine_similarity),
        "l2_distance": float(np.linalg.norm(icl - no)),
    }


def extract_vectors_for_model(demo, resources, args, contexts):
    tokenizer = resources["tokenizer"]
    model = resources["model"]
    train_set = resources["train_set"]
    test_set = list(resources["test_set"])

    if args.max_examples and args.max_examples > 0:
        test_set = test_set[: args.max_examples]

    if not test_set:
        raise ValueError("No test examples were selected.")

    no_vectors = []
    no_prompt_tokens = []
    no_truncated = []
    icl_vectors = []
    pair_rows = []

    print(f"Extracting hidden states for {len(test_set)} test examples...")

    for local_idx, ex in enumerate(tqdm(test_set)):
        question_id = int(ex.get("question_id", local_idx))
        problem = ex["problem"]
        gold = demo.extract_gold(ex["solution"])

        no_messages = demo.make_messages_no_icl(problem)
        no_result = final_prompt_hidden_state(
            tokenizer,
            model,
            no_messages,
            args.max_length,
            args.allow_truncation,
        )
        no_vector = no_result["vector"]
        no_vectors.append(no_vector)
        no_prompt_tokens.append(no_result["prompt_tokens"])
        no_truncated.append(no_result["was_truncated"])

        question_icl_vectors = []
        for context_id, context in enumerate(contexts):
            icl_messages = demo.make_messages_icl(train_set, context, problem)
            icl_result = final_prompt_hidden_state(
                tokenizer,
                model,
                icl_messages,
                args.max_length,
                args.allow_truncation,
            )
            icl_vector = icl_result["vector"]
            question_icl_vectors.append(icl_vector)

            distances = vector_distances(no_vector, icl_vector)
            pair_rows.append({
                "question_row": local_idx,
                "question_id": question_id,
                "context_id": context_id,
                "demo_ids": ",".join(map(str, context)),
                "gold": gold,
                "problem": problem,
                "no_icl_prompt_tokens": no_result["prompt_tokens"],
                "icl_prompt_tokens": icl_result["prompt_tokens"],
                "no_icl_was_truncated": no_result["was_truncated"],
                "icl_was_truncated": icl_result["was_truncated"],
                **distances,
            })

        icl_vectors.append(question_icl_vectors)

    return {
        "test_set": test_set,
        "no_icl_vectors": np.stack(no_vectors),
        "icl_vectors": np.asarray(icl_vectors),
        "pair_metadata": pd.DataFrame(pair_rows),
        "no_prompt_tokens": np.asarray(no_prompt_tokens, dtype=np.int32),
        "no_truncated": np.asarray(no_truncated, dtype=bool),
    }


def save_vectors(run_dir, model_name, args, contexts, vector_data):
    run_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{DATASET_SLUG}_{args.math_config}_{scope_slug(args)}_{safe_slug(model_name)}"
    npz_path = run_dir / f"{prefix}_final_token_hidden_states.npz"
    metadata_path = run_dir / f"{prefix}_pairs.csv"
    contexts_path = run_dir / f"{prefix}_contexts.csv"
    config_path = run_dir / f"{prefix}_config.json"

    pair_df = vector_data["pair_metadata"]
    np.savez_compressed(
        npz_path,
        no_icl_vectors=vector_data["no_icl_vectors"],
        icl_vectors=vector_data["icl_vectors"],
        question_ids=pair_df.drop_duplicates("question_row")["question_id"].to_numpy(),
        no_icl_prompt_tokens=vector_data["no_prompt_tokens"],
        no_icl_was_truncated=vector_data["no_truncated"],
        cosine_distance=pair_df["cosine_distance"].to_numpy().reshape(
            vector_data["icl_vectors"].shape[:2]
        ),
        l2_distance=pair_df["l2_distance"].to_numpy().reshape(
            vector_data["icl_vectors"].shape[:2]
        ),
    )

    if not args.no_save_csv:
        pair_df.to_csv(metadata_path, index=False)
        pd.DataFrame({
            "context_id": list(range(len(contexts))),
            "demo_ids": [",".join(map(str, context)) for context in contexts],
        }).to_csv(contexts_path, index=False)

    config = {
        "model": model_name,
        "dataset": DATASET_NAME,
        "math_config": args.math_config,
        "test_scope": args.test_scope,
        "split_index": args.split_index if args.test_scope == "subset" else None,
        "example_id": args.example_id if args.test_scope == "single" else None,
        "max_examples": args.max_examples,
        "num_contexts": args.num_contexts,
        "k": args.k,
        "max_length": args.max_length,
        "seed": args.seed,
        "embedding_definition": (
            "outputs.hidden_states[-1] at the last non-padding prompt token after "
            "tokenizer.apply_chat_template(..., add_generation_prompt=True)"
        ),
        "paths": {
            "npz": str(npz_path),
            "pairs_csv": str(metadata_path) if not args.no_save_csv else None,
            "contexts_csv": str(contexts_path) if not args.no_save_csv else None,
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    return {
        "npz": npz_path,
        "metadata": metadata_path if not args.no_save_csv else None,
        "contexts": contexts_path if not args.no_save_csv else None,
        "config": config_path,
    }


def compute_umap(vector_data, args):
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for plotting. Install it with "
            "`python -m pip install umap-learn` or rerun with --skip-umap."
        ) from exc

    no_vectors = vector_data["no_icl_vectors"]
    icl_vectors = vector_data["icl_vectors"]
    flat_icl = icl_vectors.reshape(-1, icl_vectors.shape[-1])
    combined = np.vstack([no_vectors, flat_icl])

    if combined.shape[0] < 5:
        raise ValueError(
            f"UMAP needs more paired points for a meaningful plot; got {combined.shape[0]} vectors. "
            "Increase --max-examples or --num-contexts, or use --skip-umap for a load/extract smoke test."
        )

    n_neighbors = min(args.umap_neighbors, combined.shape[0] - 1)
    n_neighbors = max(2, n_neighbors)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=args.umap_min_dist,
        metric="cosine",
        random_state=args.seed,
    )
    coords = reducer.fit_transform(combined)
    num_questions = no_vectors.shape[0]
    no_coords = coords[:num_questions]
    icl_coords = coords[num_questions:].reshape(icl_vectors.shape[0], icl_vectors.shape[1], 2)

    return no_coords, icl_coords


def save_umap_outputs(run_dir, model_name, args, vector_data):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyArrowPatch

    no_coords, icl_coords = compute_umap(vector_data, args)
    pair_df = vector_data["pair_metadata"].copy()
    num_questions, num_contexts, _ = icl_coords.shape

    pair_df["no_umap_x"] = np.repeat(no_coords[:, 0], num_contexts)
    pair_df["no_umap_y"] = np.repeat(no_coords[:, 1], num_contexts)
    pair_df["icl_umap_x"] = icl_coords[:, :, 0].reshape(-1)
    pair_df["icl_umap_y"] = icl_coords[:, :, 1].reshape(-1)

    prefix = f"{DATASET_SLUG}_{args.math_config}_{scope_slug(args)}_{safe_slug(model_name)}"
    coords_path = run_dir / f"{prefix}_umap_pairs.csv"
    plot_path = run_dir / f"{prefix}_umap_movement.png"
    hist_path = run_dir / f"{prefix}_movement_hist.png"
    pair_df.to_csv(coords_path, index=False)

    movement = pair_df["cosine_distance"].to_numpy()
    finite_movement = movement[np.isfinite(movement)]
    vmin = float(finite_movement.min()) if finite_movement.size else 0.0
    vmax = float(finite_movement.max()) if finite_movement.size else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    ax.scatter(
        no_coords[:, 0],
        no_coords[:, 1],
        s=22,
        c="#202020",
        alpha=0.85,
        label="No ICL",
        zorder=3,
    )
    ax.scatter(
        icl_coords[:, :, 0].reshape(-1),
        icl_coords[:, :, 1].reshape(-1),
        s=14,
        c=cmap(norm(movement)),
        alpha=0.78,
        label="With ICL",
        zorder=4,
    )

    arrow_indices = np.arange(len(pair_df))
    if args.max_arrows and len(arrow_indices) > args.max_arrows:
        rng = np.random.default_rng(args.seed)
        arrow_indices = np.sort(rng.choice(arrow_indices, size=args.max_arrows, replace=False))

    for idx in arrow_indices:
        row = pair_df.iloc[int(idx)]
        color = cmap(norm(row["cosine_distance"]))
        alpha = 0.18 + 0.62 * norm(row["cosine_distance"])
        rad = 0.08 if int(row["context_id"]) % 2 == 0 else -0.08
        patch = FancyArrowPatch(
            (row["no_umap_x"], row["no_umap_y"]),
            (row["icl_umap_x"], row["icl_umap_y"]),
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.75,
            color=color,
            alpha=float(alpha),
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
        ax.add_patch(patch)

    ax.set_title(f"ICL final-token hidden-state movement\n{model_name}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, color="#d8d8d8", linewidth=0.55, alpha=0.6)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colorbar.set_label("Cosine movement: 1 - cos(no ICL, ICL)")
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=180)
    ax.hist(finite_movement, bins=min(40, max(8, len(finite_movement) // 2)), color="#5b5bd6")
    ax.set_title(f"ICL movement magnitude\n{model_name}")
    ax.set_xlabel("Cosine movement: 1 - cos(no ICL, ICL)")
    ax.set_ylabel("Pair count")
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.55, alpha=0.6)
    fig.tight_layout()
    fig.savefig(hist_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "coords": coords_path,
        "plot": plot_path,
        "hist": hist_path,
    }


def run_for_model(demo, args, model_name):
    run_dir = model_run_dir(args, model_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print(f"Model: {model_name}")
    print(f"Output: {run_dir}")
    print("==============================")

    demo_args = make_demo_args(demo, args, model_name, run_dir)
    resources = demo.load_experiment_resources(demo_args)

    if args.load_only:
        print("Load-only smoke test completed for this model.")
        return {"run_dir": run_dir}

    contexts = demo.build_fixed_contexts(
        resources["train_set"],
        args.num_contexts,
        args.k,
    )
    vector_data = extract_vectors_for_model(demo, resources, args, contexts)
    saved = save_vectors(run_dir, model_name, args, contexts, vector_data)

    print("Saved embeddings:")
    for key, value in saved.items():
        if value is not None:
            print(f"  {key}: {value}")

    if not args.skip_umap:
        plots = save_umap_outputs(run_dir, model_name, args, vector_data)
        print("Saved UMAP outputs:")
        for key, value in plots.items():
            print(f"  {key}: {value}")
        saved.update(plots)

    del resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return saved


def main():
    args = parse_args()

    if args.num_contexts <= 0:
        raise ValueError("--num-contexts must be positive.")
    if args.k <= 0:
        raise ValueError("--k must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but torch.cuda.is_available() is False. "
            "Run this on a GPU node, or use --device auto/--device cpu intentionally."
        )
    if args.device == "auto" and not torch.cuda.is_available() and not args.allow_auto_cpu:
        raise RuntimeError(
            "--device auto would fall back to CPU because torch.cuda.is_available() is False. "
            "Run on a GPU node, pass --device cpu, or pass --allow-auto-cpu intentionally."
        )

    demo = load_demo_module()
    models = normalize_model_args(args.models)

    print("Resolved models:")
    for model in models:
        print(f"  {model}")

    results = {}
    for model_name in models:
        results[model_name] = run_for_model(demo, args, model_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
