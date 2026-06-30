#!/usr/bin/env python3
"""
MATH Oracle ICL Context Experiment.
 
Goal:
For each test problem, evaluate NUM_CONTEXTS fixed ICL contexts.
Then compute:
 
1. No ICL accuracy
2. Random-context accuracy
3. Oracle-context accuracy
 
This answers:
 
Do useful ICL contexts exist at all?
 
Example:
  python ICL-demo.py \
    --mode oracle \
    --test-scope subset \
    --split-index 0 \
    --num-contexts 20 \
    --k 4 \
    --math-config algebra
"""
 
import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
 
import random
import numpy as np
import pandas as pd
import torch
 
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
 

DEFAULT_CACHE_ROOT = "/p/vast1/dutta5/cache-dirs"
DEFAULT_DATASET_ROOT = "/p/vast1/dutta5/datasets"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TEST_CHUNK_SIZE = 400

 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--cache-root", type=str, default=DEFAULT_CACHE_ROOT)
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--output-dir", type=str, default=".")
    p.add_argument("--mode", choices=["oracle", "no_icl"], default="oracle")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--math-config", type=str, default="algebra")
    p.add_argument("--test-scope", choices=["subset", "full", "single"], default="subset")
    p.add_argument("--example-id", type=int, default=0)
    p.add_argument("--split-index", type=int, default=0)
    p.add_argument("--num-test", type=int, default=500)
    p.add_argument("--num-contexts", type=int, default=20)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--max-length", type=int, default=7500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print-examples", type=int, default=3)
    p.add_argument("--save-results", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def cache_safe_name(name):
    resolved = os.path.basename(os.path.abspath(name)) if os.path.exists(name) else name
    return re.sub(r"[^A-Za-z0-9._-]+", "__", resolved).strip("_")


def model_output_prefix(model_name):
    base = os.path.basename(model_name.rstrip("/"))
    base = re.sub(r"(?i)-?instruct.*$", "", base)
    base = base.replace("-", "")
    base = re.sub(r"^Qwen", "qwen", base)
    base = re.sub(r"[^A-Za-z0-9.]+", "_", base).strip("_")
    return base or "model"


def result_paths(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = experiment_output_prefix(args)

    return {
        "no_icl_csv": output_dir / f"{prefix}_no_icl.csv",
        "no_icl_parquet": output_dir / f"{prefix}_no_icl.parquet",
        "no_icl_summary_csv": output_dir / f"{prefix}_no_icl_summary.csv",
        "fixed_contexts_csv": output_dir / f"{prefix}_fixed_contexts.csv",
        "oracle_csv": output_dir / f"{prefix}_oracle_context.csv",
        "oracle_parquet": output_dir / f"{prefix}_oracle_context.parquet",
        "success_matrix_csv": output_dir / f"{prefix}_success_matrix.csv",
        "oracle_summary_csv": output_dir / f"{prefix}_oracle_summary.csv",
    }


def experiment_output_prefix(args):
    prefix = model_output_prefix(args.model)

    if args.test_scope == "subset":
        return f"{prefix}_split{args.split_index}"
    if args.test_scope == "single":
        return f"{prefix}_example{args.example_id}"

    return prefix


def prepare_cache_dirs(cache_root, dataset_root, model_name, math_config):
    cache_root = Path(cache_root).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    model_cache_dir = cache_root / "models" / cache_safe_name(model_name)
    dataset_cache_dir = dataset_root / "EleutherAI__hendrycks_math" / math_config

    model_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(dataset_root)

    return model_cache_dir, dataset_cache_dir


def chunk_dir(dataset_cache_dir):
    return Path(dataset_cache_dir) / "test_chunks"


def chunk_manifest_path(dataset_cache_dir):
    return chunk_dir(dataset_cache_dir) / "manifest.json"


def chunk_file_path(dataset_cache_dir, split_index):
    return chunk_dir(dataset_cache_dir) / f"{split_index}.jsonl"


def ensure_test_chunks(test_split, dataset_cache_dir, chunk_size=DEFAULT_TEST_CHUNK_SIZE):
    chunks_dir = chunk_dir(dataset_cache_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    total = len(test_split)
    num_chunks = (total + chunk_size - 1) // chunk_size
    chunks = []

    for split_index in range(num_chunks):
        start = split_index * chunk_size
        end = min(start + chunk_size, total)
        path = chunk_file_path(dataset_cache_dir, split_index)

        with path.open("w", encoding="utf-8") as f:
            for question_id in range(start, end):
                ex = test_split[int(question_id)]
                record = {
                    "question_id": question_id,
                    "problem": ex["problem"],
                    "solution": ex["solution"],
                }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        chunks.append({
            "split_index": split_index,
            "start": start,
            "end": end,
            "num_examples": end - start,
            "path": str(path),
        })

    manifest = {
        "chunk_size": chunk_size,
        "total_examples": total,
        "num_chunks": num_chunks,
        "chunks": chunks,
    }

    with chunk_manifest_path(dataset_cache_dir).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    return manifest


def load_test_chunk(dataset_cache_dir, split_index):
    path = chunk_file_path(dataset_cache_dir, split_index)
    if not path.exists():
        raise FileNotFoundError(f"Test chunk does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_causal_lm(model_name, model_cache_dir, device):
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    kwargs = {
        "cache_dir": str(model_cache_dir),
        "torch_dtype": torch.bfloat16,
    }

    if device == "auto" and has_accelerate:
        kwargs["device_map"] = "auto"
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        model = model.to("cuda")
    elif device == "cpu" or device == "auto":
        model = model.to("cpu")

    if device == "auto" and not has_accelerate:
        print("Warning: accelerate is not installed; loaded without device_map='auto'.")

    return model
 
 
def last_boxed_only_string(s):
    idx = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if idx < 0:
        return None
 
    start = s.find("{", idx)
    if start < 0:
        return None
 
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:i]
 
    return None
 
 
def normalize_math_answer(s):
    if s is None:
        return None
 
    s = str(s).strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "")
    s = s.replace("\\,", "")
    s = s.replace("\\;", "")
    s = s.replace("\\:", "")
    s = s.replace(" ", "")
    s = s.replace("\n", "")
    s = s.replace("$", "")
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    s = s.replace(",", "")
    s = s.rstrip(".")
 
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
 
    return s
 
 
def extract_gold(solution):
    boxed = last_boxed_only_string(solution)
    if boxed is not None:
        return normalize_math_answer(boxed)
 
    nums = re.findall(r"-?\d+(?:\.\d+)?", solution.replace(",", ""))
    if nums:
        return normalize_math_answer(nums[-1])
 
    return None
 
 
def extract_pred(generation):
    boxed = last_boxed_only_string(generation)
    if boxed is not None:
        return normalize_math_answer(boxed)
 
    text = generation.replace(",", "")
 
    patterns = [
        r"(?:final answer|answer is|the answer is)\s*:?\s*\$?([^.\n$]+)",
        r"therefore.*?(?:answer).*?(?:is)\s*:?\s*\$?([^.\n$]+)",
    ]
 
    for pat in patterns:
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            cand = m[-1].strip()
            cand = cand.split("\\]")[0].strip()
            return normalize_math_answer(cand)
 
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if nums:
        return normalize_math_answer(nums[-1])
 
    return None
 
 
def make_messages_no_icl(problem):
    return [
        {
            "role": "system",
            "content": (
                "You are a careful mathematical problem solver. "
                "Solve the problem step by step. "
                "Put the final answer in LaTeX boxed form: \\boxed{answer}."
            ),
        },
        {
            "role": "user",
            "content": f"Problem:\n{problem}",
        },
    ]
 
 
def make_messages_icl(train_set, context_ids, problem):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful mathematical problem solver. "
                "Solve each problem step by step. "
                "Put the final answer in LaTeX boxed form: \\boxed{answer}."
            ),
        }
    ]
 
    for idx in context_ids:
        ex = train_set[int(idx)]
 
        messages.append({
            "role": "user",
            "content": f"Problem:\n{ex['problem']}",
        })
 
        messages.append({
            "role": "assistant",
            "content": ex["solution"].strip(),
        })
 
    messages.append({
        "role": "user",
        "content": f"Problem:\n{problem}",
    })
 
    return messages
 
 
@torch.no_grad()
def generate_from_messages(tokenizer, model, messages, max_new_tokens, max_length):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
 
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)
 
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
 
    new_tokens = outputs[0, inputs.input_ids.shape[1]:]
 
    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )
 
 
def evaluate_one(tokenizer, model, messages, gold, args):
    gen = generate_from_messages(
        tokenizer,
        model,
        messages,
        args.max_new_tokens,
        args.max_length,
    )
 
    pred = extract_pred(gen)
    correct = int(pred == gold)
 
    return pred, correct, gen
 
 
def build_fixed_contexts(train_set, num_contexts, k):
    contexts = []
 
    for _ in range(num_contexts):
        ids = random.sample(range(len(train_set)), k)
        contexts.append(ids)
 
    return contexts


def make_experiment_args(**overrides):
    values = {
        "model": DEFAULT_MODEL,
        "cache_root": DEFAULT_CACHE_ROOT,
        "dataset_root": DEFAULT_DATASET_ROOT,
        "output_dir": ".",
        "mode": "no_icl",
        "device": "auto",
        "math_config": "algebra",
        "test_scope": "full",
        "example_id": 0,
        "split_index": 0,
        "num_test": None,
        "num_contexts": 20,
        "k": 4,
        "max_new_tokens": 768,
        "max_length": 7500,
        "seed": 42,
        "print_examples": 3,
        "save_results": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def add_question_id(example, question_id):
    record = dict(example)
    record["question_id"] = question_id
    return record


def select_test_examples(test_split, args, dataset_cache_dir):
    if args.test_scope == "full":
        return [
            add_question_id(test_split[int(question_id)], question_id)
            for question_id in range(len(test_split))
        ]

    if args.test_scope == "single":
        if args.example_id < 0 or args.example_id >= len(test_split):
            raise ValueError(
                f"--example-id must be in [0, {len(test_split) - 1}], got {args.example_id}"
            )
        return [add_question_id(test_split[int(args.example_id)], args.example_id)]

    manifest = ensure_test_chunks(test_split, dataset_cache_dir)
    if args.split_index < 0 or args.split_index >= manifest["num_chunks"]:
        raise ValueError(
            f"--split-index must be in [0, {manifest['num_chunks'] - 1}], "
            f"got {args.split_index}"
        )

    return load_test_chunk(dataset_cache_dir, args.split_index)


def load_experiment_resources(args):
    model_cache_dir, dataset_cache_dir = prepare_cache_dirs(
        args.cache_root,
        args.dataset_root,
        args.model,
        args.math_config,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Cache root: {Path(args.cache_root).expanduser().resolve()}")
    print(f"Dataset root: {Path(args.dataset_root).expanduser().resolve()}")
    print(f"Model cache: {model_cache_dir}")
    print(f"Dataset cache: {dataset_cache_dir}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=str(model_cache_dir),
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = load_causal_lm(
        args.model,
        model_cache_dir,
        args.device,
    )
    model.eval()

    print("Loading MATH dataset...")
    print(f"Dataset config: {args.math_config}")

    ds = load_dataset(
        "EleutherAI/hendrycks_math",
        args.math_config,
        cache_dir=str(dataset_cache_dir),
    )

    train_set = ds["train"]
    test_set = select_test_examples(ds["test"], args, dataset_cache_dir)

    print(f"Train size: {len(train_set)}")
    print(f"Test size used: {len(test_set)}")
    print(f"Test scope: {args.test_scope}")
    if args.test_scope == "subset":
        print(f"Split index: {args.split_index}")

    return {
        "tokenizer": tokenizer,
        "model": model,
        "train_set": train_set,
        "test_set": test_set,
        "model_cache_dir": model_cache_dir,
        "dataset_cache_dir": dataset_cache_dir,
    }


def run_no_icl_experiment(resources, args, save=True):
    tokenizer = resources["tokenizer"]
    model = resources["model"]
    test_set = resources["test_set"]

    all_records = []
    noicl_results = []

    print("\nRunning no-ICL inference...")

    for qid, ex in enumerate(tqdm(test_set)):
        problem = ex["problem"]
        gold = extract_gold(ex["solution"])
        question_id = ex.get("question_id", qid)
        messages = make_messages_no_icl(problem)

        pred, correct, gen = evaluate_one(
            tokenizer,
            model,
            messages,
            gold,
            args,
        )

        noicl_results.append(correct)
        all_records.append({
            "question_id": question_id,
            "method": "no_icl",
            "context_id": -1,
            "gold": gold,
            "prediction": pred,
            "correct": correct,
            "problem": problem,
            "generation": gen,
        })

        if qid < args.print_examples:
            print("\n==============================")
            print(f"Example {qid} / No ICL")
            print("==============================")
            print("Gold:", gold)
            print("Pred:", pred)
            print("Correct:", correct)
            print(gen[:1200])

    noicl_results = np.array(noicl_results)
    noicl_acc = noicl_results.mean()
    df = pd.DataFrame(all_records)

    summary = {
        "mode": "no_icl",
        "model": args.model,
        "math_config": args.math_config,
        "test_scope": args.test_scope,
        "split_index": args.split_index if args.test_scope == "subset" else None,
        "example_id": args.example_id if args.test_scope == "single" else None,
        "num_test": len(test_set),
        "noicl_acc": noicl_acc,
    }

    print("\n==========================")
    print("FINAL RESULTS")
    print("==========================")
    print(f"No ICL accuracy:              {noicl_acc:.4f}")

    if save:
        paths = result_paths(args)
        df.to_csv(paths["no_icl_csv"], index=False)
        df.to_parquet(paths["no_icl_parquet"], index=False)
        pd.DataFrame([summary]).to_csv(
            paths["no_icl_summary_csv"],
            index=False,
        )

        print("\nSaved:")
        print(f"  {paths['no_icl_csv']}")
        print(f"  {paths['no_icl_parquet']}")
        print(f"  {paths['no_icl_summary_csv']}")

    return df, summary


def run_oracle_context_experiment(resources, args, save=True):
    tokenizer = resources["tokenizer"]
    model = resources["model"]
    train_set = resources["train_set"]
    test_set = resources["test_set"]

    print("\nBuilding fixed random contexts...")
    contexts = build_fixed_contexts(
        train_set,
        args.num_contexts,
        args.k,
    )

    if save:
        paths = result_paths(args)
        context_df = pd.DataFrame({
            "context_id": list(range(args.num_contexts)),
            "demo_ids": [",".join(map(str, c)) for c in contexts],
        })
        context_df.to_csv(paths["fixed_contexts_csv"], index=False)
        print(f"Saved fixed contexts to {paths['fixed_contexts_csv']}")

    all_records = []
    noicl_results = []
    success_matrix = []

    print("\nRunning oracle-context experiment...")

    for qid, ex in enumerate(tqdm(test_set)):
        problem = ex["problem"]
        gold = extract_gold(ex["solution"])
        question_id = ex.get("question_id", qid)

        no_messages = make_messages_no_icl(problem)
        no_pred, no_correct, no_gen = evaluate_one(
            tokenizer,
            model,
            no_messages,
            gold,
            args,
        )

        noicl_results.append(no_correct)
        all_records.append({
            "question_id": question_id,
            "method": "no_icl",
            "context_id": -1,
            "gold": gold,
            "prediction": no_pred,
            "correct": no_correct,
            "problem": problem,
            "generation": no_gen,
        })

        if qid < args.print_examples:
            print("\n==============================")
            print(f"Example {qid} / No ICL")
            print("==============================")
            print("Gold:", gold)
            print("Pred:", no_pred)
            print("Correct:", no_correct)
            print(no_gen[:1200])

        row = []

        for ctx_id, ctx in enumerate(contexts):
            icl_messages = make_messages_icl(
                train_set,
                ctx,
                problem,
            )

            pred, correct, gen = evaluate_one(
                tokenizer,
                model,
                icl_messages,
                gold,
                args,
            )

            row.append(correct)
            all_records.append({
                "question_id": question_id,
                "method": "icl_context",
                "context_id": ctx_id,
                "gold": gold,
                "prediction": pred,
                "correct": correct,
                "problem": problem,
                "generation": gen,
            })

        success_matrix.append(row)

    success_matrix = np.array(success_matrix)
    noicl_results = np.array(noicl_results)

    noicl_acc = noicl_results.mean()
    random_context_acc = success_matrix.mean()
    oracle_context_acc = success_matrix.max(axis=1).mean()

    oracle_gain_vs_noicl = oracle_context_acc - noicl_acc
    oracle_gain_vs_random = oracle_context_acc - random_context_acc

    context_accs = success_matrix.mean(axis=0)
    question_context_mean = success_matrix.mean(axis=1)
    question_context_var = success_matrix.var(axis=1)

    num_context_sensitive = int((question_context_var > 0).sum())
    num_oracle_improves_noicl = int((success_matrix.max(axis=1) > noicl_results).sum())
    num_context_hurts_noicl = int((success_matrix.min(axis=1) < noicl_results).sum())

    print("\n==========================")
    print("FINAL RESULTS")
    print("==========================")
    print(f"No ICL accuracy:              {noicl_acc:.4f}")
    print(f"Random context accuracy:      {random_context_acc:.4f}")
    print(f"Oracle context accuracy:      {oracle_context_acc:.4f}")
    print(f"Oracle gain vs No ICL:        {oracle_gain_vs_noicl:.4f}")
    print(f"Oracle gain vs Random:        {oracle_gain_vs_random:.4f}")

    print("\n==========================")
    print("CONTEXT SENSITIVITY")
    print("==========================")
    print(f"Questions with mixed context outcomes:       {num_context_sensitive}/{len(test_set)}")
    print(f"Questions where oracle beats No ICL:         {num_oracle_improves_noicl}/{len(test_set)}")
    print(f"Questions where some context hurts No ICL:   {num_context_hurts_noicl}/{len(test_set)}")

    print("\n==========================")
    print("PER-CONTEXT ACCURACIES")
    print("==========================")
    for i, acc in enumerate(context_accs):
        print(f"Context {i:02d}: {acc:.4f}")

    df = pd.DataFrame(all_records)
    matrix_df = pd.DataFrame(
        success_matrix,
        columns=[f"context_{i}" for i in range(args.num_contexts)],
    )
    matrix_df.insert(0, "question_id", [ex.get("question_id", i) for i, ex in enumerate(test_set)])
    matrix_df["no_icl"] = noicl_results
    matrix_df["oracle"] = success_matrix.max(axis=1)
    matrix_df["context_mean"] = question_context_mean
    matrix_df["context_var"] = question_context_var

    summary = {
        "mode": "oracle",
        "model": args.model,
        "math_config": args.math_config,
        "test_scope": args.test_scope,
        "split_index": args.split_index if args.test_scope == "subset" else None,
        "example_id": args.example_id if args.test_scope == "single" else None,
        "num_test": len(test_set),
        "num_contexts": args.num_contexts,
        "k": args.k,
        "noicl_acc": noicl_acc,
        "random_context_acc": random_context_acc,
        "oracle_context_acc": oracle_context_acc,
        "oracle_gain_vs_noicl": oracle_gain_vs_noicl,
        "oracle_gain_vs_random": oracle_gain_vs_random,
        "num_context_sensitive": num_context_sensitive,
        "num_oracle_improves_noicl": num_oracle_improves_noicl,
        "num_context_hurts_noicl": num_context_hurts_noicl,
    }

    if save:
        paths = result_paths(args)
        df.to_csv(paths["oracle_csv"], index=False)
        df.to_parquet(paths["oracle_parquet"], index=False)
        matrix_df.to_csv(paths["success_matrix_csv"], index=False)
        pd.DataFrame([summary]).to_csv(
            paths["oracle_summary_csv"],
            index=False,
        )

        print("\nSaved:")
        print(f"  {paths['oracle_csv']}")
        print(f"  {paths['oracle_parquet']}")
        print(f"  {paths['success_matrix_csv']}")
        print(f"  {paths['oracle_summary_csv']}")
        print(f"  {paths['fixed_contexts_csv']}")

    return df, matrix_df, summary, contexts
 
 
def main():
    args = parse_args()

    resources = load_experiment_resources(args)

    if args.mode == "no_icl":
        run_no_icl_experiment(resources, args, save=args.save_results)
    else:
        run_oracle_context_experiment(resources, args, save=args.save_results)
 
 
if __name__ == "__main__":
    main()
