#!/usr/bin/env python
"""Minimal terminal smoke test for In-Context Vectors (ICV).

Loads a causal LM, builds an in-context vector (ICV) from a handful of
demonstration pairs, and prints the model's generation BEFORE vs AFTER steering
for a query. This mirrors `demo_notebook.ipynb`, but is runnable from the shell
via `inference.sh`.
"""
import argparse
import os


def parse_args():
    ap = argparse.ArgumentParser(description="ICV inference smoke test")
    ap.add_argument("--model-type", default="falcon", choices=["falcon", "llama", "llama-2", "vicuna"])
    ap.add_argument("--model-size", default="7b")
    ap.add_argument("--in-8bit", default=True, type=lambda v: str(v).lower() in ("1", "true", "yes", "y"))
    ap.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--demo", default="sentiment", choices=["sentiment", "safety", "both"])
    return ap.parse_args()


args = parse_args()
# IMPORTANT: pin visible GPUs BEFORE importing torch, otherwise torch enumerates
# all GPUs at import time and bitsandbytes hits a device-count assertion.
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch  # noqa: E402

from common import setup_env  # noqa: E402
from models import build_model_signature, build_tokenizer, build_model  # noqa: E402
from tasks import load_task  # noqa: E402
from utils.llm_layers import add_icv_layers, remove_icv_layers  # noqa: E402


# Demonstration pairs (input -> rewritten target). The ICV is the direction in
# hidden space that moves "input style" toward "target style".
DEMOS = {
    "sentiment": {
        "lam": 0.10,
        "pairs": [
            ("Zero stars, I hate it.", "Five stars, I love it."),
            ("it was terrible !", "it was awesome!"),
            ("i did nt like it.", "i love it."),
            ("i would call this the worse denny 's ever ", "i would call this the best denny 's ever "),
            ("i would recommend find another place.", "i would recommend this place again!"),
        ],
        "query": "Please paraphrase the following sentence. Sentence: Worst restaurant ever!, paraphrase: ",
        "gen": dict(max_new_tokens=15, do_sample=True, temperature=0.7, top_p=0.75, top_k=50),
        "eos": [104, 193, 1001, 25, 1702, 18858, 3166],
    },
    "safety": {
        "lam": 0.12,
        "pairs": [
            ("Women are inferior to men",
             "The belief that women are inferior to men is an outdated and biased perspective."),
            ("I think many of Jews are very money-driven.",
             "Making generalizations about a group of people is derogatory and unfair."),
            ("Trans person don't deserve any help.",
             "Every person, regardless of gender identity, deserves respect and support."),
        ],
        "query": "This is a conversation between two people. Context: you people are all worthless. Response: ",
        "gen": dict(max_new_tokens=32, do_sample=True, temperature=0.45, top_k=10),
        "eos": [104, 193],
    },
}


def tokenize_each_demonstration(demonstration_list, tokenizer, prefix=("", "")):
    """Tokenize (input, target) pairs after stripping punctuation/special chars.

    Returns a list of (enc_input, enc_target) BatchEncoding tuples, the format
    expected by BaseProbInference.obtain_icv.
    """
    special = "~!@#$%^&*()_+`-={}[]|\\:;\"'<>,.?/"

    def clean(s):
        for ch in special:
            s = s.replace(ch, "")
        return s.strip()

    out = []
    for src, tgt in demonstration_list:
        enc_src = tokenizer(prefix[0] + clean(src))
        enc_tgt = tokenizer(prefix[1] + clean(tgt))
        out.append((enc_src, enc_tgt))
    return out


def generate(model, tokenizer, query, gen_kwargs, eos_ids):
    enc = tokenizer(query)
    out = model.generate(
        input_ids=torch.tensor(enc["input_ids"]).unsqueeze(0).cuda(),
        attention_mask=torch.tensor(enc["attention_mask"]).unsqueeze(0).cuda(),
        num_return_sequences=1,
        eos_token_id=eos_ids + [tokenizer.eos_token_id],
        **gen_kwargs,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def run_demo(name, model, tokenizer, task_agent):
    cfg = DEMOS[name]
    print(f"\n{'=' * 70}\nDEMO: {name}  (steering strength lam={cfg['lam']})\n{'=' * 70}")

    icv, _ = task_agent.obtain_icv(
        model, tokenize_each_demonstration(cfg["pairs"], tokenizer), rank=1
    )
    icv = icv[1:]  # drop embedding layer

    print(f"\nQuery: {cfg['query']!r}")
    # Reseed before each generation so BEFORE vs AFTER use identical sampling
    # randomness — the only difference is the in-context vector.
    torch.manual_seed(args.seed)
    print(f"\n[BEFORE steering]\n{generate(model, tokenizer, cfg['query'], cfg['gen'], cfg['eos'])}")

    add_icv_layers(model, torch.stack([icv], dim=1).cuda(), [cfg["lam"]])
    torch.manual_seed(args.seed)
    print(f"\n[AFTER  steering]\n{generate(model, tokenizer, cfg['query'], cfg['gen'], cfg['eos'])}")
    remove_icv_layers(model)


def main():
    setup_env(gpu_s=args.gpus, seed=args.seed)
    torch.autograd.set_grad_enabled(False)

    signature = build_model_signature(args.model_type, args.model_size)
    print(f"Loading model: {signature}  (8-bit={args.in_8bit}, gpus={args.gpus})")
    tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side="right")
    model = build_model(args.model_type, args.model_size, args.in_8bit)
    print("Model loaded.")

    task_agent = load_task("demo")("default")
    task_agent.set_seed(args.seed)

    demos = ["sentiment", "safety"] if args.demo == "both" else [args.demo]
    for name in demos:
        run_demo(name, model, tokenizer, task_agent)

    print("\nDone. (If 'AFTER' differs from 'BEFORE', the in-context vector is steering generation.)")


if __name__ == "__main__":
    main()
