#!/usr/bin/env python3
"""
End-to-end pipeline for reproducing the CoT-faithfulness paper results
on Pythia-70M / GSM8K.

Steps performed:
  1. Collect CoT   activations  →  sae/cot_acts_l{layer}_{layer_loc}/
  2. Collect NoCoT activations  →  sae/nocot_acts_l{layer}_{layer_loc}/
  3. Train CoT   SAE (ratio 4 and 8)  →  sae/cot_sae_l{layer}_r{ratio}/learned_dicts.pt
  4. Train NoCoT SAE (ratio 4 and 8)  →  sae/nocot_sae_l{layer}_r{ratio}/learned_dicts.pt

If activation or SAE files already exist for a given step the step is skipped.

After the pipeline finishes it prints the exact commands to run
activation_patching.py and patch_curve.py.

Usage:
    python pipeline.py \\
        --model EleutherAI/pythia-70m-deduped \\
        --layer 2 --layer_loc residual \\
        --dict_ratios 4 8 \\
        --n_chunks 16 --n_epochs 5 --batch_size 1024
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# ─── resolve sparse_coding on sys.path ────────────────────────────────────────
_REPO_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SPARSE_CODING = os.path.join(_REPO_ROOT, "sparse_coding")
if _SPARSE_CODING not in sys.path:
    sys.path.insert(0, _SPARSE_CODING)

from transformer_lens import HookedTransformer
from activation_dataset import (
    make_sentence_dataset_with_cot,
    make_sentence_dataset_with_Nocot,
    make_activation_dataset_tl,
    get_activation_size,
    simple_tokenize_per_sample,
    collate_fn,
    MAX_SENTENCE_LEN,
    MODEL_BATCH_SIZE,
)

import train_sae  # same directory


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _act_dir(base: str, mode: str, layer: int, layer_loc: str) -> str:
    return os.path.join(base, f"{mode}_acts_l{layer}_{layer_loc}")


def _sae_dir(base: str, mode: str, layer: int, ratio: int) -> str:
    return os.path.join(base, f"{mode}_sae_l{layer}_r{ratio}")


def _chunk_count(directory: str) -> int:
    """Return the number of numbered activation chunks already saved."""
    if not os.path.isdir(directory):
        return 0
    return sum(
        1 for f in os.listdir(directory)
        if f.endswith(".pt") and f[:-3].isdigit()
    )


def _sae_exists(sae_dir: str) -> bool:
    return os.path.isfile(os.path.join(sae_dir, "learned_dicts.pt"))


# ──────────────────────────────────────────────────────────────────────────────
# Activation collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_activations(
    mode: str,
    model_name: str,
    dataset_name: str,
    act_dir: str,
    layer: int,
    layer_loc: str,
    n_chunks: int,
    device: str,
    chunk_size_gb: float = 2.0,
) -> None:
    """
    Run the model on GSM8K and save residual-stream activations.

    mode : "cot" prepends few-shot CoT examples; "nocot" just wraps the question.
    Saves numbered chunk files  0.pt, 1.pt, … and
    input_ids_layer{layer}_chunk{i}.pt  in act_dir.
    """
    os.makedirs(act_dir, exist_ok=True)

    activation_width = get_activation_size(model_name, layer_loc)
    # conservative upper bound on lines needed
    max_lines = int((chunk_size_gb * 1e9 * n_chunks) / (activation_width * 1000 * 2))

    print(f"\n>>> Loading model {model_name}")
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        torch_dtype=torch.bfloat16,
        fold_ln=False,
        center_writing_weights=False,
    )
    model.eval()

    print(f">>> Building {'CoT' if mode == 'cot' else 'NoCoT'} dataset")
    if mode == "cot":
        dataset = make_sentence_dataset_with_cot(dataset_name, max_lines=max_lines)
    else:
        dataset = make_sentence_dataset_with_Nocot(dataset_name, max_lines=max_lines)

    tokenized = simple_tokenize_per_sample(
        dataset, model.tokenizer, max_length=MAX_SENTENCE_LEN
    )
    loader = DataLoader(
        tokenized,
        batch_size=MODEL_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print(f">>> Collecting activations  →  {act_dir}")
    make_activation_dataset_tl(
        sentence_dataset   = loader,
        model              = model,
        activation_width   = activation_width,
        dataset_folders    = [act_dir],
        layers             = [layer],
        tensor_loc         = layer_loc,
        chunk_size_gb      = chunk_size_gb,
        device             = device,
        n_chunks           = n_chunks,
        max_length         = MAX_SENTENCE_LEN,
        model_batch_size   = MODEL_BATCH_SIZE,
    )

    del model
    torch.cuda.empty_cache()
    print(f">>> Done — {_chunk_count(act_dir)} chunks saved to {act_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(args) -> None:
    sae_base = os.path.join(os.path.dirname(__file__), "sae")
    os.makedirs(sae_base, exist_ok=True)

    activation_width = get_activation_size(args.model, args.layer_loc)
    print(f"Model: {args.model}  |  layer: {args.layer}  |  layer_loc: {args.layer_loc}")
    print(f"activation_width = {activation_width}")

    # ── Step 1 & 2 : activation collection ────────────────────────────────────
    for mode in ("cot", "nocot"):
        act_dir = _act_dir(sae_base, mode, args.layer, args.layer_loc)
        existing = _chunk_count(act_dir)
        if existing >= args.n_chunks:
            print(f"\n[SKIP] {mode.upper()} activations already present ({existing} chunks) in {act_dir}")
        else:
            print(f"\n[RUN ] Collecting {mode.upper()} activations ({args.n_chunks} chunks requested, {existing} found)")
            collect_activations(
                mode         = mode,
                model_name   = args.model,
                dataset_name = args.dataset,
                act_dir      = act_dir,
                layer        = args.layer,
                layer_loc    = args.layer_loc,
                n_chunks     = args.n_chunks,
                device       = args.device,
                chunk_size_gb= args.chunk_size_gb,
            )

    # ── Step 3 & 4 : SAE training ──────────────────────────────────────────────
    for mode in ("cot", "nocot"):
        act_dir = _act_dir(sae_base, mode, args.layer, args.layer_loc)
        for ratio in args.dict_ratios:
            sae_dir   = _sae_dir(sae_base, mode, args.layer, ratio)
            out_path  = os.path.join(sae_dir, "learned_dicts.pt")
            if _sae_exists(sae_dir):
                print(f"\n[SKIP] {mode.upper()} SAE (ratio={ratio}) already trained: {out_path}")
            else:
                print(f"\n[RUN ] Training {mode.upper()} SAE (ratio={ratio})")
                os.makedirs(sae_dir, exist_ok=True)
                train_sae.train_and_save(
                    act_dir          = act_dir,
                    output_path      = out_path,
                    layer            = args.layer,
                    activation_width = activation_width,
                    dict_ratio       = ratio,
                    batch_size       = args.batch_size,
                    n_epochs         = args.n_epochs,
                    lr               = args.lr,
                    device           = args.device,
                )

    # ── Print patching commands ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE.  Run the following commands to evaluate:\n")
    for ratio in args.dict_ratios:
        cot_dict   = os.path.join(_sae_dir(sae_base, "cot",   args.layer, ratio), "learned_dicts.pt")
        nocot_dict = os.path.join(_sae_dir(sae_base, "nocot", args.layer, ratio), "learned_dicts.pt")
        cot_acts   = _act_dir(sae_base, "cot",   args.layer, args.layer_loc)
        nocot_acts = _act_dir(sae_base, "nocot", args.layer, args.layer_loc)

        results_dir = os.path.join(os.path.dirname(__file__), "results")

        print(f"# ── dict_ratio={ratio}  (rank 6 ≈ l1_alpha=0.0027, best for Pythia-70M) ──")
        print(
            f"python activation_patching.py \\\n"
            f"    --model {args.model} \\\n"
            f"    --layer {args.layer} --layer_loc {args.layer_loc} --rank 6 \\\n"
            f"    --dict_cot   {cot_dict} \\\n"
            f"    --dict_nocot {nocot_dict} \\\n"
            f"    --acts_cot_dir   {cot_acts} \\\n"
            f"    --acts_nocot_dir {nocot_acts} \\\n"
            f"    --topk 20 --max_samples 2000 \\\n"
            f"    --out {results_dir}/patch_hist_r{ratio}.png\n"
        )
        print(
            f"python patch_curve.py \\\n"
            f"    --model {args.model} \\\n"
            f"    --layer {args.layer} --layer_loc {args.layer_loc} --rank 6 \\\n"
            f"    --dict_cot   {cot_dict} \\\n"
            f"    --dict_nocot {nocot_dict} \\\n"
            f"    --acts_cot_dir   {cot_acts} \\\n"
            f"    --acts_nocot_dir {nocot_acts} \\\n"
            f"    --max_samples 2000 \\\n"
            f"    --out {results_dir}/patch_curve_r{ratio}.png \\\n"
            f"    --save_stats {results_dir}/ttest_r{ratio}.txt\n"
        )
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full CoT-faithfulness pipeline: collect activations + train SAEs."
    )
    parser.add_argument("--model",         default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--dataset",       default="openai/gsm8k")
    parser.add_argument("--layer",         type=int, default=2)
    parser.add_argument("--layer_loc",     default="residual")
    parser.add_argument("--dict_ratios",   type=int, nargs="+", default=[4, 8],
                        help="SAE dictionary ratios to train (default: 4 8)")
    parser.add_argument("--n_chunks",      type=int,   default=16)
    parser.add_argument("--n_epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",    type=int,   default=1024)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--chunk_size_gb", type=float, default=2.0)
    parser.add_argument("--device",        default="cuda:0" if torch.cuda.is_available() else "cpu")

    run_pipeline(parser.parse_args())
