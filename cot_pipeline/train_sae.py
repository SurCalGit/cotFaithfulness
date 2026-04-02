#!/usr/bin/env python3
"""
Standalone Sparse Autoencoder (SAE) training script.

All SAE architecture and training code is inlined here — no imports from
sparse_coding/. Trains a FunctionalTiedSAE ensemble over 9 L1 penalty values:
    l1_values = [0.0] + np.logspace(-4, -2, 8)

Index-to-l1_alpha mapping (use as --rank in patching scripts):
    rank 0 : 0.0000   (no sparsity)
    rank 1 : 0.0001
    rank 2 : 0.0002
    rank 3 : 0.0004
    rank 4 : 0.0007
    rank 5 : 0.0014
    rank 6 : 0.0027   ← used in the paper for Pythia-70M
    rank 7 : 0.0051
    rank 8 : 0.0100

Saves a nested dict in PyTorch format:
    {layer: {rank: (TiedSAE_instance, {"l1_alpha": ..., "dict_size": ...})}}

Usage:
    python train_sae.py \\
        --act_dir sae/cot_acts_l2_residual \\
        --output_path sae/cot_sae_l2_r4/learned_dicts.pt \\
        --layer 2 --activation_width 512 --dict_ratio 4 \\
        --batch_size 1024 --n_epochs 5
"""

import os
import argparse
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
import optree
import torch
import torch.nn as nn
import torchopt
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble helpers  (inlined from autoencoders/ensemble.py)
# ──────────────────────────────────────────────────────────────────────────────

class DictSignature:
    @staticmethod
    def to_learned_dict(params, buffers):
        pass

    @staticmethod
    def loss(params, buffers, batch):
        pass


def construct_stacked_leaf(tensors, device=None):
    all_rg = all(t.requires_grad for t in tensors)
    none_rg = all(not t.requires_grad for t in tensors)
    if not all_rg and not none_rg:
        raise RuntimeError("Mixed requires_grad across tensors")
    result = torch.stack(list(tensors)).to(device=device)
    if all_rg:
        result = result.detach().requires_grad_()
    return result


def stack_dict(models, device=None):
    tensors, treespecs = zip(*[optree.tree_flatten(m) for m in models])
    tensors = list(zip(*tensors))
    stacked = [construct_stacked_leaf(ts, device=device) for ts in tensors]
    return optree.tree_unflatten(treespecs[0], stacked)


def unstack_dict(params, n_models, device=None):
    tensors, treespec = optree.tree_flatten(params)
    tensors_ = [[] for _ in range(n_models)]
    for t in tensors:
        for i in range(n_models):
            tensors_[i].append(t[i].to(device=device))
    return [optree.tree_unflatten(treespec, ts) for ts in tensors_]


# ──────────────────────────────────────────────────────────────────────────────
# LearnedDict base class  (inlined from autoencoders/learned_dict.py)
# ──────────────────────────────────────────────────────────────────────────────

class LearnedDict(ABC):
    n_feats: int
    activation_size: int

    @abstractmethod
    def get_learned_dict(self):
        pass

    @abstractmethod
    def encode(self, batch):
        pass

    @abstractmethod
    def to_device(self, device):
        pass

    def decode(self, code):
        ld = self.get_learned_dict()
        return torch.einsum("nd,bn->bd", ld, code)

    def center(self, batch):
        return batch

    def uncenter(self, batch):
        return batch


# ──────────────────────────────────────────────────────────────────────────────
# TiedSAE  (inlined from autoencoders/learned_dict.py)
# ──────────────────────────────────────────────────────────────────────────────

class TiedSAE(LearnedDict):
    """
    Sparse autoencoder where decoder = normalised encoder^T.
    Supports optional centering (translation, rotation, scaling).
    """

    def __init__(self, encoder, encoder_bias, centering=(None, None, None), norm_encoder=True):
        self.encoder = encoder
        self.encoder_bias = encoder_bias
        self.norm_encoder = norm_encoder
        self.n_feats, self.activation_size = encoder.shape
        t, r, s = centering
        self.center_trans  = t if t is not None else torch.zeros(self.activation_size)
        self.center_rot    = r if r is not None else torch.eye(self.activation_size)
        self.center_scale  = s if s is not None else torch.ones(self.activation_size)

    def _ensure_centering(self):
        if not hasattr(self, "center_trans"):
            self.center_trans = torch.zeros(self.activation_size, device=self.encoder.device)
        if not hasattr(self, "center_rot"):
            self.center_rot = torch.eye(self.activation_size, device=self.encoder.device)
        if not hasattr(self, "center_scale"):
            self.center_scale = torch.ones(self.activation_size, device=self.encoder.device)

    def get_learned_dict(self):
        norms = torch.norm(self.encoder, 2, dim=-1)
        return self.encoder / torch.clamp(norms, 1e-8)[:, None]

    def center(self, batch):
        self._ensure_centering()
        return (
            torch.einsum("cu,bu->bc", self.center_rot, batch - self.center_trans[None, :])
            * self.center_scale[None, :]
        )

    def uncenter(self, batch):
        self._ensure_centering()
        return (
            torch.einsum("cu,bc->bu", self.center_rot, batch / self.center_scale[None, :])
            + self.center_trans[None, :]
        )

    def to_device(self, device):
        self._ensure_centering()
        self.encoder      = self.encoder.to(device)
        self.encoder_bias = self.encoder_bias.to(device)
        self.center_trans = self.center_trans.to(device)
        self.center_rot   = self.center_rot.to(device)
        self.center_scale = self.center_scale.to(device)

    def encode(self, batch):
        enc = (
            self.encoder / torch.clamp(torch.norm(self.encoder, 2, dim=-1), 1e-8)[:, None]
            if self.norm_encoder
            else self.encoder
        )
        c = torch.einsum("nd,bd->bn", enc, batch) + self.encoder_bias
        return torch.clamp(c, min=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# FunctionalTiedSAE  (inlined from autoencoders/sae_ensemble.py)
# ──────────────────────────────────────────────────────────────────────────────

class FunctionalTiedSAE(DictSignature):
    """
    Functional (stateless) tied SAE — stores all state in params/buffers dicts
    so that torch.vmap can vectorise across a batch of models.
    """

    @staticmethod
    def init(
        activation_size,
        n_dict_components,
        l1_alpha,
        bias_decay=0.0,
        device=None,
        dtype=None,
        translation=None,
        rotation=None,
        scaling=None,
    ):
        params, buffers = {}, {}

        buffers["center_rot"]   = rotation    if rotation    is not None else torch.eye(activation_size, device=device, dtype=dtype)
        buffers["center_trans"] = translation if translation is not None else torch.zeros(activation_size, device=device, dtype=dtype)
        buffers["center_scale"] = scaling     if scaling     is not None else torch.ones(activation_size, device=device, dtype=dtype)

        params["encoder"] = torch.empty((n_dict_components, activation_size), device=device, dtype=dtype)
        nn.init.xavier_uniform_(params["encoder"])

        params["encoder_bias"] = torch.zeros((n_dict_components,), device=device, dtype=dtype)

        buffers["l1_alpha"]   = torch.tensor(l1_alpha,   device=device, dtype=dtype)
        buffers["bias_decay"] = torch.tensor(bias_decay, device=device, dtype=dtype)

        return params, buffers

    @staticmethod
    def to_learned_dict(params, buffers):
        return TiedSAE(
            params["encoder"],
            params["encoder_bias"],
            centering=(
                buffers["center_trans"],
                buffers["center_rot"],
                buffers["center_scale"],
            ),
            norm_encoder=True,
        )

    @staticmethod
    def loss(params, buffers, batch):
        norms = torch.norm(params["encoder"], 2, dim=-1)
        ld = params["encoder"] / torch.clamp(norms, 1e-8)[:, None]

        # centre
        bc = (
            torch.einsum("cu,bu->bc", buffers["center_rot"], batch - buffers["center_trans"][None, :])
            * buffers["center_scale"][None, :]
        )

        c = torch.clamp(
            torch.einsum("nd,bd->bn", ld, bc) + params["encoder_bias"],
            min=0.0,
        )
        xhc = torch.einsum("nd,bn->bd", ld, c)

        l_rec = (xhc - bc).pow(2).mean()
        l_l1  = buffers["l1_alpha"]   * torch.norm(c, 1, dim=-1).mean()
        l_bd  = buffers["bias_decay"] * torch.norm(params["encoder_bias"], 2)

        loss_val = l_rec + l_l1 + l_bd
        loss_data = {"loss": loss_val, "l_reconstruction": l_rec, "l_l1": l_l1}
        return loss_val, (loss_data, {"c": c})


# ──────────────────────────────────────────────────────────────────────────────
# FunctionalEnsemble  (inlined from autoencoders/ensemble.py)
# ──────────────────────────────────────────────────────────────────────────────

class FunctionalEnsemble:
    """
    Vectorises a batch of SAE models using torch.vmap so all N models are
    trained in a single forward/backward pass.
    """

    def __init__(self, models, sig, optimizer_func, optimizer_kwargs, device=None):
        self.n_models = len(models)
        params_list, buffers_list = zip(*models)

        self.device = device if device is not None else params_list[0]["encoder"].device
        self.params  = stack_dict(list(params_list),  device=self.device)
        self.buffers = stack_dict(list(buffers_list), device=self.device)
        self.sig = sig

        self.optimizer_func   = optimizer_func
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer        = optimizer_func(**optimizer_kwargs)
        self.optim_states     = torch.vmap(self.optimizer.init)(self.params)

        def _calc_grads(p, b, batch):
            return torch.func.grad(self.sig.loss, has_aux=True)(p, b, batch)

        self.calc_grads = torch.vmap(_calc_grads)
        self.update     = torch.vmap(self.optimizer.update)

    def step_batch(self, minibatch):
        """
        minibatch: [batch_size, activation_width]
        Internally expands to [n_models, batch_size, activation_width].
        """
        with torch.no_grad():
            mb = minibatch.expand(self.n_models, *minibatch.shape)
            grads, (losses, aux) = self.calc_grads(self.params, self.buffers, mb)
            updates, new_optim_states = self.update(grads, self.optim_states)

            # write new optimizer states in-place
            new_leaves, _ = optree.tree_flatten(new_optim_states)
            old_leaves, _ = optree.tree_flatten(self.optim_states)
            for new_leaf, old_leaf in zip(new_leaves, old_leaves):
                old_leaf.copy_(new_leaf)

            torchopt.apply_updates(self.params, updates)
        return losses, aux

    def unstack(self, device=None):
        params_u  = unstack_dict(self.params,  self.n_models, device=device)
        buffers_u = unstack_dict(self.buffers, self.n_models, device=device)
        return list(zip(params_u, buffers_u))


# ──────────────────────────────────────────────────────────────────────────────
# Training function
# ──────────────────────────────────────────────────────────────────────────────

def train_and_save(
    act_dir: str,
    output_path: str,
    layer: int,
    activation_width: int,
    dict_ratio: int,
    batch_size: int = 1024,
    n_epochs: int = 5,
    lr: float = 1e-3,
    device: str = "cuda:0",
    dtype=torch.float32,
) -> None:
    """
    Train a FunctionalTiedSAE ensemble on pre-saved activation chunks.

    Args:
        act_dir:          Directory containing numbered activation chunks (0.pt, 1.pt, …).
        output_path:      Where to save the nested learned_dicts.pt.
        layer:            Layer index (used only as key in the output dict).
        activation_width: d_model of the base transformer (512 for Pythia-70M).
        dict_ratio:       SAE dictionary size = activation_width * dict_ratio.
        batch_size:       Mini-batch size during SGD.
        n_epochs:         Number of full passes over all chunks.
        lr:               Adam learning rate.
        device:           Torch device string.
        dtype:            Float dtype for SAE weights (float32 recommended).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    dict_size = activation_width * dict_ratio

    # 9 models: l1=0 plus 8 log-spaced values
    l1_values = np.concatenate([[0.0], np.logspace(-4, -2, 8)])
    print(f"L1 values: {[f'{v:.4f}' for v in l1_values]}")

    models = [
        FunctionalTiedSAE.init(
            activation_width,
            dict_size,
            float(l1),
            bias_decay=0.0,
            device=device,
            dtype=dtype,
        )
        for l1 in l1_values
    ]

    ensemble = FunctionalEnsemble(
        models,
        FunctionalTiedSAE,
        torchopt.adam,
        {"lr": lr},
        device=device,
    )

    # Discover activation chunk files
    chunk_files = sorted(
        [
            os.path.join(act_dir, f)
            for f in os.listdir(act_dir)
            if f.endswith(".pt") and f[:-3].isdigit()
        ],
        key=lambda p: int(os.path.basename(p)[:-3]),
    )
    if not chunk_files:
        raise FileNotFoundError(f"No numbered .pt chunk files found in {act_dir}")
    print(f"Found {len(chunk_files)} activation chunks in {act_dir}")

    torch.set_grad_enabled(False)

    for epoch in range(n_epochs):
        chunk_order = np.random.permutation(len(chunk_files))
        epoch_loss = []

        for ci, chunk_idx in enumerate(chunk_order):
            chunk = torch.load(chunk_files[chunk_idx], map_location="cpu").to(dtype=torch.float32)
            n = chunk.shape[0]

            perm = torch.randperm(n)
            chunk_loss = []

            pbar = tqdm(
                range(0, n, batch_size),
                desc=f"Epoch {epoch+1}/{n_epochs} | Chunk {ci+1}/{len(chunk_files)}",
                leave=False,
            )
            for start in pbar:
                idx = perm[start : start + batch_size]
                batch = chunk[idx].to(device)
                losses, _ = ensemble.step_batch(batch)
                mean_loss = losses.mean().item()
                chunk_loss.append(mean_loss)
                pbar.set_postfix(loss=f"{mean_loss:.4f}")

            epoch_loss.append(np.mean(chunk_loss))
            del chunk
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{n_epochs} — mean loss: {np.mean(epoch_loss):.4f}")

    # Unstack and convert to TiedSAE objects
    unstacked = ensemble.unstack(device="cpu")
    nested = {
        layer: {
            rank: (
                FunctionalTiedSAE.to_learned_dict(params, buffers),
                {"l1_alpha": float(l1_values[rank]), "dict_size": dict_size},
            )
            for rank, (params, buffers) in enumerate(unstacked)
        }
    }

    torch.save(nested, output_path)
    print(f"Saved {len(l1_values)} SAEs → {output_path}")
    print("Rank-to-l1_alpha mapping:")
    for rank, l1 in enumerate(l1_values):
        print(f"  rank {rank}: l1_alpha = {l1:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TiedSAE ensemble on pre-saved activations.")
    parser.add_argument("--act_dir",          required=True,  help="Directory of numbered activation .pt chunks")
    parser.add_argument("--output_path",      required=True,  help="Output path for learned_dicts.pt")
    parser.add_argument("--layer",            type=int, required=True, help="Layer index (used as dict key)")
    parser.add_argument("--activation_width", type=int, required=True, help="d_model of base transformer (512 for Pythia-70M)")
    parser.add_argument("--dict_ratio",       type=int, default=4,    help="SAE dict size = activation_width * dict_ratio")
    parser.add_argument("--batch_size",       type=int, default=1024)
    parser.add_argument("--n_epochs",         type=int, default=5)
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--device",           default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_and_save(
        act_dir          = args.act_dir,
        output_path      = args.output_path,
        layer            = args.layer,
        activation_width = args.activation_width,
        dict_ratio       = args.dict_ratio,
        batch_size       = args.batch_size,
        n_epochs         = args.n_epochs,
        lr               = args.lr,
        device           = args.device,
    )
