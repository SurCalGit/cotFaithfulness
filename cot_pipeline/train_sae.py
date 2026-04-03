#!/usr/bin/env python3
"""
Standalone Sparse Autoencoder (SAE) training script.

All SAE architecture and training code is inlined here.
Uses standard torch.optim.Adam — no torchopt or vmap dependencies.

Trains 9 TiedSAE models in parallel (one per L1 penalty value):
    l1_values = [0.0] + np.logspace(-4, -2, 8)

Index-to-l1_alpha mapping (use as --rank in patching scripts):
    rank 0 : 0.0000   (no sparsity)
    rank 1 : 0.0001
    rank 2 : 0.0002
    rank 3 : 0.0004
    rank 4 : 0.0007
    rank 5 : 0.0014
    rank 6 : 0.0027   ← used in the paper for Pythia-70M
    rank 7 : 0.0052
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

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# LearnedDict base class  (used by patching scripts via to_device / encode / decode)
# ──────────────────────────────────────────────────────────────────────────────

class LearnedDict(ABC):
    n_feats: int
    activation_size: int

    @abstractmethod
    def get_learned_dict(self): pass

    @abstractmethod
    def encode(self, batch): pass

    @abstractmethod
    def to_device(self, device): pass

    def decode(self, code):
        return torch.einsum("nd,bn->bd", self.get_learned_dict(), code)


# ──────────────────────────────────────────────────────────────────────────────
# TiedSAE  — inference-only class used by patching scripts
# ──────────────────────────────────────────────────────────────────────────────

class TiedSAE(LearnedDict):
    """
    Tied sparse autoencoder (decoder = normalised encoder^T).
    Supports optional centering (translation, rotation, scaling).
    This class is used only for saving/loading and inference in patching scripts.
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

    def to_device(self, device):
        self._ensure_centering()
        self.encoder       = self.encoder.to(device)
        self.encoder_bias  = self.encoder_bias.to(device)
        self.center_trans  = self.center_trans.to(device)
        self.center_rot    = self.center_rot.to(device)
        self.center_scale  = self.center_scale.to(device)

    def encode(self, batch):
        enc = (
            self.encoder / torch.clamp(torch.norm(self.encoder, 2, dim=-1), 1e-8)[:, None]
            if self.norm_encoder else self.encoder
        )
        return torch.clamp(torch.einsum("nd,bd->bn", enc, batch) + self.encoder_bias, min=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# TiedSAEModule  — nn.Module used during training
# ──────────────────────────────────────────────────────────────────────────────

class TiedSAEModule(nn.Module):
    """
    nn.Module wrapper around TiedSAE for training with torch.optim.
    Uses the same architecture as TiedSAE but supports autograd.
    """

    def __init__(self, activation_size: int, n_dict_components: int, l1_alpha: float, bias_decay: float = 0.0):
        super().__init__()
        self.activation_size  = activation_size
        self.n_feats          = n_dict_components
        self.l1_alpha         = l1_alpha
        self.bias_decay       = bias_decay

        self.encoder      = nn.Parameter(torch.empty(n_dict_components, activation_size))
        self.encoder_bias = nn.Parameter(torch.zeros(n_dict_components))
        nn.init.xavier_uniform_(self.encoder)

        self.register_buffer("center_trans", torch.zeros(activation_size))
        self.register_buffer("center_rot",   torch.eye(activation_size))
        self.register_buffer("center_scale", torch.ones(activation_size))

    def _center(self, batch):
        return (
            torch.einsum("cu,bu->bc", self.center_rot, batch - self.center_trans[None, :])
            * self.center_scale[None, :]
        )

    def forward(self, batch):
        """Returns reconstruction loss + sparsity loss."""
        norms = torch.norm(self.encoder, 2, dim=-1)
        ld    = self.encoder / torch.clamp(norms, 1e-8)[:, None]

        bc  = self._center(batch)
        c   = torch.clamp(torch.einsum("nd,bd->bn", ld, bc) + self.encoder_bias, min=0.0)
        xhc = torch.einsum("nd,bn->bd", ld, c)

        l_rec = (xhc - bc).pow(2).mean()
        l_l1  = self.l1_alpha  * torch.norm(c, 1, dim=-1).mean()
        l_bd  = self.bias_decay * torch.norm(self.encoder_bias, 2)
        return l_rec + l_l1 + l_bd

    def to_tied_sae(self) -> TiedSAE:
        """Export to TiedSAE for saving / use in patching scripts."""
        return TiedSAE(
            self.encoder.detach().cpu().clone(),
            self.encoder_bias.detach().cpu().clone(),
            centering=(
                self.center_trans.detach().cpu().clone(),
                self.center_rot.detach().cpu().clone(),
                self.center_scale.detach().cpu().clone(),
            ),
            norm_encoder=True,
        )


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
    Train one TiedSAEModule per l1_alpha value on pre-saved activation chunks.

    Args:
        act_dir:          Directory containing numbered activation chunks (0.pt, 1.pt, …).
        output_path:      Where to save the nested learned_dicts.pt.
        layer:            Layer index (used as key in the output dict).
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
    l1_values = np.concatenate([[0.0], np.logspace(-4, -2, 8)])
    print(f"L1 values: {[f'{v:.4f}' for v in l1_values]}")

    # Initialise one module + one Adam optimiser per l1_alpha
    models = [
        TiedSAEModule(activation_width, dict_size, float(l1)).to(device)
        for l1 in l1_values
    ]
    optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]

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

    for epoch in range(n_epochs):
        chunk_order = np.random.permutation(len(chunk_files))
        epoch_losses = [[] for _ in models]

        for ci, chunk_idx in enumerate(chunk_order):
            chunk = torch.load(chunk_files[chunk_idx], map_location="cpu").to(dtype=torch.float32)
            n     = chunk.shape[0]
            perm  = torch.randperm(n)

            pbar = tqdm(
                range(0, n, batch_size),
                desc=f"Epoch {epoch+1}/{n_epochs} | Chunk {ci+1}/{len(chunk_files)}",
                leave=False,
            )
            for start in pbar:
                batch = chunk[perm[start : start + batch_size]].to(device)
                batch_losses = []
                for model, optimizer in zip(models, optimizers):
                    optimizer.zero_grad()
                    loss = model(batch)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                for i, l in enumerate(batch_losses):
                    epoch_losses[i].append(l)
                pbar.set_postfix(loss=f"{np.mean(batch_losses):.4f}")

            del chunk
            torch.cuda.empty_cache()

        mean_losses = [np.mean(el) for el in epoch_losses]
        print(
            f"Epoch {epoch+1}/{n_epochs} — "
            f"mean loss: {np.mean(mean_losses):.4f}  "
            f"(min {min(mean_losses):.4f} / max {max(mean_losses):.4f})"
        )

    # Export to TiedSAE and save in nested format expected by patching scripts
    nested = {
        layer: {
            rank: (
                model.to_tied_sae(),
                {"l1_alpha": float(l1_values[rank]), "dict_size": dict_size},
            )
            for rank, model in enumerate(models)
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
