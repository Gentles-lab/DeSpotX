"""End-to-end CLI for DeSpotX."""

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io as sio
import torch


def main():
    p = argparse.ArgumentParser(description="DeSpotX decontamination")
    p.add_argument("--h5ad", required=True, help="Path to input .h5ad")
    p.add_argument("--cell-type-col", default="cell_type",
                   help="adata.obs column with cell-type labels")
    p.add_argument("--spatial-key", default="spatial",
                   help="adata.obsm key with 2D spatial coordinates")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"] +
                   [f"cuda:{i}" for i in range(8)],
                   help="'auto' selects cuda if available")
    # Anchor settings
    p.add_argument("--adaptive-k", type=float, default=0.3,
                   help="Adaptive per-gene threshold multiplier")
    p.add_argument("--k-neighbors", type=int, default=12,
                   help="Spatial K-NN graph degree")
    # Model & training
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--anchor-weight", type=float, default=50.0)
    p.add_argument("--anchor-output-mask",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Set counts at anchor positions to zero in the output "
                        "(default: enabled). --no-anchor-output-mask keeps the "
                        "anchor penalty during training but returns the soft "
                        "reweighted counts at anchor positions.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--amp", action="store_true",
                   help="Use CUDA automatic mixed precision")
    p.add_argument("--eps-prior-weight", type=float, default=1.0,
                   help="Weight on the Beta prior over per-cell ε. "
                        "0 = ablation: ε is no longer pulled toward μ_ε.")
    p.add_argument("--eps-head-init", type=float, default=0.20,
                   help="Initial sigmoid output of the ε-head before training "
                        "(default: 0.20, the paper-default architectural prior). "
                        "Set to 0.5 to remove the architectural ε prior — used by "
                        "the strong identifiability ablation.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from . import (DeSpotX, SpatialDataset, compute_anchors,
                     train, adjust_counts)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[despotx] loading {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    if args.cell_type_col not in adata.obs.columns:
        raise ValueError(f"adata.obs is missing '{args.cell_type_col}'")
    if args.spatial_key not in adata.obsm:
        raise ValueError(f"adata.obsm is missing '{args.spatial_key}'")
    adata.obs[args.cell_type_col] = adata.obs[args.cell_type_col].astype("category")
    print(f"[despotx]   shape={adata.shape}  types={adata.obs[args.cell_type_col].nunique()}")

    # NB likelihood + χ estimator require raw integer counts.
    X = adata.X
    sample = X.data[:200000] if sp.issparse(X) else np.asarray(X).ravel()[:200000]
    sample = np.asarray(sample, dtype=np.float64)
    nz = sample[sample != 0]
    if nz.size and (nz < 0).any():
        raise ValueError(
            "adata.X has negative values; DeSpotX requires raw integer counts.")
    integer_frac = float(((nz % 1) == 0).mean()) if nz.size else 1.0
    if integer_frac < 0.99:
        max_val = float(nz.max()) if nz.size else 0.0
        raise ValueError(
            f"adata.X does not look like raw integer counts "
            f"(integer_frac={integer_frac:.3f}, max_nonzero={max_val:.3f}). "
            f"DeSpotX requires raw counts, not log/size-normalised values. "
            f"If your h5ad has raw counts in a layer (e.g. layers['count']), "
            f"copy them to .X before running.")
    print(f"[despotx]   counts validated: integer_frac={integer_frac:.3f}")

    print(f"[despotx] computing anchor mask  adaptive_k={args.adaptive_k}")
    anchor_mask = compute_anchors(
        adata, cell_type_col=args.cell_type_col,
        adaptive=True, adaptive_k=args.adaptive_k,
        hierarchical=False, homogeneity_min=None,
        spatial_key=args.spatial_key, k_neighbors=args.k_neighbors,
    )

    dataset = SpatialDataset(adata, cell_type_col=args.cell_type_col,
                              spatial_key=args.spatial_key,
                              k_neighbors=args.k_neighbors)
    print(f"[despotx]   k-NN graph built, k={args.k_neighbors}")

    # --anchor-weight 0 also disables inference-time (1 - A_{t,g}) zeroing.
    eff_anchor_mask = anchor_mask if args.anchor_weight > 0 else None
    model = DeSpotX(n_genes=dataset.n_genes, n_types=dataset.n_types,
                     z_dim=args.z_dim, anchor_mask=eff_anchor_mask,
                     anchor_weight=args.anchor_weight,
                     edge_feat="kernel",
                     chi_mode="other_type_hard",
                     chi_soft_leak=0.1,
                     eps_prior_weight=args.eps_prior_weight,
                     mu_eps_strategy="clamp",
                     mu_eps_hyperprior_weight=1.0,
                     n_gat_layers=1,
                     eps_head_init=args.eps_head_init,
                     apply_output_mask=args.anchor_output_mask)
    model.warm_start_decoder(adata, cell_type_col=args.cell_type_col,
                              set_decoder_shift=True)

    print(f"[despotx] training for {args.epochs} epochs")
    train(model, dataset, n_epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, device=args.device, num_workers=args.num_workers,
          amp=args.amp)

    print(f"[despotx] inference")
    result = adjust_counts(model, dataset, batch_size=256,
                             device=args.device,
                             num_workers=args.num_workers)
    adjusted = result["adjusted"]
    epsilon = result["epsilon"]

    # MTX is genes × cells.
    mat = sp.csr_matrix(np.round(adjusted).astype(np.int32).T)
    sio.mmwrite(str(out / "adjusted.mtx"), mat)
    np.save(out / "adjusted_float.npy", adjusted)
    np.save(out / "anchor_mask.npy", anchor_mask)
    pd.Series(adata.var_names).to_csv(out / "genes.tsv", header=False, index=False)
    pd.Series(adata.obs_names).to_csv(out / "cells.tsv", header=False, index=False)
    pd.DataFrame({"cell_id": adata.obs_names,
                  "contamination": epsilon}).to_csv(
        out / "contamination.tsv", sep="\t", index=False)

    adj_adata = adata.copy()
    adj_adata.X = sp.csr_matrix(adjusted.astype(np.float32))
    adj_adata.obs["epsilon"] = epsilon
    adj_adata.write_h5ad(out / "adjusted.h5ad", compression="gzip")

    removed_frac = 1.0 - adjusted.sum() / (
        adata.X.sum() if sp.issparse(adata.X) else adata.X.sum())
    print(f"[despotx] done. wrote {out}")
    print(f"[despotx]   removed fraction : {100 * removed_frac:.2f}%")
    print(f"[despotx]   mean epsilon     : {epsilon.mean():.3f}")


if __name__ == "__main__":
    main()
