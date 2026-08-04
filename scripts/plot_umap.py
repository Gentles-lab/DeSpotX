"""Plot side-by-side UMAP of raw vs DeSpotX-decontaminated counts."""

import argparse
import sys
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import warnings


def umap(adata, seed=0):
    a = adata.copy()
    a.X = a.X.astype(np.float32)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=30, random_state=seed)
    sc.pp.neighbors(a, n_neighbors=15, random_state=seed)
    sc.tl.umap(a, random_state=seed)
    return a


def main():
    p = argparse.ArgumentParser(
        description="Side-by-side UMAP of raw vs DeSpotX output")
    p.add_argument("--raw", required=True,
                   help="Path to raw input .h5ad")
    p.add_argument("--adjusted", required=True,
                   help="Path to DeSpotX-adjusted .h5ad")
    p.add_argument("--cell-type-col", default="cell_type",
                   help="adata.obs column with cell-type labels")
    p.add_argument("--out", required=True,
                   help="Output PNG path")
    p.add_argument("--min-cells-per-type", type=int, default=100,
                   help="Cell types with fewer cells get grouped into 'Other'")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    warnings.filterwarnings("ignore")

    # Pin global RNG in addition to the per-call random_state below, so the
    # figure is byte-reproducible across runs at a fixed --seed.
    np.random.seed(args.seed)
    sc.settings.seed = args.seed

    print(f"[plot_umap] loading {args.raw}")
    raw = ad.read_h5ad(args.raw)
    print(f"[plot_umap] loading {args.adjusted}")
    adj = ad.read_h5ad(args.adjusted)

    print(f"[plot_umap] computing UMAP (raw)")
    raw_u = umap(raw, seed=args.seed)
    print(f"[plot_umap] computing UMAP (DeSpotX)")
    adj_u = umap(adj, seed=args.seed)

    ct = raw.obs[args.cell_type_col].astype(str)
    counts = ct.value_counts()
    keep = set(counts[counts >= args.min_cells_per_type].index)
    ct_lab = ct.map(lambda x: x if x in keep else "Other")
    raw_u.obs["ct_plot"] = ct_lab.values
    adj_u.obs["ct_plot"] = ct_lab.values

    unique = sorted(ct_lab.unique())
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(unique), 20)))
    colors = {t: cmap[i] for i, t in enumerate(unique)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, ad_obj, title in [(axes[0], raw_u, "Raw"),
                                (axes[1], adj_u, "DeSpotX-decontaminated")]:
        coords = ad_obj.obsm["X_umap"]
        for t in unique:
            m = (ad_obj.obs["ct_plot"] == t).values
            ax.scatter(coords[m, 0], coords[m, 1],
                        s=4, alpha=0.7, color=colors[t], label=t,
                        edgecolors="none", rasterized=True)
        ax.set_title(title, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    fontsize=8, frameon=False, markerscale=2)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_umap] wrote {out}")


if __name__ == "__main__":
    main()
