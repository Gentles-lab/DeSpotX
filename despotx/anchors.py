"""Anchor mask: (cell-type, gene) positions where the gene is non-expressed."""

import numpy as np


def compute_anchors(
    adata,
    cell_type_col,
    min_non_expr_frac = 0.99,
    adaptive = False,
    adaptive_k = 0.2,
    adaptive_min = 0.005,
    adaptive_max = 0.25,
    hierarchical = False,
    specificity_quantile = 0.5,
    homogeneity_min = None,
    spatial_key = "spatial",
    k_neighbors = 12,
    verbose = True,
):
    """Compute the (n_types, n_genes) Boolean anchor mask.

    An entry ``A[t, g] = True`` means gene g is expressed in < ``threshold_g``
    fraction of type-t cells, so observed counts at (type t, gene g) must be
    attributed to ambient contamination."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    X = adata.X
    ct = np.asarray(adata.obs[cell_type_col].values)
    types, ct_int = np.unique(ct, return_inverse=True)
    n_types, n_genes = len(types), adata.n_vars

    cell_mask = np.ones(len(ct), dtype=bool)
    if homogeneity_min is not None and spatial_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        if coords.shape[1] == 2:
            tree = cKDTree(coords)
            _, nbr_idx = tree.query(coords, k=k_neighbors + 1)
            nbr_idx = nbr_idx[:, 1:]
            same = (ct_int[nbr_idx] == ct_int[:, None]).mean(axis=1)
            cell_mask = same >= homogeneity_min
            if verbose:
                print(f"[anchors] homogeneity_min={homogeneity_min}: "
                      f"{cell_mask.sum()}/{len(ct)} cells kept")

    rate = np.zeros((n_types, n_genes), dtype=np.float32)
    for t in range(n_types):
        rows = np.where((ct_int == t) & cell_mask)[0]
        if len(rows) == 0:
            rows = np.where(ct_int == t)[0]
        if len(rows) == 0:
            continue
        Xt = X[rows]
        if sp.issparse(Xt):
            rate[t] = np.asarray((Xt > 0).mean(axis=0)).ravel()
        else:
            rate[t] = (np.asarray(Xt) > 0).mean(axis=0)

    if adaptive:
        thresholds = np.clip(adaptive_k * rate.max(axis=0),
                             adaptive_min, adaptive_max)
    else:
        thresholds = np.full(n_genes, 1.0 - min_non_expr_frac)

    mask = rate < thresholds[None, :]

    if hierarchical:
        specificity = rate.std(axis=0)
        threshold = np.quantile(specificity, specificity_quantile)
        is_state = specificity < threshold
        mask[:, is_state] = False
        if verbose:
            print(f"[anchors] hierarchical: {is_state.sum()}/{n_genes} genes "
                  f"flagged as state markers (anchor disabled)")

    if verbose:
        print(f"[anchors] mask: {mask.sum()}/{mask.size} positions "
              f"({100*mask.mean():.1f}% anchored)")
    return mask
