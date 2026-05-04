"""Spatial dataset: AnnData + K-NN spatial graph yielding per-cell star batches."""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset


def build_knn(coords, k):
    tree = cKDTree(coords)
    dists, idx = tree.query(coords, k=k + 1)
    return idx[:, 1:].astype(np.int64), dists[:, 1:].astype(np.float32)


class SpatialDataset(Dataset):
    """Torch dataset that indexes into an AnnData row by row."""

    def __init__(self, adata, cell_type_col, spatial_key = "spatial",
                 k_neighbors = 12):
        import scipy.sparse as sp

        X = adata.X
        self.sparse = sp.issparse(X)
        self.X = X if self.sparse else np.asarray(X, dtype=np.float32)

        ct = np.asarray(adata.obs[cell_type_col].values)
        self.ct_categories, self.ct_int = np.unique(ct, return_inverse=True)
        self.ct_int = self.ct_int.astype(np.int64)

        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float32)
        assert coords.shape[1] == 2, "Expected 2D spatial coordinates"
        self.coords = coords

        self.nbr_idx, self.nbr_dist = build_knn(coords, k_neighbors)
        self.k = k_neighbors
        self.n_cells, self.n_genes = self.X.shape
        self.n_types = len(self.ct_categories)

    def _row(self, i):
        if self.sparse:
            return self.X[i].toarray().ravel().astype(np.float32)
        return self.X[i].astype(np.float32)

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        nbr = self.nbr_idx[idx]
        offsets = (self.coords[nbr] - self.coords[idx]).astype(np.float32)
        return {
            "counts": torch.from_numpy(self._row(idx)),
            "cell_type": torch.tensor(self.ct_int[idx], dtype=torch.long),
            "neighbor_counts": torch.from_numpy(
                np.stack([self._row(j) for j in nbr], axis=0)),
            "neighbor_types": torch.from_numpy(self.ct_int[nbr]),
            "neighbor_dists": torch.from_numpy(self.nbr_dist[idx]),
            "neighbor_offsets": torch.from_numpy(offsets),
        }


def collate(items):
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def move_batch(batch, device):
    """Send a batch dict to the target device."""
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}
