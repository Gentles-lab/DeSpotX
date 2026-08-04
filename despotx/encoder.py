"""GATv2 spatial encoder over a per-cell star graph (deterministic z)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


def _dist_weights(dists):
    # Floor sigma at 1e-6 (not 1.0) so kernel scales with the data's coordinate units.
    sigma = dists.median(dim=-1, keepdim=True).values.clamp(min=1e-6)
    w = torch.exp(-dists / sigma)
    return w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def _build_star_edge_index(B, K, device):
    blocks = []
    for i in range(B):
        c = i * (K + 1)
        nbrs = torch.arange(1, K + 1, device=device) + c
        center = torch.full((K,), c, device=device, dtype=torch.long)
        blocks.append(torch.stack([nbrs, center], dim=0))
        blocks.append(torch.stack([center, nbrs], dim=0))
    return torch.cat(blocks, dim=1)


class GATEncoder(nn.Module):
    def __init__(self, n_genes, n_types, z_dim = 16,
                 type_embed_dim = 32, hidden = 256, heads = 4,
                 dropout = 0.1, chi_mode = "other_type_hard",
                 chi_soft_leak = 0.1, edge_feat = "kernel",
                 n_gat_layers = 1):
        super().__init__()
        if edge_feat not in ("raw", "kernel"):
            raise ValueError(f"edge_feat must be 'raw' or 'kernel', got {edge_feat!r}")
        if chi_mode not in ("other_type_hard", "other_type_soft", "global"):
            raise ValueError(
                f"chi_mode must be one of 'other_type_hard', 'other_type_soft', "
                f"'global'; got {chi_mode!r}")
        if n_gat_layers < 0:
            raise ValueError(f"n_gat_layers must be >= 0, got {n_gat_layers}")
        self.n_genes = n_genes
        self.z_dim = z_dim
        self.chi_mode = chi_mode
        self.chi_soft_leak = chi_soft_leak
        self.edge_feat = edge_feat
        self.n_gat_layers = n_gat_layers
        self.type_embed = nn.Embedding(n_types, type_embed_dim)
        self.node_in = nn.Linear(n_genes + type_embed_dim, hidden)
        self.gat_layers = nn.ModuleList([
            GATv2Conv(hidden, hidden, heads=heads, concat=False,
                      edge_dim=1, dropout=dropout)
            for _ in range(n_gat_layers)
        ])
        self.head_z_loc = nn.Linear(hidden, z_dim)
        # global_ambient buffer is registered later via set_global_ambient().

    def _encode_graph(self, batch):
        x = batch["counts"]
        ct = batch["cell_type"]
        nbr_x = batch["neighbor_counts"]
        nbr_ct = batch["neighbor_types"]
        nbr_d = batch["neighbor_dists"]
        B, K, G = nbr_x.shape
        device = x.device

        all_x = torch.cat([x.unsqueeze(1), nbr_x], dim=1).reshape(B * (K + 1), G)
        all_ct = torch.cat([ct.unsqueeze(1), nbr_ct], dim=1).reshape(B * (K + 1))
        node_feat = torch.log1p(all_x)
        e = self.type_embed(all_ct)
        node_h = self.node_in(torch.cat([node_feat, e], dim=-1))

        # n_gat_layers=0 short-circuits the spatial encoder (centre-only h).
        if self.n_gat_layers == 0:
            center_idx = torch.arange(B, device=device) * (K + 1)
            return node_h[center_idx]
        edge_index = _build_star_edge_index(B, K, device)
        if self.edge_feat == "kernel":
            sigma = nbr_d.median(dim=-1, keepdim=True).values.clamp(min=1e-6)
            edge_w = torch.exp(-nbr_d / sigma)              # (B, K) in (0, 1]
        else:
            edge_w = nbr_d
        # edge_index is built per-cell (cell0 fwd+bwd, cell1 fwd+bwd, ...), so
        # edge_attr must interleave the same way: for each cell, its K weights
        # twice (fwd then bwd). The old `reshape(B*K,1).repeat(2,1)` ordered all
        # cells then repeated the block, which mis-paired every edge for B>1
        # (fixed 2026-08-04; only aligned at batch=1).
        edge_attr = edge_w.reshape(B, K, 1).repeat(1, 2, 1).reshape(2 * B * K, 1)
        h = node_h
        for i, gat in enumerate(self.gat_layers):
            h = gat(h, edge_index, edge_attr)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)

        center_idx = torch.arange(B, device=device) * (K + 1)
        center_h = h[center_idx]
        return center_h

    def set_global_ambient(self, profile):
        """Register fallback profile for cells with no other-type neighbours."""
        import numpy as np
        if isinstance(profile, np.ndarray):
            profile = torch.from_numpy(profile.astype(np.float32))
        else:
            profile = profile.float()
        self.register_buffer("global_ambient", profile, persistent=True)

    def chi_from_neighbors(self, batch):
        global_ambient = getattr(self, "global_ambient", None)
        return _explicit_chi(batch, self.chi_mode, self.chi_soft_leak,
                              global_ambient=global_ambient)


def _explicit_chi(batch, mode, soft_leak,
                   global_ambient = None):
    """χ from neighbours; same-type cells dropped (Proposition 1).
    Cells with no other-type neighbours fall back to ``global_ambient``."""
    if mode == "global":
        x_nbr = batch["neighbor_counts"]
        B, K, G = x_nbr.shape
        if global_ambient is not None:
            return global_ambient.to(x_nbr.device).unsqueeze(0).expand(B, G).contiguous()
        return x_nbr.new_full((B, G), 1.0 / G)
    leak = 0.0 if mode == "other_type_hard" else soft_leak
    x_nbr = batch["neighbor_counts"]
    w = _dist_weights(batch["neighbor_dists"])
    center_ct = batch["cell_type"].unsqueeze(-1)
    nbr_ct = batch["neighbor_types"]
    same = (nbr_ct == center_ct).float()
    soft_mask = (1.0 - same) + leak * same
    w2 = w * soft_mask
    total = w2.sum(dim=-1, keepdim=True)
    has_other = (total > 1e-6)                                    # (B, 1)
    w_norm = w2 / total.clamp(min=1e-8)
    # Paper Eq. 3: χ_g = (1/Z) sum_j w_ij · 1[t_j!=t_i] · (x_jg / d_j).
    nbr_lib = x_nbr.sum(dim=-1, keepdim=True).clamp(min=1.0)      # (B, K, 1)
    nbr_prof = x_nbr / nbr_lib                                    # (B, K, G), rows sum to 1
    chi = (nbr_prof * w_norm.unsqueeze(-1)).sum(dim=1)            # (B, G), sums to 1 across g
    if global_ambient is not None:
        fb = global_ambient.to(chi.device).unsqueeze(0).expand_as(chi)
    else:
        G = chi.shape[-1]
        fb = chi.new_full((1, G), 1.0 / G).expand_as(chi)
    return torch.where(has_other, chi, fb)
