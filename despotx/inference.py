"""Batch inference: produces corrected counts for the entire dataset."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import collate, move_batch
from .train import _resolve_device


@torch.no_grad()
def adjust_counts(
    model,
    dataset,
    batch_size = 256,
    device = "auto",
    num_workers = 4,
):
    """Decontaminate the full dataset; returns dict {adjusted, epsilon}."""
    device = _resolve_device(device)
    model.to(device).eval()
    pin_memory = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=num_workers,
                        pin_memory=pin_memory)
    adj_parts, eps_parts = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        adj, eps = model.adjust_counts(batch)
        adj_parts.append(adj.cpu().numpy().astype(np.float32))
        eps_parts.append(eps.cpu().numpy().astype(np.float32))
    return {"adjusted": np.concatenate(adj_parts),
            "epsilon": np.concatenate(eps_parts)}
