"""Training loop. Adam, gradient clip 10.0, optional CUDA AMP."""

import time
import torch
from torch.utils.data import DataLoader

from .dataset import collate, move_batch


def _resolve_device(device):
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train(
    model,
    dataset,
    n_epochs = 10,
    batch_size = 128,
    lr = 1e-3,
    device = "auto",
    num_workers = 4,
    amp = False,
    grad_clip = 10.0,
    shuffle = True,
    verbose = True,
):
    """Train a DeSpotX model."""
    device = _resolve_device(device)
    model.to(device)
    if verbose:
        print(f"[train] device = {device}")
        if device.type == "cuda":
            print(f"[train]   torch.cuda.get_device_name(): {torch.cuda.get_device_name(device)}")

    pin_memory = device.type == "cuda"
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=collate, num_workers=num_workers,
                        pin_memory=pin_memory, persistent_workers=num_workers > 0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if (amp and device.type == "cuda") else None
    hist = []

    t0_all = time.time()
    for epoch in range(n_epochs):
        model.set_epoch(epoch)
        t0 = time.time()
        loss_sum, n_batches, stats_sum = 0.0, 0, {}
        for batch in loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss, stats = model.training_step(batch)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss, stats = model.training_step(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            loss_sum += loss.item()
            n_batches += 1
            for k, v in stats.items():
                stats_sum[k] = stats_sum.get(k, 0.0) + v

        avg_loss = loss_sum / max(n_batches, 1)
        elapsed = time.time() - t0
        avg_stats = {k: v / max(n_batches, 1) for k, v in stats_sum.items()}
        prev_loss = hist[-1]["loss"] if hist else None
        hist.append({"epoch": epoch, "loss": avg_loss,
                     "time": elapsed, **avg_stats})
        if verbose:
            stats_str = "  ".join(f"{k}={v:.3f}" for k, v in avg_stats.items())
            if prev_loss is None:
                delta_str = ""
            else:
                delta = avg_loss - prev_loss
                arrow = "↓" if delta < 0 else ("↑" if delta > 0 else "·")
                delta_str = f" {arrow}{abs(delta):.3f}"
            print(f"[train] epoch {epoch+1}/{n_epochs}: "
                  f"loss={avg_loss:.3f}{delta_str}  {stats_str}  ({elapsed:.1f}s)")

    if verbose:
        print(f"[train] total: {time.time() - t0_all:.1f}s")
    return {"history": hist, "device": str(device)}
