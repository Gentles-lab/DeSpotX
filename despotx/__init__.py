"""DeSpotX: spatial-aware ambient RNA decontamination."""
from .anchors import compute_anchors
from .dataset import SpatialDataset
from .inference import adjust_counts
from .model import DeSpotX
from .train import train

__all__ = [
    "DeSpotX",
    "SpatialDataset",
    "compute_anchors",
    "train",
    "adjust_counts",
]
