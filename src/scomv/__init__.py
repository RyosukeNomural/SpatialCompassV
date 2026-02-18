"""Top-level package for SpatialCompassV."""
__author__ = "Ryosuke Nomura"
__version__ = "0.1.0"

# --- Public API (entry points) ---
from .cell_pipeline import CellPolarPipeline
from .gene_pipeline import SCOMVPipeline
from .spatial_deg import Spatial_DEG
from .dendrogram import dendrogram2newick
from .plot_3d import plot_3d

__all__ = [
    "CellPolarPipeline",
    "SCOMVPipeline",
    "Spatial_DEG",
    "dendrogram2newick",
    "plot_3d",
]
