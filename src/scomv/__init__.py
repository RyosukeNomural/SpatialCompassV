"""Top-level package for SpatialCompassV."""

__author__ = """Ryosuke Nomura"""
#__email__ = 'nomubare123@g.ecc.u-tokyo.ac.jp'

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scomv")
except PackageNotFoundError:
    __version__ = "unknown"

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
