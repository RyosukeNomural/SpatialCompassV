"""Top-level package for SpatialCompassV."""
__author__ = "Ryosuke Nomura"
__version__ = "0.1.7"

# --- Public API (entry points) ---
from .cell_pipeline import CellPolarPipeline
from .gene_pipeline import SCOMVPipeline
from .spatial_deg import Spatial_DEG
from .dendrogram import dendrogram2newick
from .plot_3d import plot_3d
from .polar_plot import plot_polar_heatmap
from .preparation.io import save_preprocessed, load_preprocessed
from .peripheral_score import (
    IMMUNE_BLOOD_MARKERS,
    compute_peripheral_score,
    rank_genes_by_peripheral_score,
    add_pattern_label,
    plot_angle_profile,
    plot_marker_profiles,
)
from .signature import (
    PolarROI,
    HALLMARK_CATEGORY,
    signature_polar_enrichment,
    hallmark_polar_enrichment,
    load_hallmark_gene_sets,
    plot_signature_polar,
    plot_significant_hallmarks,
    signature_polar_histogram,
    plot_signature_backprojection,
)
from .pgi import compute_pgi, plot_pgi_ranking, plot_pgi_polar
from .gene_clustering import (
    compute_gene_features,
    cluster_genes_two_stage,
    plot_gene_clusters,
    cluster_mean_maps,
    plot_cluster_mean_polar,
    plot_cluster_presence,
    cluster_enrichment,
)
from .solid_clustering import (
    solid_gene_uniformity,
    solid_cell_uniformity,
    cluster_solids,
    plot_solid_clusters,
)

__all__ = [
    "CellPolarPipeline",
    "SCOMVPipeline",
    "Spatial_DEG",
    "dendrogram2newick",
    "plot_3d",
    "plot_polar_heatmap",
    "save_preprocessed",
    "load_preprocessed",
    "IMMUNE_BLOOD_MARKERS",
    "compute_peripheral_score",
    "rank_genes_by_peripheral_score",
    "add_pattern_label",
    "plot_angle_profile",
    "plot_marker_profiles",
    "PolarROI",
    "HALLMARK_CATEGORY",
    "signature_polar_enrichment",
    "hallmark_polar_enrichment",
    "load_hallmark_gene_sets",
    "plot_signature_polar",
    "plot_significant_hallmarks",
    "signature_polar_histogram",
    "plot_signature_backprojection",
    "compute_pgi",
    "plot_pgi_ranking",
    "plot_pgi_polar",
    "compute_gene_features",
    "cluster_genes_two_stage",
    "plot_gene_clusters",
    "cluster_mean_maps",
    "plot_cluster_mean_polar",
    "plot_cluster_presence",
    "cluster_enrichment",
    "solid_gene_uniformity",
    "solid_cell_uniformity",
    "cluster_solids",
    "plot_solid_clusters",
]
