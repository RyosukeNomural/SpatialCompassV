from __future__ import annotations

from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scanpy as sc
import squidpy as sq
import skbio
from skbio.stats.ordination import pcoa
import plotly.express as px
from .preparation.choose_roi import extract_roi, contour_regions
from .preparation.scomv_calc_vector import compute_min_vectors_polar
from .polar_plot import plot_polar_heatmap


def plot_gene_polar_hist2d(
    subset_grid,
    min_vector_df,
    gene: str,
    bin_size_um: int = 10,
    radius_bins: Optional[np.ndarray] = None,
    angle_bins_deg: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    clim_ratio: float = 0.25,
    figsize: Tuple[int, int] = (6, 6),
    xlim: Tuple[float, float] = (-150, 300),
    ylim: Tuple[float, float] = (-180, 180),
    show_grid: bool = True,
    grid_kwargs: Optional[dict] = None,
    style: str = "polar",
    polar_kwargs: Optional[dict] = None,
    dpi: Optional[int] = None,
    show: bool = True,
):
    """
    Plot a polar 2D histogram weighted by gene expression in subset_grid.X.
    min_vector_df is assumed to be aligned with subset_grid (iloc-compatible order).

    style : "polar" (default) draws the histogram as a filled polar map
        (``scomv.polar_plot.plot_polar_heatmap``; extra options via
        ``polar_kwargs``; the dashed circle is the boundary, radius 0).
        "rect" draws the previous radius x angle rectangle (uses xlim, ylim,
        show_grid, grid_kwargs).
    dpi : int or None
        Figure resolution (None: matplotlib default).
    show : bool
        Call ``plt.show()``. Set False to adjust the figure (``plt.gcf()``)
        before showing it.

    Returns
    -------
    A_counts : np.ndarray
    total_n : float
        Total expression (sum)
    gene_index : int
    """

    if radius_bins is None:
        radius_bins = np.arange(-150, 310, 10)
    if angle_bins_deg is None:
        angle_bins_deg = np.arange(-180, 181, 30)
    if grid_kwargs is None:
        grid_kwargs = dict(which="major", color="white", linewidth=0.6, alpha=0.6)

    # gene -> index
    genes = list(subset_grid.var.index)
    if gene not in genes:
        raise ValueError(f"Gene '{gene}' not found in subset_grid.var.index")
    N = genes.index(gene)

    # expression vector
    expr = subset_grid.X[:, N]
    if hasattr(expr, "toarray"):
        expr = expr.toarray().ravel()
    else:
        expr = np.asarray(expr).ravel()

    total_n = int(expr.sum())
    end_gene = subset_grid.var.index[N]
    print(f"{end_gene}: n={total_n}")

    gene_angles = []
    gene_radiis = []
    gene_weights = []

    # collect weighted vectors
    for i, expression in enumerate(expr):
        # do NOT cast to int (keep normalized values valid)
        if expression <= 0:
            continue

        angles = min_vector_df["angle"].iloc[i]
        radiis = min_vector_df["radii"].iloc[i]
        n_vecs = len(angles)
        if n_vecs == 0:
            continue

        w = float(expression) / n_vecs

        for ang, rad in zip(angles, radiis):
            gene_angles.append(ang)
            gene_radiis.append(rad * bin_size_um)
            gene_weights.append(w)

    if len(gene_angles) == 0:
        raise ValueError(
            "No vectors collected (expression nearly zero or index mismatch "
            "between min_vector_df and subset_grid)"
        )

    degree_list = np.degrees(gene_angles)

    gene_weights = np.asarray(gene_weights, dtype=float)
    gene_weights = gene_weights / gene_weights.sum()

    if style not in ("polar", "rect"):
        raise ValueError(f"style must be 'polar' or 'rect', got {style!r}")

    A_counts, _, _ = np.histogram2d(
        gene_radiis, degree_list,
        bins=[radius_bins, angle_bins_deg],
        weights=gene_weights,
    )
    max_val = np.max(A_counts) if A_counts.size else 0
    vmax = max_val * float(clim_ratio) if (max_val > 0 and clim_ratio is not None) else None

    if style == "polar":
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw={"projection": "polar"})
        kw = dict(cmap=cmap, vmin=0, vmax=vmax, cbar_label="Density",
                  title=f"{end_gene} (n = {total_n})")
        kw.update(polar_kwargs or {})
        plot_polar_heatmap(A_counts, radius_bins, angle_bins_deg, ax=ax, **kw)
        plt.tight_layout()
        if show:
            plt.show()
        return A_counts, total_n, N

    # style == "rect"
    plt.figure(figsize=figsize, dpi=dpi)
    img = plt.pcolormesh(radius_bins, angle_bins_deg, A_counts.T, cmap=cmap)
    if vmax is not None:
        img.set_clim(0, vmax)

    cbar = plt.colorbar(img)
    cbar.set_label("Density", fontsize=15)
    cbar.ax.tick_params(labelsize=12)

    plt.xlabel("Radius", fontsize=15)
    plt.ylabel("Angle", fontsize=15)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.yticks(np.arange(-180, 181, 30), fontsize=13)
    plt.xticks(fontsize=13)

    if show_grid:
        plt.grid(True, **grid_kwargs)

    plt.title(f"{end_gene} (n = {total_n})", fontsize=20)
    plt.tight_layout()
    if show:
        plt.show()

    return A_counts, total_n, N


def plot_2d_expression(gene, grid):
    """
    Plot the spatial expression pattern of a specified gene using Squidpy.

    This function visualizes gene expression stored in a grid-like AnnData object
    (e.g., spatially binned or aggregated data) using `sq.pl.spatial_scatter`.
    The figure is intended for interactive display only (no saving by default).

    Parameters
    ----------
    gene : str
        Gene name to be visualized. Must be present in `grid.var_names`.

    grid : AnnData
        AnnData object containing:
        - gene expression matrix in `grid.X`
        - spatial coordinates in `grid.obsm["spatial"]`

    Notes
    -----
    - The color scale upper bound is set to 20% of the maximum expression value
      of the specified gene (`vmax = max * 0.2`).
    - Axis limits are explicitly matched to the original spatial coordinates.
    - The y-axis is inverted to follow image-like coordinate conventions.
    - This function only displays the figure using `plt.show()` and does not save files.
    """

    gene_idx_grid = np.where(grid.var_names == gene)[0][0]
    max_val_grid = grid.X[:, gene_idx_grid].max()
    sum_expr_grid = int(grid.X[:, gene_idx_grid].sum())

    coords = grid.obsm["spatial"]  # original coordinates (cells × 2)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    ax = sq.pl.spatial_scatter(
        grid,
        cmap="viridis",
        color=[gene],
        shape=None,
        size=80,
        img=False,
        vmax=max_val_grid * 0.2,
        figsize=(6, 6),
        return_ax=True,
    )

    ax.set_title(f"{gene} (n={sum_expr_grid})", fontsize=27)

    # match axis limits to original coordinates
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # invert y-axis for image-like orientation
    ax.invert_yaxis()

    ax.axis("on")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    for s in ax.spines.values():
        s.set_visible(True)

    xticks = np.linspace(x_min, x_max, 5)
    yticks = np.linspace(y_min, y_max, 5)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # tick label formatting
    ax.set_xticklabels([f"{v:.0f}" for v in xticks], fontsize=15)
    ax.set_yticklabels([f"{v:.0f}" for v in yticks], fontsize=15)

    plt.show()
