import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict


def extract_roi(
    grid,
    roi: Tuple[int, int, int, int] = (0, 4000, 0, 4000),
    bin_size: int = 10,
    region_col: str = "region_10",
):
    """
    Extract grid points within ROI, remove NaN regions,
    bin spatial coordinates, and match shortest-distance info.

    Returns
    -------
    subset_grid : AnnData-like
    filtered_shortest : pd.DataFrame
    xy_bins : List[Tuple[int, int]]
    """
    xmin, xmax, ymin, ymax = roi

    cond = (
        (grid.obs["imagecol"] >= xmin) &
        (grid.obs["imagecol"] <= xmax) &
        (grid.obs["imagerow"] >= ymin) &
        (grid.obs["imagerow"] <= ymax)
    )
    subset_grid = grid[cond].copy()

    subset_grid = subset_grid[~subset_grid.obs[region_col].isna()].copy()

    xy_bins = subset_grid.obs.apply(
        lambda r: (int(r["imagecol"] // bin_size), int(r["imagerow"] // bin_size)),
        axis=1
    ).tolist()

    filtered_shortest = grid.shortest[grid.shortest.index.isin(xy_bins)]
    return subset_grid, filtered_shortest, xy_bins


# -------------------------
# contour / inside utilities
# -------------------------

def _xy_from_shortest_index(
    shortest_df: pd.DataFrame,
    adata,
    min_x: float, max_x: float, min_y: float, max_y: float,
    bin_size: int = 10,
):
    """shortest_df.index =  Convert (xbin, ybin) back to real coordinates and filter by ROI"""
    if shortest_df is None or len(shortest_df) == 0:
        return np.array([]), np.array([]), [], []

    points = list(shortest_df.index)
    xb, yb = zip(*points)

    # bin -> µm
    x = np.array([i * bin_size + adata.obs.imagecol.min() for i in xb])
    y = np.array([j * bin_size + adata.obs.imagerow.min() for j in yb])

    m = (min_x < x) & (x < max_x) & (min_y < y) & (y < max_y)
    x = x[m]
    y = y[m]

    gx = [int(v // bin_size) for v in x]
    gy = [int(v // bin_size) for v in y]
    return x, y, gx, gy


def _plot_points_black_bg(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    outpath: Optional[str] = None,
    color: str = "yellow",
    marker: str = "s",
    s: int = 80,
    margin: int = 300,
    show: bool = True,
    ax=None,
    label: Optional[str] = None,
):
    """ Scatter plot with a black background; draw on the given ax if provided"""
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
        ax.set_facecolor("black")
        created_fig = True
    else:
        fig = ax.figure

    if len(x) > 0:
        ax.scatter(x, y, color=color, s=s, marker=marker, label=label)
        ax.set_xlim(x.min() - margin, x.max() + margin)
        ax.set_ylim(y.min() - margin, y.max() + margin)

    ax.set_title(title, fontsize=10, color="white", weight="bold")
    ax.invert_yaxis()
    ax.autoscale(enable=False)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")

    if label is not None:
        ax.legend(
            facecolor="black",
            edgecolor="white",
            fontsize=14,
            labelcolor="white",
            frameon=True,
            loc="upper right",
            borderpad=0.5,
            handlelength=1.5,
        )

    if outpath is not None and created_fig:
        fig.savefig(outpath, dpi=300, facecolor="black", bbox_inches="tight")

    if created_fig and show:
        plt.show()
    elif created_fig and not show:
        plt.close(fig)

    return ax


def contour_regions(
    filter_grid_shortest,
    adata,
    min_x: float, max_x: float, min_y: float, max_y: float,
    out_dir_base: Optional[str] = None,
    bin_size: int = 10,
    outline_band=(-1.0, 0.0),
    inside_threshold: float = -1.0,
    margin: int = 300,
    show: bool = True,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Returns
    -------
    g_x_cont, g_y_cont, g_x_inside, g_y_inside
    """
    # --- output dir ---
    if out_dir_base is None:
        out_dir_base = os.getcwd()
    out_dir = os.path.join(out_dir_base, f"{min_x}_{max_x}_{min_y}_{max_y}")
    os.makedirs(out_dir, exist_ok=True)

    def xy_from_df(df):
        if len(df) == 0:
            return np.array([]), np.array([]), [], []
        xb, yb = zip(*list(df.index))
        x = np.array([i * bin_size + adata.obs.imagecol.min() for i in xb])
        y = np.array([j * bin_size + adata.obs.imagerow.min() for j in yb])
        m = (min_x < x) & (x < max_x) & (min_y < y) & (y < max_y)
        x, y = x[m], y[m]
        gx = [int(v // bin_size) for v in x]
        gy = [int(v // bin_size) for v in y]
        return x, y, gx, gy

    # outline
    lo, hi = outline_band
    outline_df = filter_grid_shortest[
        (filter_grid_shortest["euclidean"] >= lo) & (filter_grid_shortest["euclidean"] <= hi)
    ]
    x_out, y_out, g_x_cont, g_y_cont = xy_from_df(outline_df)

    # inside
    inside_df = filter_grid_shortest[filter_grid_shortest["euclidean"] < inside_threshold]
    x_in, y_in, g_x_inside, g_y_inside = xy_from_df(inside_df)

    # plots
    def plot_one(x, y, title, path, color):
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
        ax.set_facecolor("black")
        if len(x) > 0:
            ax.scatter(x, y, color=color, s=80, marker="s")
            ax.set_xlim(x.min() - margin, x.max() + margin)
            ax.set_ylim(y.min() - margin, y.max() + margin)
        ax.set_title(title, fontsize=10, color="white", weight="bold")
        ax.invert_yaxis()
        ax.autoscale(enable=False)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("white")
        fig.savefig(path, dpi=300, facecolor="black", bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    plot_one(x_out, y_out, "Tumor Contour", os.path.join(out_dir, "tumor_contour_region.png"), "yellow")
    plot_one(x_in, y_in, "Tumor Inside",  os.path.join(out_dir, "tumor_inside_region.png"),  "yellow")

    # combined
    fig, ax = plt.subplots(figsize=(5, 5), facecolor="black")
    ax.set_facecolor("black")
    if len(x_out) > 0:
        ax.scatter(x_out, y_out, color="red", s=80, marker="s", label="Outline")
    if len(x_in) > 0:
        ax.scatter(x_in, y_in, color="yellow", s=80, marker="s", label="Inside")
    ax.set_title("Tumor Contour and Inside", fontsize=10, color="white", weight="bold")
    if len(x_out) + len(x_in) > 0:
        xs = np.concatenate([x_out, x_in]) if len(x_out) and len(x_in) else (x_out if len(x_out) else x_in)
        ys = np.concatenate([y_out, y_in]) if len(y_out) and len(y_in) else (y_out if len(y_out) else y_in)
        ax.set_xlim(xs.min() - margin, xs.max() + margin)
        ax.set_ylim(ys.min() - margin, ys.max() + margin)
    ax.invert_yaxis()
    ax.autoscale(enable=False)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("white")
    ax.legend(facecolor="black", edgecolor="white", fontsize=14, labelcolor="white")
    fig.savefig(os.path.join(out_dir, "tumor_combined_contour_inside.png"), dpi=300, facecolor="black", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return g_x_cont, g_y_cont, g_x_inside, g_y_inside



__all__ = ["extract_roi", "contour_regions"]
