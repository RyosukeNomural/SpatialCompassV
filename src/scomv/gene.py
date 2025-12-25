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
from preparation.choose_roi import extract_roi, contour_regions
from preparation.scomv_calc_vector import compute_min_vectors_polar


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
):
    """
    Plot a polar 2D histogram weighted by gene expression in subset_grid.X.
    min_vector_df is assumed to be aligned with subset_grid (iloc-compatible order).

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

    # plot
    plt.figure(figsize=figsize)
    A_counts, xedges, yedges, img = plt.hist2d(
        gene_radiis, degree_list,
        bins=[radius_bins, angle_bins_deg],
        weights=gene_weights,
        cmap=cmap
    )

    max_val = np.max(A_counts) if A_counts.size else 0
    if max_val > 0 and clim_ratio is not None:
        img.set_clim(0, max_val * float(clim_ratio))

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
    plt.show()

    return A_counts, total_n, N


# ----------------------------
# Basic: subset by ROI
# ----------------------------
def subset_by_roi(adata, roi: Tuple[float, float, float, float]):
    min_x, max_x, min_y, max_y = roi
    return adata[
        (adata.obs["imagecol"] >= min_x) & (adata.obs["imagecol"] <= max_x) &
        (adata.obs["imagerow"] >= min_y) & (adata.obs["imagerow"] <= max_y)
    ].copy()


# ----------------------------
# Compute Moran's I
# ----------------------------
def compute_moran_df(
    adata,
    roi: Tuple[float, float, float, float],
    subsample_fraction: float = 0.5,
    delaunay: bool = True,
    n_perms: int = 100,
    n_jobs: int = 1,
    plot_cdf: bool = True,
) -> pd.DataFrame:
    """
    Subsample within ROI, compute spatial neighbors, and calculate Moran's I.
    Returns results as a DataFrame.
    """
    ad = subset_by_roi(adata, roi)

    ad_sub = sc.pp.subsample(ad, fraction=subsample_fraction, copy=True)

    sq.gr.spatial_neighbors(ad_sub, coord_type="generic", delaunay=delaunay)
    sq.gr.spatial_autocorr(ad_sub, mode="moran", n_perms=n_perms, n_jobs=n_jobs)

    moran_df = ad_sub.uns["moranI"].copy()  # columns: I, pval_norm, pval_sim, ...
    if plot_cdf:
        I_values = moran_df["I"].to_numpy()
        sorted_vals = np.sort(I_values)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        plt.figure(figsize=(6, 4))
        plt.plot(sorted_vals, cdf, color="steelblue", linewidth=2)
        plt.xlabel("Moran's I")
        plt.ylabel("Cumulative Frequency")
        plt.title("CDF of Moran's I")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return moran_df


# ----------------------------
# Gene length (total counts) as int
# ----------------------------
def gene_total_counts_int(adata) -> np.ndarray:
    """
    Sum adata.X along the gene axis and return an integer array.
    """
    X = adata.X
    gene_lens = np.array(X.sum(axis=0)).ravel().astype(int)
    return gene_lens


def plot_gene_length_cdf(gene_lens: np.ndarray, log_scale: bool = False):
    sorted_lengths = np.sort(gene_lens)
    cum_freq = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

    plt.figure(figsize=(6, 4))
    plt.plot(sorted_lengths, cum_freq, drawstyle="steps-post")
    plt.xlabel("Gene total counts" + (" (log)" if log_scale else ""))
    plt.ylabel("Cumulative Relative Frequency")
    if log_scale:
        plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Polar histograms for all genes
# ----------------------------
def build_polar_distributions_for_genes(
    adata_grid,
    min_vector_df: pd.DataFrame,
    moran_df: Optional[pd.DataFrame] = None,
    min_gene_count: int = 0,
    require_moran_nonneg: bool = True,
    bin_size_um: int = 10,
    radius_bins: Optional[np.ndarray] = None,
    angle_bins_deg: Optional[np.ndarray] = None,
    clim_ratio: float = 0.15,
    make_plots: bool = False,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    For all genes in adata_grid (subset_grid / filter_subset_grid):
    - Construct expression-weighted, tie-aware vectors
    - Compute hist2d (radius x angle) counts
    - Return only genes satisfying filtering criteria

    Note:
    min_vector_df must be aligned with adata_grid (iloc-compatible order).
    """

    if radius_bins is None:
        radius_bins = np.arange(-150, 310, 10)
    if angle_bins_deg is None:
        angle_bins_deg = np.arange(-180, 181, 30)

    # Assume Moran's I index corresponds to gene names
    moran_I = None
    if moran_df is not None:
        moran_I = moran_df["I"]

    X = adata_grid.X
    n_genes = adata_grid.n_vars
    n_cells = adata_grid.n_obs

    polar_counts_list: List[np.ndarray] = []
    selected_genes: List[str] = []

    for n in range(n_genes):
        gene_name = adata_grid.var.index[n]
        expr = X[:, n]
        if hasattr(expr, "toarray"):
            expr = expr.toarray().ravel()
        else:
            expr = np.asarray(expr).ravel()

        total_n = int(expr.sum())
        if total_n <= min_gene_count:
            continue

        if moran_I is not None and require_moran_nonneg:
            # Assume moran_df index corresponds to gene_name
            if gene_name in moran_I.index:
                if float(moran_I.loc[gene_name]) < 0:
                    continue
            else:
                # Exclude genes missing from Moran's I results
                continue

        gene_angles = []
        gene_radiis = []
        gene_weights = []

        for i in range(n_cells):
            e = int(expr[i])
            if e == 0:
                continue

            angles = min_vector_df["angle"].iloc[i]
            radiis = min_vector_df["radii"].iloc[i]
            n_vecs = len(angles)
            if n_vecs == 0:
                continue

            w = e / n_vecs
            for ang, rad in zip(angles, radiis):
                gene_angles.append(ang)
                gene_radiis.append(rad * bin_size_um)
                gene_weights.append(w)

        if len(gene_angles) == 0:
            continue

        deg = np.degrees(gene_angles)
        w = np.asarray(gene_weights, dtype=float)
        w = w / w.sum()

        # hist2d counts
        fig = plt.figure(figsize=(6, 6)) if make_plots else plt.figure(figsize=(1, 1))
        counts, xedges, yedges, img = plt.hist2d(
            gene_radiis, deg,
            bins=[radius_bins, angle_bins_deg],
            weights=w,
            cmap="viridis"
        )

        if make_plots:
            max_val = np.max(counts) if counts.size else 0
            if max_val > 0:
                img.set_clim(0, max_val * float(clim_ratio))
            plt.title(f"{gene_name} (n={total_n})")
            plt.tight_layout()
            plt.show()

        plt.clf()
        plt.close(fig)

        selected_genes.append(gene_name)
        polar_counts_list.append(counts)

    return polar_counts_list, selected_genes


# ----------------------------
# MINAS similarity -> distance matrix
# ----------------------------
def minas_similarity_matrix(
    polar_counts_list: List[np.ndarray],
    labels: List[str],
    make_distance: bool = True,
    fill_diagonal_zero: bool = True,
) -> pd.DataFrame:
    """
    Compute MINAS similarity (sum of element-wise minima) between count matrices.
    """
    n = len(labels)
    table = np.zeros((n, n), dtype=float)

    for i in range(n):
        A = np.asarray(polar_counts_list[i])
        for j in range(i, n):
            B = np.asarray(polar_counts_list[j])
            m = np.minimum(A, B)
            sim = float(m.sum())
            table[i, j] = sim
            table[j, i] = sim

    df = pd.DataFrame(table, index=labels, columns=labels)

    if make_distance:
        df = 1.0 - df

    if fill_diagonal_zero:
        np.fill_diagonal(df.values, 0.0)

    return df


# ----------------------------
# PCoA + Plotly scatter
# ----------------------------
def run_pcoa_from_distance_df(dist_df: pd.DataFrame):
    dm = skbio.DistanceMatrix(dist_df.values, ids=list(dist_df.index))
    res = pcoa(dm)
    coords = res.samples
    explained = res.proportion_explained
    return res, coords, explained


def plot_pcoa_plotly(
    coords: pd.DataFrame,
    labels: List[str],
    width: int = 1200,
    height: int = 1500,
    margin: float = 0.05,
):
    df_plot = pd.DataFrame({
        "PCoA1": coords.iloc[:, 0].values,
        "PCoA2": coords.iloc[:, 1].values,
        "gene": labels
    })

    fig = px.scatter(
        df_plot,
        x="PCoA1",
        y="PCoA2",
        hover_name="gene",
    )

    fig.update_traces(marker=dict(size=12))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    xmin, xmax = df_plot["PCoA1"].min(), df_plot["PCoA1"].max()
    ymin, ymax = df_plot["PCoA2"].min(), df_plot["PCoA2"].max()

    fig.update_xaxes(range=[xmin - margin, xmax + margin])
    fig.update_yaxes(range=[ymin - margin, ymax + margin])

    fig.update_layout(
        width=width,
        height=height,
        title="PCoA Plot",
        xaxis_title="PCoA1",
        yaxis_title="PCoA2",
        font=dict(size=18),
    )
    fig.show()
    return fig


# ----------------------------
# Entry point: run the full pipeline
# ----------------------------
def run_full_pipeline(
    adata,
    grid,
    roi,
    bin_size=10,
    region_col="region_10",
    subsample_fraction=0.5,
    min_gene_percentile=30,
    max_gene_percentile=95,
    require_moran_nonneg=True,
    contour_bounds=(0, 10000, 0, 10000),
    invert_y=True,
    make_inside_negative=True,
):
    """
    Run the full SCOMV pipeline for a single ROI.
    """

    # ------------------
    # 1. Extract ROI
    # ------------------
    subset_grid, filtered_shortest, xy_list = extract_roi(
        grid=grid,
        roi=roi,
        bin_size=bin_size,
        region_col=region_col,
    )

    # ------------------
    # 2. Extract tumor contour
    # ------------------
    min_x, max_x, min_y, max_y = contour_bounds
    g_x_cont, g_y_cont, g_x_inside, g_y_inside = contour_regions(
        filtered_shortest,
        adata,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        out_dir_base=None,
        show=False,
    )

    outline_points = list(zip(g_x_cont, g_y_cont))
    inside_points  = list(zip(g_x_inside, g_y_inside))

    # ------------------
    # 3. Compute minimum vectors
    # ------------------
    min_vector_df = compute_min_vectors_polar(
        xy_list=xy_list,
        outline_points=outline_points,
        inside_points=inside_points,
        invert_y=invert_y,
        make_inside_negative=make_inside_negative,
    )

    # 4. Moran's I
    moran_df = compute_moran_df(
        adata=adata,
        roi=roi,
        subsample_fraction=subsample_fraction,
        plot_cdf=False,
    ).sort_index()

    # 5. Gene counts (int)
    gene_lens = gene_total_counts_int(subset_grid)

    p_low  = int(np.percentile(gene_lens, min_gene_percentile))
    p_high = int(np.percentile(gene_lens, max_gene_percentile))  # optional
    # print(p_low, p_high)

    # 6. Polar distributions (use adata_grid)
    polar_counts_list, selected_genes = build_polar_distributions_for_genes(
        adata_grid=subset_grid,
        min_vector_df=min_vector_df,
        moran_df=moran_df,
        min_gene_count=p_low,
        require_moran_nonneg=require_moran_nonneg,
        make_plots=False,
    )

    # 7. Distance matrix
    dist_df = minas_similarity_matrix(
        polar_counts_list=polar_counts_list,
        labels=selected_genes,
        make_distance=True,
        fill_diagonal_zero=True,
    )

    # 8. PCoA
    pcoa_res, coords, explained = run_pcoa_from_distance_df(dist_df)

    return {
        "roi": roi,
        "subset_grid": subset_grid,
        "dist_df": dist_df,
    }
