import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Sequence
import skbio
from skbio.stats.ordination import pcoa


def annotate_cells(
    cell_df: pd.DataFrame,
    roi: Tuple[float, float, float, float],
    bin_size: int = 10,
    x_col: int = 0,
    y_col: int = 1,
    coord_col: str = "coord_tuple",
) -> pd.DataFrame:
    """
    Filter cells by ROI and add a grid coordinate tuple.

    Parameters
    ----------
    cell_df : pd.DataFrame
        Cell-level table (x, y are in the first columns by default)
    roi : (min_x, max_x, min_y, max_y)
        ROI in image coordinates (µm)
    bin_size : int
        Spatial bin size (µm)
    x_col, y_col : int
        Column indices for x and y
    coord_col : str
        Name of the output grid-coordinate column

    Returns
    -------
    filtered_df : pd.DataFrame
        Filtered DataFrame with a coord_tuple column
    """

    min_x, max_x, min_y, max_y = roi

    filtered_df = cell_df[
        (cell_df.iloc[:, x_col] > min_x) & (cell_df.iloc[:, x_col] < max_x) &
        (cell_df.iloc[:, y_col] > min_y) & (cell_df.iloc[:, y_col] < max_y)
    ].copy()

    filtered_df[coord_col] = list(zip(
        np.floor(filtered_df.iloc[:, x_col] / bin_size).astype(int),
        np.floor(filtered_df.iloc[:, y_col] / bin_size).astype(int),
    ))

    return filtered_df
    

def compute_cluster_polar_distributions(
    cell_df_filtered,
    min_vector_df,
    cluster_col: str = "Cluster",
    coord_col: str = "coord_tuple",
    unlabeled_name: str = "Unlabeled",
    min_cells: int = 30,
    bin_size_um: int = 10,  # scale factor to convert radii back to real distance (in your code: *10)
    radius_bins: Optional[np.ndarray] = None,
    angle_bins_deg: Optional[np.ndarray] = None,
    clim_ratio: float = 0.5,
    plot: bool = True,
    cmap: str = "viridis",
    xlim: Tuple[float, float] = (-150, 300),
    ylim: Tuple[float, float] = (-180, 180),
) -> Tuple[List[np.ndarray], List[str], Dict[str, np.ndarray]]:
    """
    For each cluster in cell_df_filtered, collect polar (radius, angle) samples from min_vector_df,
    apply tie-aware weighting, and build a weighted 2D histogram.

    Returns
    -------
    polar_counts_list : List[np.ndarray]
        List of 2D histogram counts (A_counts) per selected cluster
    selected_clusters : List[str]
        Cluster names used (>= min_cells, excluding the unlabeled group)
    counts_by_cluster : Dict[str, np.ndarray]
        Mapping: cluster_name -> A_counts
    """

    if radius_bins is None:
        radius_bins = np.arange(-150, 310, 10)
    if angle_bins_deg is None:
        angle_bins_deg = np.arange(-180, 181, 30)

    polar_counts_list: List[np.ndarray] = []
    selected_clusters: List[str] = []
    counts_by_cluster: Dict[str, np.ndarray] = {}

    # Stabilize coord_tuple types (e.g., list -> tuple)
    if coord_col in cell_df_filtered.columns:
        cell_df_filtered = cell_df_filtered.copy()
        cell_df_filtered[coord_col] = cell_df_filtered[coord_col].apply(lambda x: tuple(x))
    else:
        raise KeyError(f"'{coord_col}' not found in cell_df_filtered.columns")

    for target_cluster in cell_df_filtered[cluster_col].unique():
        if target_cluster == unlabeled_name:
            continue

        sub = cell_df_filtered[cell_df_filtered[cluster_col].eq(target_cluster)].copy()
        total_n = len(sub)
        if total_n < min_cells:
            continue

        selected_clusters.append(target_cluster)

        cluster_angles = []
        cluster_radiis = []
        cluster_weights = []

        # Row-wise lookup: coord_tuple -> min_vector_df
        for _, row in sub.iterrows():
            x, y = map(int, row[coord_col])
            key = (x, y)

            if key in min_vector_df.index:
                angles = min_vector_df.at[key, "angle"]
                radiis = min_vector_df.at[key, "radii"]

                n_vecs = len(angles)
                if n_vecs == 0:
                    continue

                w = 1.0 / n_vecs  # tie-aware weighting

                for ang, rad in zip(angles, radiis):
                    cluster_angles.append(ang)
                    cluster_radiis.append(rad * bin_size_um)  # convert to real distance
                    cluster_weights.append(w)

        # Skip clusters with no samples
        if len(cluster_angles) == 0:
            continue

        degree_list = np.degrees(cluster_angles)

        w = np.asarray(cluster_weights, dtype=float)
        w = w / w.sum()  # normalize

        # 2D histogram
        if plot:
            plt.figure(figsize=(6, 6))

        A_counts, xedges, yedges, img = plt.hist2d(
            cluster_radiis, degree_list,
            bins=[radius_bins, angle_bins_deg],
            weights=w,
            cmap=cmap
        )

        # Adjust intensity range
        if clim_ratio is not None:
            max_val = np.max(A_counts) if A_counts.size else 0
            if max_val > 0:
                img.set_clim(0, max_val * float(clim_ratio))

        if plot:
            cbar = plt.colorbar(img)
            cbar.set_label("Density", fontsize=15)
            cbar.ax.tick_params(labelsize=12)

            plt.xlabel("Radius", fontsize=15)
            plt.ylabel("Angle", fontsize=15)
            plt.ylim(*ylim)
            plt.yticks(np.arange(-180, 181, 30), fontsize=13)
            plt.xlim(*xlim)
            plt.xticks(fontsize=13)
            plt.grid(True)
            plt.title(f"{target_cluster} (n = {total_n})", fontsize=20)
            plt.tight_layout()
            plt.show()
        else:
            # If plot=False, close the figure (hist2d still creates an internal artist)
            plt.close("all")

        polar_counts_list.append(A_counts)
        counts_by_cluster[target_cluster] = A_counts

    return polar_counts_list, selected_clusters, counts_by_cluster


def cell_similarity_matrix(
    polar_counts_list: List[np.ndarray],
    labels: Sequence[str],
    make_distance: bool = True,
    fill_diagonal_zero: bool = True,
) -> pd.DataFrame:
    """
    Build a matrix using min-sum similarity between A_counts (2D histograms).
    If make_distance=True, returns (1 - similarity).

    Parameters
    ----------
    polar_counts_list : list of 2D arrays
        2D histograms (A_counts) for each cluster
    labels : list-like
        Names in the same order as polar_counts_list
    make_distance : bool
        True: convert to 1 - similarity (distance-like)
    fill_diagonal_zero : bool
        True: set diagonal to 0

    Returns
    -------
    df : pd.DataFrame (n x n)
    """
    n = len(polar_counts_list)
    if n != len(labels):
        raise ValueError(f"len(polar_counts_list)={n} != len(labels)={len(labels)}")

    # Convert to ndarray and check shapes
    mats = [np.asarray(m) for m in polar_counts_list]
    shape0 = mats[0].shape
    if any(m.shape != shape0 for m in mats):
        raise ValueError(
            f"All A_counts must have the same shape. First={shape0}, got={[m.shape for m in mats]}"
        )

    table = np.zeros((n, n), dtype=float)

    # Compute upper triangle only (same as your original implementation)
    for i in range(n):
        A = mats[i]
        for j in range(i, n):
            B = mats[j]
            sim = np.minimum(A, B).sum()   # equivalent to nested sums in the original code
            table[i, j] = sim
            table[j, i] = sim

    df = pd.DataFrame(table, index=labels, columns=labels).fillna(0)

    if make_distance:
        df = 1.0 - df

    if fill_diagonal_zero:
        np.fill_diagonal(df.values, 0.0)

    return df


def cell_pcoa(
    dist_df: pd.DataFrame,
    n_components: int = 10,
    figsize: Tuple[int, int] = (8, 4),
    color: str = "steelblue",
    outpath: Optional[str] = None,
    title: str = "PCoA Explained Variance",
    ylabel: str = "Proportion Explained",
    xlabel: str = "PCoA Axis",
    show_values: bool = True,
    show_plot: bool = True,
):
    """
    Distance matrix (DataFrame) -> PCoA -> explained-variance bar plot.

    Parameters
    ----------
    dist_df : pd.DataFrame
        Square distance matrix (index == columns)
    n_components : int
        Number of PCoA components to display
    outpath : str or None
        If provided, save the bar plot
    show_values : bool
        Annotate bars with explained variance values
    show_plot : bool
        Whether to display the plot

    Returns
    -------
    pcoa_res : skbio OrdinationResults
    coords : pd.DataFrame
        PCoA coordinates (samples)
    explained_variance_ratio : np.ndarray
    """

    # --- PCoA ---
    dm = skbio.DistanceMatrix(dist_df.values, ids=list(dist_df.index))
    pcoa_res = pcoa(dm)
    coords = pcoa_res.samples
    explained_variance_ratio = pcoa_res.proportion_explained

    # --- Plot explained variance ---
    evr = np.asarray(explained_variance_ratio)[:n_components]
    x = np.arange(1, len(evr) + 1)

    plt.figure(figsize=figsize)
    bars = plt.bar(x, evr, color=color)

    if show_values:
        for b, v in zip(bars, evr):
            plt.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(x, [f"PCoA{i}" for i in x])
    plt.ylim(0, max(evr) * 1.15)
    plt.tight_layout()

    if outpath is not None:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return pcoa_res, coords, explained_variance_ratio


def cell_heatmap(
    dist_df: pd.DataFrame,
    method: str = "ward",
    metric: str = "euclidean",
    cmap: str = "viridis",
    figsize=(10, 10),
    font_size: int = 35,
    rotation: int = 90,
    outpath: Optional[str] = None,
):
    """
    Draw a seaborn clustermap using dist_df.
    If dist_df is a distance matrix, you can cluster using dist_df itself,
    while visualizing 1 - dist_df to display a similarity-like heatmap.
    """
    sns.set(context="notebook")

    g = sns.clustermap(
        1 - dist_df,          # visualize as similarity-like values
        method=method,
        metric=metric,
        cmap=cmap,
        figsize=figsize
    )

    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(),
        fontsize=font_size,
        rotation=rotation
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(),
        fontsize=font_size
    )

    # if outpath is not None:
    #     g.fig.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
    return g
