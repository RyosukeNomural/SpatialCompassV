"""Tumor solid extraction and per-solid analysis.

Splits the workflow that used to live as a notebook-local
``convert_individual_solid`` function into smaller, reusable pieces:

- :func:`label_solid_regions` — connected-component labeling of tumor solids
  from the marker delineation image.
- :func:`extract_tumor_solids` — aggregate gene expression per solid into a
  solid-level AnnData.
- :func:`assign_solid_ids` — map arbitrary (x, y) points (e.g. individual
  cells) onto a solid label image.
- :func:`compute_solid_min_vectors` — grid-square subset and per-square
  minimum-distance polar vectors around one solid.
- :func:`compute_per_solid_polar_counts` — per-(solid, cell type) polar
  2D-histogram counts.
- :func:`compute_uniformity` — per-(solid, cell type, distance bin)
  uniformity score from those histogram counts.
- :func:`distance_weighted_mean` — combine per-distance-bin matrices into one
  weighted mean matrix.
- :func:`ward_cluster` — standardize + Ward hierarchical clustering.
- :func:`remap_clusters_by_reference` — relabel cluster ids by a reference
  column's per-cluster mean, for stable/interpretable cluster numbering.
- :func:`add_significance_bars` — pairwise Mann-Whitney significance bars for
  a categorical plot.
- :func:`plot_uniformity_by_cluster` — violin+strip plot of one cell type's
  score across clusters, with significance bars, in one call.
- :func:`plot_uniformity_heatmap` — row-clustered heatmap of a feature
  matrix, using a precomputed linkage.
- :func:`plot_cluster_map` — spatial map of solids colored by cluster, with
  solid ids and a legend.
- :func:`plot_cell_type_over_clusters` — one cell type's spatial positions
  overlaid on a faded cluster map; call once per cell type.

Two-group (solid or cluster) comparison — every ``solid_ids`` argument below
accepts either a single solid id or a list of solid ids (e.g. all solids in
one Ward cluster):

- :func:`compute_expression_fold_factor` — total-expression fold factor
  between two solid groups.
- :func:`compute_gene_uniformity` — per-gene, per-distance-bin uniformity
  for one solid group.
- :func:`compute_gene_uniformity_pair` — the above for two groups, on a
  comparable expression scale.
- :func:`weighted_mean_across_bins` — collapse a gene x distance-bin table
  into one score per gene.
- :func:`plot_gene_uniformity_scatter` — 2D scatter of per-gene uniformity,
  group A vs group B.
- :func:`compute_deg_between_solid_groups` — Wilcoxon DEG between two solid
  groups' cells.
- :func:`plot_deg_volcano` — volcano plot from a DEG table.
- :func:`run_pathway_enrichment` / :func:`plot_pathway_enrichment` — Enrichr
  pathway enrichment on DEG-derived gene lists (requires network access).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import anndata as ad
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text
from itertools import combinations
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import issparse
from scipy.stats import entropy as _scipy_entropy
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler

from scomv.cell import annotate_cells, compute_cluster_polar_distributions
from scomv.preparation.scomv_calc_vector import compute_min_vectors_polar


def _as_solid_id_list(solid_ids: Union[int, Sequence[int]]) -> List[int]:
    if isinstance(solid_ids, (int, np.integer)):
        return [int(solid_ids)]
    return [int(s) for s in solid_ids]


def label_solid_regions(
    grid,
    min_area: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Connected-component label image of tumor solids.

    Reads ``grid.uns["marker_delineation"]`` (set by
    ``scomv.preparation.skny_calc_distance.calculate_distance``), drops
    components smaller than ``min_area``, and relabels the remainder
    consecutively.

    Parameters
    ----------
    grid : AnnData
        Grid AnnData with ``.uns["marker_delineation"]``.
    min_area : int
        Minimum connected-component area, in grid pixels, to keep as a solid.

    Returns
    -------
    labels : np.ndarray, shape (H, W)
        Label image; 0 = background, 1..N = individual solids (consecutive).
    stats : np.ndarray, shape (N+1, 5)
        Per-label stats from ``cv2.connectedComponentsWithStats``
        (x, y, width, height, area), indexed by label id; row 0 is background.
    centroids : np.ndarray, shape (N+1, 2)
        Per-label (x, y) centroid, indexed by label id; row 0 is background.
    """
    img_color = grid.uns["marker_delineation"]
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(img_gray)

    for i in range(1, retval):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            labels[labels == i] = 0

    labels = labels.astype(np.uint8)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(labels)

    return labels, stats, centroids


def extract_tumor_solids(
    grid,
    section_um: Optional[Union[Tuple[int, int], Sequence[int]]] = None,
    min_area: int = 100,
    verbose: bool = True,
):
    """
    Extract individual tumor solids and aggregate gene expression per solid
    within a specified distance range.

    Parameters
    ----------
    grid : AnnData
        Grid AnnData produced by ``st.tl.cci.grid`` and
        ``calculate_distance`` (must have ``.uns["grid_yedges"]``,
        ``.uns["grid_xedges"]``, ``.uns["marker_delineation"]``, and a
        ``.shortest`` attribute).
    section_um : tuple[int, int] or None
        None -> tumor interior only (distance <= 0).
        (-60, 0) -> 60 um outside to the boundary.
        (-60, 60) -> 60 um outside to 60 um inside.
        Values must be multiples of 30.
    min_area : int
        Minimum connected-component area in grid pixels, passed to
        :func:`label_solid_regions`.
    verbose : bool
        Print which distance range is being aggregated.

    Returns
    -------
    AnnData
        One observation per solid. ``.var_names`` = genes, ``.obs`` holds the
        solid centroid (imagecol, imagerow) and solid id,
        ``.uns["individual_tumor_solid"]`` holds the label image annotated
        with bounding boxes and solid ids, and ``.shortest`` (set via
        ``setattr``) holds the per-grid-square distance table with an added
        "solid" column.
    """
    N_ROW = len(grid.uns["grid_yedges"]) - 1
    N_COL = len(grid.uns["grid_xedges"]) - 1

    if section_um is None:
        if verbose:
            print("Calculating gene expression of (-∞, 0]")
        section_range = None
    else:
        if not isinstance(section_um, (tuple, list)) or len(section_um) != 2:
            raise ValueError("section_um must be None or a tuple/list like (-60, 0)")
        low_um, high_um = sorted(section_um)
        if (low_um % 30 != 0) or (high_um % 30 != 0):
            raise ValueError("section_um values must be multiples of 30")
        section_range = (int(low_um / 10), int(high_um / 10))
        if verbose:
            print(f"Calculating gene expression of [{low_um}, {high_um}] μm")

    labels, stats, centroids = label_solid_regions(grid, min_area=min_area)

    img_color = grid.uns["marker_delineation"].copy()
    for i in range(1, stats.shape[0]):
        x, y, width, height, area = stats[i]
        cv2.rectangle(img_color, (x, y), (x + width, y + height), (0, 0, 255), thickness=1)
        cv2.putText(img_color, f"{i}", (x, y),
                    cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1, cv2.LINE_8)

    df_shortest = getattr(grid, "shortest").copy()
    df_shortest["right"] = df_shortest["region"].apply(lambda x: x.right)

    df_solid_labels = pd.DataFrame(
        labels.reshape(N_COL, N_ROW).reshape(N_ROW * N_COL),
        index=df_shortest.index, columns=["solid"]
    )
    df_shortest = pd.merge(df_shortest, df_solid_labels,
                            right_index=True, left_index=True, how="left")

    solid_ls = []
    if section_range is None:
        for i, s in zip(df_shortest["right"], df_shortest["solid"]):
            if pd.isna(i):
                solid_ls.append(0)
            elif (i > 0) and (s != 0):
                solid_ls.append(0)
            else:
                solid_ls.append(s)
    else:
        low, high = section_range
        for i, s in zip(df_shortest["right"], df_shortest["solid"]):
            if pd.isna(i):
                solid_ls.append(0)
            elif (not (low <= i <= high)) and (s != 0):
                solid_ls.append(0)
            else:
                solid_ls.append(s)
    df_shortest["solid"] = solid_ls

    df_grid = grid.to_df()
    all_grid_index = [f"grid_{i + 1}" for i in range(N_ROW * N_COL)]
    df_grid = pd.merge(pd.DataFrame(index=all_grid_index),
                        df_grid, right_index=True, left_index=True, how="left").fillna(np.nan)

    df_solid = pd.DataFrame(
        np.array(df_shortest["solid"]).reshape(N_ROW, N_COL).T.reshape(N_ROW * N_COL),
        index=all_grid_index, columns=["solid"]
    )
    df_grid_solid = pd.merge(df_grid, df_solid, right_index=True, left_index=True, how="left")

    df_solid_mean = df_grid_solid.dropna().groupby("solid").mean()
    df_solid_mean = df_solid_mean[df_solid_mean.index != 0]

    if df_solid_mean.shape[0] == 0:
        raise ValueError("No tumor solids remained after filtering.")

    df_solid_centroid = pd.DataFrame(
        centroids[df_solid_mean.index.astype(int)],
        index=[f"solid_{int(i)}" for i in df_solid_mean.index],
        columns=["imagecol", "imagerow"]
    )
    df_solid_centroid["solid"] = df_solid_mean.index.astype(int)

    solid = ad.AnnData(
        df_solid_mean.values,
        uns={"spatial": {"data": {
            "scalefactors": {"tissue_hires_scalef": 1, "spot_diameter_fullres": 10},
            "use_quality": "hires",
            "images": {"hires": np.array([[255, 255, 255, 255]])}
        }}}
    )
    solid.obs_names = [f"solid_{int(i)}" for i in df_solid_mean.index]
    solid.var_names = df_solid_mean.columns.tolist()
    solid.obs = df_solid_centroid
    solid.uns["individual_tumor_solid"] = img_color
    setattr(solid, "shortest", df_shortest)

    return solid


def assign_solid_ids(
    df: pd.DataFrame,
    labels_img: np.ndarray,
    bin_size: int = 10,
    x_col: str = "imagecol",
    y_col: str = "imagerow",
) -> np.ndarray:
    """
    Map each row's (x_col, y_col) spatial coordinate onto a solid label image.

    Parameters
    ----------
    df : pd.DataFrame
        Table with spatial coordinate columns (e.g. per-cell ``adata.obs``).
    labels_img : np.ndarray
        Label image, as returned by :func:`label_solid_regions` (or the
        first element of the tuple it returns).
    bin_size : int
        Pixel size of one label-image cell, in the same units as x_col/y_col.
    x_col, y_col : str
        Column names holding the spatial coordinates.

    Returns
    -------
    np.ndarray, shape (len(df),)
        Solid id per row. Points that fall outside ``labels_img`` bounds are
        assigned 0 (same as unlabeled background).
    """
    cols = (df[x_col].to_numpy() // bin_size).astype(int)
    rows = (df[y_col].to_numpy() // bin_size).astype(int)
    h, w = labels_img.shape
    valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    solid_ids = np.zeros(len(df), dtype=int)
    solid_ids[valid] = labels_img[rows[valid], cols[valid]]
    return solid_ids


def compute_solid_min_vectors(
    grid,
    solid,
    solid_id: int,
    bin_size: int = 10,
    margin_bins: int = 15,
):
    """
    Grid-square subset around one solid and the per-square minimum-distance
    polar vectors (angle/radius to the nearest tumor-boundary point) within
    it.

    Parameters
    ----------
    grid : AnnData
        Grid AnnData with ``.obs[["imagecol", "imagerow"]]``.
    solid : AnnData
        Output of :func:`extract_tumor_solids` (must carry ``.shortest``
        with "solid" and "euclidean" columns).
    solid_id : int
        Solid id to process (as in ``solid.obs["solid"]``).
    bin_size : int
        Grid bin size, in the same spatial units as ``grid.obs`` columns.
    margin_bins : int
        Number of extra grid bins to include around the solid's bounding box.

    Returns
    -------
    subset_grid : AnnData
        Grid squares within the solid's bounding box (+ margin).
    min_vector_df : pd.DataFrame
        Indexed by (x_bin, y_bin); "angle"/"radii" columns hold the vectors
        from that grid square to the nearest boundary point(s).
    roi : tuple[float, float, float, float]
        (min_x, max_x, min_y, max_y) in raw spatial units, sized to match
        the strict-inequality filtering used by
        ``scomv.cell.annotate_cells``.

    Raises
    ------
    ValueError
        If the solid has no interior grid squares, or its bounding box (+
        margin) contains no grid squares at all.
    """
    df_sh = getattr(solid, "shortest")
    df_interior = df_sh[
        (df_sh["solid"] == solid_id) & df_sh["euclidean"].notna()
    ]
    if len(df_interior) == 0:
        raise ValueError(f"solid_id={solid_id}: no interior grid squares found")

    xbins = [xy[0] for xy in df_interior.index]
    ybins = [xy[1] for xy in df_interior.index]
    xmin_b = min(xbins) - margin_bins
    xmax_b = max(xbins) + margin_bins
    ymin_b = min(ybins) - margin_bins
    ymax_b = max(ybins) + margin_bins

    def to_bin(r):
        return (int(r["imagecol"] // bin_size), int(r["imagerow"] // bin_size))

    bin_series = grid.obs.apply(to_bin, axis=1)
    in_box = bin_series.apply(
        lambda t: xmin_b <= t[0] <= xmax_b and ymin_b <= t[1] <= ymax_b
    )
    subset_grid = grid[in_box].copy()

    if subset_grid.n_obs == 0:
        raise ValueError(f"solid_id={solid_id}: empty subset_grid")

    xy_list = [to_bin(r) for _, r in subset_grid.obs.iterrows()]

    outline_pts = list(df_interior.index[
        (df_interior["euclidean"] >= -1.0) & (df_interior["euclidean"] <= 0.0)
    ])
    inside_pts = list(df_interior.index[df_interior["euclidean"] < -1.0])

    if len(outline_pts) == 0:
        outline_pts = list(df_interior.index)
        inside_pts = []

    min_vector_df = compute_min_vectors_polar(
        xy_list=xy_list,
        outline_points=outline_pts,
        inside_points=inside_pts,
        invert_y=True,
        make_inside_negative=True,
    )

    roi = (
        (xmin_b - 1) * bin_size,
        (xmax_b + 2) * bin_size,
        (ymin_b - 1) * bin_size,
        (ymax_b + 2) * bin_size,
    )

    return subset_grid, min_vector_df, roi


def compute_per_solid_polar_counts(
    grid,
    solid,
    cell_df: pd.DataFrame,
    cell_types: Sequence[str],
    bin_size: int = 10,
    margin_bins: int = 15,
    min_cells: int = 5,
    cluster_col: str = "Cluster",
) -> Dict[str, Dict[int, Tuple[np.ndarray, int]]]:
    """
    For each solid and each cell type, compute the polar 2D-histogram counts
    of that cell type's cells relative to the solid boundary.

    Parameters
    ----------
    grid : AnnData
    solid : AnnData
        Output of :func:`extract_tumor_solids`.
    cell_df : pd.DataFrame
        Per-cell table with spatial coordinates in its first two columns
        (see ``scomv.cell.annotate_cells``) and a ``cluster_col`` column
        holding the cell type label (e.g. a broad cell type).
    cell_types : Sequence[str]
        Which values of ``cluster_col`` to compute results for.
    bin_size, margin_bins : int
        Passed through to :func:`compute_solid_min_vectors`.
    min_cells : int
        Minimum number of cells of a given type within a solid's ROI, and
        minimum vector samples per cell type, required to compute a polar
        histogram (passed to ``scomv.cell.compute_cluster_polar_distributions``).
    cluster_col : str
        Column in ``cell_df`` holding the cell type label.

    Returns
    -------
    dict[str, dict[int, tuple[np.ndarray, int]]]
        ``results[cell_type][solid_id] = (A_counts, n_cells)``, where
        ``A_counts`` is the (radius bin x angle bin) histogram and
        ``n_cells`` is the number of cells of that type used to build it.
    """
    results: Dict[str, Dict[int, Tuple[np.ndarray, int]]] = {ct: {} for ct in cell_types}

    for solid_id in solid.obs["solid"]:
        sid = int(solid_id)
        try:
            _, min_vector_df, roi = compute_solid_min_vectors(
                grid, solid, solid_id=sid, bin_size=bin_size, margin_bins=margin_bins
            )
        except ValueError:
            continue

        cell_sub = annotate_cells(cell_df, roi=roi, bin_size=bin_size)

        for ct in cell_types:
            sub = cell_sub[cell_sub[cluster_col] == ct].copy()
            if len(sub) < min_cells:
                continue
            try:
                _, _, counts = compute_cluster_polar_distributions(
                    sub, min_vector_df, cluster_col=cluster_col,
                    plot=False, min_cells=min_cells,
                )
            except Exception:
                continue
            if ct in counts:
                results[ct][sid] = (counts[ct], len(sub))

    return results


def compute_uniformity(
    solid_results: Mapping[str, Mapping[int, Tuple[np.ndarray, int]]],
    cell_types: Sequence[str],
    dist_bins: Mapping[str, int],
    n_ref: float = 12,
    angle_bins_deg: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Compute a per-(cell type, solid, distance bin) uniformity score from
    polar histogram counts.

    The score combines:

    - normalized Shannon entropy across angle bins at a given radius bin
      (0 = concentrated in one direction, 1 = spread uniformly across all
      directions);
    - a cell-count confidence factor, ``1 - exp(-n_bin / n_ref)``, so that
      bins backed by very few cells don't get an artificially high score.

    Parameters
    ----------
    solid_results : dict[str, dict[int, tuple[np.ndarray, int]]]
        Output of :func:`compute_per_solid_polar_counts`:
        ``solid_results[cell_type][solid_id] = (A_counts, n_cells)``.
    cell_types : Sequence[str]
        Cell types to include as columns (in this order).
    dist_bins : dict[str, int]
        Mapping from a distance-bin label (used as a key in the returned
        dicts) to the row index into ``A_counts`` along the radius axis.
    n_ref : float
        Reference cell count for the confidence factor
        ``1 - exp(-n_bin / n_ref)``.
    angle_bins_deg : np.ndarray or None
        Angle bin edges (degrees) matching the columns of ``A_counts``.
        Defaults to ``np.arange(-180, 181, 30)`` — must match whatever was
        used to build ``A_counts``
        (e.g. in ``scomv.cell.compute_cluster_polar_distributions``).

    Returns
    -------
    uniformity : dict[str, pd.DataFrame]
        ``uniformity[bin_name]``, indexed by solid_id, columns =
        ``cell_types``; values = uniformity score, 0.0 where data is missing.
    r_bar : dict[str, pd.DataFrame]
        Same shape; values = mean resultant vector length (0 = spread
        uniformly, 1 = all in one direction). Not used downstream by
        default — kept for reference.
    """
    if angle_bins_deg is None:
        angle_bins_deg = np.arange(-180, 181, 30)
    angle_centers = (angle_bins_deg[:-1] + angle_bins_deg[1:]) / 2
    rad = np.deg2rad(angle_centers)

    def _entropy_metrics(slice_: np.ndarray) -> Tuple[float, float]:
        total = slice_.sum()
        if total == 0 or np.isnan(total):
            return np.nan, np.nan
        p = slice_ / total
        uniformity = float(_scipy_entropy(p + 1e-12) / np.log(len(p)))
        r_bar = float(np.abs(np.sum(p * np.exp(1j * rad))))
        return uniformity, r_bar

    uniformity_mats: Dict[str, pd.DataFrame] = {}
    r_bar_mats: Dict[str, pd.DataFrame] = {}

    for bin_name, bin_idx in dist_bins.items():
        uni_records = []
        rbar_records = []
        for ct in cell_types:
            for sid, (A, n_cells) in solid_results.get(ct, {}).items():
                uni, r_bar = _entropy_metrics(A[bin_idx, :])
                if np.isnan(uni):
                    score, r_bar_val = 0.0, 0.0
                else:
                    n_bin = A[bin_idx, :].sum() * n_cells
                    weight = 1.0 - np.exp(-n_bin / n_ref)
                    score = float(uni * weight)
                    r_bar_val = r_bar
                uni_records.append({"cell_type": ct, "solid_id": sid, "score": score})
                rbar_records.append({"cell_type": ct, "solid_id": sid, "score": r_bar_val})

        uni_mat = pd.DataFrame(uni_records).pivot(
            index="solid_id", columns="cell_type", values="score"
        )
        uniformity_mats[bin_name] = uni_mat.reindex(columns=cell_types).fillna(0.0)

        rbar_mat = pd.DataFrame(rbar_records).pivot(
            index="solid_id", columns="cell_type", values="score"
        )
        r_bar_mats[bin_name] = rbar_mat.reindex(columns=cell_types).fillna(0.0)

    return uniformity_mats, r_bar_mats


def distance_weighted_mean(
    feature_mats: Mapping[str, pd.DataFrame],
    bin_weights: Mapping[str, float],
) -> pd.DataFrame:
    """
    Combine per-distance-bin feature matrices into a single distance-weighted
    matrix: ``sum(bin_weights[b] * feature_mats[b]) / sum(bin_weights)``.

    Parameters
    ----------
    feature_mats : dict[str, pd.DataFrame]
        One DataFrame per distance-bin label (index = solid_id, columns =
        e.g. cell types), such as the first element returned by
        :func:`compute_uniformity`. All DataFrames used (i.e. the ones with
        a key in ``bin_weights``) must share the same index and columns.
    bin_weights : dict[str, float]
        Weight per distance-bin label; keys must be a subset of
        ``feature_mats``' keys. E.g. an inverse-square-root falloff with bin
        centers ``d`` (µm) and a reference distance ``d0``:
        ``{name: np.sqrt(d0 / d_center) for name, d_center in ...}``.

    Returns
    -------
    pd.DataFrame
        Weighted mean, same index/columns as the input matrices, with NaNs
        (from bins with no overlapping index/columns) filled as 0.0.
    """
    if not bin_weights:
        raise ValueError("bin_weights must not be empty")

    w_sum = sum(bin_weights.values())
    bin_names = list(bin_weights)
    combined = bin_weights[bin_names[0]] * feature_mats[bin_names[0]]
    for bin_name in bin_names[1:]:
        combined = combined + bin_weights[bin_name] * feature_mats[bin_name]

    return (combined / w_sum).fillna(0.0)


def ward_cluster(
    feat_df: pd.DataFrame,
    n_clusters: int,
    method: str = "ward",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize columns, hierarchically cluster rows, and cut into
    ``n_clusters`` flat clusters.

    Parameters
    ----------
    feat_df : pd.DataFrame
        Feature matrix, rows = samples (e.g. solids), columns = features.
    n_clusters : int
        Number of flat clusters to cut the dendrogram into
        (``scipy.cluster.hierarchy.fcluster(..., criterion="maxclust")``).
    method : str
        Linkage method, passed to ``scipy.cluster.hierarchy.linkage``.

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        0-indexed cluster label per row of ``feat_df`` (row order preserved).
    Z : np.ndarray
        Linkage matrix, e.g. for ``sns.clustermap(..., row_linkage=Z)`` or
        ``scipy.cluster.hierarchy.dendrogram``.
    """
    X = StandardScaler().fit_transform(feat_df.values)
    Z = linkage(X, method=method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
    return labels, Z


def remap_clusters_by_reference(
    labels: Sequence[int],
    feat_df: pd.DataFrame,
    ref_col: str,
    ascending: bool = False,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Relabel cluster ids by a reference column's per-cluster mean, so cluster
    identities are interpretable (e.g. "cluster 0 = highest mean CD8+ T cell
    score") and stable across reruns of :func:`ward_cluster` (which assigns
    ids arbitrarily).

    Parameters
    ----------
    labels : array-like of int
        Cluster id per row of ``feat_df`` (as returned by
        :func:`ward_cluster`), in the same row order as ``feat_df``.
    feat_df : pd.DataFrame
        Feature matrix the clustering was computed on (or any DataFrame
        sharing row order with ``labels``).
    ref_col : str
        Column in ``feat_df`` to rank clusters by.
    ascending : bool
        If True, cluster 0 = lowest mean ``ref_col``; if False (default),
        cluster 0 = highest.

    Returns
    -------
    labels_remapped : np.ndarray
        Same shape as ``labels``, with ids reassigned by rank.
    remap : dict[int, int]
        old_label -> new_label.
    """
    labels = np.asarray(labels)
    ref = feat_df[ref_col]
    unique_labels = sorted(set(labels.tolist()))
    mean_by_cluster = {
        k: ref[feat_df.index[labels == k]].mean() for k in unique_labels
    }
    ranked = sorted(mean_by_cluster, key=mean_by_cluster.get, reverse=not ascending)
    remap = {old_k: new_k for new_k, old_k in enumerate(ranked)}
    labels_remapped = np.array([remap[k] for k in labels])
    return labels_remapped, remap


def add_significance_bars(
    ax,
    data_by_cluster: Mapping[int, np.ndarray],
    n_clusters: int,
    alpha: float = 0.05,
    max_bars: Optional[int] = None,
    start: float = 1.13,
):
    """
    Draw pairwise Mann-Whitney significance bars above a categorical plot's
    axes frame (e.g. on top of a violin/strip plot of cluster-wise values).

    p-values are Bonferroni-corrected across all pairs, drawn most- to
    least-significant from the bottom, and use a blended transform so bars
    sit in axes-fraction y-space (independent of data ylim).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    data_by_cluster : dict[int, np.ndarray]
        Cluster index (0..n_clusters-1) -> array of values.
    n_clusters : int
        Number of clusters (bars are drawn between x-positions 0..n_clusters-1).
    alpha : float
        Significance threshold on the Bonferroni-corrected p-value.
    max_bars : int or None
        Keep only the most significant ``max_bars`` pairs. Does not modify
        ``ax``'s ylim.
    start : float
        y position (axes fraction) of the first bar.
    """
    pairs = list(combinations(range(n_clusters), 2))
    n_tests = len(pairs)

    all_vals = np.concatenate([v for v in data_by_cluster.values() if len(v) > 0])
    if len(all_vals) == 0:
        return

    sig_pairs = []
    for i, j in pairs:
        v1 = data_by_cluster.get(i, np.array([]))
        v2 = data_by_cluster.get(j, np.array([]))
        if len(v1) < 2 or len(v2) < 2:
            continue
        _, p = mannwhitneyu(v1, v2, alternative="two-sided")
        p_adj = min(p * n_tests, 1.0)
        if p_adj < alpha:
            stars = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*"
            sig_pairs.append((i, j, stars, p_adj))

    sig_pairs.sort(key=lambda x: x[3])
    if max_bars is not None:
        sig_pairs = sig_pairs[:max_bars]
    sig_pairs.sort(key=lambda x: (x[1] - x[0], x[0]))

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    step = 0.040
    gap = 0.010

    height = start
    for i, j, stars, _ in sig_pairs:
        bar_h = height + gap
        ax.plot([i, i, j, j], [height, bar_h, bar_h, height],
                lw=1.0, color="black",
                transform=trans, clip_on=False, zorder=5)
        ax.text((i + j) / 2, bar_h, stars,
                ha="center", va="bottom", fontsize=8,
                transform=trans, clip_on=False, zorder=5)
        height += step


_DEFAULT_CLUSTER_HEX = [
    "#e6194b", "#4363d8", "#3cb44b", "#f58231", "#ffe119",
    "#42d4f4", "#f032e6", "#a9a9a9",
]


def plot_uniformity_by_cluster(
    cell_type: str,
    feat_df: pd.DataFrame,
    labels: Sequence[int],
    n_clusters: Optional[int] = None,
    palette: Optional[Mapping[int, str]] = None,
    ylim: Tuple[float, float] = (-0.05, 1.0),
    ylabel: str = "Uniformity",
    max_bars: Optional[int] = None,
    sig_start: float = 1.04,
    top_margin: float = 0.72,
    ax=None,
):
    """
    Violin + strip plot of one cell type's score across clusters, with
    pairwise Mann-Whitney significance bars on top.

    Parameters
    ----------
    cell_type : str
        Column of ``feat_df`` to plot (e.g. a cell type name).
    feat_df : pd.DataFrame
        Feature matrix indexed by solid_id, with ``cell_type`` as a column
        (e.g. output of :func:`distance_weighted_mean`).
    labels : array-like of int
        0-indexed cluster id per row of ``feat_df``, in the same row order
        (e.g. from :func:`ward_cluster` + :func:`remap_clusters_by_reference`).
    n_clusters : int or None
        Number of clusters; inferred as ``max(labels) + 1`` if None.
    palette : dict[int, str] or None
        Cluster id -> hex color. Defaults to a built-in qualitative palette.
    ylim : tuple[float, float]
        y-axis limits.
    ylabel : str
        y-axis label.
    max_bars : int or None
        Passed to :func:`add_significance_bars` — cap the number of
        significance bars drawn (most significant first).
    sig_start : float
        Passed to :func:`add_significance_bars` as ``start``.
    top_margin : float
        ``fig.subplots_adjust(top=...)`` value, i.e. how much headroom to
        leave above the axes for the title + significance bars. If bars
        overlap the title, lower this (e.g. 0.6); if there's too much empty
        space, raise it (e.g. 0.85). Has no effect when ``ax`` is given.
    ax : matplotlib.axes.Axes or None
        Draw into an existing axes instead of creating a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = int(labels.max()) + 1
    if palette is None:
        palette = {k: _DEFAULT_CLUSTER_HEX[k % len(_DEFAULT_CLUSTER_HEX)] for k in range(n_clusters)}

    clust_labels = [f"c{k}" for k in range(n_clusters)]
    str_palette = {f"c{k}": palette[k] for k in range(n_clusters)}

    vals_col = feat_df[cell_type]
    rows = []
    data_by_cluster: Dict[int, np.ndarray] = {}
    for k in range(n_clusters):
        vals = vals_col[feat_df.index[labels == k]].values
        data_by_cluster[k] = vals
        for v in vals:
            rows.append({"cluster": f"c{k}", "score": v})
    df_plot = pd.DataFrame(rows)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    else:
        fig = ax.figure

    sns.violinplot(data=df_plot, x="cluster", y="score",
                   hue="cluster", order=clust_labels,
                   palette=str_palette, inner="box", linewidth=1.2, legend=False, ax=ax)
    sns.stripplot(data=df_plot, x="cluster", y="score",
                  hue="cluster", order=clust_labels,
                  palette=str_palette, size=4, jitter=True,
                  alpha=0.6, linewidth=0.3, edgecolor="white", legend=False, ax=ax)

    ax.set_ylim(*ylim)
    add_significance_bars(ax, data_by_cluster, n_clusters, max_bars=max_bars, start=sig_start)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticklabels(
        [f"c{k}\n(n={len(data_by_cluster[k])})" for k in range(n_clusters)], fontsize=14
    )
    ax.tick_params(axis="y", labelsize=12)
    fig.suptitle(cell_type, fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=top_margin)

    return ax


def plot_uniformity_heatmap(
    feat_df: pd.DataFrame,
    Z: np.ndarray,
    cmap: str = "coolwarm",
    vmin: float = 0,
    vmax: float = 1,
    figsize: Tuple[float, float] = (16, 12),
    cbar_label: str = "Uniformity",
    xtick_fontsize: int = 20,
    ytick_fontsize: int = 13,
    save_path: Optional[str] = None,
):
    """
    Row-clustered heatmap of a feature matrix, using a precomputed row
    linkage (columns are left in their given order).

    Parameters
    ----------
    feat_df : pd.DataFrame
        Feature matrix (rows = samples, columns = features), e.g. output of
        :func:`distance_weighted_mean`.
    Z : np.ndarray
        Row linkage matrix, e.g. the second element returned by
        :func:`ward_cluster`.
    cmap, vmin, vmax : passed to ``sns.clustermap``.
    figsize : tuple[float, float]
    cbar_label : str
        Label on the color bar.
    xtick_fontsize, ytick_fontsize : int
    save_path : str or None
        If given, save the figure to this path.

    Returns
    -------
    seaborn.matrix.ClusterGrid
    """
    g = sns.clustermap(
        feat_df,
        row_linkage=Z,
        col_cluster=False,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        figsize=figsize,
        dendrogram_ratio=(0.2, 0),
        cbar_pos=(1.02, 0.3, 0.02, 0.4),
        xticklabels=True,
        yticklabels=True,
        linewidths=0.3,
        linecolor="gray",
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=xtick_fontsize, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=ytick_fontsize, rotation=0)
    g.ax_heatmap.set_ylabel("")
    g.cax.set_ylabel(cbar_label, fontsize=9, rotation=270, labelpad=12)

    if save_path is not None:
        g.savefig(save_path, dpi=300)

    return g


def plot_cluster_map(
    labels_img: np.ndarray,
    solid_obs: pd.DataFrame,
    cluster_of_solid: Mapping[int, int],
    n_clusters: Optional[int] = None,
    palette: Optional[Mapping[int, str]] = None,
    background_rgb: Tuple[int, int, int] = (40, 40, 40),
    figsize: Tuple[float, float] = (14, 11),
    dpi: int = 120,
    title: str = "Solid regions — cluster map",
    legend_title: Optional[str] = None,
    id_fontsize: int = 20,
    id_col: str = "solid",
    x_col: str = "imagecol",
    y_col: str = "imagerow",
    save_path: Optional[str] = None,
    ax=None,
) -> Tuple[plt.Axes, np.ndarray]:
    """
    Colorize a solid label image by cluster id, with per-solid id numbers
    and a cluster-size legend.

    Parameters
    ----------
    labels_img : np.ndarray
        Solid label image, as returned by :func:`label_solid_regions`
        (0 = background, 1..N = solid ids).
    solid_obs : pd.DataFrame
        Per-solid table with centroid coordinates and solid id (e.g.
        ``solid.obs`` from :func:`extract_tumor_solids`), used to place the
        id number text.
    cluster_of_solid : dict[int, int]
        solid_id -> 0-indexed cluster_id (e.g.
        ``dict(zip(feat_df.index, labels))``). Solids not present are left
        as background.
    n_clusters : int or None
        Number of clusters; inferred as ``max(cluster_of_solid.values()) + 1``
        if None.
    palette : dict[int, str] or None
        cluster_id -> hex color. Defaults to a built-in qualitative palette.
    background_rgb : tuple[int, int, int]
        Fill color for pixels with no assigned cluster.
    id_fontsize : int
        Font size for the per-solid id number.
    id_col, x_col, y_col : str
        Columns of ``solid_obs`` holding the solid id and centroid.
    save_path : str or None
        If given, save the figure (``fig.savefig(..., bbox_inches="tight")``).
    ax : matplotlib.axes.Axes or None
        Draw into an existing axes instead of creating a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
    img : np.ndarray, shape (H, W, 3), uint8
        The colorized image — reuse it as ``cluster_img`` in
        :func:`plot_cell_type_over_clusters` to avoid recomputing it.
    """
    if n_clusters is None:
        n_clusters = max(cluster_of_solid.values()) + 1
    if palette is None:
        palette = {k: _DEFAULT_CLUSTER_HEX[k % len(_DEFAULT_CLUSTER_HEX)] for k in range(n_clusters)}
    rgb = {k: tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for k, c in palette.items()}

    h, w = labels_img.shape
    img = np.full((h, w, 3), background_rgb, dtype=np.uint8)
    for sid in range(1, int(labels_img.max()) + 1):
        mask = labels_img == sid
        if not mask.any():
            continue
        k = cluster_of_solid.get(sid)
        if k is not None:
            img[mask] = rgb[k]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    ax.imshow(img, interpolation="nearest")
    for _, row in solid_obs.iterrows():
        sid = int(row[id_col])
        if sid in cluster_of_solid:
            ax.text(row[x_col], row[y_col], str(sid),
                    fontsize=id_fontsize, color="white", fontweight="bold",
                    ha="center", va="center", zorder=4)

    cluster_sizes = {k: sum(1 for v in cluster_of_solid.values() if v == k) for k in range(n_clusters)}
    legend_elements = [
        Line2D([0], [0], marker="s", color="w",
               label=f"c{k}  (n={cluster_sizes[k]})",
               markerfacecolor=np.array(rgb[k]) / 255,
               markeredgecolor="none", markersize=12)
        for k in range(n_clusters)
    ]
    ax.legend(handles=legend_elements, title=legend_title or f"Cluster (k={n_clusters})",
              loc="upper right", fontsize=9, title_fontsize=10, frameon=True)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax, img


def plot_cell_type_over_clusters(
    cell_type: str,
    cell_df: pd.DataFrame,
    cluster_img: np.ndarray,
    origin: Tuple[float, float],
    cluster_sizes: Mapping[int, int],
    bin_size: int = 10,
    cluster_col: str = "Cluster",
    x_col: str = "imagecol",
    y_col: str = "imagerow",
    palette: Optional[Mapping[int, str]] = None,
    dot_color: str = "#00ffff",
    dot_size: float = 2,
    fade: float = 0.1,
    background_rgb: Tuple[int, int, int] = (220, 220, 220),
    figsize: Tuple[float, float] = (8, 7),
    dpi: int = 150,
    save_path: Optional[str] = None,
    ax=None,
):
    """
    Overlay one cell type's spatial positions (dots) on a faded cluster map,
    to inspect its distribution relative to clusters.

    Everything except ``cell_type`` is normally the same across calls, so
    bind it once with ``functools.partial`` to get a "just pass the cell
    type name" plotting function::

        from functools import partial
        show = partial(
            plot_cell_type_over_clusters,
            cell_df=cell_df_broad, cluster_img=cluster_img,
            origin=(adata.obs.imagecol.min(), adata.obs.imagerow.min()),
            cluster_sizes=cluster_sizes, cluster_col="BroadType",
        )
        show("CD8+ T cells")
        show("Fibroblasts")

    Parameters
    ----------
    cell_type : str
        Value of ``cluster_col`` in ``cell_df`` to scatter.
    cell_df : pd.DataFrame
        Per-cell table with spatial coordinates and ``cluster_col``.
    cluster_img : np.ndarray, shape (H, W, 3)
        Colorized cluster image, as returned by :func:`plot_cluster_map`.
    origin : tuple[float, float]
        (x_min, y_min), in the same spatial units as ``x_col``/``y_col``,
        used to align cell coordinates onto ``cluster_img``'s pixel grid —
        e.g. ``(adata.obs.imagecol.min(), adata.obs.imagerow.min())``.
    cluster_sizes : dict[int, int]
        cluster_id -> count, for the legend (e.g.
        ``pd.Series(labels).value_counts().to_dict()``).
    bin_size : int
        Pixel size of one ``cluster_img`` cell, in the same units as
        ``x_col``/``y_col``.
    palette : dict[int, str] or None
        cluster_id -> hex color; should match what ``cluster_img`` was built
        with. Defaults to the same built-in palette as :func:`plot_cluster_map`.
    dot_color, dot_size : scatter style for the cell-type dots.
    fade : float
        How much to fade ``cluster_img`` toward ``background_rgb`` (0 = no
        fade, 1 = fully background), so the overlaid dots stand out.
    save_path : str or None
        If given, save the figure (``fig.savefig(..., bbox_inches="tight")``).
    ax : matplotlib.axes.Axes or None
        Draw into an existing axes instead of creating a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    n_clusters = len(cluster_sizes)
    if palette is None:
        palette = {k: _DEFAULT_CLUSTER_HEX[k % len(_DEFAULT_CLUSTER_HEX)] for k in range(n_clusters)}

    h_px, w_px = cluster_img.shape[:2]
    bg = np.full_like(cluster_img, background_rgb)
    img_faded = (cluster_img * (1 - fade) + bg * fade).astype(np.uint8)

    x_min, y_min = origin
    sub = cell_df[cell_df[cluster_col] == cell_type]
    x_px = (sub[x_col] - x_min) / bin_size
    y_px = (sub[y_col] - y_min) / bin_size

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    ax.imshow(img_faded, interpolation="nearest")
    ax.scatter(x_px, y_px, s=dot_size, c=dot_color, alpha=0.85, linewidths=0, zorder=3)
    ax.set_xlim(0, w_px)
    ax.set_ylim(h_px, 0)
    ax.axis("off")
    ax.set_title(f"{cell_type}  (n={len(sub):,})", fontsize=11, fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor=palette[k], edgecolor="none",
                       label=f"c{k}  (n={cluster_sizes[k]})")
        for k in sorted(cluster_sizes)
    ] + [mpatches.Patch(facecolor=dot_color, edgecolor="none", label=cell_type)]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return ax


def compute_expression_fold_factor(
    grid,
    solid,
    solid_ids_a: Union[int, Sequence[int]],
    solid_ids_b: Union[int, Sequence[int]],
    bin_size: int = 10,
    margin_bins: int = 15,
) -> float:
    """
    Total-expression fold factor between two solid groups, to correct for a
    systematic expression offset before comparing gene-level uniformity or
    running DEG between them.

    Each group's total expression is the per-gene sum of its solids'
    grid-square values (as in :func:`compute_solid_min_vectors`'
    ``subset_grid``), summed across solids within the group.

    Parameters
    ----------
    grid, solid : AnnData
    solid_ids_a, solid_ids_b : int or list[int]
        **The tumor number(s) you are comparing** — e.g. ``solid_ids_a=7,
        solid_ids_b=12`` to compare tumor 7 against tumor 12, or
        ``solid_ids_a=[1, 7]`` for a group made of several solids (e.g. one
        Ward cluster). These are the same numbers shown as text labels in
        :func:`plot_cluster_map` and stored in ``solid.obs["solid"]`` — look
        there first to decide what to pass.
    bin_size, margin_bins : int
        Passed to :func:`compute_solid_min_vectors`.

    Returns
    -------
    float
        k = median(total_expr_b / total_expr_a) over genes with nonzero
        expression in group a. k >= 1 means group b's total expression is
        higher; k < 1 means group a's is higher.
    """
    def total_expr(solid_ids):
        totals = None
        var_names = None
        for sid in _as_solid_id_list(solid_ids):
            sg, _, _ = compute_solid_min_vectors(
                grid, solid, solid_id=sid, bin_size=bin_size, margin_bins=margin_bins
            )
            X = sg.X
            if issparse(X):
                X = X.toarray()
            s = np.asarray(X).sum(axis=0)
            if totals is None:
                totals = s.copy()
                var_names = list(sg.var_names)
            elif list(sg.var_names) != var_names:
                raise ValueError("solids within the same group have mismatched var_names")
            else:
                totals = totals + s
        return dict(zip(var_names, totals.tolist()))

    expr_a = total_expr(solid_ids_a)
    expr_b = total_expr(solid_ids_b)

    common = sorted(set(expr_a) & set(expr_b))
    ratios = [expr_b[g] / expr_a[g] for g in common if expr_a[g] > 0]
    if not ratios:
        raise ValueError("no genes with nonzero expression in group a to compute a fold factor")
    return float(np.median(ratios))


def _gene_weighted_polar_samples(subset_grid, min_vector_df, gene, bin_size_um=10):
    genes = list(subset_grid.var.index)
    if gene not in genes:
        raise ValueError(f"Gene '{gene}' not found in subset_grid.var.index")
    idx = genes.index(gene)

    expr = subset_grid.X[:, idx]
    if hasattr(expr, "toarray"):
        expr = expr.toarray().ravel()
    else:
        expr = np.asarray(expr).ravel()

    radiis, degrees, weights = [], [], []
    for i, expression in enumerate(expr):
        if expression <= 0:
            continue
        angles = min_vector_df["angle"].iloc[i]
        radiis_i = min_vector_df["radii"].iloc[i]
        n_vecs = len(angles)
        if n_vecs == 0:
            continue
        w = float(expression) / n_vecs
        for ang, rad in zip(angles, radiis_i):
            degrees.append(np.degrees(ang))
            radiis.append(rad * bin_size_um)
            weights.append(w)

    return np.asarray(radiis), np.asarray(degrees), np.asarray(weights)


def compute_gene_uniformity(
    grid,
    solid,
    solid_ids: Union[int, Sequence[int]],
    dist_bins: Mapping[str, int],
    genes: Optional[Sequence[str]] = None,
    n_ref: float = 24,
    n_scale: float = 1.0,
    bin_size: int = 10,
    margin_bins: int = 15,
    radius_bins: Optional[np.ndarray] = None,
    angle_bins_deg: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Per-gene, per-distance-bin uniformity for one solid group (a single
    solid id, or a list — e.g. all solids in one cluster — whose weighted
    expression samples are pooled before computing uniformity, the same way
    :func:`compute_uniformity` does for cell types).

    Parameters
    ----------
    grid, solid : AnnData
    solid_ids : int or list[int]
        **The tumor number(s) making up this group** — e.g. ``solid_ids=7``
        for tumor 7 alone, or ``solid_ids=[1, 7]`` to pool several solids
        (e.g. one Ward cluster). These are the same numbers shown as text
        labels in :func:`plot_cluster_map` and stored in
        ``solid.obs["solid"]`` — look there first to decide what to pass.
    dist_bins : dict[str, int]
        Distance-bin label -> row index into the (radius x angle) histogram,
        as in :func:`compute_uniformity`.
    genes : sequence of str or None
        Genes to compute. Defaults to genes common to all solids in
        ``solid_ids``.
    n_ref : float
        Reference cell count for the confidence factor (same formula as
        :func:`compute_uniformity`, ``1 - exp(-n_bin / n_ref)``). Defaults
        to 24 here (vs. 12 for cell-type-level :func:`compute_uniformity`)
        — gene-level counts run lower, so a higher reference count is used
        for full confidence.
    n_scale : float
        Multiplier on the pooled expressing-cell count — e.g. from
        :func:`compute_expression_fold_factor`, to correct for a systematic
        total-expression offset against another group being compared (see
        :func:`compute_gene_uniformity_pair`).
    bin_size, margin_bins : int
        Passed to :func:`compute_solid_min_vectors`.
    radius_bins, angle_bins_deg : np.ndarray or None
        Histogram bin edges; default to the same values used elsewhere in
        this module (``np.arange(-150, 310, 10)`` / ``np.arange(-180, 181, 30)``).

    Returns
    -------
    pd.DataFrame
        Indexed by gene, columns = ``dist_bins`` keys, values = uniformity
        score. Genes with no expression anywhere in the group's ROI (across
        all solids in ``solid_ids``) are dropped, not zero-filled.
    """
    if radius_bins is None:
        radius_bins = np.arange(-150, 310, 10)
    if angle_bins_deg is None:
        angle_bins_deg = np.arange(-180, 181, 30)
    angle_centers = (angle_bins_deg[:-1] + angle_bins_deg[1:]) / 2
    rad = np.deg2rad(angle_centers)

    subsets = []
    gene_sets = []
    for sid in _as_solid_id_list(solid_ids):
        sg, mv, _ = compute_solid_min_vectors(
            grid, solid, solid_id=sid, bin_size=bin_size, margin_bins=margin_bins
        )
        subsets.append((sg, mv))
        gene_sets.append(set(sg.var_names))

    if genes is None:
        genes = sorted(set.intersection(*gene_sets)) if gene_sets else []

    def _entropy(slice_):
        total = slice_.sum()
        if total == 0 or np.isnan(total):
            return np.nan
        p = slice_ / total
        return float(_scipy_entropy(p + 1e-12) / np.log(len(p)))

    records = []
    for gene in genes:
        all_radii, all_deg, all_w = [], [], []
        n_expr_cells = 0
        for sg, mv in subsets:
            r, d, w = _gene_weighted_polar_samples(sg, mv, gene, bin_size_um=bin_size)
            all_radii.append(r)
            all_deg.append(d)
            all_w.append(w)

            idx = list(sg.var_names).index(gene)
            expr = sg.X[:, idx]
            expr = expr.toarray().ravel() if hasattr(expr, "toarray") else np.asarray(expr).ravel()
            n_expr_cells += int((expr > 0).sum())

        radii_cat = np.concatenate(all_radii) if all_radii else np.array([])
        if len(radii_cat) == 0:
            # No solid in the group expresses this gene anywhere in its ROI
            # (mirrors the original notebook's per-solid ValueError -> skip).
            continue

        row = {"gene": gene}
        deg_cat = np.concatenate(all_deg)
        w_cat = np.concatenate(all_w)
        w_norm = w_cat / w_cat.sum()

        A_counts, _, _ = np.histogram2d(
            radii_cat, deg_cat, bins=[radius_bins, angle_bins_deg], weights=w_norm
        )

        n_cells = n_expr_cells * n_scale
        for bin_name, bin_idx in dist_bins.items():
            uni = _entropy(A_counts[bin_idx, :])
            if np.isnan(uni):
                row[bin_name] = 0.0
            else:
                n_bin = A_counts[bin_idx, :].sum() * n_cells
                weight = 1.0 - np.exp(-n_bin / n_ref)
                row[bin_name] = float(uni * weight)
        records.append(row)

    return pd.DataFrame(records).set_index("gene")


def compute_gene_uniformity_pair(
    grid,
    solid,
    solid_ids_a: Union[int, Sequence[int]],
    solid_ids_b: Union[int, Sequence[int]],
    dist_bins: Mapping[str, int],
    genes: Optional[Sequence[str]] = None,
    n_ref: float = 24,
    correct_expression: bool = True,
    bin_size: int = 10,
    margin_bins: int = 15,
    radius_bins: Optional[np.ndarray] = None,
    angle_bins_deg: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Per-gene uniformity for two solid groups, on a comparable expression
    scale.

    If ``correct_expression`` is True (default), the total-expression fold
    factor between the two groups (:func:`compute_expression_fold_factor`)
    is used to scale up the pooled expressing-cell count of whichever group
    has the lower total expression, before computing uniformity — so a
    systematic total-expression offset between the groups doesn't
    masquerade as a uniformity difference.

    Parameters
    ----------
    solid_ids_a, solid_ids_b : int or list[int]
        **The tumor number(s) you are comparing**, e.g. ``solid_ids_a=7,
        solid_ids_b=12``. See :func:`compute_gene_uniformity`'s
        ``solid_ids`` for how to find these numbers.
    dist_bins, genes, n_ref, bin_size, margin_bins, radius_bins,
    angle_bins_deg : see :func:`compute_gene_uniformity`.
    correct_expression : bool
        Apply the fold-factor correction described above.

    Returns
    -------
    wu_a, wu_b : pd.DataFrame
        Per-gene, per-distance-bin uniformity for each group, restricted to
        genes common to both.
    k : float
        The fold factor used (1.0 if ``correct_expression`` is False).
    """
    ids_a = _as_solid_id_list(solid_ids_a)
    ids_b = _as_solid_id_list(solid_ids_b)

    n_scale_a, n_scale_b, k = 1.0, 1.0, 1.0
    if correct_expression:
        k = compute_expression_fold_factor(
            grid, solid, ids_a, ids_b, bin_size=bin_size, margin_bins=margin_bins
        )
        if k >= 1:
            n_scale_a = k
        else:
            n_scale_b = 1.0 / k

    common_kwargs = dict(
        dist_bins=dist_bins, genes=genes, n_ref=n_ref,
        bin_size=bin_size, margin_bins=margin_bins,
        radius_bins=radius_bins, angle_bins_deg=angle_bins_deg,
    )
    wu_a = compute_gene_uniformity(grid, solid, ids_a, n_scale=n_scale_a, **common_kwargs)
    wu_b = compute_gene_uniformity(grid, solid, ids_b, n_scale=n_scale_b, **common_kwargs)

    common = sorted(set(wu_a.index) & set(wu_b.index))
    return wu_a.loc[common], wu_b.loc[common], k


def weighted_mean_across_bins(
    df: pd.DataFrame,
    bin_weights: Mapping[str, float],
) -> pd.Series:
    """
    Row-wise weighted mean of a DataFrame whose columns are distance-bin
    labels (e.g. :func:`compute_gene_uniformity`'s output — one row per
    gene, one column per distance bin).

    This is the column-wise counterpart to :func:`distance_weighted_mean`,
    which instead combines several separate per-bin matrices; use this one
    when a single DataFrame already has bins as columns.

    Parameters
    ----------
    df : pd.DataFrame
    bin_weights : dict[str, float]
        Weight per column name; keys must be a subset of ``df.columns``.

    Returns
    -------
    pd.Series
        Weighted mean per row, indexed like ``df``.
    """
    cols = list(bin_weights)
    w = np.array([bin_weights[c] for c in cols])
    return pd.Series((df[cols].values * w).sum(axis=1) / w.sum(), index=df.index)


def plot_gene_uniformity_scatter(
    wu_a: pd.DataFrame,
    wu_b: pd.DataFrame,
    bin_weights: Mapping[str, float],
    label_a: str = "Group A",
    label_b: str = "Group B",
    n_annotate: int = 10,
    figsize: Tuple[float, float] = (7, 7),
    point_size: float = 30,
    color_up: str = "#e6194b",
    color_down: str = "#4363d8",
    ax=None,
) -> Tuple[plt.Axes, List[str]]:
    """
    2D scatter of one distance-weighted uniformity score per gene: group A
    (x-axis) vs group B (y-axis), a diagonal reference line, and the most
    divergent genes on each side annotated.

    Parameters
    ----------
    wu_a, wu_b : pd.DataFrame
        Per-gene, per-distance-bin uniformity, e.g. from
        :func:`compute_gene_uniformity_pair`.
    bin_weights : dict[str, float]
        Passed to :func:`weighted_mean_across_bins` to collapse each
        group's per-distance-bin table into one score per gene.
    label_a, label_b : str
        Used in the axis labels (e.g. "tumor 7", "tumor 12").
    n_annotate : int
        Number of most-divergent genes to label on *each* side of the
        diagonal.
    ax : matplotlib.axes.Axes or None
        Draw into an existing axes instead of creating a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
    ann_genes : list[str]
        The annotated gene names (both directions), for reuse elsewhere
        (e.g. spatial follow-up plots).
    """
    common = sorted(set(wu_a.index) & set(wu_b.index))
    score_a = weighted_mean_across_bins(wu_a.loc[common], bin_weights)
    score_b = weighted_mean_across_bins(wu_b.loc[common], bin_weights)

    x = score_a.values
    y = score_b.values
    diff = y - x
    colors = np.where(diff > 0, color_up, color_down)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)

    ax.scatter(x, y, c=colors, s=point_size, alpha=0.7, linewidths=0)
    lim = (0, 1)
    ax.plot(lim, lim, color="black", lw=1.0, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    above_idx = np.where(diff > 0)[0]
    below_idx = np.where(diff < 0)[0]
    top_above = above_idx[np.argsort(diff[above_idx])[-n_annotate:]]
    top_below = below_idx[np.argsort(diff[below_idx])[:n_annotate]]
    ann_idx = np.concatenate([top_above, top_below])
    ann_genes = [common[i] for i in ann_idx]

    texts = [ax.text(x[i], y[i], common[i], fontsize=14) for i in ann_idx]
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        expand=(1.4, 1.6), force_text=(0.5, 0.7),
    )

    ax.set_xlabel(f"Uniformity of {label_a}", fontsize=18)
    ax.set_ylabel(f"Uniformity of {label_b}", fontsize=18)
    ax.tick_params(labelsize=14)

    return ax, ann_genes


def compute_deg_between_solid_groups(
    adata,
    cell_solid_ids: pd.Series,
    solid_ids_a: Union[int, Sequence[int]],
    solid_ids_b: Union[int, Sequence[int]],
    label_a: str = "group_a",
    label_b: str = "group_b",
    normalize: bool = True,
    target_sum: float = 1e4,
    method: str = "wilcoxon",
) -> pd.DataFrame:
    """
    Wilcoxon rank-sum DEG between cells belonging to two solid groups
    (group b tested against group a as the reference).

    Parameters
    ----------
    adata : AnnData
        Per-cell AnnData (raw/filtered counts).
    cell_solid_ids : pd.Series
        solid_id per cell, aligned to ``adata.obs.index`` — e.g.
        ``pd.Series(assign_solid_ids(adata.obs, labels_img), index=adata.obs.index)``.
    solid_ids_a, solid_ids_b : int or list[int]
        **The tumor number(s) you are comparing**, e.g. ``solid_ids_a=7,
        solid_ids_b=12``. See :func:`compute_gene_uniformity`'s
        ``solid_ids`` for how to find these numbers.
    label_a, label_b : str
        Group names used as categories and in the returned table (group b
        is tested against group a as the reference).
    normalize, target_sum : bool, float
        Passed to ``sc.pp.normalize_total`` (skip if ``adata`` is already
        normalized).
    method : str
        Passed to ``sc.tl.rank_genes_groups``.

    Returns
    -------
    pd.DataFrame
        Indexed by gene: scores, logfoldchanges, pvals, pvals_adj,
        -log10padj, pts, pts_rest.
    """
    ids_a = set(_as_solid_id_list(solid_ids_a))
    ids_b = set(_as_solid_id_list(solid_ids_b))

    group_of_cell = cell_solid_ids.map(
        lambda s: label_b if s in ids_b else (label_a if s in ids_a else None)
    )
    focal_mask = group_of_cell.notna()

    adata_focal = adata[adata.obs.index.isin(group_of_cell.index[focal_mask])].copy()
    adata_focal.obs["_group"] = group_of_cell.reindex(adata_focal.obs.index)

    if normalize:
        sc.pp.normalize_total(adata_focal, target_sum=target_sum)
        sc.pp.log1p(adata_focal)

    key = f"deg_{label_b}_vs_{label_a}"
    sc.tl.rank_genes_groups(
        adata_focal, groupby="_group", groups=[label_b], reference=label_a,
        method=method, key_added=key, pts=True,
    )

    deg_df = sc.get.rank_genes_groups_df(adata_focal, group=label_b, key=key)
    deg_df = deg_df.rename(columns={"names": "gene"}).set_index("gene")
    deg_df["-log10padj"] = -np.log10(deg_df["pvals_adj"].clip(lower=1e-300))

    return deg_df


def plot_deg_volcano(
    deg_df: pd.DataFrame,
    sig_padj: float = 0.05,
    sig_lfc: float = 0.5,
    n_annotate: int = 20,
    label_up: str = "Up",
    label_down: str = "Down",
    color_up: str = "#e6194b",
    color_down: str = "#4363d8",
    color_ns: str = "#cccccc",
    figsize: Tuple[float, float] = (9, 7),
    ax=None,
) -> Tuple[plt.Axes, List[str], List[str]]:
    """
    Volcano plot (log2FC vs -log10 adjusted p) from a DEG table, e.g. from
    :func:`compute_deg_between_solid_groups`.

    Parameters
    ----------
    deg_df : pd.DataFrame
        Must have "logfoldchanges", "pvals_adj", "-log10padj" columns.
    sig_padj, sig_lfc : float
        Significance thresholds for adjusted p-value and |log2FC|.
    n_annotate : int
        Number of significant genes to label (highest -log10padj first).
    label_up, label_down : str
        Legend labels for the two significant groups.
    ax : matplotlib.axes.Axes or None
        Draw into an existing axes instead of creating a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
    genes_up, genes_down : list[str]
        Significant gene names on each side, sorted by effect size —
        useful as input to :func:`run_pathway_enrichment`.
    """
    lfc = deg_df["logfoldchanges"]
    logp = deg_df["-log10padj"]
    padj = deg_df["pvals_adj"]

    up = (padj < sig_padj) & (lfc > sig_lfc)
    down = (padj < sig_padj) & (lfc < -sig_lfc)
    ns = ~(up | down)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)

    ax.scatter(lfc[ns], logp[ns], c=color_ns, s=16, alpha=0.6, linewidths=0)
    ax.scatter(lfc[down], logp[down], c=color_down, s=35, alpha=0.85, linewidths=0,
               label=f"{label_down}  (n={int(down.sum())})")
    ax.scatter(lfc[up], logp[up], c=color_up, s=35, alpha=0.85, linewidths=0,
               label=f"{label_up}  (n={int(up.sum())})")

    ax.axhline(-np.log10(sig_padj), color="black", lw=1.0, ls="--")
    ax.axvline(sig_lfc, color="black", lw=1.0, ls="--")
    ax.axvline(-sig_lfc, color="black", lw=1.0, ls="--")

    top_label = deg_df[up | down].nlargest(n_annotate, "-log10padj")
    texts = [
        ax.text(row["logfoldchanges"], row["-log10padj"], gene, fontsize=12)
        for gene, row in top_label.iterrows()
    ]
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6),
        expand=(1.4, 1.6), force_text=(0.5, 0.7),
    )

    ax.set_xlabel("log$_2$ FC", fontsize=16)
    ax.set_ylabel("-log$_{10}$(adjusted $p$-value)", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=12, loc="best", framealpha=0.85, edgecolor="#aaaaaa")

    genes_up = deg_df[up].sort_values("logfoldchanges", ascending=False).index.tolist()
    genes_down = deg_df[down].sort_values("logfoldchanges").index.tolist()

    return ax, genes_up, genes_down


def run_pathway_enrichment(
    gene_lists: Mapping[str, Sequence[str]],
    gene_sets: Sequence[str] = ("GO_Biological_Process_2023", "Reactome_2022", "MSigDB_Hallmark_2020"),
    organism: str = "human",
) -> Dict[str, pd.DataFrame]:
    """
    Run Enrichr pathway enrichment (via ``gseapy``) for each named gene list.

    Requires network access (calls the Enrichr web API) and the optional
    ``gseapy`` dependency.

    Parameters
    ----------
    gene_lists : dict[str, list[str]]
        e.g. ``{"Up": genes_up, "Down": genes_down}`` from
        :func:`plot_deg_volcano`.
    gene_sets : sequence of str
        Enrichr gene-set library names.
    organism : str
        Passed to ``gseapy.enrichr``.

    Returns
    -------
    dict[str, pd.DataFrame]
        One results table per non-empty key in ``gene_lists``.
    """
    import gseapy as gp

    results = {}
    for name, genes in gene_lists.items():
        if not genes:
            continue
        enr = gp.enrichr(gene_list=list(genes), gene_sets=list(gene_sets), organism=organism, outdir=None)
        results[name] = enr.results.copy()
    return results


def plot_pathway_enrichment(
    results: Mapping[str, pd.DataFrame],
    gene_sets: Sequence[str],
    top_n: int = 15,
    colors: Optional[Mapping[str, str]] = None,
    sig_padj: float = 0.05,
    figsize_per_panel: Tuple[float, float] = (7, 5),
):
    """
    Bar-chart panels of top enriched terms per (gene-list, gene-set) pair,
    from :func:`run_pathway_enrichment`'s output.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Output of :func:`run_pathway_enrichment`.
    gene_sets : sequence of str
        Which Enrichr gene sets (columns of the grid) to plot, in order.
    top_n : int
        Number of top terms (by adjusted p-value) per panel.
    colors : dict[str, str] or None
        Bar color per key in ``results``. Defaults to a built-in palette.
    sig_padj : float
        Reference line drawn at ``-log10(sig_padj)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    default_colors = ["#e6194b", "#4363d8", "#3cb44b", "#f58231"]
    if colors is None:
        colors = {name: default_colors[i % len(default_colors)] for i, name in enumerate(results)}

    n_rows = len(results)
    n_cols = len(gene_sets)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows), dpi=120,
    )
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    for row_i, (name, res) in enumerate(results.items()):
        res = res.copy()
        res["-log10padj"] = -np.log10(res["Adjusted P-value"].clip(lower=1e-300))
        color = colors[name]
        for col_i, gs in enumerate(gene_sets):
            ax = axes[row_i][col_i]
            sub = res[res["Gene_set"] == gs].nlargest(top_n, "-log10padj")
            if sub.empty:
                ax.axis("off")
                continue
            terms = sub["Term"].str.replace(r" \(GO:\d+\)", "", regex=True).str[:55]
            y_pos = range(len(sub))
            ax.barh(list(y_pos), sub["-log10padj"].values, color=color, alpha=0.8,
                    edgecolor="white", linewidth=0.5)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(terms.values, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel("-log$_{10}$(adjusted $p$-value)", fontsize=12)
            ax.set_title(f"{name}\n{gs}", fontsize=12, pad=6)
            ax.tick_params(axis="x", labelsize=11)
            ax.axvline(-np.log10(sig_padj), color="black", lw=1.0, ls="--")

    fig.tight_layout()
    return fig
