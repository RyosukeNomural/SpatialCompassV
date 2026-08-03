"""Cluster genes by spatial distribution pattern within one ROI.

Builds on top of ``scomv.gene_pipeline.SCOMVPipeline`` (which computes one
polar histogram per gene for an ROI, plus a PCoA embedding of those
histograms) to find "gene modules" — groups of genes with similar spatial
distribution around a reference boundary (e.g. a tumor edge) — and to
characterize/annotate them:

- :func:`compute_gene_uniformity` — per-gene, per-distance-bin uniformity
  score from already-computed polar histograms.
- :func:`compute_inside_outside_density_ratio` — per-gene inside/outside
  density ratio ("A" score).
- :func:`plot_pcoa_scored` — PCoA scatter colored by a per-gene scalar
  (e.g. uniformity or density ratio).
- :func:`plot_pcoa_groups` — PCoA scatter with named, colored gene groups
  (e.g. curated marker gene sets) on a gray background.
- :func:`cluster_genes_by_pcoa` — HDBSCAN clustering on (PC1, PC2, any
  number of extra weighted per-gene scalars, e.g. uniformity and/or
  density ratio). Pass ``extra_axes={}`` for PCoA-only clustering.
- :func:`select_pcoa_cluster_params` — sweep HDBSCAN granularity settings
  (and, if given, extra-axis weights) and score each combination by DBCV
  (a density-based cluster validity index), so settings are chosen from
  the feature space's own structure rather than by eye (e.g. "does gene X
  land where I expect").
- :func:`combine_cluster_summaries` / :func:`select_cluster_annotation_genes`
  — merge multiple clustering passes into one numbering, and pick which
  genes to label per cluster.
- :func:`compute_presence_fraction_map` / :func:`plot_presence_fraction_grid`
  — per-cluster spatial maps of "fraction of the cluster's genes expressed".
- :func:`run_go_enrichment_per_cluster` / :func:`plot_go_enrichment_per_cluster`
  — Enrichr over-representation analysis (ORA) per cluster (requires
  network access).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from scipy.stats import entropy as _scipy_entropy


def compute_gene_uniformity(
    subset_grid,
    selected_genes: Sequence[str],
    polar_counts_list: Sequence[np.ndarray],
    dist_bins: Mapping[str, int],
    n_ref: float = 24,
) -> pd.DataFrame:
    """
    Per-gene, per-distance-bin uniformity score, from polar histograms
    already computed for many genes at once (e.g. by
    ``scomv.gene_pipeline.SCOMVPipeline.run``).

    Same entropy + confidence-weighting formula as
    ``scomv.tumor_solid.compute_uniformity`` / ``compute_gene_uniformity``,
    but reads already-computed histograms instead of recomputing them per
    gene — ``SCOMVPipeline`` already builds one polar histogram per gene
    for a single ROI, so there is nothing left to recompute here.

    Parameters
    ----------
    subset_grid : AnnData
        The grid subset the histograms were computed on (e.g.
        ``gene_pipe.subset_grid``) — used to look up each gene's
        expressing grid-square count.
    selected_genes : list[str]
        Gene names, in the same order as ``polar_counts_list`` (e.g.
        ``gene_pipe.selected_genes``).
    polar_counts_list : list[np.ndarray]
        One (radius bin x angle bin) histogram per gene, same order as
        ``selected_genes`` (e.g. ``gene_pipe.polar_counts_list``).
    dist_bins : dict[str, int]
        Distance-bin label -> row index into each histogram, as in
        ``scomv.tumor_solid.compute_uniformity``.
    n_ref : float
        Reference "expressing grid square" count for the confidence
        factor ``1 - exp(-n_bin / n_ref)``.

    Returns
    -------
    pd.DataFrame
        Indexed by gene, columns = ``dist_bins`` keys, values = uniformity
        score. Combine across distance bins with
        ``scomv.tumor_solid.weighted_mean_across_bins``.
    """
    var_names = list(subset_grid.var_names)

    def _entropy(slice_: np.ndarray) -> float:
        total = slice_.sum()
        if total == 0 or np.isnan(total):
            return np.nan
        p = slice_ / total
        return float(_scipy_entropy(p + 1e-12) / np.log(len(p)))

    records = []
    for gene, A in zip(selected_genes, polar_counts_list):
        if gene in var_names:
            idx = var_names.index(gene)
            expr = subset_grid.X[:, idx]
            expr = expr.toarray().ravel() if hasattr(expr, "toarray") else np.asarray(expr).ravel()
            n_cells = int((expr > 0).sum())
        else:
            n_cells = 0

        row = {"gene": gene}
        for bin_name, bin_idx in dist_bins.items():
            uni = _entropy(A[bin_idx, :])
            if np.isnan(uni):
                row[bin_name] = 0.0
            else:
                n_bin = A[bin_idx, :].sum() * n_cells
                weight = 1.0 - np.exp(-n_bin / n_ref)
                row[bin_name] = float(uni * weight)
        records.append(row)

    return pd.DataFrame(records).set_index("gene")


def compute_inside_outside_density_ratio(
    selected_genes: Sequence[str],
    polar_counts_list: Sequence[np.ndarray],
    radius_bins: np.ndarray,
    n_inside: int,
    n_total: int,
) -> pd.Series:
    """
    Ratio of each gene's weighted expression density inside a reference
    boundary (e.g. the tumor edge) to its density outside, normalized by
    the number of inside vs. outside grid squares in the ROI.

    A ratio > 1 means the gene's spatial mass is denser inside the
    boundary than outside (per grid square); < 1 means the opposite.

    Parameters
    ----------
    selected_genes : list[str]
    polar_counts_list : list[np.ndarray]
        One (radius bin x angle bin) histogram per gene, e.g.
        ``gene_pipe.polar_counts_list`` (the radius axis is signed
        distance to the boundary; negative = inside).
    radius_bins : np.ndarray
        Radius bin edges the histograms were built with (e.g.
        ``gene_pipe.radius_bins``).
    n_inside : int
        Number of grid squares inside the boundary in the ROI (e.g.
        ``len(gene_pipe.inside_points)``).
    n_total : int
        Total number of grid squares in the ROI (e.g.
        ``len(gene_pipe.xy_list)``).

    Returns
    -------
    pd.Series
        Indexed by gene; NaN where the outside density is zero.
    """
    dist_centers = (radius_bins[:-1] + radius_bins[1:]) / 2
    inside_idx = np.where(dist_centers < 0)[0]
    outside_idx = np.where(dist_centers >= 0)[0]
    n_outside = max(n_total - n_inside, 1)
    n_inside = max(n_inside, 1)

    ratios = {}
    for gene, A in zip(selected_genes, polar_counts_list):
        A = np.asarray(A)
        w_in = float(A[inside_idx, :].sum())
        w_out = float(A[outside_idx, :].sum())
        density_in = w_in / n_inside
        density_out = w_out / n_outside
        ratios[gene] = density_in / density_out if density_out > 0 else np.nan

    return pd.Series(ratios, name="density_ratio")


def plot_pcoa_scored(
    coords: pd.DataFrame,
    values: Mapping[str, float],
    annotate_genes: Optional[Sequence[str]] = None,
    cmap: str = "RdYlBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_label: str = "",
    point_size: float = 150,
    figsize: Tuple[float, float] = (20, 15),
    label_fontsize: float = 35,
    tick_fontsize: float = 30,
    annotate_fontsize: float = 28,
    cbar_fontsize: float = 30,
    ax=None,
):
    """
    PCoA scatter (PC1 vs PC2) colored by a per-gene scalar, with optional
    gene-name labels.

    Parameters
    ----------
    coords : pd.DataFrame
        Indexed by gene, with "PC1"/"PC2" columns (e.g.
        ``gene_pipe.coords``).
    values : dict[str, float] or pd.Series
        Per-gene scalar to color by (e.g. from :func:`compute_gene_uniformity`
        + ``scomv.tumor_solid.weighted_mean_across_bins``, or
        :func:`compute_inside_outside_density_ratio`). Genes not present
        here, or with a NaN value, are skipped.
    annotate_genes : sequence of str or None
        Gene names to label on the plot (e.g. a curated marker gene list).
    cmap, vmin, vmax : passed to ``plt.scatter``; vmin/vmax default to the
        min/max of ``values``.
    cbar_label : str
        Color bar label.
    ax : matplotlib.axes.Axes or None
        Draw into an existing axes instead of creating a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    genes_in = [g for g in coords.index if g in values and not pd.isna(values[g])]
    if not genes_in:
        raise ValueError("no genes in `coords` have a non-NaN value in `values`")

    pc1 = np.array([coords.loc[g, "PC1"] for g in genes_in])
    pc2 = np.array([coords.loc[g, "PC2"] for g in genes_in])
    val = np.array([values[g] for g in genes_in], dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sca = ax.scatter(
        pc1, pc2, c=val, cmap=cmap, s=point_size, alpha=0.85, linewidths=0,
        vmin=val.min() if vmin is None else vmin,
        vmax=val.max() if vmax is None else vmax,
    )
    cbar = plt.colorbar(sca, ax=ax, pad=0.02)
    cbar.set_label(cbar_label, fontsize=cbar_fontsize)
    cbar.ax.tick_params(labelsize=max(tick_fontsize - 5, 8))

    if annotate_genes:
        annotate_set = set(annotate_genes)
        texts = [
            ax.text(pc1[i], pc2[i], g, fontsize=annotate_fontsize)
            for i, g in enumerate(genes_in) if g in annotate_set
        ]
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
            only_move={"points": "xy", "text": "xy"},
            force_text=1.5, force_points=1.0,
            expand_text=(1.2, 1.2), expand_points=(1.2, 1.2),
            lim=200, precision=0.01, autoalign="y",
        )

    ax.set_xlabel("PCoA1", fontsize=label_fontsize)
    ax.set_ylabel("PCoA2", fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)

    return ax


def plot_pcoa_groups(
    coords: pd.DataFrame,
    groups: Sequence[Tuple[Optional[str], Sequence[str], str, float, bool]],
    background_genes: Optional[Sequence[str]] = None,
    background_color: str = "gray",
    background_alpha: float = 0.3,
    background_size: float = 150,
    annotate_fontsize: float = 25,
    figsize: Tuple[float, float] = (20, 15),
    legend_fontsize: float = 23,
    label_fontsize: float = 30,
    tick_fontsize: float = 25,
    ax=None,
):
    """
    PCoA scatter with named, colored, optionally-labeled gene groups drawn
    on top of a gray background of "everything else" — e.g. curated marker
    gene sets per cell type.

    Parameters
    ----------
    coords : pd.DataFrame
        Indexed by gene, with "PC1"/"PC2" columns.
    groups : list of (label, genes, color, size, add_text)
        One entry per group: legend label (``None`` = no legend entry),
        gene names, marker color, marker size, and whether to text-label
        each point. E.g.::

            [("T cell", ["CD3D", "CD3E", "CD8A"], "#1f77b4", 200, True),
             (None,     ["TP53", "EGFR"],         "black",   200, True)]

    background_genes : sequence of str or None
        Genes to draw in ``background_color`` behind the groups (e.g.
        ``set(coords.index) - {g for _, gs, *_ in groups for g in gs}``).
        Pass None to skip drawing a background.
    ax : matplotlib.axes.Axes or None
        Draw into an existing axes instead of creating a new figure.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if background_genes:
        for g in background_genes:
            if g in coords.index:
                ax.scatter(coords.loc[g, "PC1"], coords.loc[g, "PC2"],
                           color=background_color, alpha=background_alpha, s=background_size)

    texts = []
    for label, gene_list, color, size, add_text in groups:
        for g in gene_list:
            if g not in coords.index:
                continue
            if label is None:
                ax.scatter(coords.loc[g, "PC1"], coords.loc[g, "PC2"], color=color, s=size)
            else:
                ax.scatter(coords.loc[g, "PC1"], coords.loc[g, "PC2"], color=color, label=label, s=size)
            if add_text:
                texts.append(ax.text(coords.loc[g, "PC1"], coords.loc[g, "PC2"], g, fontsize=annotate_fontsize))

    ax.set_xlabel("PCoA1", fontsize=label_fontsize)
    ax.set_ylabel("PCoA2", fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="best", prop={"size": legend_fontsize}, ncol=1)

    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
        only_move={"points": "xy", "text": "xy"},
        force_text=1.5, force_points=1.0,
        expand_text=(1.2, 1.2), expand_points=(1.2, 1.2),
        lim=200, precision=0.01, autoalign="y",
    )

    return ax


def _soft_assign_noise(raw_labels: np.ndarray, clusterer) -> np.ndarray:
    """
    Reassign HDBSCAN noise points (label -1) to their most likely cluster,
    via ``hdbscan.all_points_membership_vectors``.

    Handles two degenerate cases that method chokes on directly: no real
    clusters found at all (falls back to a single cluster, label 0 — the
    only sensible grouping when there's nothing to soft-assign against),
    and exactly one real cluster found (its membership vector is 1D, not
    the usual (n_points, n_clusters) — every noise point trivially belongs
    to that one cluster).
    """
    import hdbscan

    raw_labels = raw_labels.copy()
    if not (raw_labels == -1).any():
        return raw_labels

    real_cluster_ids = sorted(set(raw_labels) - {-1})
    if len(real_cluster_ids) == 0:
        raw_labels[:] = 0
        return raw_labels
    if len(real_cluster_ids) == 1:
        raw_labels[raw_labels == -1] = real_cluster_ids[0]
        return raw_labels

    soft = np.asarray(hdbscan.all_points_membership_vectors(clusterer))
    noise_idx = np.where(raw_labels == -1)[0]
    raw_labels[noise_idx] = soft[noise_idx].argmax(axis=1)
    return raw_labels


def cluster_genes_by_pcoa(
    coords: pd.DataFrame,
    extra_axes: Mapping[str, Tuple[Mapping[str, float], float]] = {},
    genes: Optional[Sequence[str]] = None,
    min_cluster_size: int = 12,
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
) -> Tuple[np.ndarray, Dict[int, Dict]]:
    """
    Cluster genes by (PC1, PC2, *extra_axes) using HDBSCAN, after z-scoring
    every dimension.

    HDBSCAN noise points are soft-assigned to their nearest cluster (via
    ``hdbscan.all_points_membership_vectors``) rather than left as an
    unclustered "-1" group.

    Parameters
    ----------
    coords : pd.DataFrame
        Indexed by gene, with "PC1"/"PC2" columns.
    extra_axes : dict[str, (dict[str, float] or pd.Series, float)]
        Additional per-gene scalar dimensions to cluster on, beyond PCoA —
        e.g. uniformity and/or inside/outside density ratio. Maps an axis
        name (used as the key for that axis's per-cluster mean in the
        returned summary, and to restrict ``genes`` by default) to a
        ``(values, weight)`` pair::

            {"mean_wu": (wu_score, 1.0), "mean_A": (density_ratio, 0.5)}

        Each axis is z-scored *first* (independently, to zero mean / unit
        variance) and *then* multiplied by its weight — 1.0 = that axis
        carries about as much influence on the clustering distance as one
        PCoA dimension alone (PC1 and PC2 combined therefore carry roughly
        twice the influence of a single weight=1.0 axis); smaller values
        let PCoA proximity dominate more, 0.0 = no influence at all.
        (Weighting *before* z-scoring, the naive approach, doesn't work —
        for any weight > 0 it cancels out exactly once z-scored, since
        z-score is scale-invariant.)
        Pass ``{}`` (the default) for **PCoA-only clustering**. Use
        :func:`select_pcoa_cluster_params` to choose these weights (and
        the HDBSCAN settings below) by DBCV instead of eyeballing where
        specific genes land.
    genes : sequence of str or None
        Which genes to cluster. Defaults to genes present in ``coords``
        and in every axis of ``extra_axes`` (with a non-NaN value) — e.g.
        pass a gene list here to cluster only genes *not* already assigned
        to an earlier pass.
    min_cluster_size : int
        Passed to ``hdbscan.HDBSCAN``.
    cluster_selection_method : str
        Passed to ``hdbscan.HDBSCAN``. ``"eom"`` (default) favors fewer,
        more stable clusters; ``"leaf"`` selects the condensed tree's leaf
        clusters directly, which is typically the more effective way to get
        *more, finer* clusters — more so than shrinking
        ``min_cluster_size`` alone, which can leave the result almost
        unchanged under ``"eom"``. ``"leaf"`` can easily over-split, though
        — if so, raise ``min_cluster_size`` and/or
        ``cluster_selection_epsilon`` rather than switching back to
        ``"eom"`` (which tends to jump straight back to very few clusters).
    cluster_selection_epsilon : float
        Passed to ``hdbscan.HDBSCAN``. Merges any clusters separated by
        less than this distance (in the standardized feature space) — a
        straightforward knob to pull ``"leaf"`` back from over-splitting
        without losing its finer granularity. 0.0 (default) = no merging.

    Returns
    -------
    labels : np.ndarray
        Cluster id per gene in ``genes`` order (0-indexed, renumbered by
        ascending mean PC1; no -1 noise ids remain).
    cluster_summary : dict[int, dict]
        Per cluster: "n", "pc1", "pc2", one mean-per-axis entry keyed by
        each name in ``extra_axes``, and "genes".
    """
    import hdbscan
    from sklearn.preprocessing import StandardScaler

    if genes is None:
        genes = [
            g for g in coords.index
            if all(g in vals and not pd.isna(vals[g]) for vals, _ in extra_axes.values())
        ]
    genes = list(genes)

    pc1 = np.array([coords.loc[g, "PC1"] for g in genes])
    pc2 = np.array([coords.loc[g, "PC2"] for g in genes])

    # Standardize each dimension independently first, *then* apply the
    # weight — weighting before z-scoring is a no-op for weight > 0 (it
    # cancels out exactly once scaled to unit variance).
    pcoa_sc = StandardScaler().fit_transform(np.column_stack([pc1, pc2]))
    feat_cols = [pcoa_sc[:, 0], pcoa_sc[:, 1]]

    axis_arrays: Dict[str, np.ndarray] = {}
    for name, (vals, weight) in extra_axes.items():
        arr = np.array([vals[g] for g in genes], dtype=float)
        axis_arrays[name] = arr
        arr_sc = StandardScaler().fit_transform(arr.reshape(-1, 1)).ravel()
        feat_cols.append(arr_sc * weight)

    feat_sc = np.column_stack(feat_cols)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=1,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True,
    )
    raw_labels = clusterer.fit_predict(feat_sc).copy()
    raw_labels = _soft_assign_noise(raw_labels, clusterer)

    unique_cids = sorted(set(raw_labels))
    order = sorted(unique_cids, key=lambda c: pc1[raw_labels == c].mean())
    remap = {cid: new_i for new_i, cid in enumerate(order)}
    labels = np.array([remap[c] for c in raw_labels])

    cluster_summary: Dict[int, Dict] = {}
    for new_i in range(len(order)):
        mask = labels == new_i
        info = {
            "n": int(mask.sum()),
            "pc1": float(pc1[mask].mean()),
            "pc2": float(pc2[mask].mean()),
            "genes": [genes[i] for i in range(len(genes)) if mask[i]],
        }
        for name, arr in axis_arrays.items():
            info[name] = float(arr[mask].mean())
        cluster_summary[new_i] = info

    return labels, cluster_summary


def select_pcoa_cluster_params(
    coords: pd.DataFrame,
    extra_axes: Mapping[str, Tuple[Mapping[str, float], Sequence[float]]] = {},
    genes: Optional[Sequence[str]] = None,
    min_cluster_sizes: Sequence[int] = (5, 8, 10, 12, 15, 20),
    cluster_selection_methods: Sequence[str] = ("eom", "leaf"),
    cluster_selection_epsilons: Sequence[float] = (0.0,),
) -> pd.DataFrame:
    """
    Sweep HDBSCAN settings — granularity (``min_cluster_size``,
    ``cluster_selection_method``, ``cluster_selection_epsilon``) and, if
    given, how much weight each of ``extra_axes`` (e.g. uniformity,
    inside/outside density ratio) should carry alongside PCoA — and score
    every combination by DBCV (density-based cluster validity index: how
    dense/well-separated the resulting clusters are).

    Use this to pick all of these settings from the feature space's own
    structure — e.g. the combination with the highest ``dbcv`` — rather
    than by eye (e.g. tuning weights/granularity until some particular
    gene lands in "the right" cluster, which biases the result toward
    what you already expected to find). Feed the chosen row's settings
    into :func:`cluster_genes_by_pcoa`.

    Parameters
    ----------
    coords : pd.DataFrame
        Indexed by gene, with "PC1"/"PC2" columns.
    extra_axes : dict[str, (dict[str, float] or pd.Series, sequence[float])]
        Additional per-gene scalar dimensions to include, each with a grid
        of candidate weights to try — e.g.::

            {"mean_wu": (wu_score, [0.25, 0.5, 1.0]),
             "mean_A":  (density_ratio, [0.25, 0.5, 1.0])}

        Pass ``{}`` (the default) to sweep PCoA-only settings, matching
        :func:`cluster_genes_by_pcoa` with ``extra_axes={}``. The sweep
        tries every combination of weights across all axes (Cartesian
        product), so keep each grid short — the number of HDBSCAN fits is
        ``len(cluster_selection_methods) * len(min_cluster_sizes) *
        len(cluster_selection_epsilons) * prod(len(grid) for grid in
        extra_axes.values())``.
    genes : sequence of str or None
        Which genes to cluster. Defaults to genes present in ``coords``
        and in every axis of ``extra_axes`` (with a non-NaN value).
    min_cluster_sizes, cluster_selection_methods, cluster_selection_epsilons : sequence
        Grids of HDBSCAN settings to try, in the same sense as
        :func:`cluster_genes_by_pcoa`'s equivalently-named parameters.

    Returns
    -------
    pd.DataFrame
        One row per settings combination, with a weight column per
        ``extra_axes`` entry (named ``"{axis_name}_weight"``),
        "n_clusters", "n_noise_before_soft_assign" (HDBSCAN-native noise
        points before the nearest-cluster soft-assignment
        :func:`cluster_genes_by_pcoa` applies), and "dbcv" (higher = more
        valid; NaN where a combination degenerates to a single cluster and
        DBCV isn't defined). Sorted by "dbcv" descending.
    """
    import hdbscan
    from hdbscan.validity import validity_index
    from itertools import product
    from sklearn.preprocessing import StandardScaler

    if genes is None:
        genes = [
            g for g in coords.index
            if all(g in vals and not pd.isna(vals[g]) for vals, _ in extra_axes.values())
        ]
    genes = list(genes)

    pc1 = np.array([coords.loc[g, "PC1"] for g in genes])
    pc2 = np.array([coords.loc[g, "PC2"] for g in genes])

    # Standardize PCoA and every extra axis independently, once, up front
    # (identical across all combos below) — then apply each combo's
    # weights afterward. Weighting *before* z-scoring would be a no-op for
    # weight > 0 (it cancels out exactly once scaled to unit variance).
    pcoa_sc = StandardScaler().fit_transform(np.column_stack([pc1, pc2]))
    axis_names = list(extra_axes.keys())
    axis_values_sc = {}
    for name, (vals, _) in extra_axes.items():
        arr = np.array([vals[g] for g in genes], dtype=float)
        axis_values_sc[name] = StandardScaler().fit_transform(arr.reshape(-1, 1)).ravel()
    weight_grids = [extra_axes[name][1] for name in axis_names]

    records = []
    for method in cluster_selection_methods:
        for min_cluster_size in min_cluster_sizes:
            for epsilon in cluster_selection_epsilons:
                for weight_combo in product(*weight_grids) if weight_grids else [()]:
                    feat_cols = [pcoa_sc[:, 0], pcoa_sc[:, 1]] + [
                        axis_values_sc[name] * weight for name, weight in zip(axis_names, weight_combo)
                    ]
                    feat_sc = np.column_stack(feat_cols)

                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size, min_samples=1,
                        cluster_selection_method=method,
                        cluster_selection_epsilon=epsilon,
                        prediction_data=True,
                    )
                    raw_labels = clusterer.fit_predict(feat_sc).copy()
                    n_noise = int((raw_labels == -1).sum())
                    raw_labels = _soft_assign_noise(raw_labels, clusterer)

                    n_clusters = len(set(raw_labels))
                    try:
                        dbcv = float(validity_index(feat_sc, raw_labels))
                    except (ValueError, ZeroDivisionError):
                        dbcv = np.nan

                    record = {
                        "cluster_selection_method": method,
                        "min_cluster_size": min_cluster_size,
                        "cluster_selection_epsilon": epsilon,
                    }
                    for name, weight in zip(axis_names, weight_combo):
                        record[f"{name}_weight"] = weight
                    record["n_clusters"] = n_clusters
                    record["n_noise_before_soft_assign"] = n_noise
                    record["dbcv"] = dbcv
                    records.append(record)

    return (
        pd.DataFrame(records)
        .sort_values("dbcv", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def combine_cluster_summaries(
    *labeled_summaries: Tuple[str, Mapping[int, Dict]],
) -> Dict[int, Dict]:
    """
    Merge cluster summaries from multiple clustering passes (e.g.
    :func:`cluster_genes_by_pcoa` called once per pass) into one combined,
    sequential 1-indexed numbering.

    Parameters
    ----------
    *labeled_summaries : (str, dict[int, dict])
        One (pass_label, cluster_summary) pair per pass, e.g.
        ``("internal", internal_summary), ("remaining", remaining_summary)``.

    Returns
    -------
    dict[int, dict]
        1-indexed combined cluster id -> ``{**original cluster info,
        "source": pass_label, "source_id": original cluster id}``.
    """
    combined: Dict[int, Dict] = {}
    num = 1
    for pass_label, summary in labeled_summaries:
        for cid, info in summary.items():
            combined[num] = {**info, "source": pass_label, "source_id": cid}
            num += 1
    return combined


def select_cluster_annotation_genes(
    cluster_summary: Mapping[Any, Dict],
    coords: pd.DataFrame,
    preset_genes: Optional[Sequence[str]] = None,
    min_annotations: int = 3,
) -> Dict[Any, set]:
    """
    Pick which genes to text-label per cluster on a PCoA plot: all of a
    cluster's genes that are in ``preset_genes`` (e.g. curated marker
    genes), topped up — if fewer than ``min_annotations`` — with the genes
    closest to the cluster's PCoA centroid.

    Parameters
    ----------
    cluster_summary : dict[Any, dict]
        From :func:`cluster_genes_by_pcoa` or :func:`combine_cluster_summaries`;
        each value must have a "genes" list.
    coords : pd.DataFrame
        Indexed by gene, with "PC1"/"PC2" columns.
    preset_genes : sequence of str or None
        Genes to always prefer labeling (e.g. curated marker genes).
    min_annotations : int
        Minimum number of labeled genes per cluster (best-effort — skipped
        if the cluster has fewer genes than this).

    Returns
    -------
    dict[Any, set[str]]
        cluster id -> set of gene names to label.
    """
    preset = set(preset_genes) if preset_genes else set()
    result: Dict[Any, set] = {}
    for cid, info in cluster_summary.items():
        genes_c = info["genes"]
        from_preset = [g for g in genes_c if g in preset]
        n_need = min_annotations - len(from_preset)
        if n_need > 0 and genes_c:
            cx = float(np.mean([coords.loc[g, "PC1"] for g in genes_c]))
            cy = float(np.mean([coords.loc[g, "PC2"] for g in genes_c]))
            rest = [g for g in genes_c if g not in preset]
            rest_sorted = sorted(
                rest, key=lambda g: np.hypot(coords.loc[g, "PC1"] - cx, coords.loc[g, "PC2"] - cy)
            )
            result[cid] = set(from_preset) | set(rest_sorted[:n_need])
        else:
            result[cid] = set(from_preset)
    return result


def compute_presence_fraction_map(
    subset_grid,
    genes: Sequence[str],
    bin_size_um: float = 20,
    x_col: str = "imagecol",
    y_col: str = "imagerow",
) -> np.ndarray:
    """
    For a set of genes, the fraction of them expressed (count > 0 in at
    least one cell) per spatial bin, over ``subset_grid``.

    Parameters
    ----------
    subset_grid : AnnData
        e.g. ``gene_pipe.subset_grid``.
    genes : sequence of str
        Genes to average presence over (e.g. one gene cluster's members).
        Genes not in ``subset_grid.var_names`` are skipped.
    bin_size_um : float
        Spatial bin size, in the same units as ``x_col``/``y_col``.
    x_col, y_col : str
        Columns of ``subset_grid.obs`` holding spatial coordinates.

    Returns
    -------
    np.ndarray, shape (n_y_bins, n_x_bins)
        Fraction of ``genes`` expressed in each bin (all zero if ``genes``
        is empty after filtering).
    """
    var_names = list(subset_grid.var_names)
    genes_present = [g for g in genes if g in var_names]

    x_vals = subset_grid.obs[x_col].to_numpy(dtype=float)
    y_vals = subset_grid.obs[y_col].to_numpy(dtype=float)
    ix = np.floor((x_vals - x_vals.min()) / bin_size_um).astype(int)
    iy = np.floor((y_vals - y_vals.min()) / bin_size_um).astype(int)
    n_x = ix.max() + 1
    n_y = iy.max() + 1

    if not genes_present:
        return np.zeros((n_y, n_x))

    presence_sum = np.zeros((n_y, n_x))
    for gene in genes_present:
        g_idx = var_names.index(gene)
        expr = subset_grid.X[:, g_idx]
        expr = expr.toarray().ravel() if hasattr(expr, "toarray") else np.asarray(expr).ravel()
        expressed = expr > 0
        pmap = np.zeros((n_y, n_x), dtype=bool)
        pmap[iy[expressed], ix[expressed]] = True
        presence_sum += pmap

    return presence_sum / len(genes_present)


def plot_presence_fraction_grid(
    maps: Mapping[str, np.ndarray],
    colors: Optional[Mapping[str, str]] = None,
    max_cols: int = 4,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    cbar_label: str = "Fraction of genes expressed",
    panel_size: Tuple[float, float] = (4, 4.2),
    suptitle: Optional[str] = None,
):
    """
    Grid of presence-fraction spatial maps, one panel per cluster.

    Parameters
    ----------
    maps : dict[str, np.ndarray]
        Panel title -> 2D presence-fraction map (e.g. from
        :func:`compute_presence_fraction_map`), in display order.
    colors : dict[str, str] or None
        Panel title -> title color (e.g. to match a cluster's scatter
        color in :func:`cluster_genes_by_pcoa`'s plots). Defaults to black.
    max_cols : int
        Maximum panels per row.
    vmax : float or None
        Shared color-scale maximum. Defaults to the max across all maps
        (the "global" mode from the original notebook); pass each map
        already normalized yourself for "per-panel" scaling.
    suptitle : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    import math

    labels = list(maps.keys())
    n = len(labels)
    n_cols = min(n, max_cols)
    n_rows = math.ceil(n / n_cols)
    if vmax is None:
        vmax = max(m.max() for m in maps.values())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows), dpi=130,
        sharex=True, sharey=True, squeeze=False,
    )

    im = None
    for i, label in enumerate(labels):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        im = ax.imshow(maps[label], cmap=cmap, aspect="equal", origin="upper",
                        vmin=0, vmax=vmax, interpolation="nearest")
        color = (colors or {}).get(label, "black")
        ax.set_title(label, fontsize=12, fontweight="bold", color=color)
        ax.axis("off")

    for i in range(n, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes.flat[n - 1], fraction=0.046, pad=0.04, label=cbar_label)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()

    return fig


def run_go_enrichment_per_cluster(
    cluster_genes: Mapping[Any, Sequence[str]],
    background_genes: Optional[Sequence[str]] = None,
    gene_sets: Sequence[str] = ("GO_Biological_Process_2023", "GO_Molecular_Function_2023"),
    padj_thresh: float = 0.05,
    min_genes: int = 3,
    organism: str = "human",
) -> Dict[Any, pd.DataFrame]:
    """
    Over-representation analysis (ORA, hypergeometric test) per gene
    cluster, against a custom background gene set.

    Each named library in ``gene_sets`` is downloaded once (via
    ``gseapy.get_library``) and the test itself then runs fully offline
    (``gseapy.enrich``, local hypergeometric test) — this still needs
    network access for that one-time download per library, but avoids
    Enrichr's online "custom background" endpoint
    (``gseapy.enrichr(..., background=...)``, which goes through
    Enrichr's Speedrichr service). That endpoint has been unreliable in
    practice (e.g. intermittent HTTP 500s from
    ``speedrichr/api/backgroundenrich``); this path sidesteps it entirely.
    Requires the optional ``gseapy`` package.

    Parameters
    ----------
    cluster_genes : dict[Any, list[str]]
        Cluster id/label -> its gene list (e.g. a
        :func:`cluster_genes_by_pcoa` summary's "genes" field per cluster).
    background_genes : sequence of str or None
        Background/universe gene set for the ORA. **For a targeted panel
        (e.g. a ~300-gene Xenium/CosMx panel), pass the actual measured
        gene set here** (e.g. all genes considered in the PCoA) — not
        ``None``. The hypergeometric test assumes the background is the
        pool your gene list could have been drawn from; a targeted panel
        is deliberately curated around specific biology (immune, cancer,
        stromal markers, ...), so testing against the whole genome makes
        genes from that curated pool look "enriched" for the panel's own
        design bias rather than anything specific to your gene list. The
        cost is lower power — with only a few hundred background genes,
        real signals can fail to reach significance — but that is the
        statistically valid trade-off for panel data, not a reason to
        widen the background.
    gene_sets : sequence of str
        Enrichr gene-set library names.
    padj_thresh : float
        Keep only terms with adjusted p-value below this.
    min_genes : int
        Skip clusters with fewer genes than this.
    organism : str
        Passed to ``gseapy.get_library`` (e.g. "human", "mouse").

    Returns
    -------
    dict[Any, pd.DataFrame]
        cluster id -> significant terms, sorted by adjusted p-value.
        Clusters with fewer than ``min_genes`` genes are omitted; clusters
        with zero significant terms get an empty DataFrame.
    """
    import gseapy as gp

    # Fetch each library once (shared across all clusters), not once per
    # cluster — also cuts down the number of network calls considerably.
    library_dicts = {name: gp.get_library(name=name, organism=organism) for name in gene_sets}

    results: Dict[Any, pd.DataFrame] = {}
    for cid, genes_c in cluster_genes.items():
        if len(genes_c) < min_genes:
            continue
        per_library_dfs = []
        for lib_name, lib_dict in library_dicts.items():
            kwargs = dict(gene_list=list(genes_c), gene_sets=lib_dict, outdir=None, verbose=False)
            if background_genes is not None:
                kwargs["background"] = list(background_genes)
            enr = gp.enrich(**kwargs)
            df = enr.results
            df["Gene_set"] = lib_name
            per_library_dfs.append(df)
        df = pd.concat(per_library_dfs, ignore_index=True)
        df = df[df["Adjusted P-value"] < padj_thresh].copy()
        df = df.sort_values("Adjusted P-value")
        results[cid] = df
    return results


def plot_go_enrichment_per_cluster(
    results: Mapping[Any, pd.DataFrame],
    top_n: int = 8,
    padj_thresh: float = 0.05,
    colors: Optional[Mapping[Any, str]] = None,
    n_cols: int = 2,
    panel_size: Tuple[float, float] = (8, 5),
    term_fontsize: float = 14,
    axis_fontsize: float = 15,
    title_fontsize: float = 18,
    suptitle_fontsize: float = 18,
    hspace: float = 0.6,
    suptitle: Optional[str] = None,
):
    """
    Horizontal bar chart of top enriched GO terms per cluster, from
    :func:`run_go_enrichment_per_cluster`'s output.

    Parameters
    ----------
    results : dict[Any, pd.DataFrame]
        Output of :func:`run_go_enrichment_per_cluster`.
    top_n : int
        Number of top terms (by adjusted p-value) per panel.
    padj_thresh : float
        Reference line drawn at ``-log10(padj_thresh)``.
    colors : dict[Any, str] or None
        Panel color per cluster id. Defaults to a built-in qualitative
        palette.
    n_cols : int
        Panels per row.
    term_fontsize, axis_fontsize, title_fontsize, suptitle_fontsize : float
        Font sizes for the term labels, x-axis label, per-panel title, and
        figure suptitle, respectively.
    hspace : float
        Vertical spacing between panel rows (fraction of panel height),
        passed to ``fig.subplots_adjust``. Increase if a row's x-axis
        label overlaps the next row's title (common with ``n_cols=1`` and
        large ``title_fontsize``/``axis_fontsize``, since
        ``tight_layout()`` alone doesn't reserve enough room for big
        fonts).

    Returns
    -------
    matplotlib.figure.Figure or None
        None if no cluster has any significant term.
    """
    import math

    nonempty = [cid for cid, df in results.items() if not df.empty]
    if not nonempty:
        return None

    default_colors = plt.cm.tab20.colors
    if colors is None:
        colors = {cid: default_colors[i % len(default_colors)] for i, cid in enumerate(nonempty)}

    ncols = min(len(nonempty), n_cols)
    nrows = math.ceil(len(nonempty) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                              dpi=100, squeeze=False)

    for i, cid in enumerate(nonempty):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        df_top = results[cid].head(top_n)
        terms = df_top["Term"].str.split(r" \(GO:").str[0]
        vals = df_top["Adjusted P-value"].apply(lambda x: -np.log10(x))
        color = colors[cid]
        ax.barh(range(len(terms)), vals.values[::-1], color=color, alpha=0.85)
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms.values[::-1], fontsize=term_fontsize)
        ax.set_xlabel("-log10(adjusted p-value)", fontsize=axis_fontsize)
        ax.tick_params(axis="x", labelsize=term_fontsize)
        ax.axvline(-np.log10(padj_thresh), color="gray", ls="--", lw=0.8)
        ax.set_title(f"{cid}", fontsize=title_fontsize, fontweight="bold", color=color)

    for i in range(len(nonempty), nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(hspace=hspace)

    return fig
