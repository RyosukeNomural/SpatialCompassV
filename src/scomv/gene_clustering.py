"""Gene-level workflow after ``SCOMVPipeline.run``: PCoA -> clustering -> cluster mean maps -> enrichment.

Each step is one function (the gene_analysis tutorial, §2-§6)::

    from scomv.gene_clustering import (compute_gene_features, cluster_genes_two_stage, plot_gene_clusters,
                                       cluster_mean_maps, plot_cluster_mean_polar, plot_cluster_presence,
                                       cluster_enrichment)
    feats = compute_gene_features(gene_pipe)                 # PC1, PC2, uniformity, A per gene
    gc = cluster_genes_two_stage(feats)                      # internal (A >= 1.2) by PCoA + A, the rest by PCoA + U
    plot_gene_clusters(gc)
    maps = cluster_mean_maps(gene_pipe, gc)                  # mean polar histogram + presence fraction per cluster
    plot_cluster_mean_polar(maps); plot_cluster_presence(maps)
    go = cluster_enrichment(gc)                              # GO over-representation per cluster (network)

``gc.labels`` (gene -> "C1", ...; "-" = unclassified) can be passed to ``scomv.pgi.plot_pgi_ranking``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .gene_module import (cluster_genes_by_pcoa, combine_cluster_summaries, compute_gene_uniformity,
                          compute_inside_outside_density_ratio, compute_presence_fraction_map,
                          plot_go_enrichment_per_cluster, run_go_enrichment_per_cluster,
                          select_cluster_annotation_genes)

DEFAULT_MARKERS = {"CD3D", "CD3E", "CD4", "CD247", "CD8A", "GZMA", "TRAC", "CD19", "BANK1", "MS4A1", "TCL1A", "CD79A",
                   "CD68", "CD80", "CD86", "CD163", "CDH1", "EPCAM", "KRT8", "CD93", "PECAM1", "VWF", "POSTN", "LUM",
                   "MMP2"}


def default_distance_bins(radius_bins, d_from: float = 10, d_to: float = 100, step: float = 10):
    """{label: row index} of the d_from..d_to µm rings and their weights sqrt(15 / ring centre) (tutorial)."""
    lo = np.asarray(radius_bins, float)[:-1]
    bins, weights = {}, {}
    for d in np.arange(d_from, d_to, step):
        idx = np.flatnonzero(np.isclose(lo, d))
        if len(idx):
            lab = f"{d:g}-{d + step:g}µm"
            bins[lab] = int(idx[0])
            weights[lab] = math.sqrt(15 / (d + step / 2))
    return bins, weights


def compute_gene_features(pipe, *, dist_bins: Optional[Mapping[str, int]] = None,
                          bin_weights: Optional[Mapping[str, float]] = None, n_ref: float = 24,
                          n_pc: int = 2) -> pd.DataFrame:
    """Per gene: PCoA coordinates (``PC1``.. ``PC{n_pc}``), distance-weighted ``uniformity`` (10-100 µm) and the
    inside/outside density ratio ``A``. ``pipe``: a run ``SCOMVPipeline``."""
    from .tumor_solid import weighted_mean_across_bins
    if pipe.coords is None:
        raise ValueError("run the pipeline first (pipe.run(roi=...))")
    if dist_bins is None:
        dist_bins, w = default_distance_bins(pipe.radius_bins)
        bin_weights = w if bin_weights is None else bin_weights
    if bin_weights is None:
        bin_weights = {k: 1.0 for k in dist_bins}
    wu = compute_gene_uniformity(pipe.subset_grid, pipe.selected_genes, pipe.polar_counts_list, dist_bins, n_ref=n_ref)
    u = weighted_mean_across_bins(wu, bin_weights)
    a = compute_inside_outside_density_ratio(pipe.selected_genes, pipe.polar_counts_list, pipe.radius_bins,
                                             n_inside=len(pipe.inside_points), n_total=len(pipe.xy_list))
    out = pipe.coords[[f"PC{i}" for i in range(1, n_pc + 1)]].copy()
    out["uniformity"] = pd.Series(u).reindex(out.index)
    out["A"] = a.reindex(out.index)
    return out


@dataclass
class GeneClusters:
    clusters: Dict[int, Dict]           # 1-based id -> info ("genes", "n", "source", "mean_A" / "mean_wu", ...)
    features: pd.DataFrame
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> pd.Series:
        """gene -> "C<k>" ("-" = unclassified), for every gene of ``features``."""
        lab = pd.Series("-", index=self.features.index, name="cluster")
        for k, info in self.clusters.items():
            lab[info["genes"]] = f"C{k}"
        return lab

    @property
    def gene_sets(self) -> Dict[str, list]:
        return {f"C{k}": list(info["genes"]) for k, info in self.clusters.items()}

    @property
    def colors(self) -> Dict[str, Any]:
        return {f"C{k}": plt.cm.tab20.colors[i % 20] for i, k in enumerate(self.clusters)}

    def summary(self) -> pd.DataFrame:
        rows = []
        for k, info in self.clusters.items():
            g = info["genes"]
            rows.append({"cluster": f"C{k}", "source": info["source"], "n": info["n"],
                         "mean_A": self.features.loc[g, "A"].mean(),
                         "mean_uniformity": self.features.loc[g, "uniformity"].mean(),
                         "genes": ", ".join(sorted(g)[:12]) + (" ..." if len(g) > 12 else "")})
        return pd.DataFrame(rows).set_index("cluster")


def cluster_genes_two_stage(
    features: pd.DataFrame,
    *,
    a_thresh: float = 1.2,
    a_weight: float = 0.3,
    min_cluster_size_internal: int = 12,
    internal_method: str = "leaf",
    internal_epsilon: float = 0.0,
    u_weight: float = 0.5,
    min_cluster_size_remaining: int = 10,
    remaining_method: str = "leaf",
    remaining_epsilon: float = 0.1,
    keep_noise: bool = False,
    verbose: bool = True,
) -> GeneClusters:
    """Two-stage HDBSCAN of the tutorial.

    1. Genes with ``A >= a_thresh`` (decided per gene) are "internal" and clustered by PC1, PC2 and A
       (weight ``a_weight``).
    2. The other genes are clustered by PC1, PC2 and uniformity (weight ``u_weight``).
    Clusters are numbered C1.. (internal first). ``features``: output of :func:`compute_gene_features`.
    """
    f = features.dropna(subset=["PC1", "PC2", "A"])
    coords = f[["PC1", "PC2"]]
    internal = [g for g in f.index if f.loc[g, "A"] >= a_thresh]
    remaining = [g for g in f.index if g not in set(internal) and np.isfinite(f.loc[g, "uniformity"])]
    parts = []
    if len(internal) >= max(min_cluster_size_internal, 2):
        _, s_in = cluster_genes_by_pcoa(coords, extra_axes={"mean_A": (f["A"], a_weight)}, genes=internal,
                                        min_cluster_size=min_cluster_size_internal,
                                        cluster_selection_method=internal_method,
                                        cluster_selection_epsilon=internal_epsilon, keep_noise=keep_noise)
        parts.append(("internal", s_in))
    if len(remaining) >= max(min_cluster_size_remaining, 2):
        _, s_rem = cluster_genes_by_pcoa(coords, extra_axes={"mean_wu": (f["uniformity"], u_weight)}, genes=remaining,
                                         min_cluster_size=min_cluster_size_remaining,
                                         cluster_selection_method=remaining_method,
                                         cluster_selection_epsilon=remaining_epsilon, keep_noise=keep_noise)
        parts.append(("remaining", s_rem))
    res = GeneClusters(combine_cluster_summaries(*parts), features,
                       dict(a_thresh=a_thresh, a_weight=a_weight, u_weight=u_weight,
                            min_cluster_size_internal=min_cluster_size_internal,
                            min_cluster_size_remaining=min_cluster_size_remaining))
    if verbose:
        print(f"internal genes (A >= {a_thresh}): {len(internal)}, remaining: {len(remaining)} -> "
              f"{len(res.clusters)} clusters")
    return res


def plot_gene_clusters(gc: GeneClusters, *, sources: Optional[Sequence[str]] = None,
                       marker_genes: Optional[Sequence[str]] = None, min_annotations: int = 5, ax=None,
                       figsize=(18, 16), point_size: float = 200, fontsize: float = 33, label_fontsize: float = 45,
                       tick_fontsize: float = 40, legend_fontsize: float = 31, legend_ncol: int = 4,
                       legend_value: Optional[str] = None, title: Optional[str] = None):
    """PCoA of all genes coloured by cluster (circles: internal, squares: remaining; grey: the rest).

    ``sources``: colour only the clusters of these passes (e.g. ``("internal",)``); the other genes are grey.
    ``legend_value``: add each cluster's mean of this summary key to the legend (e.g. "mean_A", "mean_wu").
    Up to ``min_annotations`` genes per cluster are labelled, preferring ``marker_genes``. Returns the axes.
    """
    markers = DEFAULT_MARKERS if marker_genes is None else set(marker_genes)
    coords = gc.features[["PC1", "PC2"]]
    col = gc.colors
    shown = {k: i for k, i in gc.clusters.items() if sources is None or i["source"] in sources}
    annot = select_cluster_annotation_genes(shown, coords, preset_genes=markers, min_annotations=min_annotations)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
    colored = set(g for info in shown.values() for g in info["genes"])
    rest = [g for g in coords.index if g not in colored]
    ax.scatter(coords.loc[rest, "PC1"], coords.loc[rest, "PC2"], color="lightgray", s=point_size * 0.3, alpha=0.3,
               linewidths=0, zorder=1)
    texts = []
    for k, info in shown.items():
        g = info["genes"]
        lab = f"C{k} n={info['n']}" + (f"  {legend_value}={info[legend_value]:.2f}" if legend_value in info else "")
        ax.scatter(coords.loc[g, "PC1"], coords.loc[g, "PC2"], color=col[f"C{k}"], s=point_size, alpha=0.9,
                   edgecolors="black", linewidths=0.8, zorder=3, marker="o" if info["source"] == "internal" else "s",
                   label=lab)
        texts += [ax.text(coords.loc[x, "PC1"], coords.loc[x, "PC2"], x, fontsize=fontsize, color="black")
                  for x in g if x in annot[k]]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color="gray", lw=0.5),
                    only_move={"points": "xy", "text": "xy"}, force_text=1.5, force_points=1.0,
                    expand_text=(1.2, 1.2), expand_points=(1.2, 1.2), lim=200, precision=0.01, autoalign="y")
    except Exception:
        pass
    ax.set_xlabel("PCoA1", fontsize=label_fontsize)
    ax.set_ylabel("PCoA2", fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    if title:
        ax.set_title(title, fontsize=legend_fontsize)
    ax.figure.tight_layout()
    # legend above the panel (added after tight_layout so the panel keeps its size)
    ax.legend(fontsize=legend_fontsize, markerscale=1.2, ncol=legend_ncol, loc="lower left",
              bbox_to_anchor=(0.0, 1.01 if not title else 1.06), borderaxespad=0.0, columnspacing=1.0,
              handletextpad=0.3)
    return ax


@dataclass
class ClusterMaps:
    polar: Dict[str, np.ndarray]        # cluster -> mean polar histogram of the member genes (radius x angle)
    presence: Dict[str, np.ndarray]     # cluster -> fraction of member genes expressed per spatial bin
    labels: Dict[str, str]              # cluster -> panel title
    colors: Dict[str, Any]
    radius_bins: np.ndarray
    angle_bins_deg: np.ndarray


def cluster_mean_maps(pipe, gc: GeneClusters, *, presence_bin_um: float = 20) -> ClusterMaps:
    """Per cluster: the mean of the members' polar histograms (where the cluster sits around the tumor) and the
    presence-fraction map on the tissue (``compute_presence_fraction_map``)."""
    hist = dict(zip(pipe.selected_genes, pipe.polar_counts_list))
    polar, pres, lab = {}, {}, {}
    for k, info in gc.clusters.items():
        name = f"C{k}"
        polar[name] = np.mean([hist[g] for g in info["genes"] if g in hist], axis=0)
        pres[name] = compute_presence_fraction_map(pipe.subset_grid, info["genes"], bin_size_um=presence_bin_um)
        score = (f"A={info['mean_A']:.2f}" if info["source"] == "internal" else f"U={info['mean_wu']:.2f}")
        lab[name] = f"{name}  n={info['n']}  {score}"
    return ClusterMaps(polar, pres, lab, gc.colors, np.asarray(pipe.radius_bins), np.asarray(pipe.angle_bins_deg))


def plot_cluster_mean_polar(maps: ClusterMaps, *, ncols: int = 4, panel_size: float = 5.5, r_min: float = -100,
                            clim_percentile: float = 99):
    """Mean polar histogram of every cluster (each panel on its own colour scale, capped at the
    ``clim_percentile`` of its cells: the boundary squares, r = 0 with angle 0° by convention, would otherwise
    take the whole scale)."""
    from .polar_plot import plot_polar_heatmap
    keep = maps.radius_bins[:-1] >= r_min
    rb = maps.radius_bins[np.r_[keep, True]]
    names = list(maps.polar)
    ncols = min(ncols, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows * 1.05),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    for ax, n in zip(axes.ravel(), names):
        A = maps.polar[n][keep]
        vmax = float(np.nanpercentile(A, clim_percentile)) or None
        plot_polar_heatmap(A, rb, maps.angle_bins_deg, ax=ax, vmin=0, vmax=vmax,
                           hole_frac=0.15, boundary_kwargs=dict(color="white", lw=2, ls="--"),
                           radius_ticks=[t for t in (-100, 0, 100, 200, 300) if rb[0] <= t <= rb[-1]],
                           cbar_label="mean density", fontsize=12)
        ax.set_title(maps.labels[n], fontsize=15, weight="bold", pad=22)
        ax.text(-0.02, 1.1, "\u25A0", transform=ax.transAxes, fontsize=22, color=maps.colors.get(n, "k"),
                ha="right", va="center")
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_cluster_presence(maps: ClusterMaps, *, ncols: int = 4, per_cluster_scale: bool = True,
                          panel_size=(5.6, 5.8), title_fontsize: float = 38, tick_fontsize: float = 24):
    """Presence fraction (fraction of the cluster's genes detected) per spatial bin, one panel per cluster.

    ``per_cluster_scale``: each panel min-max scaled on its own, colour-bar ticks show that cluster's actual values
    (otherwise one common 0-max scale). Returns the figure.
    """
    names = list(maps.presence)
    ncols = min(ncols, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), dpi=130,
                             sharex=True, sharey=True, squeeze=False)
    ticks = np.linspace(0, 1, 5)
    vmax_all = max(float(m.max()) for m in maps.presence.values())
    for ax, n in zip(axes.ravel(), names):
        m = maps.presence[n]
        lo, hi = (float(m.min()), float(m.max())) if per_cluster_scale else (0.0, vmax_all)
        im = ax.imshow((m - lo) / (hi - lo) if hi > lo else np.zeros_like(m), cmap="viridis", aspect="equal",
                       origin="upper", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(maps.labels[n].split("  ")[0], fontsize=title_fontsize, fontweight="bold",
                     color=maps.colors.get(n, "k"))
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=ticks)
        cbar.ax.set_yticklabels([f"{lo + t * (hi - lo):.1f}" for t in ticks])
        cbar.ax.tick_params(labelsize=tick_fontsize)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def cluster_enrichment(gc: GeneClusters, *, background: Optional[Sequence[str]] = None,
                       gene_sets: Sequence[str] = ("GO_Biological_Process_2023", "GO_Molecular_Function_2023"),
                       plot: bool = True, top_n: int = 5, **plot_kwargs):
    """Over-representation analysis of every cluster (background: all clustered-feature genes = the panel).
    Requires ``gseapy`` and network access for the libraries. Returns {cluster: DataFrame} (and plots it)."""
    bg = list(gc.features.index) if background is None else list(background)
    res = run_go_enrichment_per_cluster(gc.gene_sets, background_genes=bg, gene_sets=list(gene_sets))
    if plot and any(not d.empty for d in res.values()):
        plot_go_enrichment_per_cluster(res, top_n=top_n, colors=gc.colors, **plot_kwargs)
    return res
