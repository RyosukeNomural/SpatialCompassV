"""Group tumor solids by the uniformity of chosen genes and / or cell types around them.

Simplifies the per-solid workflows of the tumor_solid tutorial (cell types) and of the Atera analysis
(``revice/atera_wta/atera_wta_postn_uniformity_pgi.ipynb``: POSTN uniformity -> Gaussian mixture, k by BIC):

1. Features per solid (rows = solid ids):
   - :func:`solid_gene_uniformity` — distance-weighted uniformity (10-100 µm) of any list of genes, optionally
     size-factor corrected (median-of-ratios over the solids' ROIs), as in ``atera_wta_uniformity.ipynb`` §7.
   - :func:`solid_cell_uniformity` — the same for the selected cell types (tumor_solid tutorial §4-6).
   Both tables can be joined (``features = genes.join(cells)``).
2. :func:`cluster_solids` — Gaussian mixture with k chosen by BIC (default) or Ward with a fixed k, on the
   selected columns (z-scored when there are several). Groups are ordered by the mean of ``order_by``.
3. :func:`plot_solid_clusters` — BIC curve, feature distribution per group and a violin plot per feature;
   ``scomv.tumor_solid.plot_cluster_map(labels_img, solid.obs, result.cluster_of_solid)`` maps them on the slide.

Example::

    from scomv.solid_clustering import solid_gene_uniformity, solid_cell_uniformity, cluster_solids
    gu = solid_gene_uniformity(grid, solid, ["POSTN"])
    res = cluster_solids(gu)                       # k by BIC
    cu = solid_cell_uniformity(grid, solid, cell_df, ["CD8+ T cells", "Macrophages / monocytes"])
    res2 = cluster_solids(cu, method="ward", n_clusters=4)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from .signature import polar_weight_matrix


def default_bins(d_from: float = 10, d_to: float = 100, step: float = 10, radius_from: float = -150):
    """Distance bins {label: row} of the radius histogram (edges ``radius_from``.. by ``step``) and weights
    sqrt(15 / ring centre), normalised to 1 for the first ring (tumor_solid tutorial)."""
    bins, w = {}, {}
    for d in np.arange(d_from, d_to, step):
        lab = f"{d:g}-{d + step:g}µm"
        bins[lab] = int(round((d - radius_from) / step))
        w[lab] = math.sqrt((d_from + step / 2) / (d + step / 2))
    return bins, w


def _solid_ids(solid, solid_ids):
    if solid_ids is None:
        return [int(s) for s in solid.obs["solid"]]
    return [int(s) for s in solid_ids]


def solid_gene_uniformity(grid, solid, genes: Sequence[str], *, solid_ids: Optional[Sequence[int]] = None,
                          dist_bins: Optional[Mapping[str, int]] = None, bin_weights: Optional[Mapping[str, float]] = None,
                          n_ref: float = 24, size_factor: bool = True, bin_size: int = 10, margin_bins: int = 15,
                          mask_other_solids: bool = False, return_bins: bool = False, verbose: bool = True):
    """Distance-weighted uniformity of ``genes`` around every solid (solids x genes).

    Same score as ``scomv.tumor_solid.compute_gene_uniformity`` per solid (normalised entropy over the 30° sectors
    x confidence 1 - exp(-n / n_ref)), combined over the distance bins with ``bin_weights``.

    ``size_factor``: scale each solid's expressing-square count by 1/s, s = median over genes of
    (ROI expression / geometric mean across solids) — removes the "more stroma -> higher uniformity for every
    gene" effect (Atera analysis). The factors use all genes expressed in every ROI and are stored in
    ``result.attrs["size_factor"]``. ``return_bins``: also return {bin: solids x genes}.
    """
    from .tumor_solid import compute_solid_min_vectors
    if dist_bins is None:
        dist_bins, w0 = default_bins()
        bin_weights = w0 if bin_weights is None else bin_weights
    if bin_weights is None:
        bin_weights = {k: 1.0 for k in dist_bins}
    rb, ab = np.arange(-150, 310, 10), np.arange(-180, 181, 30)
    var = list(grid.var_names)
    vidx = {g: i for i, g in enumerate(var)}
    missing = [g for g in genes if g not in vidx]
    if missing:
        raise KeyError(f"genes not in the grid: {missing}")
    genes = list(dict.fromkeys(genes))
    sids = _solid_ids(solid, solid_ids)
    hists, n_expr, totals = {}, {}, {}
    for sid in sids:
        sg, mv, _ = compute_solid_min_vectors(grid, solid, sid, bin_size=bin_size, margin_bins=margin_bins,
                                              mask_other_solids=mask_other_solids)
        X = sg.X.tocsr() if sparse.issparse(sg.X) else sparse.csr_matrix(np.asarray(sg.X))
        X = X.astype(np.float64)
        X.data[X.data < 0] = 0
        X.eliminate_zeros()
        M, has_vec = polar_weight_matrix(mv, rb, ab, bin_size, n_rows=sg.n_obs)
        Xg = X[:, [vidx[g] for g in genes]]
        raw = np.asarray((Xg.T @ M).todense())                             # genes x cells
        tot = np.asarray(Xg[has_vec].sum(axis=0)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            hists[sid] = (raw / tot[:, None]).reshape(len(genes), len(rb) - 1, len(ab) - 1)
        n_expr[sid] = np.asarray((Xg > 0).sum(axis=0)).ravel()
        totals[sid] = np.asarray(X[has_vec].sum(axis=0)).ravel() if size_factor else None
    if size_factor:
        T = pd.DataFrame({s: totals[s] for s in sids}, index=var).T
        ok = (T > 0).all()
        if ok.sum() == 0:
            raise ValueError("no gene is expressed in every solid's ROI; use size_factor=False")
        logt = np.log(T.loc[:, ok])
        sf = np.exp((logt - logt.mean()).median(axis=1))
    else:
        sf = pd.Series(1.0, index=sids)
    n_a = len(ab) - 1
    per_bin = {b: {} for b in dist_bins}
    for sid in sids:
        A = hists[sid]
        for b, idx in dist_bins.items():
            s = A[:, idx, :]
            t = s.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                p = s / t[:, None] + 1e-12
                p = p / p.sum(axis=1, keepdims=True)
                H = -(p * np.log(p)).sum(axis=1) / np.log(n_a)
            u = H * (1.0 - np.exp(-(t * n_expr[sid] / sf[sid]) / n_ref))
            u[~(t > 0)] = 0.0
            per_bin[b][sid] = u
    per_bin = {b: pd.DataFrame(v, index=genes).T.rename_axis("solid") for b, v in per_bin.items()}
    wsum = sum(bin_weights.values())
    out = sum(bin_weights[b] * per_bin[b] for b in bin_weights) / wsum
    out.attrs["size_factor"] = sf
    if verbose:
        print(f"uniformity of {len(genes)} gene(s) around {len(sids)} solids"
              + (f" (size factor: median {sf.median():.2f}, range {sf.min():.2f}-{sf.max():.2f})" if size_factor else ""))
    return (out, per_bin) if return_bins else out


def solid_cell_uniformity(grid, solid, cell_df: pd.DataFrame, cell_types: Sequence[str], *,
                          cluster_col: str = "Cluster", solid_ids: Optional[Sequence[int]] = None,
                          dist_bins: Optional[Mapping[str, int]] = None, bin_weights: Optional[Mapping[str, float]] = None,
                          n_ref: float = 12, min_cells: int = 5, bin_size: int = 10, margin_bins: int = 15,
                          mask_other_solids: bool = False) -> pd.DataFrame:
    """Distance-weighted uniformity of the selected cell types around every solid (solids x cell types).

    ``cell_types``: the cell types to use (values of ``cell_df[cluster_col]``); only these are computed.
    """
    from .tumor_solid import compute_per_solid_polar_counts, compute_uniformity, distance_weighted_mean
    cell_types = list(cell_types)
    unknown = set(cell_types) - set(cell_df[cluster_col].unique())
    if unknown:
        raise KeyError(f"cell types not in cell_df[{cluster_col!r}]: {sorted(unknown)}")
    if dist_bins is None:
        dist_bins, w0 = default_bins()
        bin_weights = w0 if bin_weights is None else bin_weights
    if bin_weights is None:
        bin_weights = {k: 1.0 for k in dist_bins}
    res = compute_per_solid_polar_counts(grid, solid, cell_df[cell_df[cluster_col].isin(cell_types)], cell_types,
                                         bin_size=bin_size, margin_bins=margin_bins, min_cells=min_cells,
                                         cluster_col=cluster_col, mask_other_solids=mask_other_solids)
    mats, _ = compute_uniformity(res, cell_types, dist_bins, n_ref=n_ref)
    out = distance_weighted_mean(mats, bin_weights)
    if solid_ids is not None:
        out = out.reindex([int(s) for s in solid_ids]).fillna(0.0)
    return out.rename_axis("solid")


@dataclass
class SolidClusters:
    labels: pd.Series                       # solid -> group index (0 = lowest mean of order_by)
    group_names: list
    features: pd.DataFrame                  # the columns used
    method: str
    k: int
    selection: Optional[pd.DataFrame] = None   # BIC per k (gmm) and bootstrap choice
    model: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def named(self) -> pd.Series:
        return self.labels.map(dict(enumerate(self.group_names)))

    @property
    def cluster_of_solid(self) -> Dict[int, int]:
        """For ``scomv.tumor_solid.plot_cluster_map``."""
        return {int(s): int(c) for s, c in self.labels.items()}

    def summary(self) -> pd.DataFrame:
        g = self.features.groupby(self.named)
        out = g.mean().add_prefix("mean ")
        out.insert(0, "n_solids", g.size())
        return out.reindex(self.group_names)


def cluster_solids(features: pd.DataFrame, columns: Optional[Sequence[str]] = None, *, method: str = "gmm",
                   n_clusters: Optional[int] = None, k_range: Sequence[int] = range(1, 7), log: bool = False,
                   order_by: Optional[str] = None, n_boot: int = 0, random_state: int = 0,
                   verbose: bool = True) -> SolidClusters:
    """Group the solids on the selected feature columns.

    method="gmm": Gaussian mixture; k = ``n_clusters`` or the lowest BIC over ``k_range`` (``n_boot`` > 0 repeats
    the BIC choice on bootstrap resamples). method="ward": ``scomv.tumor_solid.ward_cluster`` with ``n_clusters``.
    Several columns are z-scored first (one column: used as is, as in the Atera POSTN analysis). ``log``: cluster
    log(value) (values <= 0 are clipped to the smallest positive value). Groups are numbered by increasing mean of
    ``order_by`` (default: the first column); with 2 or 3 groups they are named low / (mid) / high.
    """
    from sklearn.mixture import GaussianMixture
    cols = list(features.columns if columns is None else columns)
    F = features[cols].astype(float)
    X = F.to_numpy()
    if log:
        pos = X[X > 0]
        X = np.log(np.clip(X, pos.min() if pos.size else 1e-6, None))
    if X.shape[1] > 1:
        sd = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1)
    selection, model = None, None
    if method == "gmm":
        ks = [k for k in k_range if k <= len(X)]
        bic = pd.Series({k: GaussianMixture(k, n_init=10, random_state=random_state).fit(X).bic(X) for k in ks},
                        name="BIC")
        selection = bic.to_frame()
        if n_boot:
            rng = np.random.default_rng(random_state)
            picks = []
            for _ in range(n_boot):
                Xb = X[rng.integers(0, len(X), len(X))]
                bb = {k: GaussianMixture(k, n_init=3, random_state=random_state).fit(Xb).bic(Xb) for k in ks}
                picks.append(min(bb, key=bb.get))
            selection["bootstrap choice"] = pd.Series(picks).value_counts().reindex(ks, fill_value=0)
        k = int(n_clusters) if n_clusters else int(bic.idxmin())
        model = GaussianMixture(k, n_init=20, random_state=random_state).fit(X)
        raw = model.predict(X)
    elif method == "ward":
        if not n_clusters:
            raise ValueError("method='ward' needs n_clusters")
        from .tumor_solid import ward_cluster
        raw, model = ward_cluster(pd.DataFrame(X, index=F.index), n_clusters)
        k = int(n_clusters)
    else:
        raise ValueError("method must be 'gmm' or 'ward'")
    ref = F[order_by if order_by else cols[0]]
    order = ref.groupby(raw).mean().sort_values().index
    rank = {c: i for i, c in enumerate(order)}
    labels = pd.Series([rank[c] for c in raw], index=F.index, name="group")
    k_used = len(order)
    names = {1: ["all"], 2: ["low", "high"], 3: ["low", "mid", "high"]}.get(k_used, [f"G{i + 1}" for i in range(k_used)])
    res = SolidClusters(labels, names, F, method, k_used, selection, model)
    if verbose:
        if selection is not None:
            print(f"BIC choice: k = {int(selection['BIC'].idxmin())}" + (f" (using k = {k})" if n_clusters else ""))
        print(res.summary().round(3).to_string())
    return res


def plot_solid_clusters(res: SolidClusters, columns: Optional[Sequence[str]] = None, *, colors=None):
    """BIC curve (gmm), histogram of the first feature per group and a violin plot per feature (with pairwise
    Mann-Whitney bars). Returns the figure."""
    from .tumor_solid import plot_uniformity_by_cluster
    cols = list(res.features.columns if columns is None else columns)
    if colors is None:
        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, res.k)) if res.k > 2 else ["#f58231", "#4363d8"][:res.k]
        colors = {i: c for i, c in enumerate(cmap)}
    n_top = (res.selection is not None) + 1
    ncol = max(n_top, min(len(cols), 4))
    nrow = 1 + int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 4.8 * nrow), squeeze=False)
    j = 0
    if res.selection is not None:
        ax = axes[0, 0]
        ax.plot(res.selection.index, res.selection["BIC"], "o-")
        ax.axvline(res.k, color="k", ls=":")
        ax.set_xlabel("Gaussian mixture components k")
        ax.set_ylabel("BIC")
        j = 1
    ax = axes[0, j]
    v = res.features[cols[0]]
    bins = np.linspace(v.min(), v.max(), 35)
    for i, name in enumerate(res.group_names):
        ax.hist(v[res.labels == i], bins=bins, color=colors[i], alpha=0.85, label=f"{name} (n={int((res.labels == i).sum())})")
    ax.set_xlabel(f"{cols[0]} uniformity")
    ax.set_ylabel("solids")
    ax.legend()
    for a in axes[0, j + 1:]:
        a.axis("off")
    for a, c in zip(axes[1:].ravel(), cols):
        plot_uniformity_by_cluster(c, res.features, res.labels.to_numpy(), n_clusters=res.k, palette=colors,
                                   ylim=(min(-0.05, res.features[c].min() - 0.05), max(1.0, res.features[c].max() + 0.05)),
                                   ax=a)
        a.set_ylabel(f"{c}\nuniformity", fontsize=13)
        a.set_xticklabels([f"{n}\n(n={int((res.labels == i).sum())})" for i, n in enumerate(res.group_names)])
    for a in axes[1:].ravel()[len(cols):]:
        a.axis("off")
    fig.suptitle("")                       # plot_uniformity_by_cluster titles the whole figure; not wanted here
    fig.subplots_adjust(hspace=0.75, wspace=0.35, top=0.95, bottom=0.08)   # room for the significance bars
    return fig
