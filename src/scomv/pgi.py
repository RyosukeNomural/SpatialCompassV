"""Gap infiltration score (PGI) for any reference gene (POSTN in the paper).

The reference gene surrounds the tumor, but not evenly: at a given distance from the boundary some directions are
poor in it. The gap infiltration score asks which genes are concentrated in these reference-poor parts of the
peritumoral stroma (``revice/postn_gap/postn_gap_infiltration.ipynb``; gene_analysis tutorial §7):

0. Stroma only: a grid square is stromal if it has <= ``epi_max`` transcripts of the epithelial markers.
1. Area-corrected density: ``G_g(r, a)`` = polar histogram of gene g over the stromal squares (sums to 1);
   ``U(r, a)`` = the same with every stromal square weighted 1; ``D_g = G_g / U`` (1 = stromal average).
2. Band B: the rings with ``band_um[0] <= r < band_um[1]``.
3. Gaps Γ: in each ring, the ``gap_frac`` of the sectors (with stroma) with the lowest reference density.
4. ``PGI_g = log2 (sum_Γ G_g / sum_Γ U) / (sum_B G_g / sum_B U)``; PGI > 0: denser in the gaps than in the band.
5. ``rho_ref``: area-weighted Pearson correlation of log(D_g + 0.05) and log(D_ref + 0.05) over the band
   (< 0: mutually exclusive with the reference gene).
6. Rotation null: the gene's map rotated by 1 .. n_sectors-1 sectors; ``z_rot`` = (PGI - mean) / SD (descriptive).

Typical use::

    from scomv.pgi import compute_pgi, plot_pgi_ranking, plot_pgi_polar
    pgi = compute_pgi(gene_pipe, ref_gene="POSTN")
    pgi.scored.head(20)
    plot_pgi_ranking(pgi, cluster=gene_clusters.labels)
    plot_pgi_polar(pgi, pgi.scored.index[:7])
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from .preparation.scomv_calc_vector import flatten_min_vectors, polar_hist_for_expression
from .signature import as_polar_roi, polar_weight_matrix

EPI_GENES = ("CDH1", "EPCAM", "KRT7", "KRT8", "TACSTD2")


@dataclass
class PGIResult:
    ref_gene: str
    table: pd.DataFrame            # every gene with counts in the band: PGI, z_rot, rho_ref, densities, band_counts
    min_band_counts: float
    U: np.ndarray                  # stromal area share per polar cell (n_r x n_a)
    D: dict                        # gene -> density map relative to the stromal average (n_r x n_a)
    band: np.ndarray
    gap: np.ndarray
    radius_bins: np.ndarray
    angle_bins_deg: np.ndarray

    @property
    def scored(self) -> pd.DataFrame:
        """Genes with >= ``min_band_counts`` transcripts in the stromal band, by decreasing PGI (with ``rank``)."""
        s = self.table[self.table.band_counts >= self.min_band_counts].sort_values("PGI", ascending=False).copy()
        s["rank"] = np.arange(1, len(s) + 1)
        return s


def compute_pgi(
    roi,
    ref_gene: str = "POSTN",
    genes: Optional[Sequence[str]] = None,
    *,
    epi_genes: Sequence[str] = EPI_GENES,
    epi_max: float = 1,
    band_um: Sequence[float] = (0, 300),
    gap_frac: float = 1 / 3,
    min_band_counts: float = 200,
    radius_bins=None,
    angle_bins_deg=None,
    pseudo: float = 0.05,
    verbose: bool = True,
) -> PGIResult:
    """Gap infiltration score of every gene relative to the gaps of ``ref_gene`` (see the module docstring).

    Parameters
    ----------
    roi : SCOMVPipeline (after ``run``), scomv.signature.PolarROI or (subset_grid, min_vector_df)
    ref_gene : gene whose low-density sectors define the gaps.
    genes : genes to score (default: all genes of the ROI grid).
    epi_genes, epi_max : epithelial markers and the max. number of their transcripts in a stromal square
        (markers missing from the data are ignored; pass ``epi_genes=()`` to use every square).
    band_um : radius range of the band (µm, from the boundary).
    gap_frac : fraction of the sectors of each ring that are gaps.
    min_band_counts : genes with fewer transcripts in the stromal band are not in ``result.scored``.
    radius_bins, angle_bins_deg : polar bins (default: the pipeline's, else -150..300 µm by 10 x 30°).
    """
    roi_obj = roi
    roi = as_polar_roi(roi)
    sg = roi.subset_grid
    if radius_bins is None:
        radius_bins = getattr(roi_obj, "radius_bins", None)
    if angle_bins_deg is None:
        angle_bins_deg = getattr(roi_obj, "angle_bins_deg", None)
    RB = np.asarray(np.arange(-150, 310, 10) if radius_bins is None else radius_bins, float)
    AB = np.asarray(np.arange(-180, 181, 30) if angle_bins_deg is None else angle_bins_deg, float)
    n_r, n_a = len(RB) - 1, len(AB) - 1

    var = list(sg.var_names)
    vidx = {g: i for i, g in enumerate(var)}
    if ref_gene not in vidx:
        raise KeyError(f"reference gene {ref_gene!r} is not in the ROI grid")
    genes = var if genes is None else [g for g in dict.fromkeys(genes) if g in vidx]
    if ref_gene not in genes:
        genes = list(genes) + [ref_gene]
    X = sg.X.tocsc() if sparse.issparse(sg.X) else sparse.csc_matrix(np.asarray(sg.X))
    epi = [g for g in epi_genes if g in vidx]
    epi_counts = np.asarray(X[:, [vidx[g] for g in epi]].sum(axis=1)).ravel() if epi else np.zeros(sg.n_obs)
    keep = (epi_counts <= epi_max).astype(float)

    M, has_vec = polar_weight_matrix(roi.min_vector_df, RB, AB, roi.bin_size, n_rows=sg.n_obs)
    MT = M.T.tocsr()
    u = keep * has_vec
    U = np.asarray(MT @ keep).ravel() / max(u.sum(), 1e-12)
    U = U.reshape(n_r, n_a)

    # polar histograms of every gene over the stroma (same numbers as polar_hist_for_expression: counts cast to int)
    Xg = X[:, [vidx[g] for g in genes]].astype(np.float64)
    Xg.data = np.trunc(Xg.data)
    Xg = (sparse.diags(keep) @ Xg).tocsc()
    raw = np.asarray((MT @ Xg).todense()).T.reshape(len(genes), n_r, n_a)        # genes x r x a
    tot = np.asarray(Xg[has_vec].sum(axis=0)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        Gm = np.where(tot[:, None, None] > 0, raw / tot[:, None, None], 0.0)
        D = np.where(U > 0, Gm / U, np.nan)

    r_lo = RB[:-1]
    band_rows = np.flatnonzero((r_lo >= band_um[0]) & (r_lo < band_um[1]))
    band = np.zeros((n_r, n_a), bool)
    band[band_rows] = True
    band &= U > 0
    gi = genes.index(ref_gene)
    D_ref = D[gi]
    # rank the sectors with the package histogram (polar_hist_for_expression) so that sectors with equal density
    # are ordered exactly as in the per-gene SCOMV histograms (the matrix products above differ by ~1e-16)
    flat = flatten_min_vectors(roi.min_vector_df.iloc[:sg.n_obs])
    U_rank = polar_hist_for_expression(keep, flat, RB, AB, roi.bin_size)
    G_rank = polar_hist_for_expression(X[:, vidx[ref_gene]].toarray().ravel() * keep, flat, RB, AB, roi.bin_size)
    if U_rank is None or G_rank is None:
        raise ValueError(f"no stromal {ref_gene} counts in the ROI")
    with np.errstate(invalid="ignore", divide="ignore"):
        D_rank = np.where(U_rank > 0, G_rank / U_rank, np.nan)
    gap = np.zeros_like(band)
    k_gap = int(round(n_a * gap_frac))
    for r in band_rows:
        ok = np.flatnonzero(U[r] > 0)
        gap[r, ok[np.argsort(D_rank[r, ok], kind="stable")[:k_gap]]] = True

    def pgi_of(G):                      # G: genes x r x a (area-weighted maps)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log2((G[:, gap].sum(axis=1) / U[gap].sum()) / (G[:, band].sum(axis=1) / U[band].sum()))

    obs = pgi_of(Gm)
    Dn = np.nan_to_num(D)
    null = np.stack([pgi_of(np.roll(Dn, s, axis=2) * U) for s in range(1, n_a)], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (obs - null.mean(axis=1)) / null.std(axis=1, ddof=1)

    w = U[band]
    lx = np.log(Dn[:, band] + pseudo)
    ly = np.log(np.nan_to_num(D_ref)[band] + pseudo)
    mx, my = (lx * w).sum(axis=1) / w.sum(), (ly * w).sum() / w.sum()
    cov = ((lx - mx[:, None]) * (ly - my) * w).sum(axis=1) / w.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = cov / np.sqrt(((lx - mx[:, None]) ** 2 * w).sum(axis=1) / w.sum() * ((ly - my) ** 2 * w).sum() / w.sum())

    n_vec, vec_row, _, rad = flatten_min_vectors(roi.min_vector_df.iloc[:sg.n_obs])
    in_band = (rad * roi.bin_size >= band_um[0]) & (rad * roi.bin_size < band_um[1])
    row_w = np.bincount(vec_row[in_band], weights=1.0 / n_vec[vec_row[in_band]], minlength=len(n_vec)) * keep
    band_counts = np.asarray(X[:, [vidx[g] for g in genes]].T @ row_w).ravel()

    with np.errstate(invalid="ignore", divide="ignore"):
        dens_gap = Gm[:, gap].sum(axis=1) / U[gap].sum()
        dens_band = Gm[:, band].sum(axis=1) / U[band].sum()
    has_band = Gm[:, band].sum(axis=1) > 0
    table = pd.DataFrame({"PGI": obs, "z_rot": z, f"rho_{ref_gene}": rho, "density_in_gaps": dens_gap,
                          "density_in_band": dens_band, "band_counts": band_counts}, index=pd.Index(genes, name="gene"))
    table = table[has_band]
    res = PGIResult(ref_gene, table, min_band_counts, U, {g: D[i] for i, g in enumerate(genes)}, band, gap, RB, AB)
    if verbose:
        n_st = int((keep > 0).sum())
        print(f"{ref_gene} gaps: {int(gap.sum())} of {int(band.sum())} band cells ({band_um[0]:g}-{band_um[1]:g} µm); "
              f"the gaps hold {U[gap].sum() / U[band].sum():.0%} of the band's stroma and "
              f"{Gm[gi][gap].sum() / Gm[gi][band].sum():.0%} of its {ref_gene}.  Stromal squares: {n_st:,} / {sg.n_obs:,} "
              f"(epithelial markers: {', '.join(epi) if epi else 'none'})")
        print(f"{len(res.scored)} of {len(table)} genes scored (>= {min_band_counts:g} transcripts in the stromal band); "
              f"PGI of {ref_gene} itself: {table.loc[ref_gene, 'PGI']:.2f}")
    return res


def _cluster_colors(labels: Optional[pd.Series], colors: Optional[Mapping[str, Any]] = None):
    """Colour per cluster label: ``colors`` if given, else "C<k>" -> the k-th tab20 colour (same as
    ``GeneClusters.colors``), other labels in sorted order."""
    if labels is None:
        return None, None
    import re
    order = [c for c in dict.fromkeys(labels.values) if c not in ("-", None)]
    order = sorted(order, key=lambda c: (len(str(c)), str(c)))
    col = {}
    for i, c in enumerate(order):
        m = re.fullmatch(r"C(\d+)", str(c))
        col[c] = plt.cm.tab20.colors[(int(m.group(1)) - 1 if m else i) % 20]
    if colors:
        col.update({c: colors[c] for c in order if c in colors})
    col["-"] = "lightgray"
    return col, order + ["-"]


def plot_pgi_ranking(res: PGIResult, *, top: int = 20, cluster: Optional[Mapping[str, Any]] = None,
                     label_genes: Sequence[str] = (), n_label_top: int = 10, bold_genes: Optional[Sequence[str]] = None,
                     colors: Optional[Mapping[str, Any]] = None, fontsize: float = 21, bar_figsize=(13, 10),
                     scatter_figsize=(15, 9)):
    """(a) Top genes by PGI (bar) and (b) PGI vs. the mutual-exclusion coefficient rho (scatter).

    ``cluster``: gene -> cluster label (e.g. ``GeneClusters.labels``) to colour the genes; ``colors``: label ->
    colour (default: "C<k>" -> k-th tab20 colour, as in ``GeneClusters.colors``).
    The scatter labels the ``n_label_top`` best genes, the reference gene and ``label_genes``; ``bold_genes``
    (default: the reference gene) are printed in bold. ``fontsize`` sets the gene names of the bar chart; the
    other texts scale with it. Returns (fig_bar, fig_scatter).
    """
    from matplotlib.patches import Patch
    f = fontsize / 21
    s = res.scored
    rho_col = f"rho_{res.ref_gene}"
    cl = pd.Series(cluster).reindex(s.index).fillna("-") if cluster is not None else None
    ccol, order = _cluster_colors(cl, colors)
    colour = (lambda idx: [ccol[c] for c in cl[idx]]) if cl is not None else (lambda idx: "#4363d8")
    bold = {res.ref_gene} if bold_genes is None else set(bold_genes)
    label = lambda c: "unclassified" if c == "-" else c

    t = s.head(top)
    fig1, ax = plt.subplots(figsize=bar_figsize)
    ax.barh(np.arange(len(t))[::-1], t.PGI, color=colour(t.index), height=0.75)
    ax.set_yticks(np.arange(len(t))[::-1], list(t.index), fontsize=21 * f)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.tick_params(axis="x", labelsize=22 * f)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("PGI", fontsize=26 * f)
    if cl is not None:
        ax.legend(handles=[Patch(color=ccol[c], label=label(c)) for c in order if c in set(cl[t.index])],
                  loc="lower right", fontsize=22 * f, handlelength=1.2, borderpad=0.4)
    fig1.tight_layout()

    fig2, ax = plt.subplots(figsize=scatter_figsize)
    ax.scatter(s[rho_col], s.PGI, c=colour(s.index), s=70, lw=0)
    lab = list(t.index[:n_label_top]) + [res.ref_gene] + list(label_genes)
    texts = [ax.text(s.loc[g, rho_col], s.loc[g, "PGI"], g, fontsize=26 * f,
                     fontweight="bold" if g in bold else "normal")
             for g in dict.fromkeys(lab) if g in s.index]
    ax.set_ylim(s.PGI.min() - 0.08, s.PGI.max() + 0.22)
    ax.set_xlim(s[rho_col].min() - 0.12, s[rho_col].max() + 0.12)
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, expand=(1.3, 1.7), force_text=(0.8, 1.2), max_move=(100, 100), iter_lim=3000,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
    except Exception:
        pass
    if cl is not None:
        for c in order[:-1]:
            ax.scatter([], [], color=ccol[c], s=120, label=label(c))
        ax.legend(ncol=3, loc="upper right", fontsize=28 * f, columnspacing=0.8, handletextpad=0.1, borderpad=0.3)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel(f"mutual-exclusion coefficient ρ (vs {res.ref_gene})", fontsize=35 * f)
    ax.set_ylabel("PGI", fontsize=35 * f)
    ax.tick_params(labelsize=32 * f)
    fig2.tight_layout()
    return fig1, fig2


def plot_pgi_polar(res: PGIResult, genes: Sequence[str], *, with_gap_panel: bool = True, r_min: float = -100,
                   ncols: int = 4, panel_size: float = 6.5):
    """Stromal density maps (D; 1 = stromal average) of the genes, optionally after a panel showing the gaps."""
    from .polar_plot import plot_polar_heatmap
    keep = res.radius_bins[:-1] >= r_min
    rb = res.radius_bins[np.r_[keep, True]]
    ticks = [t for t in (-100, 0, 100, 200, 300) if rb[0] <= t <= rb[-1]]
    kw = dict(radius_ticks=ticks, hole_frac=0.15, fontsize=14)
    panels = (["__gaps__"] if with_gap_panel else []) + [g for g in genes if g in res.D]
    ncols = min(ncols, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    axes = axes.ravel()
    for ax, g in zip(axes, panels):
        if g == "__gaps__":
            m = np.where(res.band, res.gap.astype(float), np.nan)[keep]
            plot_polar_heatmap(m, rb, res.angle_bins_deg, ax=ax, cmap="Greys", vmin=0, vmax=1.3, colorbar=False,
                               boundary_kwargs=dict(color="black", lw=2, ls="--"), **kw)
            title = f"{res.ref_gene} gaps (black)"
        else:
            Dg = res.D[g]
            vmax = np.nanpercentile(Dg[res.band], 95) if np.isfinite(Dg[res.band]).any() else None
            plot_polar_heatmap(Dg[keep], rb, res.angle_bins_deg, ax=ax, vmin=0, vmax=vmax,
                               boundary_kwargs=dict(color="white", lw=2, ls="--"),
                               cbar_label="density / stromal average", cbar_pad=0.16, **kw)
            title = g if g not in res.table.index or g == res.ref_gene else f"{g}  PGI={res.table.loc[g, 'PGI']:.2f}"
        ax.set_title(title, fontsize=18, weight="bold", pad=30)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    return fig
