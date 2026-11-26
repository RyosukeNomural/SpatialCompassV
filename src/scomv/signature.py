"""Multi-gene signatures in the tumor-relative polar space (MSigDB Hallmark or any gene list).

The polar histogram of SCOMV only needs one non-negative weight per grid square, so a **signature score** can be
mapped around the tumor boundary exactly like one gene's counts. This module packages the analysis of
``revice/signature_polar_hallmark.ipynb`` / ``atera_wta_solid*_hallmark_polar_r150_a30.ipynb``:

1. Score per grid square (method B): counts / total counts of the square x median total -> log1p -> z-score per
   panel gene over the tissue squares -> mean z of the member genes, clipped at 0.
2. Polar cells: ``r_step`` µm rings from the boundary out to ``r_max`` µm x 30° sectors (the boundary squares
   themselves, r = 0, are not "outside"). With ``include_inside=True`` the tumor (every vector with r < 1 µm:
   boundary squares and squares inside) is one extra cell, drawn as the centre disc.
3. Enrichment of the signature relative to the tissue itself:
   ``log2 E_c = log2 (S_c / sum S) / (T_c / sum T)`` (S: signature, T: number of tissue squares).
4. Null: random gene sets of the same size drawn from the same expression deciles of the panel; one-sided empirical
   p per cell, BH within each signature (``q``) and across all signatures x cells (``q_global``).

Typical use (after ``gene_pipe = SCOMVPipeline(...); gene_pipe.run(roi=...)``)::

    from scomv.signature import signature_polar_enrichment, plot_signature_polar
    res = signature_polar_enrichment(gene_pipe, {"T cell": ["CD3E", "CD8A", "GZMA"]}, r_max=300)
    plot_signature_polar(res)

    from scomv.signature import hallmark_polar_enrichment, plot_significant_hallmarks
    hm = hallmark_polar_enrichment(gene_pipe, r_max=150, include_inside=True)
    plot_significant_hallmarks(hm)

A single tumor solid works the same way: ``PolarROI.from_solid(grid, solid, solid_id)`` instead of the pipeline.
"""
from __future__ import annotations

import json
import textwrap
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from .preparation.scomv_calc_vector import flatten_min_vectors

ANGLE_STEP = 30          # sector width of every polar map in this module (degrees; fixed)

HALLMARK_CATEGORY = {    # Liberzon et al. 2015, Cell Systems 1:417 (process categories of the 50 Hallmarks)
    "immune": ["ALLOGRAFT_REJECTION", "COAGULATION", "COMPLEMENT", "INTERFERON_ALPHA_RESPONSE",
               "INTERFERON_GAMMA_RESPONSE", "IL6_JAK_STAT3_SIGNALING", "INFLAMMATORY_RESPONSE"],
    "development": ["ADIPOGENESIS", "ANGIOGENESIS", "EPITHELIAL_MESENCHYMAL_TRANSITION", "MYOGENESIS",
                    "SPERMATOGENESIS", "PANCREAS_BETA_CELLS"],
    "signaling": ["ANDROGEN_RESPONSE", "ESTROGEN_RESPONSE_EARLY", "ESTROGEN_RESPONSE_LATE", "IL2_STAT5_SIGNALING",
                  "KRAS_SIGNALING_UP", "KRAS_SIGNALING_DN", "MTORC1_SIGNALING", "NOTCH_SIGNALING",
                  "PI3K_AKT_MTOR_SIGNALING", "HEDGEHOG_SIGNALING", "TGF_BETA_SIGNALING", "TNFA_SIGNALING_VIA_NFKB",
                  "WNT_BETA_CATENIN_SIGNALING"],
    "proliferation": ["E2F_TARGETS", "G2M_CHECKPOINT", "MYC_TARGETS_V1", "MYC_TARGETS_V2", "P53_PATHWAY",
                      "MITOTIC_SPINDLE"],
    "metabolic": ["BILE_ACID_METABOLISM", "CHOLESTEROL_HOMEOSTASIS", "FATTY_ACID_METABOLISM", "GLYCOLYSIS",
                  "HEME_METABOLISM", "OXIDATIVE_PHOSPHORYLATION", "XENOBIOTIC_METABOLISM"],
    "DNA damage": ["DNA_REPAIR", "UV_RESPONSE_DN", "UV_RESPONSE_UP"],
    "cellular component": ["APICAL_JUNCTION", "APICAL_SURFACE", "PEROXISOME"],
    "pathway": ["APOPTOSIS", "HYPOXIA", "PROTEIN_SECRETION", "UNFOLDED_PROTEIN_RESPONSE",
                "REACTIVE_OXYGEN_SPECIES_PATHWAY"],
}


# ===================================================================================================== ROI and binning
@dataclass
class PolarROI:
    """Grid squares of one ROI and their shortest vectors to the tumor boundary.

    ``subset_grid`` rows and ``min_vector_df`` rows are the same squares in the same order (as produced by
    ``SCOMVPipeline.run`` or ``scomv.tumor_solid.compute_solid_min_vectors``).
    """
    subset_grid: Any
    min_vector_df: pd.DataFrame
    bin_size: float = 10
    xy: Optional[np.ndarray] = None            # square positions (for maps); optional
    outline: Optional[np.ndarray] = None       # boundary squares (for maps); optional
    inside: Optional[np.ndarray] = None        # bool per square: inside the tumor (for maps); optional

    @classmethod
    def from_pipeline(cls, pipe) -> "PolarROI":
        if pipe.subset_grid is None or pipe.min_vector_df is None:
            raise ValueError("run the pipeline first (pipe.run(roi=...))")
        inside = None
        if pipe.xy_list is not None and pipe.inside_points is not None:
            s = set(map(tuple, pipe.inside_points))
            inside = np.array([tuple(p) in s for p in pipe.xy_list])
        return cls(pipe.subset_grid, pipe.min_vector_df, pipe.bin_size,
                   xy=None if pipe.xy_list is None else np.asarray(pipe.xy_list, float),
                   outline=None if pipe.outline_points is None else np.asarray(pipe.outline_points, float),
                   inside=inside)

    @classmethod
    def from_solid(cls, grid, solid, solid_id: int, bin_size: int = 10, margin_bins: int = 15,
                   mask_other_solids: bool = False) -> "PolarROI":
        from .tumor_solid import compute_solid_min_vectors
        sg, mv, _ = compute_solid_min_vectors(grid, solid, solid_id, bin_size=bin_size, margin_bins=margin_bins,
                                              mask_other_solids=mask_other_solids)
        return cls(sg, mv, bin_size)

    @property
    def n_squares(self) -> int:
        return self.subset_grid.n_obs

    def histogram(self, weights, radius_bins=None, angle_bins_deg=None) -> np.ndarray:
        """Polar histogram (radius x angle) of any non-negative per-square weights, normalised like the per-gene
        SCOMV histograms (sum over the squares with a vector = 1). Default bins: -150..300 µm by 10 x 30°."""
        rb = np.arange(-150, 310, 10) if radius_bins is None else np.asarray(radius_bins)
        ab = np.arange(-180, 181, ANGLE_STEP) if angle_bins_deg is None else np.asarray(angle_bins_deg)
        w = np.asarray(weights, float)
        M, has_vec = polar_weight_matrix(self.min_vector_df, rb, ab, self.bin_size, n_rows=self.n_squares)
        return np.asarray(M.T @ w).reshape(len(rb) - 1, len(ab) - 1) / w[has_vec].sum()


def as_polar_roi(obj) -> PolarROI:
    """A ``PolarROI`` from a ``PolarROI``, a run ``SCOMVPipeline`` or a ``(subset_grid, min_vector_df)`` pair."""
    if isinstance(obj, PolarROI):
        return obj
    if isinstance(obj, tuple) and len(obj) in (2, 3):
        return PolarROI(*obj)
    if hasattr(obj, "subset_grid") and hasattr(obj, "min_vector_df"):
        return PolarROI.from_pipeline(obj)
    raise TypeError("expected a PolarROI, a run SCOMVPipeline or a (subset_grid, min_vector_df) tuple")


def _hist_index(v, edges):
    """Bin index as np.histogram (last bin includes its right edge); -1 = outside the edges."""
    edges = np.asarray(edges, float)
    i = np.searchsorted(edges, v, side="right") - 1
    i[v == edges[-1]] = len(edges) - 2
    i[(v < edges[0]) | (v > edges[-1])] = -1
    return i


def polar_weight_matrix(min_vector_df: pd.DataFrame, radius_bins, angle_bins_deg, bin_size_um: float = 10,
                        n_rows: Optional[int] = None):
    """Sparse (squares x polar cells) matrix of the tie weights 1/n of every square's shortest vectors.

    The polar histogram of any per-square weight vector ``w`` is ``M.T @ w`` (cells ordered radius-major, as
    ``np.histogram2d(radius, angle)``); divided by ``w[has_vec].sum()`` it equals
    ``polar_hist_for_expression`` / ``SCOMVPipeline.polar_counts_list`` for integer counts.

    Returns ``(M, has_vec)``.
    """
    mv = min_vector_df if n_rows is None else min_vector_df.iloc[:n_rows]
    n_vec, vec_row, angle, radii = flatten_min_vectors(mv)
    ri = _hist_index(radii * bin_size_um, radius_bins)
    ai = _hist_index(np.degrees(angle), angle_bins_deg)
    ok = (ri >= 0) & (ai >= 0)
    n_a = len(angle_bins_deg) - 1
    M = sparse.csr_matrix((1.0 / n_vec[vec_row[ok]], (vec_row[ok], ri[ok] * n_a + ai[ok])),
                          shape=(len(n_vec), (len(radius_bins) - 1) * n_a))
    return M, n_vec > 0


def _bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = np.minimum.accumulate((pv[order] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
    tmp = np.empty_like(pv)
    tmp[order] = np.minimum(ranked, 1.0)
    q[ok] = tmp
    return q


def _dense_rows(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


# ===================================================================================================== scores
@dataclass
class SignatureScorer:
    """Per-square z-scored log-normalised expression of the panel genes (method B), shared by all signatures."""
    panel: List[str]
    Z: np.ndarray                 # squares x panel (float32)
    tissue: np.ndarray            # squares with any counts
    expr_decile: np.ndarray       # decile of each panel gene's mean log-expression (for the null)

    @property
    def gene_pos(self) -> Dict[str, int]:
        return {g: i for i, g in enumerate(self.panel)}

    def score(self, genes: Sequence[str]) -> np.ndarray:
        pos = self.gene_pos
        idx = [pos[g] for g in genes]
        return np.clip(self.Z[:, idx].mean(axis=1), 0, None) * self.tissue


def build_signature_scorer(subset_grid, panel_genes: Optional[Sequence[str]] = None, min_counts: float = 1,
                           max_gb: float = 8.0) -> SignatureScorer:
    """Normalise, log1p and z-score the panel genes of ``subset_grid`` (see the module docstring, step 1).

    ``panel_genes``: genes that may be members of a signature and of the random null sets (default: every gene of
    ``subset_grid`` with at least ``min_counts`` counts in the ROI). Totals per square use all genes of the grid.
    """
    X = subset_grid.X
    X = X.tocsr() if sparse.issparse(X) else np.asarray(X)
    var = list(subset_grid.var_names)
    vidx = {g: i for i, g in enumerate(var)}
    total = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    tissue = total > 0
    if panel_genes is None:
        gsum = np.asarray(X.sum(axis=0)).ravel()
        panel = [g for g, s in zip(var, gsum) if s >= min_counts]
    else:
        panel = [g for g in dict.fromkeys(panel_genes) if g in vidx]
    gb = X.shape[0] * len(panel) * 4 * 2 / 1e9            # Z plus its transpose (float32)
    if gb > max_gb:
        raise MemoryError(f"the z-score matrix needs ~{gb:.1f} GB ({X.shape[0]:,} squares x {len(panel):,} genes); "
                          f"pass fewer panel_genes or raise max_gb")
    scale = np.zeros_like(total)
    scale[tissue] = float(np.median(total[tissue])) / total[tissue]
    Xp = X[:, [vidx[g] for g in panel]]
    if sparse.issparse(Xp):
        Xp = (sparse.diags(scale) @ Xp).tocsr()
        Xp.data = np.log1p(Xp.data)
        logn = Xp.toarray().astype(np.float32)
    else:
        logn = np.log1p(Xp * scale[:, None]).astype(np.float32)
    mu, sd = logn[tissue].mean(axis=0), logn[tissue].std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.zeros_like(logn)
    Z[tissue] = (logn[tissue] - mu) / sd
    decile = np.asarray(pd.qcut(mu, 10, labels=False, duplicates="drop"))
    return SignatureScorer(panel, Z, tissue, decile)


def _normalise_gene_sets(gene_sets, name: str = "signature") -> Dict[str, List[str]]:
    if isinstance(gene_sets, Mapping):
        return {str(k): list(v) for k, v in gene_sets.items()}
    gs = list(gene_sets)
    if gs and isinstance(gs[0], str):
        return {name: gs}
    raise TypeError("gene_sets must be a list of gene names or a dict {name: [genes]}")


# ===================================================================================================== enrichment
@dataclass
class SignatureEnrichment:
    """Result of :func:`signature_polar_enrichment` (one entry per signature in the dicts).

    Cell layout of every per-cell array: with ``include_inside`` cell 0 is the inside, then the outside cells
    radius-major (``n_r`` rings x ``n_a`` sectors).
    """
    gene_sets: Dict[str, List[str]]
    log2e: Dict[str, np.ndarray]
    z: Dict[str, np.ndarray]
    p: Dict[str, np.ndarray]
    q: Dict[str, np.ndarray]
    q_global: Dict[str, np.ndarray]
    r_edges: np.ndarray
    a_edges: np.ndarray
    include_inside: bool
    tissue_hist: np.ndarray
    valid_cell: np.ndarray
    fdr: float
    n_perm: int
    dropped: Dict[str, int] = field(default_factory=dict)   # signatures left out: name -> genes in the panel
    roi: Optional[PolarROI] = None
    scorer: Optional[SignatureScorer] = None
    vec_cell: Optional[np.ndarray] = None                   # polar cell of every shortest vector (-1 = none)
    vec_row: Optional[np.ndarray] = None
    polar_matrix: Any = None                                # squares x polar cells (tie weights)

    def log2e_of(self, weights, tissue_weights=None) -> np.ndarray:
        """log2 E map of any per-square weights (n_squares,) or (n_squares, B) with the same polar cells, tested cells
        and baseline as this result (``tissue_weights``: other per-square baseline weights, e.g. a bootstrap)."""
        M = self.polar_matrix
        S = np.asarray(M.T @ np.asarray(weights, float))
        T = self.tissue_hist if tissue_weights is None else np.asarray(M.T @ np.asarray(tissue_weights, float))
        return _log2_enrichment(S, T, self.valid_cell)

    @property
    def names(self) -> List[str]:
        return list(self.log2e)

    @property
    def n_r(self) -> int:
        return len(self.r_edges) - 1

    @property
    def n_a(self) -> int:
        return len(self.a_edges) - 1

    def cell_labels(self) -> List[str]:
        step = self.r_edges[1] - self.r_edges[0]
        out = [f"{r:g}-{r + step:g} µm, {a:g}-{a + ANGLE_STEP:g}°" for r in self.r_edges[:-1] for a in self.a_edges[:-1]]
        return (["inside"] if self.include_inside else []) + out

    def is_sig(self, name: str, q_key: str = "q") -> np.ndarray:
        q = getattr(self, q_key)[name]
        return (q < self.fdr) & (self.log2e[name] > 0)

    def n_significant(self, q_key: str = "q") -> pd.Series:
        return pd.Series({n: int(self.is_sig(n, q_key).sum()) for n in self.names}, name=f"n_sig_cells ({q_key})")

    def significant_names(self, q_key: str = "q") -> List[str]:
        """Signatures with at least one significantly enriched cell, most significant cells first."""
        n = self.n_significant(q_key)
        return list(n[n > 0].sort_values(ascending=False, kind="stable").index)

    def outside(self, values: np.ndarray) -> np.ndarray:
        """The outside cells of a per-cell array as (n_r, n_a)."""
        v = np.asarray(values)
        return (v[1:] if self.include_inside else v).reshape(self.n_r, self.n_a)

    def table(self) -> pd.DataFrame:
        """Long table: every signature x polar cell."""
        lab = self.cell_labels()
        return pd.concat([pd.DataFrame({"signature": n, "cell": lab, "n_tissue_squares": self.tissue_hist,
                                        "valid": self.valid_cell, "log2_enrichment": self.log2e[n], "z": self.z[n],
                                        "p": self.p[n], "q": self.q[n], "q_global": self.q_global[n]})
                          for n in self.names], ignore_index=True)

    def summary(self) -> pd.DataFrame:
        """Per signature: number of significant cells and the best cell."""
        lab = np.array(self.cell_labels())
        rows = []
        for n in self.names:
            zz = np.where(self.valid_cell, self.z[n], -np.inf)
            best = int(np.nanargmax(zz))
            row = {"signature": n, "n_genes": len(self.gene_sets[n]), "n_sig_cells": int(self.is_sig(n).sum()),
                   "n_sig_cells_global": int(self.is_sig(n, "q_global").sum()), "min_p": np.nanmin(self.p[n]),
                   "best_cell": lab[best], "best_log2E": self.log2e[n][best], "best_z": self.z[n][best],
                   "best_q": self.q[n][best], "best_q_global": self.q_global[n][best]}
            if self.include_inside:
                row.update(inside_log2E=self.log2e[n][0], inside_sig=bool(self.is_sig(n, "q_global")[0]))
            if n in _CATEGORY_OF:
                row["category"] = _CATEGORY_OF[n]
            rows.append(row)
        return (pd.DataFrame(rows).set_index("signature")
                .sort_values(["n_sig_cells_global", "n_sig_cells"], ascending=False, kind="stable"))


_CATEGORY_OF = {h: c for c, hs in HALLMARK_CATEGORY.items() for h in hs}


def _log2_enrichment(S_hist, T_hist, valid):
    """log2 (S share / T share) over the valid cells; S_hist (cells,) or (cells, B), T_hist (cells,) or same shape."""
    S_hist = np.asarray(S_hist, float)
    T = np.asarray(T_hist, float)
    if S_hist.ndim == 2 and T.ndim == 1:
        T = T[:, None]
    v = valid if S_hist.ndim == 1 else valid[:, None]
    eps = 1e-3 / max(valid.sum(), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share_s = S_hist / np.where(v, S_hist, 0).sum(axis=0)
        share_t = T / np.where(v, T, 0).sum(axis=0)
        return np.where(v, np.log2((share_s + eps) / (share_t + eps)), np.nan)


def signature_polar_enrichment(
    roi,
    gene_sets: Union[Sequence[str], Mapping[str, Sequence[str]]],
    *,
    r_max: float = 300,
    r_step: float = 50,
    include_inside: bool = False,
    n_perm: int = 20000,
    fdr: float = 0.05,
    min_set_genes: int = 5,
    min_tissue_squares: int = 15,
    panel_genes: Optional[Sequence[str]] = None,
    name: str = "signature",
    scorer: Optional[SignatureScorer] = None,
    perm_chunk: int = 250,
    verbose: bool = True,
) -> SignatureEnrichment:
    """Polar enrichment map of one or several gene signatures around the tumor boundary, with permutation test.

    Parameters
    ----------
    roi : SCOMVPipeline (after ``run``), PolarROI or (subset_grid, min_vector_df)
    gene_sets : list of genes (one signature called ``name``) or dict {signature name: genes}
    r_max, r_step : outer radius and ring width of the polar cells (µm). The sector width is fixed at 30°.
    include_inside : also test the tumor (every vector with r < 1 µm) as one extra cell (cell 0).
        The baseline is then the whole tissue of the map (inside + outside).
    n_perm : random gene sets per signature (decile-matched); p has a floor of 1 / (n_perm + 1).
    fdr : significance threshold on q (and log2 E > 0).
    min_set_genes : signatures with fewer panel genes are skipped (listed in ``result.dropped``).
    min_tissue_squares : polar cells with fewer tissue squares are not tested (grey).
    panel_genes : genes for the signatures and the null (default: all genes with counts in the ROI).
    scorer : reuse a :func:`build_signature_scorer` result (e.g. for several calls on the same ROI).
    """
    roi = as_polar_roi(roi)
    sg = roi.subset_grid
    if scorer is None:
        scorer = build_signature_scorer(sg, panel_genes)
    pos = scorer.gene_pos
    sets, dropped = {}, {}
    gene_sets = _normalise_gene_sets(gene_sets, name)
    n_input = len(gene_sets)
    for n, genes in gene_sets.items():
        hit = sorted(set(genes) & set(pos))
        missing = sorted(set(genes) - set(pos))
        if len(hit) >= min_set_genes:
            sets[n] = hit
        else:
            dropped[n] = len(hit)
        if verbose and missing and n_input <= 3:
            print(f"{n}: {len(hit)} / {len(set(genes))} genes in the panel"
                  + (f" (missing: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''})" if missing else ""))
    if not sets:
        raise ValueError(f"no signature has >= {min_set_genes} panel genes: {dropped}")

    r_edges = np.arange(0, r_max + r_step / 2, r_step, dtype=float)
    a_edges = np.arange(-180, 181, ANGLE_STEP, dtype=float)
    r_edges_bin = np.r_[1.0, r_edges[1:]]              # r = 0 (boundary squares) is not "outside"
    M, _ = polar_weight_matrix(roi.min_vector_df, r_edges_bin, a_edges, roi.bin_size, n_rows=sg.n_obs)
    n_vec, vec_row, ang, rad = flatten_min_vectors(roi.min_vector_df.iloc[:sg.n_obs])
    ri, ai = _hist_index(rad * roi.bin_size, r_edges_bin), _hist_index(np.degrees(ang), a_edges)
    vec_cell = np.where((ri >= 0) & (ai >= 0), ri * (len(a_edges) - 1) + ai, -1)
    if include_inside:
        M_in, _ = polar_weight_matrix(roi.min_vector_df, np.array([-1e9, 1.0]), np.array([-180.0, 180.0]),
                                      roi.bin_size, n_rows=sg.n_obs)
        M = sparse.hstack([M_in, M]).tocsr()
        vec_cell = np.where(vec_cell >= 0, vec_cell + 1, np.where(rad * roi.bin_size < 1.0, 0, -1))
    MT = M.T.tocsr().astype(np.float32)
    tissue = scorer.tissue
    tissue_hist = np.asarray(M.T @ tissue.astype(float)).ravel()
    valid = tissue_hist >= min_tissue_squares

    def log2e(S_hist):
        return _log2_enrichment(S_hist, tissue_hist, valid)

    ZT = np.ascontiguousarray(scorer.Z.T)              # panel x squares
    n_panel = len(scorer.panel)

    def null(genes, rng):
        k = len(genes)
        member_q = pd.Series(scorer.expr_decile[[pos[g] for g in genes]]).value_counts()
        pools = {qq: np.flatnonzero(scorer.expr_decile == qq) for qq in member_q.index}
        out = []
        for start in range(0, n_perm, perm_chunk):
            m = min(perm_chunk, n_perm - start)
            idx = np.concatenate([np.concatenate([rng.choice(pools[qq], size=c, replace=False)
                                                  for qq, c in member_q.items()]) for _ in range(m)])
            R = sparse.csr_matrix((np.full(m * k, 1.0 / k, np.float32), (np.repeat(np.arange(m), k), idx)),
                                  shape=(m, n_panel))
            S = np.clip(np.asarray(R @ ZT).T, 0, None) * tissue[:, None]   # squares x m
            out.append(log2e(np.asarray(MT @ S)))
        return np.concatenate(out, axis=1)

    import time
    t0 = time.time()
    L, Zs, P, Q = {}, {}, {}, {}
    for n, genes in sets.items():
        obs = log2e(M.T @ scorer.score(genes))
        nl = null(genes, np.random.default_rng(zlib.crc32(n.encode())))
        import warnings
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # untested cells are all-NaN
            Zs[n] = (obs - np.nanmean(nl, axis=1)) / np.nanstd(nl, axis=1)
        P[n] = np.where(valid, (1 + (nl >= obs[:, None]).sum(axis=1)) / (1 + n_perm), np.nan)
        Q[n] = _bh(P[n])
        L[n] = obs
    names = list(sets)
    qg = np.split(_bh(np.concatenate([P[n] for n in names])), len(names))
    if verbose:
        print(f"{len(names)} signature(s) x {n_perm:,} permutations, {int(valid.sum())} / {valid.size} polar cells "
              f"tested ({time.time() - t0:.0f} s)" + (f"; skipped (< {min_set_genes} panel genes): {dropped}" if dropped else ""))
    return SignatureEnrichment(sets, L, Zs, P, Q, dict(zip(names, qg)), r_edges, a_edges, include_inside,
                               tissue_hist, valid, fdr, n_perm, dropped, roi, scorer, vec_cell, vec_row, M)


# ===================================================================================================== Hallmark
def load_hallmark_gene_sets(dbver: str = "2024.1.Hs", cache: Optional[Union[str, Path]] = None,
                            strip_prefix: bool = True) -> Dict[str, List[str]]:
    """MSigDB Hallmark gene sets (via ``gseapy.Msigdb``; network). ``cache``: JSON file read if present, written
    after a download."""
    raw = None
    if cache is not None and Path(cache).exists():
        raw = json.load(open(cache))
    if raw is None:
        from gseapy import Msigdb
        raw = Msigdb().get_gmt(category="h.all", dbver=dbver)
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            json.dump(raw, open(cache, "w"))
    return {(k.replace("HALLMARK_", "") if strip_prefix else k): list(v) for k, v in raw.items()}


def hallmark_polar_enrichment(roi, *, r_max: float = 150, r_step: float = 50, include_inside: bool = False,
                              hallmark: Optional[Mapping[str, Sequence[str]]] = None, dbver: str = "2024.1.Hs",
                              cache: Optional[Union[str, Path]] = None, **kwargs) -> SignatureEnrichment:
    """:func:`signature_polar_enrichment` for all MSigDB Hallmark sets (30° sectors; ``q_global`` = BH over all
    sets x cells, the threshold of :func:`plot_significant_hallmarks`)."""
    if hallmark is None:
        hallmark = load_hallmark_gene_sets(dbver, cache)
    return signature_polar_enrichment(roi, dict(hallmark), r_max=r_max, r_step=r_step, include_inside=include_inside,
                                      **kwargs)


# ===================================================================================================== plots
def _plot_one(ax, res: SignatureEnrichment, name: str, vmax: float, q_key: str, title: Optional[str],
              hole_frac: float = 0.45, fontsize: float = 16):
    from matplotlib.patches import Circle
    from .polar_plot import plot_polar_heatmap
    sig = res.is_sig(name, q_key)
    r_max = res.r_edges[-1]
    ticks = res.r_edges[1:][::2] if res.n_r > 3 else res.r_edges[1:]
    _, mesh = plot_polar_heatmap(res.outside(res.log2e[name]), res.r_edges, res.a_edges, ax=ax, cmap="RdBu_r",
                                 vmin=-vmax, vmax=vmax, theta_sub=30, hole_frac=hole_frac, mask_color="0.85",
                                 highlight=res.outside(sig), radius_ticks=ticks,
                                 center_label=None if res.include_inside else "tumor", colorbar=False,
                                 title=title, fontsize=fontsize)
    if res.include_inside:          # the inside cell = the centre disc
        v = res.log2e[name][0]
        face = "0.85" if not np.isfinite(v) else mesh.cmap(mesh.norm(v))
        hole_r = 0.5 * hole_frac / (1 + hole_frac)
        ax.add_patch(Circle((0.5, 0.5), hole_r * 0.94, transform=ax.transAxes, facecolor=face, zorder=4,
                            edgecolor="black" if sig[0] else "0.4", linewidth=2.5 if sig[0] else 0.8,
                            clip_on=False))
        ax.text(0.5, 0.5, "tumor", transform=ax.transAxes, ha="center", va="center", fontsize=fontsize * 0.8,
                zorder=6)
    return mesh


def plot_signature_polar(res: SignatureEnrichment, names: Optional[Sequence[str]] = None, *, q_key: str = "q",
                         ncols: int = 4, vmax: Optional[float] = None, panel_size: float = 6.5,
                         suptitle: Optional[str] = None, wrap: int = 22):
    """Polar enrichment maps (colour = log2 E; black frame = q < fdr and enriched; grey = not tested).

    The centre is the tumor; with ``include_inside`` it is coloured as the inside cell (black ring = significant).
    Returns the figure.
    """
    names = list(res.names if names is None else names)
    if not names:
        print("nothing to plot")
        return None
    if vmax is None:
        v = np.concatenate([np.abs(res.log2e[n][res.valid_cell]) for n in names])
        vmax = max(float(np.ceil(np.nanpercentile(v, 98) * 4) / 4), 0.25)
    ncols = min(ncols, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols * 1.1, panel_size * nrows * 1.02 + 0.8),
                             subplot_kw={"projection": "polar"}, squeeze=False,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.45})
    for ax, n in zip(axes.ravel(), names):
        mesh = _plot_one(ax, res, n, vmax, q_key, "\n".join(textwrap.wrap(n.replace("_", " "), wrap)))
        ax.title.set_y(1.12)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.02, pad=0.1, shrink=0.8)
    cbar.set_label(r"$\log_2$ enrichment vs. tissue", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    if suptitle:
        fig.suptitle(suptitle, fontsize=22, weight="bold")
    return fig


def plot_significant_hallmarks(res: SignatureEnrichment, *, q_key: str = "q_global", by_category: bool = False,
                               ncols: int = 5, panel_size: float = 6.0, suptitle: Optional[str] = None):
    """Polar maps of the Hallmarks with at least one significantly enriched cell (default: BH across all
    signatures x cells), most significant cells first. ``by_category``: one figure per Hallmark process category.
    Returns the list of figures."""
    sig = res.significant_names(q_key)
    print(f"{len(sig)} / {len(res.names)} signatures with >= 1 significant cell ({q_key} < {res.fdr})")
    if not sig:
        return []
    v = np.concatenate([np.abs(res.log2e[n][res.valid_cell]) for n in res.names])
    vmax = max(float(np.ceil(np.nanpercentile(v, 98) * 4) / 4), 0.25)       # same scale in every figure
    r_txt = f"{'inside + ' if res.include_inside else ''}0–{res.r_edges[-1]:g} µm, {ANGLE_STEP}° sectors"
    if not by_category:
        return [plot_signature_polar(res, sig, q_key=q_key, ncols=ncols, vmax=vmax, panel_size=panel_size,
                                     suptitle=suptitle or f"Significant signatures ({r_txt}; {q_key} < {res.fdr})")]
    figs = []
    groups = {c: [n for n in sig if _CATEGORY_OF.get(n) == c] for c in HALLMARK_CATEGORY}
    groups["other"] = [n for n in sig if n not in _CATEGORY_OF]
    for c, ns in groups.items():
        if ns:
            figs.append(plot_signature_polar(res, ns, q_key=q_key, ncols=ncols, vmax=vmax, panel_size=panel_size,
                                             suptitle=f"{c} ({len(ns)}) — {r_txt}; {q_key} < {res.fdr}"))
    return figs


def signature_polar_histogram(roi, genes: Sequence[str], *, radius_bins=None, scorer: Optional[SignatureScorer] = None,
                              panel_genes: Optional[Sequence[str]] = None) -> np.ndarray:
    """Score-weighted polar histogram of one signature over the full radius range (like a gene's
    ``polar_counts_list`` entry; sums to 1 over the vectors inside the bins). Draw it with
    ``scomv.polar_plot.plot_polar_heatmap(A, radius_bins, np.arange(-180, 181, 30))``."""
    roi = as_polar_roi(roi)
    if scorer is None:
        scorer = build_signature_scorer(roi.subset_grid, panel_genes)
    if radius_bins is None:
        radius_bins = np.arange(-150, 310, 10)
    a_edges = np.arange(-180, 181, ANGLE_STEP)
    genes = [g for g in genes if g in scorer.gene_pos]
    if not genes:
        raise ValueError("none of the genes is in the panel")
    return roi.histogram(scorer.score(genes), radius_bins, a_edges)


def plot_signature_backprojection(res: SignatureEnrichment, name: str, *, q_key: str = "q", point_size: float = 70,
                                  outline_size: float = 25, figsize=(26, 12), cbar_fontsize: float = 30,
                                  cbar_tick_fontsize: float = 24, title_fontsize: float = 29):
    """Where on the tissue the polar enrichment of ``name`` comes from.

    Left: signature score of every grid square (tumor squares dark grey, boundary black). Right: the squares whose
    shortest vector falls in a significant polar cell, coloured by that cell's log2 E. Needs square positions
    (``PolarROI`` from a ``SCOMVPipeline``). ``cbar_fontsize`` / ``cbar_tick_fontsize`` / ``title_fontsize``:
    colour-bar label / tick and panel-title sizes. Returns the figure.
    """
    from matplotlib.colors import TwoSlopeNorm
    roi = res.roi
    if roi is None or roi.xy is None:
        raise ValueError("square positions are needed: run on a SCOMVPipeline (or a PolarROI with xy)")
    xy = roi.xy
    tissue = res.scorer.tissue
    inside = roi.inside if roi.inside is not None else np.zeros(len(xy), bool)
    score = res.scorer.score(res.gene_sets[name])
    sig = res.is_sig(name, q_key)
    hit = (res.vec_cell >= 0) & np.isin(res.vec_cell, np.flatnonzero(sig))
    cell_val = pd.Series(res.log2e[name][res.vec_cell[hit]]).groupby(res.vec_row[hit]).mean()
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    for ax in axes:
        ax.scatter(xy[tissue & ~inside, 0], xy[tissue & ~inside, 1], c="0.9", s=point_size, linewidths=0)
        ax.scatter(xy[tissue & inside, 0], xy[tissue & inside, 1], c="0.6", s=point_size, linewidths=0)
    m = tissue & (score > 0)
    if m.any():
        sc_ = axes[0].scatter(xy[m, 0], xy[m, 1], c=score[m], cmap="Reds", s=point_size, linewidths=0,
                              vmin=0, vmax=np.percentile(score[m], 99))
        cb = fig.colorbar(sc_, ax=axes[0], fraction=0.04)
        cb.set_label("Signature score", fontsize=cbar_fontsize)
        cb.ax.tick_params(labelsize=cbar_tick_fontsize)
    axes[0].set_title(f"{name}: score", fontsize=title_fontsize, weight="bold")
    if len(cell_val):
        vmax = max(float(np.abs(cell_val).max()), 1e-6)
        sc2 = axes[1].scatter(xy[cell_val.index, 0], xy[cell_val.index, 1], c=cell_val.values, cmap="RdBu_r",
                              norm=TwoSlopeNorm(0, -vmax, vmax), s=point_size, linewidths=0)
        cb2 = fig.colorbar(sc2, ax=axes[1], fraction=0.04)
        cb2.set_label(r"$\log_2 E$ of the polar cell", fontsize=cbar_fontsize)
        cb2.ax.tick_params(labelsize=cbar_tick_fontsize)
    axes[1].set_title(f"{name}: significant polar cells ({q_key} < {res.fdr})", fontsize=title_fontsize,
                      weight="bold")
    for ax in axes:
        if roi.outline is not None:
            ax.scatter(roi.outline[:, 0], roi.outline[:, 1], c="black", s=outline_size, linewidths=0)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    return fig
