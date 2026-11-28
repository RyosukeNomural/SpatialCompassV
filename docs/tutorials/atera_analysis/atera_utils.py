"""Helpers for the Atera (Xenium WTA, ~18k genes) SCOMV analysis.

The whole-transcriptome panel makes two steps of the standard workflow
memory-bound, so they are re-implemented here with identical semantics:

* ``sparse_grid``   - same binning as ``stlearn.tl.cci.grid`` but with a sparse
  expression matrix (stlearn allocates a dense n_squares x n_genes float64
  array: 585k x 17.9k = 84 GB for this sample).
* ``solid_min_vectors`` - same output as ``scomv.tumor_solid.compute_solid_min_vectors``
  (vectorised binning instead of ``DataFrame.apply``), returning the row mask so
  that the sparse all-gene matrix can be subset in the same order.
  With ``mask_other_solids=True`` the squares of the other solids are left out of the ROI.
* ``gene_uniformity_matrix`` - same numbers as
  ``scomv.tumor_solid.compute_gene_uniformity`` for one solid, for all genes at
  once (sparse matrix products instead of a per-gene / per-bin Python loop).

Each function is checked against the package implementation in the notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

from scomv.preparation.scomv_calc_vector import compute_min_vectors_polar, flatten_min_vectors
from scomv.tumor_solid import _other_solid_squares


# ---------------------------------------------------------------- grid
def sparse_grid(adata, n_row: int, n_col: int):
    """stlearn ``tl.cci.grid`` (same bins, same edge rules, same obs) with sparse X.

    Returns (grid AnnData with an empty-gene X placeholder, sparse CSR expression
    matrix aligned to grid.obs_names, var_names).
    """
    xs = adata.obs["imagecol"].values.astype(int)
    ys = adata.obs["imagerow"].values.astype(int)
    grid_counts, xedges, yedges = np.histogram2d(xs, ys, bins=[n_col, n_row])
    grid_counts = grid_counts.astype(int)
    xedges = xedges.astype(float)
    yedges = yedges.astype(float)

    i0 = np.clip(np.searchsorted(xedges, xs, side="right") - 1, 0, n_col - 1)
    j0 = np.clip(np.searchsorted(yedges, ys, side="right") - 1, 0, n_row - 1)
    rows_cell, rows_bin = [], []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i = i0 + di
            j = j0 + dj
            ok = (i >= 0) & (i < n_col) & (j >= 0) & (j < n_row)
            ic, jc = np.clip(i, 0, n_col - 1), np.clip(j, 0, n_row - 1)
            xl, xr, yd, yu = xedges[ic], xedges[ic + 1], yedges[jc], yedges[jc + 1]
            top_left = (ic != n_col - 1) & (jc == n_row - 1)
            bottom_right = (ic == n_col - 1) & (jc != n_row - 1)
            avg = ~(top_left | bottom_right)
            inside = (
                (top_left & (xs >= xl) & (xs < xr) & (ys <= yu) & (ys > yd))
                | (bottom_right & (xs > xl) & (xs <= xr) & (ys < yu) & (ys >= yd))
                | (avg & (xs >= xl) & (xs < xr) & (ys < yu) & (ys >= yd))
            ) & ok
            rows_cell.append(np.nonzero(inside)[0])
            rows_bin.append((ic * n_row + jc)[inside])
    rc, rb = np.concatenate(rows_cell), np.concatenate(rows_bin)
    n_sq = n_row * n_col
    A = sp.csr_matrix((np.ones(len(rc), np.float32), (rb, rc)), shape=(n_sq, adata.n_obs))
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    G = (A @ X.astype(np.float32)).tocsr()
    n_cells = np.asarray(A.sum(axis=1)).ravel().astype(np.int64)

    ii, jj = np.divmod(np.arange(n_sq), n_row)
    coords = np.c_[(xedges[ii] + xedges[ii + 1]) / 2, (yedges[jj] + yedges[jj + 1]) / 2]
    keep = n_cells > 0
    obs = pd.DataFrame({"imagecol": coords[keep, 0], "imagerow": coords[keep, 1],
                        "n_cells": n_cells[keep]},
                       index=[f"grid_{n}" for n in np.nonzero(keep)[0]])
    grid = ad.AnnData(X=np.zeros((keep.sum(), 0), np.float32), obs=obs)
    grid.obsm["spatial"] = coords[keep]
    grid.uns["spatial"] = adata.uns.get("spatial", {})
    grid.uns["grid_counts"] = grid_counts
    grid.uns["grid_xedges"] = xedges
    grid.uns["grid_yedges"] = yedges
    return grid, G[keep], list(adata.var_names)


def with_genes(grid, G, var_names, genes):
    """Copy of the light grid carrying the listed genes as a dense X."""
    idx = [var_names.index(g) for g in genes]
    g = ad.AnnData(X=np.asarray(G[:, idx].todense(), dtype=np.float64), obs=grid.obs.copy(),
                   var=pd.DataFrame(index=list(genes)), uns=dict(grid.uns), obsm=dict(grid.obsm))
    return g


# ---------------------------------------------------------------- min vectors
def solid_min_vectors(grid, solid, solid_id, bin_size=10, margin_bins=15, mask_other_solids=False):
    """Same as scomv.tumor_solid.compute_solid_min_vectors; returns (row_mask, min_vector_df, roi).

    ``mask_other_solids=True`` drops the grid squares of every other solid (and the lumens
    they enclose) from the ROI, exactly like the package option of the same name.
    """
    df_sh = getattr(solid, "shortest")
    df_int = df_sh[(df_sh["solid"] == solid_id) & df_sh["euclidean"].notna()]
    if len(df_int) == 0:
        raise ValueError(f"solid_id={solid_id}: no interior grid squares found")
    xb = np.array([t[0] for t in df_int.index])
    yb = np.array([t[1] for t in df_int.index])
    xmin_b, xmax_b = xb.min() - margin_bins, xb.max() + margin_bins
    ymin_b, ymax_b = yb.min() - margin_bins, yb.max() + margin_bins

    # (column, row) from the square names: the same index as solid.shortest (scomv.preparation.grid_index)
    from scomv.preparation.grid_index import grid_square_ij
    gx, gy = grid_square_ij(grid)
    mask = (gx >= xmin_b) & (gx <= xmax_b) & (gy >= ymin_b) & (gy <= ymax_b)
    if mask_other_solids:
        mask &= ~_in_squares(gx, gy, _other_solid_squares(solid, solid_id))
    if not mask.any():
        raise ValueError(f"solid_id={solid_id}: empty subset_grid")
    xy_list = list(zip(gx[mask].tolist(), gy[mask].tolist()))

    e = df_int["euclidean"]
    outline = list(df_int.index[(e >= -1.0) & (e <= 0.0)])
    inside = list(df_int.index[e < -1.0])
    if len(outline) == 0:
        outline, inside = list(df_int.index), []
    mv = compute_min_vectors_polar(xy_list=xy_list, outline_points=outline, inside_points=inside,
                                   invert_y=True, make_inside_negative=True)
    roi = ((xmin_b - 1) * bin_size, (xmax_b + 2) * bin_size,
           (ymin_b - 1) * bin_size, (ymax_b + 2) * bin_size)
    return mask, mv, roi


def _in_squares(gx, gy, squares):
    """Boolean array: is (gx[i], gy[i]) one of ``squares`` (a set of (x_bin, y_bin))?"""
    if not squares:
        return np.zeros(len(gx), dtype=bool)
    sq = np.array(list(squares), dtype=np.int64)
    w = int(max(sq[:, 0].max(), np.max(gx)) + 1)
    keys = np.asarray(gy, dtype=np.int64) * w + np.asarray(gx, dtype=np.int64)
    return np.isin(keys, sq[:, 1] * w + sq[:, 0])


def cells_outside_other_solids(x, y, grid, solid, solid_id):
    """Boolean array over cells: True unless the cell's grid square lies inside another solid.

    Same squares as ``solid_min_vectors(..., mask_other_solids=True)`` and the cell filter of
    ``scomv.tumor_solid.compute_per_solid_polar_counts(..., mask_other_solids=True)``.
    """
    from scomv.preparation.grid_index import points_to_grid_ij
    gi, gj = points_to_grid_ij(np.asarray(x), np.asarray(y), grid)
    return ~_in_squares(gi, gj, _other_solid_squares(solid, solid_id))


# ---------------------------------------------------------------- gene uniformity
def _hist_bin_index(x, edges):
    """Bin index exactly as np.histogram2d (right edge of the last bin inclusive); -1 = outside."""
    k = np.searchsorted(edges, x, side="right") - 1
    k[x == edges[-1]] = len(edges) - 2
    k[(x < edges[0]) | (x > edges[-1])] = -1
    return k


def polar_design(mv, radius_bins, angle_bins_deg, bin_size_um=10):
    """Sparse (n_rows x n_radius*n_angle) matrix of tie weights 1/n_vec, plus has_vec mask."""
    n_vec, vec_row, angle, radii = flatten_min_vectors(mv)
    r = radii * bin_size_um
    a = np.degrees(angle)
    ri, ai = _hist_bin_index(r, np.asarray(radius_bins, float)), _hist_bin_index(a, np.asarray(angle_bins_deg, float))
    ok = (ri >= 0) & (ai >= 0)
    n_a = len(angle_bins_deg) - 1
    n_cells_hist = (len(radius_bins) - 1) * n_a
    w = 1.0 / n_vec[vec_row[ok]]
    M = sp.csr_matrix((w, (vec_row[ok], ri[ok] * n_a + ai[ok])), shape=(len(n_vec), n_cells_hist))
    return M, n_vec > 0


def gene_polar_counts(Xs, mv, radius_bins=None, angle_bins_deg=None, bin_size_um=10):
    """All-gene normalised polar histograms for one ROI (same as
    tumor_solid._gene_weighted_polar_samples + np.histogram2d(weights=w/w.sum())).

    Returns (A: genes x n_radius x n_angle, n_expr_rows per gene)."""
    if radius_bins is None:
        radius_bins = np.arange(-150, 310, 10)
    if angle_bins_deg is None:
        angle_bins_deg = np.arange(-180, 181, 30)
    M, has_vec = polar_design(mv, radius_bins, angle_bins_deg, bin_size_um)
    Xp = Xs.copy().tocsr().astype(np.float64)
    Xp.data[Xp.data < 0] = 0  # expression <= 0 is skipped in the reference
    Xp.eliminate_zeros()
    raw = np.asarray((Xp.T @ M).todense())                       # genes x cells
    total = np.asarray(Xp[has_vec].sum(axis=0)).ravel()          # sum of all sample weights
    n_expr = np.asarray((Xp > 0).sum(axis=0)).ravel()            # expressing grid squares
    with np.errstate(invalid="ignore", divide="ignore"):
        A = raw / total[:, None]
    A[total == 0] = np.nan
    return A.reshape(len(total), len(radius_bins) - 1, len(angle_bins_deg) - 1), n_expr, total


def uniformity_from_counts(A, n_cells, dist_bins, n_ref):
    """Vectorised version of the entropy x (1 - exp(-n_bin/n_ref)) score; A: genes x r x a."""
    out = {}
    n_a = A.shape[2]
    for name, b in dist_bins.items():
        s = A[:, b, :]
        tot = s.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = s / tot[:, None] + 1e-12
            p = p / p.sum(axis=1, keepdims=True)
            H = -(p * np.log(p)).sum(axis=1) / np.log(n_a)
        w = 1.0 - np.exp(-(tot * n_cells) / n_ref)
        u = H * w
        u[~(tot > 0)] = 0.0
        out[name] = u
    return pd.DataFrame(out)


# ---------------------------------------------------------------- gene x gene MINAS distance + PCoA
def minas_distance(A):
    """1 - sum(min(A_i, A_j)) for all gene pairs (same as scomv minas_similarity_matrix with
    make_distance=True, fill_diagonal_zero=True); A: genes x bins (flattened histograms).
    numba-parallel over rows; float64 accumulation."""
    import numba

    @numba.njit(parallel=True, cache=False)
    def _kernel(X):
        n, m = X.shape
        S = np.empty((n, n))
        for i in numba.prange(n):
            for j in range(i, n):
                s = 0.0
                for k in range(m):
                    a = X[i, k]; b = X[j, k]
                    s += a if a < b else b
                S[i, j] = s
                S[j, i] = s
        return S

    X = np.ascontiguousarray(np.nan_to_num(A.reshape(len(A), -1), nan=0.0), dtype=np.float64)
    keep = X.any(axis=0)                      # bins that are zero for every gene contribute nothing
    S = _kernel(np.ascontiguousarray(X[:, keep]))
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D


def pcoa_top(D, k=10):
    """Classical PCoA (as skbio.stats.ordination.pcoa, method='eigh') restricted to the top-k axes.
    Returns (coords n x k, eigenvalues k, proportion_explained k)."""
    from scipy.sparse.linalg import eigsh
    n = len(D)
    B = D.astype(np.float64) ** 2
    B *= -0.5
    B -= B.mean(axis=0, keepdims=True)
    B -= B.mean(axis=1, keepdims=True)
    total = np.trace(B)
    vals, vecs = eigsh(B, k=k, which="LA")
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    coords = vecs * np.sqrt(np.clip(vals, 0, None))
    return coords, vals, vals / total
