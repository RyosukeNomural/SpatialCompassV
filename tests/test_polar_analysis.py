"""signature / PGI / gene-clustering helpers on a small synthetic ROI."""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from scomv.preparation.scomv_calc_vector import compute_min_vectors_polar, flatten_min_vectors, polar_hist_for_expression
from scomv.signature import PolarROI, polar_weight_matrix, signature_polar_enrichment, signature_polar_histogram
from scomv.pgi import compute_pgi


def _roi(n=40, seed=0):
    """n x n grid squares around a disc-shaped tumor; Poisson counts of 30 genes."""
    rng = np.random.default_rng(seed)
    xy = [(i, j) for i in range(n) for j in range(n)]
    c = (n - 1) / 2
    d = np.array([np.hypot(i - c, j - c) for i, j in xy])
    outline = [p for p, dd in zip(xy, d) if 7.5 <= dd < 8.5]
    inside = [p for p, dd in zip(xy, d) if dd < 7.5]
    mv = compute_min_vectors_polar(xy, outline, inside, invert_y=True, make_inside_negative=True)
    genes = [f"G{k}" for k in range(30)] + ["CDH1", "REF"]
    lam = np.full((len(xy), len(genes)), 1.0)
    lam[:, -2] = np.where(d < 8.5, 5.0, 0.05)                         # epithelial marker inside
    ang = np.array([np.arctan2(-(j - c), i - c) for i, j in xy])
    lam[:, -1] = np.where((d > 8.5) & (np.cos(ang) > 0), 4.0, 0.3)     # reference gene: one side only
    lam[:, 0] = np.where((d > 8.5) & (np.cos(ang) < 0), 4.0, 0.3)      # G0: the other side (in the gaps)
    X = sp.csr_matrix(rng.poisson(lam).astype(float))
    grid = ad.AnnData(X=X, var=pd.DataFrame(index=genes))
    return PolarROI(grid, mv, 10), X, genes


def test_polar_weight_matrix_matches_polar_hist():
    roi, X, genes = _roi()
    rb, ab = np.arange(-150, 310, 10), np.arange(-180, 181, 30)
    M, has_vec = polar_weight_matrix(roi.min_vector_df, rb, ab, 10)
    flat = flatten_min_vectors(roi.min_vector_df)
    for k in (0, 5, 31):
        e = X[:, k].toarray().ravel()
        ref = polar_hist_for_expression(e, flat, rb, ab, 10)
        new = np.asarray(M.T @ e).reshape(len(rb) - 1, -1) / e[has_vec].sum()
        np.testing.assert_allclose(new, ref, atol=1e-12)


def test_signature_enrichment_runs_with_inside():
    roi, X, genes = _roi()
    res = signature_polar_enrichment(roi, {"sig": genes[:6]}, r_max=100, r_step=50, include_inside=True,
                                     n_perm=200, min_set_genes=3, verbose=False)
    assert res.log2e["sig"].shape == (1 + 2 * 12,)
    assert res.cell_labels()[0] == "inside"
    assert np.all((res.p["sig"][res.valid_cell] > 0) & (res.p["sig"][res.valid_cell] <= 1))
    assert set(res.summary().columns) >= {"n_sig_cells", "inside_log2E"}
    A = signature_polar_histogram(roi, genes[:6])
    assert A.shape == (45, 12) and np.isclose(A.sum(), 1.0)


def test_pgi_reference_gene():
    roi, X, genes = _roi()
    res = compute_pgi(roi, ref_gene="REF", epi_genes=("CDH1",), band_um=(0, 100), min_band_counts=0, verbose=False)
    t = res.table
    assert t.loc["REF", "PGI"] < 0 < t.loc["G0", "PGI"]          # G0 sits where REF is absent
    assert t.loc["G0", "rho_REF"] < 0
