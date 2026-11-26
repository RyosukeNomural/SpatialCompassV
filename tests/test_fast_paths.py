"""Optimized code paths give exactly the same results as the straightforward implementations."""
import numpy as np
import pandas as pd
import anndata as ad
import pytest

from scomv.preparation.scomv_calc_vector import (
    compute_min_vectors_polar,
    flatten_min_vectors,
    polar_hist_for_expression,
)
from scomv.preparation.io import save_preprocessed, load_preprocessed


def _min_vectors_dense(xy_list, outline_points, inside_points=None, invert_y=True, atol=1e-8):
    """Reference: dense N x M implementation (scomv <= 0.1.5)."""
    P = np.asarray(xy_list, dtype=float)
    O = np.asarray(outline_points, dtype=float)
    dx = P[:, 0:1] - O[None, :, 0]
    dy = P[:, 1:2] - O[None, :, 1]
    d2 = dx * dx + dy * dy
    tie = np.isclose(d2, d2.min(axis=1, keepdims=True), atol=atol)
    angles, radii = [], []
    for i in range(len(P)):
        dx_i, dy_i = dx[i, tie[i]], dy[i, tie[i]]
        if invert_y:
            dy_i = -dy_i
        angles.append(np.arctan2(dy_i, dx_i).tolist())
        radii.append(np.sqrt(dx_i * dx_i + dy_i * dy_i).tolist())
    df = pd.DataFrame({"angle": angles, "radii": radii}, index=xy_list)
    if inside_points:
        s = set(map(tuple, inside_points))
        m = df.index.map(lambda t: t in s)
        df.loc[m, "radii"] = df.loc[m, "radii"].apply(lambda rs: [-r for r in rs])
    return df


def _polar_hist_loop(expr, mv, radius_bins, angle_bins_deg, bin_um):
    """Reference: per-cell loop (scomv <= 0.1.5)."""
    A, R, W = [], [], []
    for i in range(len(expr)):
        e = int(expr[i])
        if e == 0:
            continue
        a, r = mv["angle"].iloc[i], mv["radii"].iloc[i]
        if len(a) == 0:
            continue
        for ang, rad in zip(a, r):
            A.append(ang)
            R.append(rad * bin_um)
            W.append(e / len(a))
    if not A:
        return None
    w = np.asarray(W, float)
    return np.histogram2d(R, np.degrees(A), bins=[radius_bins, angle_bins_deg], weights=w / w.sum())[0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_min_vectors_match_dense(seed):
    rng = np.random.default_rng(seed)
    xy = [tuple(v) for v in rng.integers(0, 200, size=(1500, 2)).tolist()]
    outline = [tuple(v) for v in rng.integers(0, 200, size=(150, 2)).tolist()]
    inside = xy[:300]
    a = _min_vectors_dense(xy, outline, inside)
    b = compute_min_vectors_polar(xy, outline, inside)
    assert a.index.equals(b.index)
    assert a["angle"].tolist() == b["angle"].tolist()
    assert a["radii"].tolist() == b["radii"].tolist()


def test_min_vectors_relative_tolerance_ties():
    # far points: np.isclose's rtol makes d2 = min + 1 a tie
    xy = [(x, 0) for x in range(0, 1500)]
    outline = [(5000, 0), (-3000, 1), (5001, 0)]
    a = _min_vectors_dense(xy, outline)
    b = compute_min_vectors_polar(xy, outline)
    assert a["angle"].tolist() == b["angle"].tolist()
    assert a["radii"].tolist() == b["radii"].tolist()


def test_polar_hist_matches_loop():
    rng = np.random.default_rng(3)
    xy = [tuple(v) for v in rng.integers(0, 60, size=(800, 2)).tolist()]
    outline = [tuple(v) for v in rng.integers(0, 60, size=(40, 2)).tolist()]
    mv = compute_min_vectors_polar(xy, outline, xy[:200])
    flat = flatten_min_vectors(mv)
    rb, ab = np.arange(-150, 310, 10), np.arange(-180, 181, 30)
    for _ in range(20):
        expr = rng.poisson(0.7, size=len(xy)).astype(float)
        ref = _polar_hist_loop(expr, mv, rb, ab, 10)
        new = polar_hist_for_expression(expr, flat, rb, ab, 10)
        assert np.array_equal(ref, new)
    assert polar_hist_for_expression(np.zeros(len(xy)), flat, rb, ab, 10) is None


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(4)
    n = 50
    obs = pd.DataFrame({"imagecol": rng.random(n), "imagerow": rng.random(n)}, index=[f"grid_{i + 1}" for i in range(n)])
    region = pd.arrays.IntervalArray.from_breaks(np.arange(-15.0, 31.0, 3.0), closed="right")
    obs["region"] = pd.arrays.IntervalArray(np.asarray(region)[rng.integers(0, len(region), n)])
    obs.loc[obs.index[:5], "region"] = np.nan
    obs["region_10"] = [str(v) for v in rng.integers(0, 9, n)]
    obs.loc[obs.index[:5], "region_10"] = np.nan
    g = ad.AnnData(X=rng.random((n, 4)), obs=obs, var=pd.DataFrame(index=list("ABCD")))
    g.uns["marker"] = rng.integers(0, 255, (5, 10, 3)).astype(np.uint8)
    g.uns["shotest"] = None
    idx = [(x, y) for y in range(5) for x in range(10)]
    sh = pd.DataFrame({"euclidean": rng.normal(size=n)}, index=idx)
    sh["euclidean_round"] = sh["euclidean"].round(0)
    sh["region"] = pd.cut(sh["euclidean"], bins=list(range(-15, 31, 3)))
    setattr(g, "shortest", sh)

    save_preprocessed(g, tmp_path / "g.h5ad")
    h = load_preprocessed(tmp_path / "g.h5ad")
    assert np.array_equal(g.X, h.X)
    assert g.obs.equals(h.obs) and list(g.obs.dtypes.astype(str)) == list(h.obs.dtypes.astype(str))
    assert np.array_equal(g.uns["marker"], h.uns["marker"]) and h.uns["shotest"] is None
    assert g.shortest.equals(h.shortest) and g.shortest.index.equals(h.shortest.index)
    assert list(g.shortest.dtypes.astype(str)) == list(h.shortest.dtypes.astype(str))


def test_label_solid_regions_many_components():
    # 300 separate 3x3 blobs: raw ids 256 must not be lost (uint8 overflow)
    import cv2
    from scomv.tumor_solid import label_solid_regions
    img = np.zeros((40, 300 * 2 + 10, 3), np.uint8)
    for k in range(300):
        img[5:8, 2 * k: 2 * k + 1] = 255
    img[20:30, 0:10] = 255  # one large component
    g = ad.AnnData(np.zeros((1, 1)))
    g.uns["marker_delineation"] = img
    labels, stats, _ = label_solid_regions(g, min_area=3)
    assert labels.max() == 301 and (stats[1:, cv2.CC_STAT_AREA] >= 3).all()


def test_calculate_distance_image_is_pixel_exact():
    # N_COL = 987: 987/100*100 != 987 in floating point
    from scomv.preparation.skny_calc_distance import calculate_distance
    n_row, n_col = 30, 987
    rng = np.random.default_rng(5)
    n = n_row * n_col
    X = np.zeros((n, 1))
    blob = np.zeros((n_col, n_row), bool)
    blob[100:300, 5:25] = True
    blob[600:900, 8:20] = True
    X[blob.ravel(), 0] = rng.integers(1, 5, blob.sum())
    obs = pd.DataFrame({"imagecol": np.repeat(np.arange(n_col) * 10 + 5.0, n_row),
                        "imagerow": np.tile(np.arange(n_row) * 10 + 5.0, n_col)},
                       index=[f"grid_{i + 1}" for i in range(n)])
    g = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["CDH1"]))
    g.uns["grid_xedges"] = np.arange(n_col + 1) * 10.0
    g.uns["grid_yedges"] = np.arange(n_row + 1) * 10.0
    g = calculate_distance(g, pos_marker_ls=["CDH1"])
    yellow = (g.uns["marker"] == np.array([36, 231, 253], np.uint8)).all(axis=2)
    assert g.uns["marker"].shape[:2] == (n_row, n_col)
    assert np.array_equal(yellow, blob.T)


def test_plot_tumor_solids_draws_sharp_ids():
    # two blobs -> two solids; boxes / ids are matplotlib artists, the mask image has no burnt-in text
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scomv.preparation.skny_calc_distance import calculate_distance
    from scomv.tumor_solid import extract_tumor_solids, label_solid_regions, plot_tumor_solids
    n_row, n_col = 40, 120
    rng = np.random.default_rng(1)
    blob = np.zeros((n_col, n_row), bool)
    blob[10:40, 8:30] = True
    blob[70:110, 10:32] = True
    X = np.zeros((n_row * n_col, 1))
    X[blob.ravel(), 0] = rng.integers(1, 5, blob.sum())
    obs = pd.DataFrame({"imagecol": np.repeat(np.arange(n_col) * 10 + 5.0, n_row),
                        "imagerow": np.tile(np.arange(n_row) * 10 + 5.0, n_col)},
                       index=[f"grid_{i + 1}" for i in range(n_row * n_col)])
    g = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["CDH1"]))
    g.uns["grid_xedges"] = np.arange(n_col + 1) * 10.0
    g.uns["grid_yedges"] = np.arange(n_row + 1) * 10.0
    g = calculate_distance(g, pos_marker_ls=["CDH1"])
    solid = extract_tumor_solids(g, min_area=50, verbose=False)
    _, stats, _ = label_solid_regions(g, min_area=50)
    assert np.array_equal(solid.uns["tumor_mask_image"], g.uns["marker_delineation"])
    assert np.array_equal(solid.uns["solid_bbox"], stats[:, :4])
    ax = plot_tumor_solids(solid)
    assert [t.get_text() for t in ax.texts] == [str(i) for i in range(1, stats.shape[0])]
    assert len(ax.patches) == stats.shape[0] - 1
    plt.close(ax.figure)


def test_other_solid_squares_include_lumens():
    # solid 2 is a ring with a 3x3 lumen; solid 1 is a separate block. Masking for solid 1 must drop
    # the ring and its lumen, and never solid 1 itself.
    from scomv.tumor_solid import _other_solid_squares
    lab = np.zeros((12, 20), int)
    lab[2:9, 2:9] = 2
    lab[4:7, 4:7] = 0          # lumen of solid 2
    lab[3:6, 13:17] = 1
    idx = [(x, y) for y in range(lab.shape[0]) for x in range(lab.shape[1])]
    solid = ad.AnnData(np.zeros((1, 1)))
    solid.shortest = pd.DataFrame({"solid": [lab[y, x] if lab[y, x] else np.nan for x, y in idx]},
                                  index=pd.Index(idx, tupleize_cols=False))
    got = _other_solid_squares(solid, 1)
    ring = {(x, y) for x, y in idx if lab[y, x] == 2}
    lumen = {(x, y) for y in range(4, 7) for x in range(4, 7)}
    assert got == ring | lumen
    assert not got & {(x, y) for x, y in idx if lab[y, x] == 1}
    # for solid 2 itself: only solid 1 is "other"
    assert _other_solid_squares(solid, 2) == {(x, y) for x, y in idx if lab[y, x] == 1}


def _random_polar_hists(n_genes, seed=0, shape=(46, 12)):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_genes):
        h = rng.random(shape) * (rng.random(shape) < 0.3)
        h[:3] = 0                                     # polar cells empty for every gene
        out.append(h / h.sum())
    return out


def test_minas_similarity_matches_loop():
    from scomv.gene_pipeline import SCOMVPipeline
    hists = _random_polar_hists(60)
    labels = [f"g{i}" for i in range(60)]
    ref = np.zeros((60, 60))
    for i in range(60):
        for j in range(i, 60):
            ref[i, j] = ref[j, i] = float(np.minimum(hists[i], hists[j]).sum())
    sim = SCOMVPipeline.minas_similarity_matrix(hists, labels, make_distance=False, fill_diagonal_zero=False)
    np.testing.assert_allclose(sim.values, ref, rtol=0, atol=1e-14)
    dist = SCOMVPipeline.minas_similarity_matrix(hists, labels)
    ref_d = 1.0 - ref
    np.fill_diagonal(ref_d, 0.0)
    np.testing.assert_allclose(dist.values, ref_d, rtol=0, atol=1e-14)
    assert list(dist.index) == labels and list(dist.columns) == labels


def test_pcoa_top_matches_skbio_full():
    import skbio
    from skbio.stats.ordination import pcoa
    from scomv.gene_pipeline import SCOMVPipeline
    hists = _random_polar_hists(300, seed=1)
    labels = [f"g{i}" for i in range(300)]
    dist = SCOMVPipeline.minas_similarity_matrix(hists, labels)
    full = pcoa(skbio.DistanceMatrix(dist.values, ids=labels))
    res, coords, expl = SCOMVPipeline.run_pcoa_from_distance_df(dist, dimensions=10)
    np.testing.assert_allclose(res.eigvals.values, full.eigvals.values[:10], rtol=1e-9)
    for c in coords.columns:
        a, b = coords[c].values, full.samples[c].values
        np.testing.assert_allclose(a * np.sign(a @ b), b, rtol=0, atol=1e-9)
    # small matrices still go through skbio's full decomposition
    res2, coords2, expl2 = SCOMVPipeline.run_pcoa_from_distance_df(dist)
    pd.testing.assert_frame_equal(coords2, full.samples)
    pd.testing.assert_series_equal(expl2, full.proportion_explained)
