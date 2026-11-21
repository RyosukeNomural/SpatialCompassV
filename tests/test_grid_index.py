"""scomv.preparation.grid_index agrees with stlearn's grid layout (histogram2d edges, column-major names)."""
import numpy as np
import pandas as pd
import anndata as ad
import pytest

from scomv.preparation.grid_index import (
    grid_index_names,
    grid_name_offset,
    grid_square_ij,
    points_to_grid_ij,
)


def _fake_grid(x, y, n_col, n_row, offset):
    """Mimic stlearn.tl.cci.grid: edges from histogram2d of int coordinates, squares named i * n_row + j (+offset)."""
    xi, yi = np.asarray(x).astype(int), np.asarray(y).astype(int)
    _, xe, ye = np.histogram2d(xi, yi, bins=[n_col, n_row])
    names, cx, cy = [], [], []
    for i in range(n_col):
        for j in range(n_row):
            names.append(f"grid_{i * n_row + j + offset}")
            cx.append((xe[i] + xe[i + 1]) / 2)
            cy.append((ye[j] + ye[j + 1]) / 2)
    g = ad.AnnData(X=np.zeros((len(names), 1)), obs=pd.DataFrame({"imagecol": cx, "imagerow": cy}, index=names))
    g.uns["grid_xedges"], g.uns["grid_yedges"] = xe, ye
    return g, xe, ye


@pytest.mark.parametrize("offset", [0, 1])
def test_names_and_points(offset):
    rng = np.random.default_rng(0)
    x = rng.uniform(3.7, 1003.7, 5000)  # origin not a multiple of 10, width not exactly 10
    y = rng.uniform(1.2, 701.9, 5000)
    n_col, n_row = 100, 70
    g, xe, ye = _fake_grid(x, y, n_col, n_row, offset)

    assert grid_name_offset(g) == offset
    assert grid_index_names(g)[:2] == [f"grid_{offset}", f"grid_{offset + 1}"]

    i, j = grid_square_ij(g)
    assert np.allclose((xe[i] + xe[i + 1]) / 2, g.obs["imagecol"])
    assert np.allclose((ye[j] + ye[j + 1]) / 2, g.obs["imagerow"])

    # points: same square as np.histogram2d on the integer-truncated coordinates (stlearn's rule)
    pi, pj = points_to_grid_ij(x, y, g)
    for k in rng.choice(len(x), 300, replace=False):
        h, _, _ = np.histogram2d([int(x[k])], [int(y[k])], bins=[xe, ye])
        ii, jj = np.argwhere(h == 1)[0]
        assert (pi[k], pj[k]) == (ii, jj)
