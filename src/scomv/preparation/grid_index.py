"""
One grid-square index for everything.

``stlearn.tl.cci.grid`` bins cells into ``n_col x n_row`` squares whose edges
are ``np.histogram2d`` edges of the *integer-truncated* cell coordinates
(``grid.uns["grid_xedges"/"grid_yedges"]``). The edges start at the minimum
coordinate (not 0) and the square width is not exactly ``bin_size`` (e.g.
10.0013 µm on the breast Xenium tutorial data). Squares are named
``grid_{i * n_row + j}`` (column ``i``, row ``j``); current stlearn numbers
them from 0, older versions from 1.

Every SCOMV step that needs a square index (the SKNY tumour image and
``grid.shortest``, the solid label image, the grid squares of an ROI / a
solid, and the square a cell falls in) must use the same ``(i, j)``:

* grid squares -> from their names (:func:`grid_square_ij`), exact;
* points (cells) -> with stlearn's own binning rule (:func:`points_to_grid_ij`).

Deriving the index from ``coordinate // bin_size`` instead is off by one square
whenever the grid origin is not a multiple of ``bin_size`` or the accumulated
width difference reaches half a square.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def grid_shape(grid) -> Tuple[int, int]:
    """(n_row, n_col) of the stlearn grid."""
    return len(grid.uns["grid_yedges"]) - 1, len(grid.uns["grid_xedges"]) - 1


def _name_numbers(obs_names) -> np.ndarray:
    return np.array([int(str(n).rsplit("_", 1)[1]) for n in obs_names], dtype=np.int64)


def grid_name_offset(grid, n_check: int = 2000) -> int:
    """
    Number of the first square in ``grid.obs_names`` (0 or 1).

    Decided by checking which numbering puts the square centres given by the
    edges at ``grid.obs[["imagecol", "imagerow"]]``.
    """
    n_row, n_col = grid_shape(grid)
    xe = np.asarray(grid.uns["grid_xedges"], dtype=float)
    ye = np.asarray(grid.uns["grid_yedges"], dtype=float)
    k = min(n_check, grid.n_obs)
    num = _name_numbers(grid.obs_names[:k])
    cx = grid.obs["imagecol"].to_numpy(dtype=float)[:k]
    cy = grid.obs["imagerow"].to_numpy(dtype=float)[:k]
    best, best_hit = 0, -1.0
    for off in (0, 1):
        n = num - off
        i, j = n // n_row, n % n_row
        ok = (n >= 0) & (i < n_col)
        if not ok.any():
            continue
        hit = np.mean(np.isclose((xe[i[ok]] + xe[i[ok] + 1]) / 2, cx[ok])
                      & np.isclose((ye[j[ok]] + ye[j[ok] + 1]) / 2, cy[ok]))
        if hit > best_hit:
            best, best_hit = off, hit
    return best


def grid_index_names(grid) -> list:
    """Names of all ``n_row * n_col`` squares in column-major order (i * n_row + j)."""
    n_row, n_col = grid_shape(grid)
    off = grid_name_offset(grid)
    return [f"grid_{k + off}" for k in range(n_row * n_col)]


def grid_square_ij(grid, obs_names=None, offset=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    (i, j) = (column, row) of grid squares, from their names.

    ``obs_names`` defaults to ``grid.obs_names``; pass a subset's names (e.g.
    ``subset_grid.obs_names``) to index a subset with the full grid's shape.
    """
    n_row, _ = grid_shape(grid)
    if offset is None:
        offset = grid_name_offset(grid)
    n = _name_numbers(grid.obs_names if obs_names is None else obs_names) - offset
    return n // n_row, n % n_row


def points_to_grid_ij(x, y, grid) -> Tuple[np.ndarray, np.ndarray]:
    """
    (i, j) of the grid square each point (e.g. a cell centroid) falls in,
    with stlearn's binning rule: integer-truncated coordinates, the grid's
    own edges, the last edge belonging to the last square.
    """
    n_row, n_col = grid_shape(grid)
    xe = np.asarray(grid.uns["grid_xedges"], dtype=float)
    ye = np.asarray(grid.uns["grid_yedges"], dtype=float)
    xs = np.asarray(x, dtype=float).astype(int)
    ys = np.asarray(y, dtype=float).astype(int)
    i = np.clip(np.searchsorted(xe, xs, side="right") - 1, 0, n_col - 1)
    j = np.clip(np.searchsorted(ye, ys, side="right") - 1, 0, n_row - 1)
    return i, j
