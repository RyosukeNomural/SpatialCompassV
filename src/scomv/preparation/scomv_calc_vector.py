import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from scipy.spatial import cKDTree


def compute_min_vectors_polar(
    xy_list: List[Tuple[int, int]],
    outline_points: List[Tuple[int, int]],
    inside_points: Optional[List[Tuple[int, int]]] = None,
    invert_y: bool = True,
    make_inside_negative: bool = True,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """
    For each point in xy_list, find all shortest vectors to outline_points
    (ties allowed), convert to polar (angle, radii). Optionally make radii
    negative for inside_points.

    Ties follow ``np.isclose(d2, min_d2, atol=atol)`` (default ``rtol=1e-5``) on
    squared distances, and tied outline points are returned in ``outline_points``
    order. A KD-tree restricts the candidates, so memory is O(N + M) instead of
    the O(N x M) dense distance matrix of the earlier implementation (results are
    identical).

    Returns
    -------
    df : pd.DataFrame with index=xy_list, columns=["angle","radii"]
         each cell is a list (ties -> multiple angles/radii)
    """

    # --- arrays ---
    P = np.asarray(xy_list, dtype=float)          # (N,2) [std_x, std_y]
    O = np.asarray(outline_points, dtype=float)   # (M,2) [x_location, y_location]

    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("xy_list must be list of (x,y)")
    if O.ndim != 2 or O.shape[1] != 2:
        raise ValueError("outline_points must be list of (x,y)")
    if len(O) == 0:
        raise ValueError("outline_points is empty")

    rtol = 1e-5  # np.isclose default
    tree = cKDTree(O)

    # --- candidates: every outline point that could satisfy the tie rule ---
    # nearest distance from the tree, then a generous search radius; the exact
    # tie rule is re-applied below on squared distances computed exactly as
    # before (dx*dx + dy*dy), so the margin only affects speed.
    d_nn, _ = tree.query(P, k=1)
    r = np.sqrt(d_nn * d_nn * (1.0 + 4.0 * rtol) + 4.0 * atol) * (1.0 + 1e-9) + 1e-9
    cand = tree.query_ball_point(P, r, return_sorted=True)

    lens = np.fromiter((len(c) for c in cand), dtype=np.int64, count=len(cand))
    row = np.repeat(np.arange(len(P)), lens)
    col = np.fromiter((j for c in cand for j in c), dtype=np.int64, count=int(lens.sum()))

    dx = P[row, 0] - O[col, 0]
    dy = P[row, 1] - O[col, 1]
    d2 = dx * dx + dy * dy

    # per point minimal squared distance (over the candidates, which contain the argmin)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    min_d2 = np.minimum.reduceat(d2, starts)
    keep = np.isclose(d2, min_d2[row], atol=atol)

    dx, dy, row = dx[keep], dy[keep], row[keep]
    if invert_y:
        dy = -dy  # Handle y-axis inversion for Xenium coordinates

    angles = np.arctan2(dy, dx)
    radii = np.sqrt(dx * dx + dy * dy)

    split_at = np.cumsum(np.bincount(row, minlength=len(P)))[:-1]
    all_angles = [a.tolist() for a in np.split(angles, split_at)]
    all_radiis = [r_.tolist() for r_ in np.split(radii, split_at)]

    df = pd.DataFrame({"angle": all_angles, "radii": all_radiis}, index=xy_list)

    # --- make inside points negative radii ---
    if make_inside_negative and inside_points is not None and len(inside_points) > 0:
        inside_set = set(map(tuple, inside_points))
        mask = df.index.map(lambda t: t in inside_set)
        df.loc[mask, "radii"] = df.loc[mask, "radii"].apply(lambda rs: [-r for r in rs])

    return df


def flatten_min_vectors(min_vector_df: pd.DataFrame):
    """
    Flatten the list-valued ``min_vector_df`` once, in row order.

    Returns
    -------
    n_vec : (N,) int array, number of (tied) vectors per row
    vec_row : (V,) row index of each vector
    angle : (V,) angle [rad]
    radii : (V,) radius [bins]
    """
    angle_lists = min_vector_df["angle"].tolist()
    radii_lists = min_vector_df["radii"].tolist()
    n_vec = np.fromiter((len(a) for a in angle_lists), dtype=np.int64, count=len(angle_lists))
    vec_row = np.repeat(np.arange(len(angle_lists)), n_vec)
    total = int(n_vec.sum())
    angle = np.fromiter((v for a in angle_lists for v in a), dtype=float, count=total)
    radii = np.fromiter((v for r in radii_lists for v in r), dtype=float, count=total)
    return n_vec, vec_row, angle, radii


def polar_hist_for_expression(expr, flat, radius_bins, angle_bins_deg, bin_size_um):
    """
    Expression-weighted, tie-aware polar histogram for one gene.

    Same numbers as the per-cell loop used before: each row's count is cast
    with ``int()``, split evenly over its tied vectors, normalized to sum 1,
    and binned with ``np.histogram2d`` (radius x angle). Vectors are taken in
    the same order, so the result is bit-identical.

    Returns None when no row with non-zero count has a vector.
    """
    n_vec, vec_row, angle, radii = flat
    e = np.asarray(expr).astype(np.int64)          # int(expr[i]) truncation
    use_row = (e != 0) & (n_vec > 0)
    sel = use_row[vec_row]
    if not sel.any():
        return None
    rows = vec_row[sel]
    w = e[rows] / n_vec[rows]
    w = w / w.sum()
    deg = np.degrees(angle[sel])
    counts, _, _ = np.histogram2d(radii[sel] * bin_size_um, deg, bins=[radius_bins, angle_bins_deg], weights=w)
    return counts
