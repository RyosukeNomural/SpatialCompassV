import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cv2
from PIL import Image
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


# =========================================================
# Utilities
# =========================================================
class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)

    def __call__(self, value, clip=None):
        x = [self.vmin, self.midpoint, self.vmax]
        y = [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def cv2pil(image):
    """Convert an OpenCV image (BGR/BGRA) to a PIL Image (RGB/RGBA)."""
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(new_image)


# =========================================================
# Main
# =========================================================
def calculate_distance(grid, pos_marker_ls, neg_marker_ls=None):
    """
    Compute signed shortest-path distances from the tumor contour on a binned grid.
    (Fix: avoid dpi=1 + tight_layout causing invalid ppem font error)
    """

    # --- Get grid shape ---
    N_ROW = len(grid.uns["grid_yedges"]) - 1
    N_COL = len(grid.uns["grid_xedges"]) - 1

    # --- Define positive grid (binary) ---
    def _marker_sum(genes):
        return sum([pd.Series(grid.obs_vector(g), index=grid.obs_names) for g in genes])

    def _binary(series):
        return pd.Series(np.where(series.to_numpy() >= 1, 1, 0).astype(np.int64), index=series.index)

    def _zero_to_minus_one(series):
        v = series.to_numpy()
        return pd.Series(np.where(v == 0, -1, v).astype(v.dtype), index=series.index)

    pos_series = _binary(_marker_sum(pos_marker_ls))

    # --- Subtract negative grid if provided ---
    if neg_marker_ls is not None:
        neg_series = _binary(_marker_sum(neg_marker_ls))
        pos_series = _zero_to_minus_one(pos_series - neg_series)
    else:
        pos_series = _zero_to_minus_one(pos_series)

    # --- Expand to all grids ---
    pos_series.name = "tumor_grid"
    # square names in column-major order; stlearn numbers them from 0 (older
    # versions from 1) -- detected from the square centres, see grid_index.py
    from scomv.preparation.grid_index import grid_index_names
    expected_index = grid_index_names(grid)
    if set(expected_index) & set(pos_series.index):
        # obs names match "grid_N" format — use index-based merge
        df_grid_tumor = (
            pd.merge(
                pd.DataFrame(index=expected_index),
                pos_series,
                right_index=True,
                left_index=True,
                how="left",
            )
            .fillna(-1)
        )
    else:
        # obs names don't match "grid_N" — assume obs are in spatial order
        if len(pos_series) == N_ROW * N_COL:
            df_grid_tumor = pd.DataFrame(
                {"tumor_grid": pos_series.values},
                index=expected_index,
            )
        else:
            raise ValueError(
                f"grid.obs index doesn't match 'grid_N' format and len(obs)={len(pos_series)} "
                f"!= N_ROW*N_COL={N_ROW * N_COL}. "
                f"Sample obs names: {list(pos_series.index[:5])}"
            )
    n_positive = int((df_grid_tumor["tumor_grid"] > 0).sum())
    print(f"[calculate_distance] N_ROW={N_ROW}, N_COL={N_COL}, "
          f"positive cells in grid={n_positive}/{len(df_grid_tumor)}")

    # =========================================================
    # Visualize tumor grid (for contour extraction)
    # FIX:
    # - Avoid dpi=1 (causes invalid ppem in freetype)
    # - Avoid tight_layout=True (forces text bbox calculations)
    # - Keep output pixel size == (N_COL, N_ROW)
    # =========================================================
    # The image is the viridis colour of each grid square (vmin=0, vmax=1,
    # values < 0 / NaN black), one pixel per square. It used to be rendered
    # through a matplotlib figure of size N/100 inch at 100 dpi; that is only
    # pixel-exact when N/100*100 rounds back to N (e.g. 987/100*100 =
    # 986.99... gives a 986-px Agg canvas, and the resize back to 987 px
    # shifted the whole mask by one square; HiDPI backends differ again).
    # Mapping the colours directly gives the same image as an exact render,
    # independent of the matplotlib backend.
    tumor_img = np.array(df_grid_tumor["tumor_grid"], dtype=float).reshape(N_COL, N_ROW).T
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("black", 1.0)
    cmap.set_under(color="black")
    rgba = cmap(mcolors.Normalize(vmin=0, vmax=1)(np.ma.masked_invalid(tumor_img)), bytes=True)
    img = cv2.cvtColor(np.ascontiguousarray(rgba), cv2.COLOR_RGBA2BGR)

    grid.uns["marker"] = img.copy()

    # --- Median filter ---
    img_med = cv2.medianBlur(img, ksize=3)
    grid.uns["marker_median"] = img_med.copy()

    # --- Binarize and extract contours ---
    threshold = 90
    img_gray = cv2.cvtColor(img_med, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gray, (3, 3))
    _, img_binary = cv2.threshold(img_blur, threshold, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_ = []
    if hierarchy is not None and len(hierarchy) > 0:
        for c, _h in zip(contours, hierarchy[0]):
            contours_.append(c)
    else:
        contours_ = contours  # fallback

    img_color_with_contours = cv2.drawContours(img_med.copy(), contours_, -1, (0, 255, 0), 1)
    grid.uns["marker_delineation"] = img_color_with_contours

    # --- Pixel graph (8-neighborhood: 1 for edge neighbors, sqrt(2) for diagonals) ---
    image = cv2pil(img_color_with_contours)
    width, height = image.size
    rgb = np.asarray(image.convert("RGB"))  # (height, width, 3)

    ys, xs = np.mgrid[0:height, 0:width]
    xs, ys = xs.ravel(), ys.ravel()
    nodes_ls = list(zip(xs.tolist(), ys.tolist()))  # row-major, (x, y)

    flat_rgb = rgb.reshape(-1, 3)
    is_contour = np.all(flat_rgb == (0, 255, 0), axis=1)          # tumor contour pixels (green)
    is_tumor = np.all(flat_rgb == (253, 231, 36), axis=1) & ~is_contour  # tumor pixels (viridis yellow)
    is_edge = (ys == 0) | (ys == height - 1) | (xs == 0) | (xs == width - 1)

    # Tumor pixels must lie strictly inside a contour polygon (pointPolygonTest == 1).
    # Only tumor pixels need the test, and only those inside a contour's bounding box.
    tumor_idx = np.flatnonzero(is_tumor)
    pending = np.ones(len(tumor_idx), dtype=bool)
    tx, ty = xs[tumor_idx], ys[tumor_idx]
    for c in contours_:
        bx, by, bw, bh = cv2.boundingRect(c)
        cand = np.flatnonzero(pending & (tx >= bx) & (tx < bx + bw) & (ty >= by) & (ty < by + bh))
        for k in cand:
            if cv2.pointPolygonTest(c, (int(tx[k]), int(ty[k])), False) == 1:
                pending[k] = False
    tumor_inside_idx = tumor_idx[~pending]

    # Remove contours that touch image edges (edge artifacts)
    edge_contour_idx = np.flatnonzero(is_contour & is_edge)
    tumor_set_idx = np.union1d(tumor_inside_idx, edge_contour_idx)
    source_idx = np.flatnonzero(is_contour & ~is_edge)
    # Fallback: if all contour pixels were at image edges, use them as sources anyway
    if len(source_idx) == 0:
        source_idx = edge_contour_idx

    idx = np.arange(height * width).reshape(height, width)
    sq2 = math.sqrt(2)
    pairs = [
        (idx[:, :-1], idx[:, 1:], 1.0),      # horizontal
        (idx[:-1, :], idx[1:, :], 1.0),      # vertical
        (idx[:-1, :-1], idx[1:, 1:], sq2),   # diagonal (x, y) - (x+1, y+1)
        (idx[:-1, 1:], idx[1:, :-1], sq2),   # diagonal (x+1, y) - (x, y+1)
    ]
    rows = np.concatenate([a.ravel() for a, _, _ in pairs])
    cols = np.concatenate([b.ravel() for _, b, _ in pairs])
    wts = np.concatenate([np.full(a.size, w) for a, _, w in pairs])
    graph = sparse.csr_matrix((wts, (rows, cols)), shape=(height * width, height * width))

    # --- Compute shortest distances from tumor surface (cutoff 31 bins) ---
    CUTOFF = 31
    dist = dijkstra(graph, directed=False, indices=source_idx, min_only=True, limit=CUTOFF + 1e-6)
    reached = np.isfinite(dist) & (dist <= CUTOFF)
    shortest_paths = ({nodes_ls[i]: float(dist[i]) for i in np.flatnonzero(reached)},)
    tumor_pixel_ls = [nodes_ls[i] for i in tumor_set_idx]

    df_shotest = pd.DataFrame.from_dict(shortest_paths[0], orient="index", columns=["euclidean"])

    inside_mask = df_shotest.index.isin(tumor_pixel_ls)
    df_shotest.loc[df_shotest.index[inside_mask]] *= -1

    df_shotest["euclidean_round"] = df_shotest["euclidean"].round(0)

    df_nodes = pd.DataFrame(index=nodes_ls)
    df_shotest = (
        pd.merge(df_nodes, df_shotest, right_index=True, left_index=True, how="left")
        .fillna(np.nan)
    )

    # --- (Safer than plt.gcf): create a dedicated debug image instead of reusing current fig ---
    # If you truly need grid.uns["shotest"], render df_shotest into an image deterministically.
    # Here we keep a minimal blank placeholder (no crash).
    grid.uns["shotest"] = None

    # --- Delineation (distance classes) ---
    v = df_shotest["euclidean_round"].to_numpy()

    def _paint(conds, colors_):
        out = np.zeros((len(v), 3), dtype=np.uint8)  # default black
        done = np.zeros(len(v), dtype=bool)
        for cnd, col in zip(conds, colors_):
            m = cnd & ~done
            out[m] = col
            done |= cnd
        return out.reshape(N_ROW, N_COL, 3)

    with np.errstate(invalid="ignore"):
        col_arr = _paint([v == 0, v < 0], [(0, 255, 0), (0, 255, 255)])
    grid.uns["marker_median_delineation"] = col_arr.copy()

    # --- Delineation at 30 µm intervals ---
    with np.errstate(invalid="ignore"):
        m3 = np.mod(v, 3) == 0
        col_arr = _paint(
            [v == 0, m3 & (v > 0), m3 & (v < 0), v < 0],
            [(0, 255, 0), (0, 0, 255), (0, 150, 255), (0, 255, 255)],
        )
    grid.uns["shotest_30_delineation"] = col_arr.copy()

    # --- Discretize distance into regions ---
    df_shotest["region"] = pd.cut(
        df_shotest.dropna()["euclidean"],
        bins=list(range(-15, 31, 3)),
    )

    region_counts = df_shotest.dropna().region.value_counts()
    for region_bin in region_counts[region_counts > 100].sort_index().index:
        m = (df_shotest["region"] == region_bin).to_numpy(dtype=bool, na_value=False)
        col_arr = np.zeros((len(m), 3), dtype=np.uint8)
        col_arr[m] = (255, 255, 255)
        col_arr = col_arr.reshape(N_ROW, N_COL, 3)
        grid.uns[f"shotest_region_{region_bin}_delineation"] = col_arr

    setattr(grid, "shortest", df_shotest)

    return grid
