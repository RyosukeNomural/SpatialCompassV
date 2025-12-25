import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import cv2
from PIL import Image
import networkx as nx


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
    pos_series = sum([grid.to_df()[g] for g in pos_marker_ls])
    pos_series = pd.Series([1 if v >= 1 else 0 for v in pos_series], index=pos_series.index)

    # --- Subtract negative grid if provided ---
    if neg_marker_ls is not None:
        neg_series = sum([grid.to_df()[g] for g in neg_marker_ls])
        neg_series = pd.Series([1 if v >= 1 else 0 for v in neg_series], index=neg_series.index)
        pos_series = pos_series - neg_series
        pos_series = pd.Series([-1 if v == 0 else v for v in pos_series], index=pos_series.index)
    else:
        pos_series = pd.Series([-1 if v == 0 else v for v in pos_series], index=pos_series.index)

    # --- Expand to all grids ---
    pos_series.name = "tumor_grid"
    df_grid_tumor = (
        pd.merge(
            pd.DataFrame(index=[f"grid_{i+1}" for i in range(N_ROW * N_COL)]),
            pos_series,
            right_index=True,
            left_index=True,
            how="left",
        )
        .fillna(-1)
    )

    # =========================================================
    # Visualize tumor grid (for contour extraction)
    # FIX:
    # - Avoid dpi=1 (causes invalid ppem in freetype)
    # - Avoid tight_layout=True (forces text bbox calculations)
    # - Keep output pixel size == (N_COL, N_ROW)
    # =========================================================
    dpi = 100  # safe DPI
    fig_w = N_COL / dpi
    fig_h = N_ROW / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)  # tight_layout removed

    cmap = plt.cm.viridis
    cmap.set_bad("black", 1.0)
    cmap.set_under(color="black")

    ax.imshow(
        np.array(df_grid_tumor["tumor_grid"]).reshape(N_COL, N_ROW).T,
        cmap=cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)  # no margins

    # Convert figure to numpy image (RGBA -> BGR)
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

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

    # --- Convert image to a graph (pixel adjacency) ---
    image = cv2pil(img_color_with_contours)
    width, height = image.size
    graph = nx.Graph()

    tumor_contour_pixel_ls = []
    tumor_pixel_ls = []
    edge_pixel_ls = []

    for y in range(height):
        for x in range(width):
            pixel_value = image.getpixel((x, y))
            graph.add_node((x, y), color=pixel_value)

            # Tumor contour pixels (green)
            if pixel_value == (0, 255, 0):
                tumor_contour_pixel_ls.append((x, y))
            # Tumor pixels (viridis yellow-ish in rendered image)
            elif pixel_value == (253, 231, 36):
                tumor_pixel_ls.append((x, y))

            if (y == 0) or (y == height - 1) or (x == 0) or (x == width - 1):
                edge_pixel_ls.append((x, y))

    nodes_ls = list(graph.nodes)

    # Determine nodes inside contour polygons
    inside_contour_ls = []
    for node in nodes_ls:
        for c in contours_:
            if cv2.pointPolygonTest(c, node, False) == 1:
                inside_contour_ls.append(node)
                break

    tumor_pixel_ls = [p for p in tumor_pixel_ls if p in inside_contour_ls]

    # Add edges (4-neighborhood + diagonals)
    for y in range(height):
        for x in range(width):
            current_node = (x, y)
            neighbors = [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]
            for nb in neighbors:
                if nb in graph.nodes:
                    graph.add_edge(current_node, nb)

            # Diagonals with Euclidean weights
            if (x + 1) < width and (y + 1) < height:
                node1 = (x, y)
                node2 = (x + 1, y + 1)
                d = math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)
                graph.add_edge(node1, node2, weight=d)

                node1 = (x + 1, y)
                node2 = (x, y + 1)
                d = math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)
                graph.add_edge(node1, node2, weight=d)

    # Remove contours that touch image edges (edge artifacts)
    tumor_edge_contour_pixel_ls = list(set(tumor_contour_pixel_ls) & set(edge_pixel_ls))
    tumor_pixel_ls = list(set(tumor_pixel_ls) | set(tumor_edge_contour_pixel_ls))
    tumor_contour_pixel_ls = list(set(tumor_contour_pixel_ls) - set(tumor_edge_contour_pixel_ls))

    # --- Compute shortest distances from tumor surface ---
    shortest_paths = nx.multi_source_dijkstra(
        graph,
        tumor_contour_pixel_ls,
        cutoff=31,
        weight="weight",
    )

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
    col_ls = []
    for v in df_shotest["euclidean_round"]:
        if v == 0:
            col_ls.append((0, 255, 0))
        elif v < 0:
            col_ls.append((0, 255, 255))
        else:
            col_ls.append((0, 0, 0))

    col_arr = np.array(col_ls, dtype=np.uint8).reshape(N_ROW, N_COL, 3)
    grid.uns["marker_median_delineation"] = col_arr.copy()

    # --- Delineation at 30 µm intervals ---
    col_ls = []
    for v in df_shotest["euclidean_round"]:
        if v == 0:
            col_ls.append((0, 255, 0))
        elif (v % 3 == 0) and (v > 0):
            col_ls.append((0, 0, 255))
        elif (v % 3 == 0) and (v < 0):
            col_ls.append((0, 150, 255))
        elif v < 0:
            col_ls.append((0, 255, 255))
        else:
            col_ls.append((0, 0, 0))

    col_arr = np.array(col_ls, dtype=np.uint8).reshape(N_ROW, N_COL, 3)
    grid.uns["shotest_30_delineation"] = col_arr.copy()

    # --- Discretize distance into regions ---
    df_shotest["region"] = pd.cut(
        df_shotest.dropna()["euclidean"],
        bins=list(range(-15, 31, 3)),
    )

    region_counts = df_shotest.dropna().region.value_counts()
    for region_bin in region_counts[region_counts > 100].sort_index().index:
        col_ls = []
        for s in df_shotest["region"]:
            col_ls.append((255, 255, 255) if s == region_bin else (0, 0, 0))

        col_arr = np.array(col_ls, dtype=np.uint8).reshape(N_ROW, N_COL, 3)
        grid.uns[f"shotest_region_{region_bin}_delineation"] = col_arr

    setattr(grid, "shortest", df_shotest)

    return grid
