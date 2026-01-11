import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Optional, Union
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go


def plot_3d(
    adata,
    genes: Union[str, Sequence[str]],                 # additional genes (1–2)
    anchor_gene: str = "CDH1",                        # anchor gene (always plotted)
    bin_size: int = 10,                               # grid size
    x_range: Optional[Tuple[float, float]] = None,    # (xmin, xmax)
    y_range: Optional[Tuple[float, float]] = None,    # (ymin, ymax)
    height_scale: float = 5.0,                        # height scale for genes
    anchor_scale: float = 1.0,                        # height scale for anchor_gene
                                                       # (set equal to height_scale to align heights)
    threshold: float = 4.0,                           # overlap threshold
                                                       # (compatible with original code: raw Z.values scale)
    sigma: float = 1.0,                               # Gaussian smoothing sigma
    reverse_x: bool = True,                           # reverse x-axis (image coordinates)
    colors: Optional[dict] = None,                    # optional color settings
):

    # ---- Normalize genes argument (limit to 1–2 genes) ----
    if isinstance(genes, str):
        genes = (genes,)
    genes = tuple(genes)
    if not (1 <= len(genes) <= 2):
        raise ValueError(
            "genes must contain 1 or 2 items, e.g. ('POSTN',) or ('POSTN', 'CD3E')"
        )

    # ---- Get indices from var_names (safe lookup) ----
    var_names = list(adata.var_names)

    def _idx(g):
        if g not in var_names:
            raise ValueError(f"Gene '{g}' not found in adata.var_names")
        return var_names.index(g)

    idx_anchor = _idx(anchor_gene)
    idxs = {g: _idx(g) for g in genes}

    # ---- Build DataFrame with coordinates and expression ----
    df = adata.obs.loc[:, ["imagecol", "imagerow"]].copy()

    X = adata.X

    def _col_as_1d(ix):
        col = X[:, ix]
        if hasattr(col, "toarray"):  # sparse matrix
            col = col.toarray()
        return np.asarray(col).reshape(-1)

    df[anchor_gene] = _col_as_1d(idx_anchor)
    for g, ix in idxs.items():
        df[g] = _col_as_1d(ix)

    # ---- Grid aggregation (bin_size units) ----
    df["grid_x"] = (df["imagecol"] // bin_size).astype(int)
    df["grid_y"] = (df["imagerow"] // bin_size).astype(int)

    agg_dict = {anchor_gene: "sum", **{g: "sum" for g in genes}}
    grid = (
        df.groupby(["grid_x", "grid_y"], as_index=False)
          .agg(agg_dict)
    )

    # Representative point (grid center)
    grid["imagecol"] = grid["grid_x"] * bin_size + bin_size / 2
    grid["imagerow"] = grid["grid_y"] * bin_size + bin_size / 2

    # ---- Range filtering (x_range / y_range) ----
    if x_range is not None:
        xmin, xmax = x_range
        grid = grid[(grid["imagecol"] >= xmin) & (grid["imagecol"] <= xmax)]
    if y_range is not None:
        ymin, ymax = y_range
        grid = grid[(grid["imagerow"] >= ymin) & (grid["imagerow"] <= ymax)]

    # ---- Normalization (avoid division by zero) ----
    def _norm100(s: pd.Series) -> pd.Series:
        m = float(s.max())
        return (s / m * 100.0) if m > 0 else s * 0.0

    for g in (anchor_gene,) + genes:
        grid[f"{g}_norm"] = _norm100(grid[g])

    # ---- Pivot table and smoothing ----
    def _pivot_and_smooth(val_col: str):
        Z = grid.pivot_table(
            index="imagerow",
            columns="imagecol",
            values=val_col,
            fill_value=0
        )
        x = Z.columns.values
        y = Z.index.values
        z = gaussian_filter(
            Z.values.astype(float),
            sigma=float(sigma),
            mode="nearest"
        )
        return x, y, Z.values.astype(float), z

    # Gene surfaces
    x, y, Z1_raw, z1 = _pivot_and_smooth(f"{genes[0]}_norm")
    z1_scaled = z1 * height_scale

    if len(genes) == 2:
        x2, y2, Z2_raw, z2 = _pivot_and_smooth(f"{genes[1]}_norm")
        if not (np.array_equal(x, x2) and np.array_equal(y, y2)):
            raise RuntimeError(
                "Pivot grids for the two genes do not align (x/y mismatch)"
            )
        z2_scaled = z2 * height_scale
    else:
        Z2_raw, z2, z2_scaled = None, None, None

    # Anchor gene surface
    xa, ya, Za_raw, za = _pivot_and_smooth(f"{anchor_gene}_norm")
    if not (np.array_equal(x, xa) and np.array_equal(y, ya)):
        raise RuntimeError(
            "Pivot grid for anchor gene does not align with genes (x/y mismatch)"
        )
    za_scaled = za * anchor_scale

    # ---- For legend: maximum z values after scaling ----
    z1_max_scaled = float(np.nanmax(z1_scaled))
    za_max_scaled = float(np.nanmax(za_scaled))

    if len(genes) == 2:
        z2_max_scaled = float(np.nanmax(z2_scaled))

    # ---- Overlap (only when two genes are provided) ----
    if len(genes) == 2:
        # Compatible with original code:
        # thresholding is applied to raw (pre-smoothed) Z.values
        mask = (Z1_raw > threshold) & (Z2_raw > threshold)
        z_overlap = np.where(
            mask,
            np.maximum(z1_scaled, z2_scaled),
            np.nan
        )
    else:
        z_overlap = None

    # ---- Color settings ----
    if colors is None:
        colors = {
            genes[0]: "Blues",
            (genes[1] if len(genes) == 2 else None): "Reds",
            anchor_gene: [[0, "#FFFF00"], [1, "#FFFF00"]],
            "overlap": [[0, "black"], [1, "black"]],
        }

    if len(genes) == 2:
        z_min_raw = min(np.nanmin(z1), np.nanmin(z2))
        z_max_raw = max(np.nanmax(z1), np.nanmax(z2))
    else:
        z_min_raw = np.nanmin(z1)
        z_max_raw = np.nanmax(z1)

    z_min_rounded = np.floor(z_min_raw / 10) * 10
    z_max_rounded = np.ceil(z_max_raw / 10) * 10

    tick_vals_original = np.arange(z_min_rounded, z_max_rounded + 1e-9, 10)
    tick_vals_scaled   = tick_vals_original * height_scale
    tick_text          = [f"{int(v)}" for v in tick_vals_original]

    # ---- Figure ----
    fig = go.Figure()

    return fig
