import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Optional, Union
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go


def plot_3d(
    adata,
    genes: Union[str, Sequence[str]],                 # Additional genes to plot (1–2)
    anchor_gene: str = "CDH1",                        # Anchor gene (always plotted)
    bin_size: int = 10,                               # Spatial grid size
    x_range: Optional[Tuple[float, float]] = None,    # X-axis range (xmin, xmax)
    y_range: Optional[Tuple[float, float]] = None,    # Y-axis range (ymin, ymax)
    height_scale: float = 5.0,                        # Height scaling factor for genes
    anchor_scale: float = 1.0,                        # Height scaling factor for anchor gene
    threshold: float = 4.0,                           # Overlap threshold (raw pivot scale)
    sigma: float = 1.0,                               # Gaussian smoothing sigma
    reverse_x: bool = True,                           # Reverse x-axis (image coordinate system)
    colors: Optional[dict] = None,                    # Optional colorscale settings
    agg: str = "mean",                                # Aggregation method ("mean" recommended)
    interpolate_missing: bool = True,                 # Interpolate missing grid values
    smooth_mode: str = "reflect",                     # Boundary handling mode for smoothing
):
    # ---- Normalize genes argument (limit to 1–2 genes) ----
    if isinstance(genes, str):
        genes = (genes,)
    genes = tuple(genes)
    if not (1 <= len(genes) <= 2):
        raise ValueError(
            "genes must contain 1 or 2 items, e.g. ('POSTN',) or ('POSTN', 'CD3E')"
        )

    # ---- Safe lookup of gene indices in adata.var_names ----
    var_names = list(adata.var_names)

    def _idx(g):
        if g not in var_names:
            raise ValueError(f"Gene '{g}' not found in adata.var_names")
        return var_names.index(g)

    idx_anchor = _idx(anchor_gene)
    idxs = {g: _idx(g) for g in genes}

    # ---- Build DataFrame with spatial coordinates and gene expression ----
    df = adata.obs.loc[:, ["imagecol", "imagerow"]].copy()
    X = adata.X

    def _col_as_1d(ix):
        col = X[:, ix]
        if hasattr(col, "toarray"):  # Handle sparse matrices
            col = col.toarray()
        return np.asarray(col).reshape(-1)

    df[anchor_gene] = _col_as_1d(idx_anchor)
    for g, ix in idxs.items():
        df[g] = _col_as_1d(ix)

    # ---- Spatial grid aggregation ----
    df["grid_x"] = (df["imagecol"] // bin_size).astype(int)
    df["grid_y"] = (df["imagerow"] // bin_size).astype(int)

    if agg not in ("sum", "mean"):
        raise ValueError("agg must be 'sum' or 'mean'")

    agg_dict = {anchor_gene: agg, **{g: agg for g in genes}}
    grid = df.groupby(["grid_x", "grid_y"], as_index=False).agg(agg_dict)

    # Define representative grid coordinates (grid center)
    grid["imagecol"] = grid["grid_x"] * bin_size + bin_size / 2
    grid["imagerow"] = grid["grid_y"] * bin_size + bin_size / 2

    # ---- Apply spatial range filtering ----
    if x_range is not None:
        xmin, xmax = x_range
        grid = grid[(grid["imagecol"] >= xmin) & (grid["imagecol"] <= xmax)]
    if y_range is not None:
        ymin, ymax = y_range
        grid = grid[(grid["imagerow"] >= ymin) & (grid["imagerow"] <= ymax)]

    if grid.shape[0] == 0:
        raise ValueError(
            "Grid became empty after x_range/y_range filtering. "
            "Check ranges or bin_size."
        )

    # ---- Normalize expression values to 0–100 ----
    def _norm100(s: pd.Series) -> pd.Series:
        m = float(s.max())
        return (s / m * 100.0) if m > 0 else s * 0.0

    for g in (anchor_gene,) + genes:
        grid[f"{g}_norm"] = _norm100(grid[g])

    # ---- Pivot table, optional interpolation, and smoothing ----
    def _pivot_and_smooth(val_col: str):
        # Keep NaNs (do not fill with zero)
        Z = grid.pivot_table(
            index="imagerow",
            columns="imagecol",
            values=val_col,
            aggfunc="mean"
        )
        x = Z.columns.values
        y = Z.index.values
        Zv = Z.values.astype(float)

        if interpolate_missing:
            # Interpolate missing grid values in both directions
            Zf = (
                pd.DataFrame(Zv)
                .interpolate(axis=1, limit_direction="both")
                .interpolate(axis=0, limit_direction="both")
                .to_numpy()
            )
        else:
            # Replace NaNs with zero (may introduce sharp edges)
            Zf = np.nan_to_num(Zv, nan=0.0)

        z = gaussian_filter(Zf, sigma=float(sigma), mode=str(smooth_mode))
        return x, y, Zv, z

    # ---- Gene surfaces ----
    x, y, Z1_raw, z1 = _pivot_and_smooth(f"{genes[0]}_norm")
    z1_scaled = z1 * float(height_scale)

    if len(genes) == 2:
        x2, y2, Z2_raw, z2 = _pivot_and_smooth(f"{genes[1]}_norm")
        if not (np.array_equal(x, x2) and np.array_equal(y, y2)):
            raise RuntimeError("Pivot grids for the two genes do not align")
        z2_scaled = z2 * float(height_scale)
    else:
        Z2_raw, z2_scaled = None, None

    # ---- Anchor gene surface ----
    xa, ya, Za_raw, za = _pivot_and_smooth(f"{anchor_gene}_norm")
    if not (np.array_equal(x, xa) and np.array_equal(y, ya)):
        raise RuntimeError("Pivot grid for anchor gene does not align with genes")
    za_scaled = za * float(anchor_scale)

    # ---- Overlap region (based on raw pivot values) ----
    if len(genes) == 2:
        Z1_cmp = np.nan_to_num(Z1_raw, nan=0.0)
        Z2_cmp = np.nan_to_num(Z2_raw, nan=0.0)
        mask = (Z1_cmp > threshold) & (Z2_cmp > threshold)
        z_overlap = np.where(mask, np.maximum(z1_scaled, z2_scaled), np.nan)
    else:
        z_overlap = None

    # ---- Z-axis ticks (display original scale) ----
    if len(genes) == 2:
        z_min_raw = min(np.nanmin(z1), np.nanmin(z2))
        z_max_raw = max(np.nanmax(z1), np.nanmax(z2))
    else:
        z_min_raw = np.nanmin(z1)
        z_max_raw = np.nanmax(z1)

    z_min_rounded = np.floor(z_min_raw / 10) * 10
    z_max_rounded = np.ceil(z_max_raw / 10) * 10
    tick_vals_original = np.arange(z_min_rounded, z_max_rounded + 1e-9, 10)
    tick_vals_scaled = tick_vals_original * float(height_scale)
    tick_text = [f"{int(v)}" for v in tick_vals_original]

    # ---- Colors ----
    if colors is None:
        colors = {
            genes[0]: "Blues",
            genes[1] if len(genes) == 2 else "gene2": "Reds",
            anchor_gene: [[0, "#FFFF00"], [1, "#FFFF00"]],
            "overlap": [[0, "black"], [1, "black"]],
        }

    cs1 = colors.get(genes[0], "Blues")
    cs2 = colors.get(genes[1], "Reds") if len(genes) == 2 else None
    cs_anchor = colors.get(anchor_gene, [[0, "#FFFF00"], [1, "#FFFF00"]])
    cs_overlap = colors.get("overlap", [[0, "black"], [1, "black"]])

    # ---- Figure ----
    fig = go.Figure()
    Xg, Yg = np.meshgrid(x, y)

    # Gene 1 surface
    fig.add_trace(go.Surface(
        x=Xg, y=Yg, z=z1_scaled,
        colorscale=cs1,
        cmin=0, cmax=max(1e-9, float(np.nanmax(z1_scaled))),
        showscale=False,
        opacity=1.0,
        name=str(genes[0]),
    ))

    # Gene 2 surface
    if len(genes) == 2:
        fig.add_trace(go.Surface(
            x=Xg, y=Yg, z=z2_scaled,
            colorscale=cs2,
            cmin=0, cmax=max(1e-9, float(np.nanmax(z2_scaled))),
            showscale=False,
            opacity=0.9,
            name=str(genes[1]),
        ))

    # Anchor gene surface
    fig.add_trace(go.Surface(
        x=Xg, y=Yg, z=za_scaled,
        colorscale=cs_anchor,
        cmin=0, cmax=max(1e-9, float(np.nanmax(za_scaled))),
        showscale=False,
        opacity=0.35,
        name=str(anchor_gene),
    ))

    # Overlap surface
    if z_overlap is not None:
        fig.add_trace(go.Surface(
            x=Xg, y=Yg, z=z_overlap,
            colorscale=cs_overlap,
            showscale=False,
            opacity=0.95,
            name="overlap",
        ))

    fig.update_layout(
        title=f"{genes[0]}" + (f" & {genes[1]}" if len(genes) == 2 else "") + " (3D)",
        scene=dict(
            xaxis=dict(title="imagecol"),
            yaxis=dict(title="imagerow"),
            zaxis=dict(
                title="expression",
                tickvals=tick_vals_scaled,
                ticktext=tick_text,
            ),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    if reverse_x:
        fig.update_layout(scene=dict(xaxis=dict(autorange="reversed")))

    return fig

